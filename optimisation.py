#!/usr/bin/env python3
"""Estimate MAYO costs with structure, multi-target reduction, both, or neither.

    python3 optimisation.py 86 78 16 10
    python3 optimisation.py 86 78 16 10 --generic --fixed-p 8
    python3 optimisation.py 88 80 16 10 --round3

Costs use the bundled CryptographicEstimators and a local persistent cache.
If the current Python lacks dependencies, the CLI retries with a working
adjacent .venv or venv. It never installs packages automatically. Each MQ cost
is minimized over all applicable algorithms exposed by MQEstimator. Tiny
generic systems also retain their
reference elementary cost as a candidate.

Round 2 structural modes use w = u + r.

--unstructured selects the emulsifier bounds from Section 3.1; otherwise the
structured bounds from Section 3.2 are used. Generic modes use the inclusive
r=0 pseudo-oil feasibility bounds and retain the specialization costs.
Centrifugation is paid once: there is no additional inverse sweep-success factor.

--round3 is a separate structural, single-target mode over GF(16). It uses
unstructured emulsifiers and one full-copy base vector in ker(Lambda). Dimension
bounds are treated as equalities: u=2*kappa+a, r=kappa*a-a*(a-1)/2, and
w=min(m,r+3*kappa+2*a). Linear terms enlarge W, not the polar-image space R. 

The positional kappa is the whipping factor. The optimized k is the paper's
additional guessing parameter: structural Reduced work uses q^(k+h-u)
assignments, whereas the endpoint is reached q^(h-u) times in expectation.
The parameters k, b0 and ell are nonnegative, with b0 = m-d-h-k-B1 in
structural modes and b0 = m-d-k-B1 in generic modes, and 0 <= ell <= b0.
Equality in the dimension bounds is allowed. Combined and multi-target-only
modes include the single-target baseline d=0 and prefer it on exact cost ties.
The positive total guessing-exponent requirement applies only when d>0.
Here h, r and w refer to the active codomain; with d>0 in combined mode these
are the paper's projected h_T, r_T and w_T. NH denotes the lower-bound model
for N_H.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

NEG_INF = -math.inf
DEFAULT_OMEGA = 2.81
DEFAULT_THETA = 2.0
MODE_NAMES = {
    "combined": "Structure + multi-target",
    "multi-target-only": "Multi-target only",
    "structural-only": "Structure only",
    "generic": "Generic",
    "round3": "Round 3 structure (single target)",
}


def log2sum(values: Iterable[float]) -> float:
    """Stable ``log2(sum(2**x for x in values))``."""

    finite = [value for value in values if value != NEG_INF]
    if not finite:
        return NEG_INF
    top = max(finite)
    return top + math.log2(sum(2.0 ** (value - top) for value in finite))


def _json_safe(value: Any) -> Any:
    """Convert estimator metadata and infinities to strict JSON values."""

    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [_json_safe(item) for item in value]
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        try:
            return float(value)
        except (TypeError, ValueError, OverflowError):
            return repr(value)


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0

def _estimator_revision(root: Path) -> str:
    """Return the vendored estimator revision for cache invalidation."""

    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        marker = root / "pyproject.toml"
        if marker.exists():
            return hashlib.sha256(marker.read_bytes()).hexdigest()
        return "unknown"


@dataclass(frozen=True)
class MQCost:
    variables: int
    equations: int
    q: int
    time_log2: float
    memory_log2: float
    algorithm: str
    parameters: Mapping[str, Any]


class EstimatorDependencyError(RuntimeError):
    """The active Python cannot import the bundled estimator's dependencies."""


class MQOracle:
    """Robust, cached wrapper around the bundled MQEstimator.

    Algorithms are queried separately.  A failure in one estimator therefore
    cannot abort all of the other candidates, which happens for some tiny
    shapes in the vendored checkout.
    """

    def __init__(
        self,
        directory: Path,
        omega: float,
        theta: float,
    ) -> None:
        self.estimator_root = directory / "CryptographicEstimators"
        self.omega = float(omega)
        self.theta = float(theta)
        self.bit_complexities = True
        self.cache_path = directory / ".optimisation_cache.json"
        self.revision = _estimator_revision(self.estimator_root)
        self._cache: dict[str, dict[str, Any]] = {}
        self._dirty = False
        self._session: dict[str, MQCost] = {}
        self._MQEstimator = None
        if self.cache_path.exists():
            try:
                loaded = json.loads(self.cache_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    self._cache = loaded
            except (OSError, json.JSONDecodeError):
                # A stale or interrupted cache is an optimization loss only.
                self._cache = {}

    def _load_library(self) -> None:
        if self._MQEstimator is not None:
            return
        sys.path.insert(0, str(self.estimator_root))
        try:
            from cryptographic_estimators.MQEstimator import MQEstimator
        except ImportError as exc:
            raise EstimatorDependencyError(
                f"Estimator dependency unavailable: {exc}.\n"
                "In the script directory, create or repair a local environment:\n"
                "  python3 -m venv .venv\n"
                "  .venv/bin/python -m pip install ./CryptographicEstimators"
            ) from exc
        self._MQEstimator = MQEstimator

    def _key(self, variables: int, equations: int, q: int) -> tuple[str, dict[str, Any]]:
        payload = {
            "schema": 3,
            "revision": self.revision,
            "variables": variables,
            "equations": equations,
            "q": q,
            "algorithm_policy": "all-applicable-mq-estimator-v1",
            "omega": self.omega,
            "theta": self.theta,
            "bit_complexities": self.bit_complexities,
            "complexity_type": 0,
            "nsolutions": 0,
            "small_instance_policy": "library-first-with-tiny-fallback-v3",
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest(), payload

    def cost(self, variables: int, equations: int, q: int) -> MQCost:
        if variables < 0 or equations < 0:
            raise ValueError("MQ dimensions must be nonnegative")
        key, payload = self._key(variables, equations, q)
        if key in self._session:
            return self._session[key]
        if key in self._cache:
            row = self._cache[key]
            try:
                result = MQCost(
                    variables=variables,
                    equations=equations,
                    q=q,
                    time_log2=float(row["time_log2"]),
                    memory_log2=float(row["memory_log2"]),
                    algorithm=str(row["algorithm"]),
                    parameters=row.get("parameters", {}),
                )
                if not (
                    math.isfinite(result.time_log2)
                    and math.isfinite(result.memory_log2)
                ):
                    raise ValueError("non-finite cached cost")
            except (KeyError, TypeError, ValueError):
                # Ignore one malformed/stale row without discarding the rest of
                # an otherwise useful persistent cache.
                self._cache.pop(key, None)
            else:
                self._session[key] = result
                return result

        # Zero-variable systems are outside MQEstimator's domain. For one
        # variable, query the library first and retain this explicit fallback
        # only if every applicable algorithm fails to give a finite estimate.
        fallback: MQCost | None = None
        if variables <= 1:
            extension_degree = math.log2(q)
            bit_conversion = (
                self.theta * math.log2(max(1.0, extension_degree))
                if self.bit_complexities
                else 0.0
            )
            time = (
                variables * extension_degree
                + math.log2(max(1, equations))
                + bit_conversion
            )
            fallback = MQCost(
                variables,
                equations,
                q,
                time,
                max(
                    0.0,
                    variables * extension_degree
                    + (
                        math.log2(max(1.0, extension_degree))
                        if self.bit_complexities
                        else 0.0
                    ),
                ),
                "small-instance exhaustive fallback",
                {},
            )
            if variables == 0 or equations == 0:
                self._remember(key, payload, fallback)
                return fallback

        self._load_library()
        assert self._MQEstimator is not None
        estimator = self._MQEstimator(
            n=variables,
            m=equations,
            q=q,
            w=self.omega,
            theta=self.theta,
            bit_complexities=int(self.bit_complexities),
            complexity_type=0,
            nsolutions=0,
            excluded_algorithms=[],
        )
        candidates: list[tuple[float, float, str, Mapping[str, Any]]] = []
        failures: list[str] = []
        for algorithm in estimator.algorithms():
            name = type(algorithm).__name__
            try:
                time = float(algorithm.time_complexity())
                memory = float(algorithm.memory_complexity())
                if not (math.isfinite(time) and math.isfinite(memory)):
                    continue
                parameters = _json_safe(algorithm.get_optimal_parameters_dict())
                candidates.append((time, memory, name, parameters))
            except Exception as exc:
                # Numerical estimators have different unsupported cases.
                # Isolate failures here without swallowing user interrupts.
                failures.append(f"{name}: {type(exc).__name__}: {exc}")
        if not candidates:
            if fallback is not None:
                self._remember(key, payload, fallback)
                return fallback
            detail = "; ".join(failures[:4])
            raise RuntimeError(
                f"No finite MQ estimate for ({variables},{equations},q={q}) "
                f"among all applicable MQEstimator algorithms. {detail}"
            )
        time, memory, name, parameters = min(
            candidates, key=lambda row: (row[0], row[1], row[2])
        )
        result = MQCost(variables, equations, q, time, memory, name, parameters)
        self._remember(key, payload, result)
        return result

    def _remember(self, key: str, payload: Mapping[str, Any], result: MQCost) -> None:
        self._session[key] = result
        self._cache[key] = {
            "input": _json_safe(payload),
            "time_log2": result.time_log2,
            "memory_log2": result.memory_log2,
            "algorithm": result.algorithm,
            "parameters": _json_safe(result.parameters),
        }
        self._dirty = True

    def flush(self) -> None:
        if not self._dirty:
            return
        temporary_name: str | None = None
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            rendered = json.dumps(self._cache, indent=2, sort_keys=True) + "\n"
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=self.cache_path.name + ".", dir=self.cache_path.parent
            )
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(rendered)
            os.replace(temporary_name, self.cache_path)
        except OSError:
            # Persistence is only a speed optimization.  Read-only folders,
            # sandboxed runs, or a full disk must not abort a completed search.
            return
        finally:
            if temporary_name is not None:
                try:
                    os.unlink(temporary_name)
                except OSError:
                    pass
        self._dirty = False


@dataclass(frozen=True)
class DPNode:
    log_cost: float
    blocks: tuple[int, ...]


@dataclass(frozen=True)
class GenericScheduleResult:
    blocks: tuple[int, ...]
    pair_sum: int
    optimized_log_cost: float
    polynomial_per_guess_log2: float


@dataclass(frozen=True)
class RawScheduleNode:
    pair_sum: int
    square_log2: float
    centrifugation_log2: float
    blocks: tuple[int, ...]


ScheduleFrontiers = dict[tuple[int, int], tuple[RawScheduleNode, ...]]


@dataclass(frozen=True)
class EndpointCosts:
    """Expected log2 stage costs per Reduced solution, including early abort."""

    ov: float
    square_only: float
    finishing: float

    @property
    def total_log2(self) -> float:
        return log2sum((self.ov, self.square_only, self.finishing))


@dataclass(frozen=True)
class Result:
    d: int
    p: int
    k: int
    a: int
    B1: int
    b0: int
    ell: int
    h: int
    blocks: tuple[int, ...]
    centrifugation: float
    square: float
    reduced_terminal: float
    endpoint: float
    specialization: float
    per_target: float
    target_factor: float
    total: float
    endpoint_costs: EndpointCosts | None = None

    def key(self) -> tuple:
        # Prefer the single-target baseline when projection gives no cost
        # improvement, then retain the established deterministic tie-breaks.
        return (
            self.total, self.d != 0,
            self.per_target, self.square, self.centrifugation,
            self.d, self.p, self.k, self.a, self.B1, self.b0, self.ell, self.h,
            self.blocks,
        )


def _pareto_prune_raw(
    nodes: Iterable[RawScheduleNode],
) -> tuple[RawScheduleNode, ...]:
    """Exact 3D skyline in (pair, Square cost, Centrifugation cost).

    Nodes are sorted by pair and Square cost.  A Fenwick tree stores the
    smallest Centrifugation cost seen at every Square-cost prefix, so each
    dominance test/update costs O(log L), not O(L).  Equal triples retain the
    lexicographically first ordered block schedule.
    """

    ordered = sorted(
        nodes,
        key=lambda node: (
            node.pair_sum,
            node.square_log2,
            node.centrifugation_log2,
            node.blocks,
        ),
    )
    if not ordered:
        return ()
    square_values = sorted({node.square_log2 for node in ordered})
    tree = [math.inf] * (len(square_values) + 1)

    def prefix_minimum(index: int) -> float:
        value = math.inf
        while index > 0:
            value = min(value, tree[index])
            index -= index & -index
        return value

    def update(index: int, value: float) -> None:
        while index < len(tree):
            if value < tree[index]:
                tree[index] = value
            index += index & -index

    retained: list[RawScheduleNode] = []
    for node in ordered:
        square_index = bisect_right(square_values, node.square_log2)
        if prefix_minimum(square_index) <= node.centrifugation_log2:
            continue
        retained.append(node)
        update(square_index, node.centrifugation_log2)
    return tuple(retained)


def build_raw_schedule_frontiers(
    *,
    B1_max: int,
    p_max: int,
    square_costs: Sequence[float],
    N: int,
    minimum_u: int,
    minimum_guesses: int,
) -> ScheduleFrontiers:
    """Run one parameter-independent DP for every (p,B1) schedule.

    The maximum pair cap occurs at the minimum supplied dimension and total
    guessing exponent g=k+h-u:

        P <= N-minimum_u-minimum_guesses+1-(minimum_u+1)B1.

    A partial state already above this cap can never become feasible after a
    positive block is prepended, so this global pruning is exact.
    """

    current: dict[int, tuple[RawScheduleNode, ...]] = {
        0: (RawScheduleNode(0, NEG_INF, NEG_INF, ()),)
    }
    frontiers: dict[tuple[int, int], tuple[RawScheduleNode, ...]] = {}
    for p in range(1, min(p_max, B1_max) + 1):
        buckets: dict[int, list[RawScheduleNode]] = {}
        for suffix_total, nodes in current.items():
            for node in nodes:
                for block in range(1, B1_max - suffix_total + 1):
                    total = suffix_total + block
                    pair_sum = node.pair_sum + suffix_total * block
                    maximum_pair = (
                        N
                        - minimum_u
                        - minimum_guesses
                        + 1
                        - (minimum_u + 1) * total
                    )
                    if pair_sum > maximum_pair:
                        continue
                    square_log = log2sum(
                        (node.square_log2, square_costs[block])
                    )
                    centrifugation_log = node.centrifugation_log2
                    if suffix_total:
                        centrifugation_log = log2sum(
                            (
                                centrifugation_log,
                                math.log2(block) + square_costs[suffix_total],
                            )
                        )
                    buckets.setdefault(total, []).append(
                        RawScheduleNode(
                            pair_sum,
                            square_log,
                            centrifugation_log,
                            (block, *node.blocks),
                        )
                    )
        current = {}
        for total, nodes in buckets.items():
            pruned = _pareto_prune_raw(nodes)
            if pruned:
                current[total] = pruned
                frontiers[(p, total)] = pruned
        if not current:
            break
    return frontiers


def raw_schedule_query_index(
    nodes: Sequence[RawScheduleNode],
    guess_log2: float,
) -> tuple[list[int], list[tuple[float, RawScheduleNode]]]:
    """Build pair-sorted prefix optima for one (p,B1,guess exponent) family."""

    ordered = sorted(nodes, key=lambda node: (node.pair_sum, node.blocks))
    pairs: list[int] = []
    prefix: list[tuple[float, RawScheduleNode]] = []
    best: tuple[tuple[Any, ...], float, RawScheduleNode] | None = None
    for node in ordered:
        score = log2sum(
            (
                node.centrifugation_log2,
                guess_log2 + node.square_log2,
            )
        )
        # If two totals round to the same double, use the raw components and
        # then path order as a deterministic tie-break.  This matters only
        # far below the reported precision.
        key = (
            score,
            node.square_log2,
            node.centrifugation_log2,
            node.blocks,
        )
        if best is None or key < best[0]:
            best = (key, score, node)
        assert best is not None
        pairs.append(node.pair_sum)
        prefix.append((best[1], best[2]))
    return pairs, prefix


def query_raw_schedule(
    index: tuple[Sequence[int], Sequence[tuple[float, RawScheduleNode]]],
    pair_cap: int,
) -> tuple[float, RawScheduleNode] | None:
    pairs, prefix = index
    position = bisect_right(pairs, pair_cap) - 1
    return None if position < 0 else prefix[position]


def minimum_pair_sum(B1: int, p: int) -> int:
    """Smallest cross-pair sum among positive p-compositions of B1."""

    if B1 < p:
        return math.inf
    return (p - 1) * (2 * B1 - p) // 2

def _maximum_remaining_squares(total: int, parts: int) -> int:
    if parts == 0:
        return 0 if total == 0 else -1
    if total < parts:
        return -1
    return (total - parts + 1) ** 2 + parts - 1


def _quadratic_operation_count(size: int) -> int:
    """Multiplications plus additions for one dense quadratic evaluation."""

    if size <= 0:
        return 0
    multiplications = size * (size + 1) // 2 + size
    additions = size * (size - 1) // 2 + size - 1
    return multiplications + additions


def _generic_specialisation_operations(
    domain_dimension: int,
    fixed: int,
) -> int:
    """Reference Pseudo-Oil specialization cost in the constant-memory model."""

    if not 0 <= fixed <= domain_dimension:
        raise ValueError("the number of fixed coordinates is out of range")
    quadratic = _quadratic_operation_count(fixed)
    remaining = domain_dimension - fixed
    matrix_vector = (
        remaining * fixed + remaining * (fixed - 1)
        if remaining > 0 and fixed > 0
        else 0
    )
    inner_product = 0 if domain_dimension <= 0 else 2 * domain_dimension - 1
    return quadratic + matrix_vector + inner_product


def _generic_polynomial_log2(
    blocks: Sequence[int],
    domain_dimension: int,
    k: int,
) -> float:
    operations = 0
    suffix = 0
    for block in reversed(blocks):
        operations += _generic_specialisation_operations(
            domain_dimension, k + suffix
        )
        suffix += block
    return NEG_INF if operations <= 0 else math.log2(operations)


def _generic_elementary_mq_cost(
    variables: int,
    equations: int,
    q: int,
) -> MQCost:
    """Reference exhaustive cost for MQ shapes with at most two rows/variables."""

    per_equation = _quadratic_operation_count(variables)
    time = (
        0.0
        if per_equation <= 0
        else (
            variables * math.log2(q)
            + math.log2(max(1, equations))
            + math.log2(per_equation)
        )
    )
    return MQCost(
        variables,
        equations,
        q,
        time,
        max(0.0, variables * math.log2(q)),
        "Pseudo-Oil elementary exhaustive fallback",
        {},
    )


def _generic_zero_variable_terminal_cost(
    linear_variables: int,
    remaining_equations: int,
    q: int,
) -> MQCost:
    """Reference dense-linear-solve/check cost when ell=b0."""

    size = linear_variables
    solve_multiplications = size**3 + size**2 + size
    solve_additions = size**3 - size
    evaluation = _quadratic_operation_count(size)
    operations = max(
        1,
        solve_multiplications
        + solve_additions
        + remaining_equations * (evaluation + 1),
    )
    return MQCost(
        0,
        remaining_equations,
        q,
        math.log2(operations),
        max(0.0, size * math.log2(q)),
        "Pseudo-Oil zero-variable terminal",
        {"eliminated_linear_variables": size},
    )


def _generic_mq_cost(
    variables: int,
    equations: int,
    q: int,
    oracle: MQOracle,
) -> MQCost:
    """Compare all library solvers with the retained tiny elementary method."""
    if variables <= 2 or equations <= 2:
        elementary = _generic_elementary_mq_cost(variables, equations, q)
        try:
            library_cost = oracle.cost(variables, equations, q)
        except RuntimeError:
            # The explicit elementary method remains usable even when the
            # library cannot provide a finite estimate for this tiny shape.
            return elementary
        return min(
            (library_cost, elementary),
            key=lambda row: (row.time_log2, row.memory_log2, row.algorithm),
        )
    return oracle.cost(variables, equations, q)


def _generic_terminal_cost(
    *,
    b0: int,
    ell: int,
    k: int,
    q: int,
    oracle: MQOracle,
) -> MQCost:
    variables = b0 - ell
    equations = variables + k
    if b0 == 0 and k == 0:
        return MQCost(0, 0, q, NEG_INF, 0.0, "empty terminal", {})
    if variables == 0:
        return _generic_zero_variable_terminal_cost(b0, k, q)
    return _generic_mq_cost(variables, equations, q, oracle)

def best_generic_schedules_by_p(
    *,
    N: int,
    M: int,
    q: int,
    p_min: int,
    p_max: int,
    k: int,
    B1: int,
    square_costs: Sequence[float],
    maximum_log_cost: float = math.inf,
) -> dict[int, GenericScheduleResult]:
    """Find requested block counts below the cutoff in one exact suffix DP."""

    b0 = M - k - B1
    if p_min < 1 or p_max < p_min or B1 < p_min or b0 < 0 or N < M:
        return {}
    p_max = min(p_max, B1)
    pair_cap = N - M + 1 - b0 * B1
    threshold = max(0, B1 * B1 - 2 * pair_cap)

    states: dict[tuple[int, int], DPNode] = {
        (0, 0): DPNode(NEG_INF, ())
    }
    schedules: dict[int, GenericScheduleResult] = {}
    guess_log = k * math.log2(q)

    for depth in range(1, p_max + 1):
        next_states: dict[tuple[int, int], DPNode] = {}
        for (suffix, square_sum), entry in states.items():
            remaining = B1 - suffix
            if remaining < 1:
                continue
            choices: Iterable[int] = (
                (remaining,)
                if depth == p_max
                else range(1, remaining + 1)
            )
            polynomial_operations = _generic_specialisation_operations(
                M, k + suffix
            )
            polynomial_term = (
                NEG_INF
                if polynomial_operations <= 0
                else guess_log + math.log2(polynomial_operations)
            )
            for block in choices:
                new_suffix = suffix + block
                exact_q = square_sum + block * block
                new_q = min(threshold, exact_q)
                terms = [
                    entry.log_cost,
                    guess_log + square_costs[block],
                    polynomial_term,
                ]
                if depth > 1:
                    terms.append(math.log2(block) + square_costs[suffix])
                candidate = DPNode(
                    log2sum(terms), entry.blocks + (block,)
                )
                # Every continuation only adds positive work.  This is a
                # safe branch-and-bound cutoff supplied by the outer search.
                if candidate.log_cost > maximum_log_cost:
                    continue

                if new_suffix == B1:
                    if depth < p_min or new_q < threshold:
                        continue
                    blocks = tuple(reversed(candidate.blocks))
                    pair_sum = (
                        B1 * B1 - sum(value * value for value in blocks)
                    ) // 2
                    incumbent = schedules.get(depth)
                    result = GenericScheduleResult(
                        blocks,
                        pair_sum,
                        candidate.log_cost,
                        _generic_polynomial_log2(blocks, M, k),
                    )
                    if incumbent is None or (
                        result.optimized_log_cost,
                        result.blocks,
                    ) < (
                        incumbent.optimized_log_cost,
                        incumbent.blocks,
                    ):
                        schedules[depth] = result
                    continue

                if depth == p_max:
                    continue
                remaining_mass = B1 - new_suffix
                minimum_future_parts = max(1, p_min - depth)
                if remaining_mass < minimum_future_parts:
                    continue
                attainable = _maximum_remaining_squares(
                    remaining_mass, minimum_future_parts
                )
                if new_q < threshold and exact_q + attainable < threshold:
                    continue
                key = (new_suffix, new_q)
                incumbent_node = next_states.get(key)
                if incumbent_node is None or (
                    candidate.log_cost,
                    candidate.blocks,
                ) < (
                    incumbent_node.log_cost,
                    incumbent_node.blocks,
                ):
                    next_states[key] = candidate

        # At fixed depth and suffix, a cheaper state with at least as much
        # accumulated block-square mass dominates every continuation.
        by_suffix: dict[int, list[tuple[int, DPNode]]] = {}
        for (suffix, square_sum), entry in next_states.items():
            by_suffix.setdefault(suffix, []).append((square_sum, entry))
        states = {}
        for suffix, frontier in by_suffix.items():
            best_higher = math.inf
            for square_sum, entry in sorted(
                frontier, reverse=True, key=lambda item: item[0]
            ):
                if best_higher <= entry.log_cost:
                    continue
                states[(suffix, square_sum)] = entry
                best_higher = entry.log_cost
        if not states and depth < p_min:
            break

    return schedules

def schedule_components(
    blocks: Sequence[int],
    guess_log2: float,
    square_costs: Sequence[float],
) -> tuple[float, float, float]:
    """Return Centrifugation and per/all-guess Square-work logarithms."""

    centrifugation = []
    suffix = 0
    for block in reversed(blocks):
        if suffix:
            centrifugation.append(math.log2(block) + square_costs[suffix])
        suffix += block
    square_per_guess = log2sum(square_costs[block] for block in blocks)
    centrifugation_log2 = log2sum(centrifugation)
    return (
        centrifugation_log2,
        square_per_guess,
        guess_log2 + square_per_guess,
    )


def linear_algebra_log2(
    rows: int,
    columns: int,
    extension_degree: int,
    omega: float,
    theta: float,
) -> float:
    """Rectangular elimination estimate in bit operations.

    The field-operation count is
    ``rows*columns*min(rows,columns)^(omega-2)``.  A GF(2^e) operation is
    charged ``e^theta`` bit operations, matching the estimator conversion.
    With zero columns this reduces to checking ``rows`` constants.
    """

    if rows <= 0:
        return NEG_INF
    effective_columns = max(1, columns)
    pivot_dimension = max(1, min(rows, effective_columns))
    return (
        math.log2(rows)
        + math.log2(effective_columns)
        + max(0.0, omega - 2.0) * math.log2(pivot_dimension)
        + theta * math.log2(max(1, extension_degree))
    )

@lru_cache(maxsize=None)
def _rank_log_probabilities(
    rows: int, columns: int, q: int,
) -> tuple[tuple[int, float], ...]:
    """All rank probabilities for a uniform matrix over GF(q), in log2.

    Start at maximal rank and recur downwards, avoiding cancellation of
    large logarithms. No rank tail is discarded, even if its probability
    would underflow outside the logarithmic domain.
    """
    log_q = math.log2(q)

    def log_factor(power: int) -> float:
        return math.log1p(-2.0 ** (-power * log_q)) / math.log(2.0)

    maximum_rank = min(rows, columns)
    log_probability = math.fsum(
        log_factor(power)
        for power in range(abs(rows - columns) + 1, max(rows, columns) + 1)
    )
    probabilities = [(maximum_rank, log_probability)]
    for rank in range(maximum_rank - 1, -1, -1):
        log_probability -= (
            (rows + columns - 2 * rank - 1) * log_q
            + log_factor(rows - rank)
            + log_factor(columns - rank)
            - log_factor(rank + 1)
        )
        probabilities.append((rank, log_probability))
    return tuple(probabilities)


@lru_cache(maxsize=None)
def _binary_gaussian_row(dimension: int) -> tuple[int, ...]:
    """Numbers of binary subspaces of each dimension, using exact integers."""
    row = [1]
    for subdimension in range(1, dimension + 1):
        row.append(
            row[-1] * ((1 << (dimension - subdimension + 1)) - 1)
            // ((1 << subdimension) - 1)
        )
    return tuple(row)


def _square_linear_kernel_counts(
    rows: int, columns: int, q: int,
) -> tuple[int, ...]:
    """Count B by binary kernel dimension for A*x^[2] + B*x over GF(q).

    A is fixed and has full column rank; B is uniform over all rows-by-columns
    field matrices. Every binary-independent set of roots is field-independent,
    so the binary kernel dimension K is at most columns, not e*columns.

    Writing s=rows, d=columns and T=q^(s*d), the aggregate Gaussian moments are
        T * E[[K choose j]_2]
          = q^(s*(d-j)) * prod_i<j(q^d-q^i) / prod_i<j(2^j-2^i).
    Their consecutive ratio gives the integer recurrence below. Triangular
    inversion then recovers exact counts with common denominator T. No floating
    subtraction or probability-tail truncation is used.
    """
    if q < 2 or not _is_power_of_two(q):
        raise ValueError("the square-linear rank model requires q = 2^e")
    if not 0 <= columns <= rows:
        raise ValueError("the square coefficient must have full column rank")
    e = q.bit_length() - 1
    total = 1 << (e * rows * columns)
    q_rows = 1 << (e * rows)
    q_columns = 1 << (e * columns)
    moments = [total]
    q_power = 1
    for j in range(columns):
        numerator = moments[-1] * (q_columns - q_power)
        denominator = q_rows * (1 << j) * ((1 << (j + 1)) - 1)
        moment, remainder = divmod(numerator, denominator)
        if remainder:
            raise ArithmeticError("nonintegral square-linear kernel moment")
        moments.append(moment)
        q_power *= q

    counts = [0] * (columns + 1)
    for kernel_dimension in range(columns, -1, -1):
        count = moments[kernel_dimension] - sum(
            _binary_gaussian_row(larger)[kernel_dimension] * counts[larger]
            for larger in range(kernel_dimension + 1, columns + 1)
        )
        if count < 0:
            raise ArithmeticError("negative square-linear kernel count")
        counts[kernel_dimension] = count
    if sum(counts) != total:
        raise ArithmeticError("square-linear kernel counts do not normalize")
    return tuple(counts)


@lru_cache(maxsize=None)
def _square_linear_rank_log_probabilities(
    rows: int, columns: int, q: int,
) -> tuple[tuple[int, float], ...]:
    """Binary rank law of a square-plus-linear map; dimensions are over GF(q)."""
    counts = _square_linear_kernel_counts(rows, columns, q)
    e = q.bit_length() - 1
    denominator_log2 = e * rows * columns
    return tuple(
        (e * columns - kernel_dimension, math.log2(count) - denominator_log2)
        for kernel_dimension, count in enumerate(counts)
        if count
    )


def _finishing_log2(variables: int, equations: int, oracle: MQOracle) -> float:
    if equations <= 0:
        return NEG_INF
    if variables <= 0:
        return math.log2(max(1, equations))
    return oracle.cost(variables, equations, 2).time_log2


@lru_cache(maxsize=None)
def _binary_endpoint_costs(
    rows: int, columns: int, equations: int,
    omega: float, theta: float, oracle: MQOracle, q: int = 2,
) -> EndpointCosts:
    """Square-Only work and finishing; input dimensions here are binary."""
    if q < 2 or not _is_power_of_two(q):
        raise ValueError("the Square-Only field must have size q = 2^e")
    e = q.bit_length() - 1
    if rows % e or columns % e:
        raise ValueError("binary dimensions must be multiples of the field degree")
    linear = linear_algebra_log2(rows, columns, 1, omega, theta)
    finishing = log2sum(
        log_probability + rank - rows
        + _finishing_log2(columns - rank, equations, oracle)
        for rank, log_probability in _square_linear_rank_log_probabilities(
            rows // e, columns // e, q,
        )
    )
    return EndpointCosts(NEG_INF, linear, finishing)


def _binary_endpoint_log2(
    rows: int, columns: int, equations: int,
    omega: float, theta: float, oracle: MQOracle, q: int = 2,
) -> float:
    return _binary_endpoint_costs(
        rows, columns, equations, omega, theta, oracle, q,
    ).total_log2


@lru_cache(maxsize=None)
def _precise_endpoint_costs(
    zero_rows: int, u: int, square_rows: int, equations: int, q: int,
    omega: float, theta: float, oracle: MQOracle,
) -> EndpointCosts:
    """Average OV and square-plus-linear ranks, with independent uniform RHS.

    Given OV rank sigma, consistency has probability q^(sigma-zero_rows).
    The Square-Only system is A*x^[2]+B*x over GF(q), with A injective and
    B uniform conditional on the OV stage. The dimension model takes the u
    pure-square generators to be independent modulo R. This makes A injective
    even when w-r>u (Round 3), and survives restriction to the OV solution space.
    All feasible Round 3 candidates are below the ambient-dimension cap.
    Its binary matrix has e*(u-sigma) columns. Given binary rank rho,
    consistency has probability 2^(rho-square_rows), and finishing retains
    e*(u-sigma)-rho Boolean variables. Each elimination is paid when reached;
    inconsistent branches incur no subsequent work.

    The square-plus-linear rank law is exact for this coefficient ensemble;
    conditional uniformity of B and the RHS remains the model's heuristic.
    """
    e = int(math.log2(q))
    field_linear = linear_algebra_log2(zero_rows, u, e, omega, theta)
    square_terms = []
    finishing_terms = []
    for rank, log_probability in _rank_log_probabilities(zero_rows, u, q):
        reach = log_probability + (rank - zero_rows) * math.log2(q)
        binary = _binary_endpoint_costs(
            square_rows, e * (u - rank), equations, omega, theta, oracle, q=q,
        )
        square_terms.append(reach + binary.square_only)
        finishing_terms.append(reach + binary.finishing)
    return EndpointCosts(
        field_linear, log2sum(square_terms), log2sum(finishing_terms),
    )


def _precise_endpoint_log2(
    zero_rows: int, u: int, square_rows: int, equations: int, q: int,
    omega: float, theta: float, oracle: MQOracle,
) -> float:
    return _precise_endpoint_costs(
        zero_rows, u, square_rows, equations, q, omega, theta, oracle,
    ).total_log2


def endpoint_costs(
    u: int, r: int, w: int, h: int, q: int,
    omega: float, theta: float, oracle: MQOracle, precise_rank: bool = True,
) -> EndpointCosts:
    """Expected stage costs per Reduced solution, before the q^(h-u) factor.

    OV is always paid. Square-Only includes the probability of reaching it;
    Finishing includes the probabilities of both linear systems being consistent.
    """
    e = int(math.log2(q))
    zero_rows = h - w
    if precise_rank:
        return _precise_endpoint_costs(
            zero_rows, u, e * (w - r), e * r, q, omega, theta, oracle,
        )
    sigma = min(zero_rows, u)
    columns = e * (u - sigma)
    square_rows = e * (w - r)
    rho = min(square_rows, columns)
    variables = columns - rho
    equations = e * r
    log_p_zero = (sigma - zero_rows) * math.log2(q)
    log_p_square = rho - square_rows

    field_linear = linear_algebra_log2(zero_rows, u, e, omega, theta)
    binary_linear = linear_algebra_log2(square_rows, columns, 1, omega, theta)
    finishing = _finishing_log2(variables, equations, oracle)
    return EndpointCosts(
        field_linear,
        log_p_zero + binary_linear,
        log_p_zero + log_p_square + finishing,
    )


def endpoint_log2(
    u: int, r: int, w: int, h: int, q: int,
    omega: float, theta: float, oracle: MQOracle, precise_rank: bool = True,
) -> float:
    """Expected endpoint cost per Reduced solution, including early abort."""
    return endpoint_costs(
        u, r, w, h, q, omega, theta, oracle, precise_rank,
    ).total_log2


def structural_terminal_log2(
    b0: int, ell: int, k: int, q: int, oracle: MQOracle,
) -> float:
    """Reduced terminal MQ(b0-ell,b0+k-ell), in structural bit units."""
    variables = b0 - ell
    if variables > 0:
        return oracle.cost(variables, variables + k, q).time_log2
    if b0 == 0:
        return NEG_INF if k == 0 else oracle.cost(0, k, q).time_log2
    # Eliminating a nonempty initial block is not free even when no
    # variables remain. Reuse the reference operation count, then convert
    # its field operations to bits consistently with structural GB costs.
    linear = _generic_zero_variable_terminal_cost(b0, k, q).time_log2
    return linear + oracle.theta * math.log2(math.log2(q))


def target_restart_log2(q: int, d: int) -> float:
    if d == 0:
        return 0.0
    return d * math.log2(q) - math.log2(d * (q - 1))

def optimize(args: argparse.Namespace) -> Result:
    """Search the selected mode and return its single best parameter set."""
    n, m, q, kappa = args.n, args.m, args.q, args.kappa
    round3 = args.mode == "round3"
    structural = args.mode in {"combined", "structural-only", "round3"}
    multi_target = args.mode in {"combined", "multi-target-only"}
    b0_min = 0

    if min(n, m) <= 0 or kappa < 2:
        raise ValueError("n and m must be positive, and kappa must be at least 2")
    if q < 2 or not _is_power_of_two(q):
        raise ValueError("q must be a power of two, at least 2")
    if structural and n < 2:
        raise ValueError("the structural construction requires n >= 2")
    if round3:
        if q != 16:
            raise ValueError("--round3 requires q = 16")
        if n - m < 2:
            raise ValueError("--round3 requires n - m >= 2 for the kernel construction")
        if kappa * (kappa + 1) // 2 > m:
            raise ValueError("--round3 requires kappa*(kappa+1)/2 <= m for the emulsifiers")
        if args.fixed_d not in (None, 0):
            raise ValueError("--round3 supports only d = 0")
    if not 2.0 <= args.omega <= 3.0:
        raise ValueError("omega must lie in [2, 3]")
    if not 0.0 <= args.theta <= 2.0:
        raise ValueError("theta must lie in [0, 2]")
    if args.fixed_p is not None and args.fixed_p < 1:
        raise ValueError("--fixed-p must be positive")
    if args.fixed_d is not None and not multi_target and not round3:
        raise ValueError("--fixed-d requires the multi-target technique")
    if args.precise_rank is not None and not structural:
        raise ValueError("rank options require the structural technique")
    if getattr(args, "unstructured", False) and not structural:
        raise ValueError("--unstructured requires the structural technique")

    N = kappa * n
    u_min = 2 * kappa if structural else 0
    p_min = args.fixed_p if args.fixed_p is not None else 1
    if multi_target:
        # Positive d still requires at least one guess. Keep d=0 available
        # even when no positive target dimension can be feasible.
        default_d_max = max(0, m - u_min - p_min - 1 - b0_min)
        d_min = args.fixed_d if args.fixed_d is not None else 0
        d_max = args.fixed_d if args.fixed_d is not None else default_d_max
        if not 0 <= d_min <= d_max < m:
            raise ValueError(f"no valid target range: require 0 <= d <= {m - 1}")
    else:
        d_min = d_max = 0

    # Shared caps/frontiers must retain the complete zero-guess baseline
    # whenever d=0 is searched. Each positive-d iteration tightens this floor.
    minimum_guesses = int(d_min > 0)
    B1_cap = m - d_min - u_min - minimum_guesses - b0_min
    if structural:
        B1_cap = min(B1_cap, (N - u_min - minimum_guesses) // (u_min + 1))
    else:
        B1_cap = min(B1_cap, N - minimum_guesses)
    p_max = min(B1_cap, args.fixed_p if args.fixed_p is not None else B1_cap)
    while p_max >= p_min:
        pairs = p_max * (p_max - 1) // 2
        feasible = (
            pairs <= N - u_min - minimum_guesses + 1 - (u_min + 1) * p_max
            if structural
            else (
                p_max + minimum_guesses <= N
                and pairs + p_max + minimum_guesses <= N + 1
            )
        )
        if feasible:
            break
        p_max -= 1
    if p_max < p_min:
        raise ValueError("no feasible positive block schedule for these inputs")

    directory = Path(__file__).resolve().parent
    if not (directory / "CryptographicEstimators").is_dir():
        raise FileNotFoundError("CryptographicEstimators must be beside this script")
    oracle = MQOracle(directory, args.omega, args.theta)
    # In either mode, feasibility implies B1+ell <= B1+b0 <= B1_cap,
    # including the zero boundaries for k,b0,ell.
    square_costs = [NEG_INF]
    for size in range(1, B1_cap + b0_min + 1):
        row = (
            oracle.cost(size, size, q)
            if structural
            else _generic_mq_cost(size, size, q, oracle)
        )
        square_costs.append(row.time_log2)

    if structural:
        best = search_structural(
            args, N, d_min, d_max, p_min, p_max,
            B1_cap, minimum_guesses, square_costs, oracle,
        )
    else:
        best = search_generic(
            args, N, d_min, d_max, p_min, p_max,
            B1_cap, minimum_guesses, square_costs, oracle,
        )
    oracle.flush()
    if best is None:
        raise ValueError("no feasible parameter set for these inputs")
    return best


def structural_dimensions(
    kappa: int, a: int, unstructured: bool = False,
) -> tuple[int, int, int]:
    """Use the chosen Table 2 bounds as the dimension model for (u, r, w)."""
    u = 2 * kappa + a
    pairs = a * (a - 1) // 2
    r = a + pairs + a * (kappa - a) if unstructured else 2 * a + pairs
    return u, r, u + r


def round3_dimensions(kappa: int, a: int, m: int) -> tuple[int, int, int]:
    """Treat the one-kernel-vector Round 3 dimension bounds as equalities.

    Polarization cancels the added linear map, so r is the unstructured polar
    dimension. The remaining linear-image bound is kappa+a, in addition to
    the u pure-square generators and r polar generators. The Square-Only
    field-row count is w-r, not u. At w=m no positive Reduced block is feasible.
    """
    u, r, quadratic_w = structural_dimensions(kappa, a, unstructured=True)
    return u, r, min(m, quadratic_w + kappa + a)


def structural_a_max(
    n: int, m: int, kappa: int, unstructured: bool = False,
) -> int:
    """Section 3.1 allows a <= kappa and assumes a third independent base vector.

    The n >= 3 guard only checks ambient dimension; existence in the required
    differential kernel remains the construction's assumption.
    """
    if unstructured:
        return kappa if n >= 3 else 0
    return 0 if m <= 2 else min(n - 2, max(0, 2 * (n - m) - 4))


def search_structural(
    args: argparse.Namespace, N: int, d_min: int, d_max: int,
    p_min: int, p_max: int, B1_cap: int, minimum_guesses: int,
    square_costs: Sequence[float], oracle: MQOracle,
) -> Result | None:
    """Search all nonnegative k,b0,ell on the quotient K_H/U.

    Use NH=N-u*(M-h), the dimension lower bound for K_H, and
    P=sum_{i<j} b_i*b_j.
    The accepted boundary conventions are:
        NH >= M,
        b0*(B1+ell+1) <= NH-u+1,
        b0*B1+P <= NH-M+1.
    Here h=M-k-B1-b0 and the Reduced guessing exponent is g=k+h-u.
    The independent NH >= M constraint also applies when b0*B1+P is zero.
    """
    n, m, q, kappa = args.n, args.m, args.q, args.kappa
    round3 = args.mode == "round3"
    unstructured = round3 or getattr(args, "unstructured", False)
    a_max = (
        kappa if n - m >= 3 else 0
    ) if round3 else structural_a_max(n, m, kappa, unstructured)
    frontiers = build_raw_schedule_frontiers(
        B1_max=B1_cap, p_max=p_max, square_costs=square_costs,
        N=N, minimum_u=2 * kappa, minimum_guesses=minimum_guesses,
    )
    indexes = {}
    log_q = math.log2(q)
    best = None

    @lru_cache(maxsize=None)
    def choose_schedule(B1: int, guesses: int, pair_cap: int):
        """The optimal ordered schedule for fixed non-schedule parameters."""
        choices = []
        for p in range(p_min, min(p_max, B1) + 1):
            if minimum_pair_sum(B1, p) > pair_cap:
                break
            nodes = frontiers.get((p, B1))
            if not nodes:
                continue
            key = (p, B1, guesses)
            if key not in indexes:
                indexes[key] = raw_schedule_query_index(nodes, guesses * log_q)
            queried = query_raw_schedule(indexes[key], pair_cap)
            if queried is not None:
                score, node = queried
                choices.append((score, p, node))
        return min(
            choices,
            key=lambda item: (
                item[0], item[2].square_log2, item[2].centrifugation_log2,
                item[1], item[2].blocks,
            ),
            default=None,
        )

    @lru_cache(maxsize=None)
    def endpoint(u: int, r: int, w: int, h: int) -> tuple[float, EndpointCosts]:
        stages = endpoint_costs(
            u, r, w, h, q, args.omega, args.theta, oracle,
            args.precise_rank is not False,
        )
        return (h - u) * log_q + stages.total_log2, stages

    @lru_cache(maxsize=None)
    def terminal(b0: int, ell: int, k: int) -> float:
        return structural_terminal_log2(b0, ell, k, q, oracle)

    for d in range(d_min, d_max + 1):
        M = m - d
        target_minimum_guesses = max(minimum_guesses, int(d > 0))
        target_factor = target_restart_log2(q, d)
        for a in range(a_max + 1):
            u, r, w = (
                round3_dimensions(kappa, a, m)
                if round3 else structural_dimensions(kappa, a, unstructured)
            )
            if w > M:
                continue
            upper_B1 = min(
                B1_cap, M - w, M - u - target_minimum_guesses, (N - M) // u,
            )
            for B1 in range(p_min, upper_B1 + 1):
                pair_floor = minimum_pair_sum(B1, p_min)
                b0_max = min(
                    M - w - B1, M - u - B1 - target_minimum_guesses,
                    (N - M) // u - B1,
                    (N - M + 1 - u * B1 - pair_floor) // (u + B1),
                )
                for b0 in range(b0_max + 1):
                    # g does not depend on k: increasing k lowers h equally.
                    guesses = M - u - B1 - b0
                    k_max = min(
                        M - w - B1 - b0,
                        (N - M) // u - B1 - b0,
                        (N - M + 1 - b0 * B1 - pair_floor) // u - B1 - b0,
                    )
                    for k in range(k_max + 1):
                        h = M - k - B1 - b0
                        NH = N - u * (M - h)
                        pair_cap = NH - M + 1 - b0 * B1
                        ell_max = (
                            0 if b0 == 0 else
                            min(b0, (NH - u + 1) // b0 - B1 - 1)
                        )
                        if ell_max < 0:
                            continue
                        choice = choose_schedule(B1, guesses, pair_cap)
                        if choice is None:
                            continue
                        schedule_cost, p, schedule = choice
                        if best is not None and target_factor + schedule_cost > best.total:
                            continue
                        endpoint_cost, endpoint_stages = endpoint(u, r, w, h)
                        subtotal = log2sum((schedule_cost, endpoint_cost))
                        if best is not None and target_factor + subtotal > best.total:
                            continue
                        for ell in range(ell_max + 1):
                            initial = (
                                NEG_INF if b0 == 0 else
                                math.log2(b0) + square_costs[B1 + ell]
                            )
                            partial = log2sum((subtotal, initial))
                            if best is not None and target_factor + partial > best.total:
                                continue
                            reduced_terminal = guesses * log_q + terminal(b0, ell, k)
                            per_target = log2sum((partial, reduced_terminal))
                            candidate = Result(
                                d=d, p=p, k=k, a=a, B1=B1, b0=b0, ell=ell, h=h,
                                blocks=schedule.blocks,
                                centrifugation=log2sum((
                                    schedule.centrifugation_log2, initial,
                                )),
                                square=guesses * log_q + schedule.square_log2,
                                reduced_terminal=reduced_terminal,
                                endpoint=endpoint_cost, specialization=NEG_INF,
                                per_target=per_target, target_factor=target_factor,
                                total=target_factor + per_target,
                                endpoint_costs=endpoint_stages,
                            )
                            if best is None or candidate.key() < best.key():
                                best = candidate
    return best


def search_generic(
    args: argparse.Namespace, N: int, d_min: int, d_max: int,
    p_min: int, p_max: int, B1_cap: int, minimum_guesses: int,
    square_costs: Sequence[float], oracle: MQOracle,
) -> Result | None:
    """Search inclusive generic bounds with the retained pseudo-oil costs.

    Require N >= M, b0*(B1+ell+1) <= N+1, and
    b0*B1+sum_{i<j} b_i*b_j <= N-M+1, including b0=ell=0.
    """
    m, q = args.m, args.q
    best = None
    for d in range(d_min, d_max + 1):
        M = m - d
        target_minimum_guesses = max(minimum_guesses, int(d > 0))
        if N < M:
            continue
        target_factor = target_restart_log2(q, d)
        for B1 in range(p_min, min(B1_cap, M - target_minimum_guesses) + 1):
            for k in range(target_minimum_guesses, M - B1 + 1):
                b0 = M - k - B1
                ell_max = (
                    0 if b0 == 0 else
                    min(b0, (N + 1) // b0 - B1 - 1)
                )
                pair_cap = N - M + 1 - b0 * B1
                if ell_max < 0 or pair_cap < minimum_pair_sum(B1, p_min):
                    continue
                local_p_max = min(p_max, B1)
                while minimum_pair_sum(B1, local_p_max) > pair_cap:
                    local_p_max -= 1
                guess_log = k * math.log2(q)
                schedules = best_generic_schedules_by_p(
                    N=N, M=M, q=q, p_min=p_min, p_max=local_p_max,
                    k=k, B1=B1, square_costs=square_costs,
                    maximum_log_cost=(
                        math.inf if best is None else best.total - target_factor
                    ),
                )
                for p, schedule in sorted(schedules.items()):
                    raw_cent, _, square = schedule_components(
                        schedule.blocks, guess_log, square_costs,
                    )
                    specialization = guess_log + schedule.polynomial_per_guess_log2
                    for ell in range(ell_max + 1):
                        fixed_cent = (
                            NEG_INF if b0 == 0 else
                            math.log2(b0) + square_costs[B1 + ell]
                        )
                        cent = log2sum((raw_cent, fixed_cent))
                        terminal = _generic_terminal_cost(
                            b0=b0, ell=ell, k=k, q=q, oracle=oracle,
                        )
                        endpoint = guess_log + terminal.time_log2
                        per_target = log2sum((
                            schedule.optimized_log_cost, fixed_cent, endpoint,
                        ))
                        candidate = Result(
                            d=d, p=p, k=k, a=0, B1=B1, b0=b0, ell=ell, h=0,
                            blocks=schedule.blocks, centrifugation=cent,
                            square=square, reduced_terminal=NEG_INF,
                            endpoint=endpoint, specialization=specialization,
                            per_target=per_target, target_factor=target_factor,
                            total=target_factor + per_target,
                        )
                        if best is None or candidate.key() < best.key():
                            best = candidate
    return best


def render_result(args: argparse.Namespace, best: Result) -> str:
    """Print only the inputs, optimum, and the terms making up its cost."""
    round3 = args.mode == "round3"
    structural = args.mode in {"combined", "structural-only", "round3"}
    multi_target = args.mode in {"combined", "multi-target-only"}
    projected = multi_target and best.d > 0
    width = 58
    lines = [
        "MAYO cost estimate",
        "=" * width,
        "",
        "Inputs",
        f"  n = {args.n}    m = {args.m}    q = {args.q}    kappa = {args.kappa}",
        f"  Mode: {MODE_NAMES[args.mode]}",
    ]
    if args.omega != DEFAULT_OMEGA:
        lines.append(f"  omega = {args.omega:g}")
    if args.theta != DEFAULT_THETA:
        lines.append(f"  theta = {args.theta:g}")
    if structural:
        regime = "unstructured" if round3 or getattr(args, "unstructured", False) else "structured"
        lines.append(f"  Emulsifiers: {regime}")
        if args.precise_rank is False:
            lines.append("  Rank estimate: full rank (maximal rank)")
    if round3:
        lines.append("  Linear-map choice: u1 in ker(Lambda)")
    if multi_target:
        if projected:
            lines.append(f"  Projected codomain: M = m - d = {args.m - best.d}")
        else:
            lines.append("  Target choice: single target (d = 0)")
    fixed = []
    if args.fixed_p is not None:
        fixed.append(f"p = {args.fixed_p}")
    if args.fixed_d is not None:
        fixed.append(f"d = {args.fixed_d}")
    if fixed:
        lines.append("  Fixed: " + ", ".join(fixed))
    lines.extend([
        "",
        "Best parameters",
        f"  d = {best.d}    p = {best.p}    k = {best.k}    B1 = {best.B1}",
    ])
    if structural:
        u, r, w = (
            round3_dimensions(args.kappa, best.a, args.m)
            if round3 else structural_dimensions(
                args.kappa, best.a, getattr(args, "unstructured", False),
            )
        )
        r_label, w_label, h_label = ("r_T", "w_T", "h_T") if projected else ("r", "w", "h")
        lines.append(f"  a = {best.a}    u = {u}    {r_label} = {r}    {w_label} = {w}")
        lines.append(f"  {h_label} = {best.h}    b0 = {best.b0}    ell = {best.ell}")
        if round3:
            lines.append(f"  Square-Only equations over GF(q): w - r = {w - r}")
    else:
        lines.append(f"  b0 = {best.b0}    ell = {best.ell}")
    lines.extend([
        f"  Blocks: {best.blocks}",
        "",
        "Estimated cost (bit operations)",
    ])

    def row(label: str, value: float, prefix: str = "") -> str:
        cost = "0" if value == NEG_INF else f"{prefix}2^{value:.4f}"
        return f"  {label:<35} {cost:>19}"

    if structural:
        stages = best.endpoint_costs
        if stages is None:
            raise ValueError("structural result is missing its endpoint breakdown")
        guessing_factor = (best.h - u) * math.log2(args.q)
        # Reduced retains its additional q^k guessing, but excludes q^(h-u).
        reduced = log2sum((best.square, best.reduced_terminal)) - guessing_factor
        lines.extend([
            row("Centrifugation (one-time)", best.centrifugation),
            # row(f"Guessing factor q^({h_label}-u)", guessing_factor, "x "),
            f"",
            # f"  Repeat q^({h_label}-u) times: 2^{math.log2(args.q)*(best.h - u)}",
            row("Repeat q^(h-u) times:", math.log2(args.q)*(best.h - u)),
            f"",
            row("Reduced", reduced),
            row("OV", stages.ov),
            row("Square-Only", stages.square_only),
            row("Finishing", stages.finishing),
        ])
    else:
        lines.extend([
            row("Centrifugation", best.centrifugation),
            row("Square solving (all guesses)", best.square),
            row("Endpoint solving (all guesses)", best.endpoint),
            row("Specialization (all guesses)", best.specialization),
        ])
    if projected:
        lines.extend([
            f"",
            row("Total per target", best.per_target),
            row("Multi-target multiplier", best.target_factor, "x "),
        ])
    lines.extend([
        "  " + "-" * (width - 2),
        row("TOTAL", best.total),
    ])
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Estimate MAYO cost; default: structure + optional multi-target (d >= 0).",
        allow_abbrev=False,
    )
    for name, help_text in (
        ("n", "base source dimension"),
        ("m", "output dimension"),
        ("q", "field size"),
        ("kappa", "whip-up factor"),
    ):
        parser.add_argument(name, type=int, help=help_text)
    modes = parser.add_mutually_exclusive_group()
    for flag, help_text in (
        ("multi-target-only", "optimize target dimension, including d=0, without structure"),
        ("structural-only", "use only the structural technique"),
        ("generic", "use neither technique"),
        ("round3", "use Round 3 single-target structure with one base vector in ker(Lambda)"),
    ):
        modes.add_argument(
            "--" + flag, dest="mode", action="store_const",
            const=flag, help=help_text,
        )
    parser.set_defaults(mode="combined")
    parser.add_argument(
        "--unstructured", action="store_true",
        help="use Section 3.1 emulsifier bounds with the structural technique (automatic in --round3)",
    )
    rank_options = parser.add_mutually_exclusive_group()
    rank_options.add_argument(
        "--precise-rank", action="store_true",
        help="average structural endpoint costs over both linear-system ranks (default)",
    )
    rank_options.add_argument(
        "--full-rank", dest="precise_rank", action="store_false",
        help="use the maximal-rank structural endpoint approximation",
    )
    # None distinguishes the structural default from an explicitly supplied
    # rank option, which remains invalid in generic modes.
    parser.set_defaults(precise_rank=None)
    parser.add_argument("--fixed-p", type=int, metavar="P", help="fix the block count")
    parser.add_argument(
        "--fixed-d", type=int, metavar="D",
        help="fix the target dimension (0 disables multi-target reduction; --round3 accepts only 0)",
    )
    parser.add_argument(
        "--omega", type=float, default=DEFAULT_OMEGA,
        help="linear algebra exponent (default: 2.81)",
    )
    parser.add_argument(
        "--theta", type=float, default=DEFAULT_THETA,
        help="field-operation bit-cost exponent (default: 2)",
    )
    return parser


def _retry_with_project_environment() -> None:
    """Retry only the CLI, using a verified local venv and the same arguments.

    Native dependencies must run under their own interpreter; never add a
    different Python version's site-packages to this process. A process already
    using either project environment does not retry, preventing restart loops.
    """
    script = Path(__file__).resolve()
    environments = [script.parent / name for name in (".venv", "venv")]
    if any(Path(sys.prefix).resolve() == env.resolve() for env in environments):
        return
    for environment in environments:
        python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        if not (environment / "pyvenv.cfg").is_file() or not os.access(python, os.X_OK):
            continue
        try:
            probe = subprocess.run(
                [
                    str(python), "-B", "-c",
                    "import sys; sys.path.insert(0, sys.argv[1]); "
                    "from cryptographic_estimators.MQEstimator import MQEstimator",
                    str(script.parent / "CryptographicEstimators"),
                ],
                capture_output=True, timeout=30, check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if probe.returncode != 0:
            continue
        command = [str(python)]
        if sys.dont_write_bytecode:
            command.append("-B")
        command.extend([str(script), *sys.argv[1:]])
        try:
            os.execv(str(python), command)
        except OSError:
            continue


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        best = optimize(args)
    except EstimatorDependencyError as exc:
        _retry_with_project_environment()
        parser.exit(2, f"error: {exc}\n")
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(render_result(args, best))


if __name__ == "__main__":
    main()
