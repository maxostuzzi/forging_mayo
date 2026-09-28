# Geometric Forgeries: MAYO Cost Optimizer

This directory contains the optimization code accompanying the paper
**“Geometric Forgeries: Cryptanalysis of MAYO.”** It reproduces the classical
bit-complexity estimates for the generic Pseudo-Oil attack and for the
multi-target, structural, and combined attacks described in the paper.

The program searches the admissible attack parameters, queries the
`MQEstimator` component of
[CryptographicEstimators](https://github.com/Crypto-TII/CryptographicEstimators)
for every relevant MQ system, and prints the best parameter set together with
a stage-by-stage cost breakdown.

## Requirements

- Python 3.10 or newer;
- `CryptographicEstimators`, placed in a directory named
  `CryptographicEstimators` beside `optimisation.py`;
- the Python dependencies declared by `CryptographicEstimators`.

## Quick start

The four positional arguments are the MAYO parameters `(n, m, q, kappa)`, where
`kappa` is the whip-up factor. For example, the default command evaluates the
combined Round 2 attack against the first Level I parameter set:

```bash
python3 optimisation.py 86 78 16 10
```

Use the virtual-environment interpreter explicitly if preferred:

```bash
.venv/bin/python optimisation.py 86 78 16 10
```

Run `python3 optimisation.py --help` for the complete command-line reference.

## Attack modes

Exactly one optional mode flag may be selected. With no mode flag, the script
runs the combined structural and multi-target optimization.

| Mode | Command-line flag | Meaning |
|---|---|---|
| Combined | *(default)* | Structural attack with optimization over the target-space dimension, including the single-target case `d = 0`. |
| Multi-target only | `--multi-target-only` | Multi-target reduction combined with the generic Pseudo-Oil attack, without the structural insight. |
| Structure only | `--structural-only` | Structural attack against a single target. |
| Generic | `--generic` | Generic single-target Pseudo-Oil attack; uses neither contribution of this paper. |
| Round 3 structure | `--round3` | Round 3 structural, single-target model over `GF(16)`, including the linear term and the Round 3 emulsifier regime. |

For the Round 2 Level I instance, the four columns in the paper are reproduced
with:

```bash
python3 optimisation.py 86 78 16 10 --generic
python3 optimisation.py 86 78 16 10 --multi-target-only
python3 optimisation.py 86 78 16 10 --structural-only
python3 optimisation.py 86 78 16 10
```

Replace the tuple by one of the following to reproduce the other Round 2 rows:

```text
Level I:    81  64  16   4
Level III: 118 108  16  11
Level V:   154 142  16  12
```

For Round 3, run the generic, Round 3 structural, and hypothetical Round 2
combined models as follows:

```bash
python3 optimisation.py 88 80 16 10 --generic
python3 optimisation.py 88 80 16 10 --round3
python3 optimisation.py 88 80 16 10
```

The remaining Round 3 parameter tuples are:

```text
Level I:    86  64  16   5
Level III: 118 108  16  11
Level V:   154 142  16  12
```

The unflagged command in the Round 3 comparison is intentionally the
counterfactual “R2 Combined” column: it models the same parameters without the
Round 3 linear term and with the old structured emulsifiers.

## Optional controls

| Option | Effect |
|---|---|
| `--unstructured` | Use the Section 3.1 emulsifier bounds in a structural Round 2 mode. This is automatic in `--round3`. |
| `--precise-rank` | Average structural endpoint costs over the relevant linear-system rank distributions. This is the default. |
| `--full-rank` | Replace the rank average by the maximal-rank approximation. |
| `--fixed-p P` | Fix the number of blocks to `P` instead of optimizing it. |
| `--fixed-d D` | Fix the target-space dimension. `D = 0` selects the single-target baseline. |
| `--omega OMEGA` | Set the linear-algebra exponent; the default is `2.81` and the accepted range is `[2, 3]`. |
| `--theta THETA` | Set the field-operation-to-bit-operation exponent; the default is `2` and the accepted range is `[0, 2]`. |

Rank options are meaningful only in structural modes. `--fixed-d` requires a
multi-target mode, except that `--round3` accepts only `--fixed-d 0`.

## Cost model

All reported values are base-2 logarithms of classical bit operations. For
each MQ instance, the program minimizes over all applicable algorithms exposed
by `MQEstimator`; tiny systems retain an explicit elementary fallback when the
library has no finite estimate.

The main modeling conventions are:

- Round 2 structural modes use `w = u + r` and treat the selected dimension
  bounds as equalities.
- Structured emulsifier bounds are used by default; `--unstructured` selects
  the bounds from Section 3.1 of the paper.
- The combined and multi-target-only searches include `d = 0`, so the reported
  optimum can fall back to a single target.
- The target-restart factor is paid only when `d > 0`.
- In Round 3 mode,
  `u = 2*kappa + a`,
  `r = kappa*a - a*(a-1)/2`, and
  `w = min(m, r + 3*kappa + 2*a)`.
  Linear terms enlarge `W`, but not the polar-image space `R`.
- Round 3 mode chooses one full-copy base vector in `ker(Lambda)` and is
  restricted to `q = 16`.

These are asymptotic operation-count estimates under the random-system and
rank-distribution assumptions stated in the paper; they are not wall-clock
benchmarks.

## Reading the output

The report has three parts:

1. **Inputs** records the selected mode and cost-model options.
2. **Best parameters** gives the optimizer's choice of `d`, `p`, `k`, `B1`,
   the block schedule, and the mode-specific dimensions.
3. **Estimated cost** gives the contribution of each attack stage and a final
   `TOTAL` of the form `2^x` bit operations.

The integer values in the paper's tables are the corresponding `TOTAL`
exponents rounded to whole bits.

## Cache and reproducibility

MQ estimates are cached in `.optimisation_cache.json` beside the script. The
cache key includes the MQ dimensions, field size, estimator revision, `omega`,
`theta`, and estimator policy, so changing any of these values does not reuse
an incompatible result. The cache affects running time only; deleting it
forces a cold run without changing the mathematical optimization.

Runs can be computationally expensive, especially with an empty cache. The
optimizer is deterministic for fixed inputs, options, estimator revision, and
cost model.

## Troubleshooting

- **`CryptographicEstimators must be beside this script`**: clone or copy the
  required checkout into `./CryptographicEstimators`.
- **`Estimator dependency unavailable`**: activate the virtual environment or
  reinstall the local estimator with
  `.venv/bin/python -m pip install ./CryptographicEstimators`.
- **No feasible parameter set**: check that the supplied MAYO tuple and any
  fixed `p` or `d` satisfy the constraints of the selected mode.
