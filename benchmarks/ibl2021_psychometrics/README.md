# IBL 2021 psychometric and training-duration replication

This benchmark attempts a **replication of numbers other people published**. It takes five
values printed by the International Brain Laboratory in *eLife* (2021), recomputes them
from the public behaviour release, and compares them against tolerances fixed in
[`PROTOCOL.md`](PROTOCOL.md) **before** the analysis was run.

That is the distinction from this repository's other six `ibl2021*` benchmarks, which run
Behavio's pipeline on IBL data and report their own numbers. Those are engineering
benchmarks. This one can be wrong in a way the IBL would recognise.

> International Brain Laboratory, Aguillon-Rodriguez V., Angelaki D., Bayer H., Bonacchi N.,
> Carandini M., et al. (2021). Standardized and reproducible measurement of decision-making
> in mice. *eLife* 10:e63711. <https://doi.org/10.7554/eLife.63711>

## Outcome: `failed-parity`

**Five of six checkable claims reproduce. One fails, and it is retained.**

| Claim | Published | Reproduced | Tolerance | Status |
| --- | ---: | ---: | ---: | --- |
| Contrast threshold σ during training (Fig 2b) | 17.8% | **16.28%** | ±1.938 | pass |
| Error rate on easy trials at proficiency (Fig 3c) | 9.5% | **9.65%** | ±2.667 | pass |
| Contrast threshold σ at proficiency (Fig 3d) | 14.3 | **14.17** | ±2.815 | pass |
| Training days to proficiency (Fig 2g) | 18.4 | **16.67** | ±2.153 | pass |
| Training kilotrials to proficiency (Fig 2 suppl. 1) | 10.8 | **10.07** | ±1.425 | pass |
| Mice reaching proficiency (Fig 2 legend) | 140 | **84** | ±5% (±7) | **fail** |
| Bias shift at 0% contrast (Fig 4d,e) | 28.5% | — | — | waived |

Because the gate `contract_passed` requires *every* non-waived claim, the benchmark is
classified **`failed-parity`**, not `published-parity`. The failure is kept rather than
engineered away; see [Why the cohort claim fails](#why-the-cohort-claim-fails).

Both Figure 3 values additionally clear the far stricter non-gating band that the
across-mice reading of the paper's dispersion would imply (±0.596 and ±0.630) — they land
within **0.15 and 0.13** of the published numbers respectively.

## Three corrections to how these values are usually described

Reading the article rather than trusting a summary changed the analysis three times. All
three are recorded in `PROTOCOL.md`.

1. **17.8% and 14.3 are the same parameter.** The fitted psychometric has no slope distinct
   from the threshold σ. The Results prose calls σ "the slope of the curves, which measures
   contrast sensitivity" while Figure 3d's own legend says "Same, for contrast threshold and
   bias". They differ only because they are measured over different session sets: all
   training sessions versus the three proficiency sessions. A replication that fits a
   "slope" and a "threshold" separately has misread the paper.
2. **The 9.5% "lapse rate" is not the fitted lapse.** The paper defines it in-line as "the
   errors made in response to easy contrasts of 50% and 100%", and Figure 3c plots
   *performance on easy trials*. It is an empirical error rate. Averaging the fitted γ and
   λ does not reproduce it.
3. **17.8% is a during-training value, not a trained-state value** (Figure 2b), which is why
   it is larger than the 14.3 measured at proficiency.

## Psychometric parameterisation

The paper's Methods print

```
P(rightward) = γ + (1 − γ − λ) · [ erf( (c − µ) / σ ) + 1 ] / 2
```

with `c` in **signed percent contrast** (−100…+100), `µ` the bias, `σ` the contrast
threshold, and `γ`, `λ` the left and right lapse rates. This is `erf_psycho_2gammas` from
`cortex-lab/psychofit`, parameter order `[bias, threshold, lapse_low, lapse_high]`.
[`psychometric.py`](psychometric.py) reimplements it, together with the released binomial
objective, its box constraints as a `1e7` penalty, and unconstrained Nelder–Mead with five
restarts. σ is reported directly, never inverted.

Behavio's own `LapsePsychometric` is deliberately **not** used: it is a logistic with a
single symmetric lapse and would not be comparable to the published parameters.

**The IBL uses two different fits, and this benchmark uses both.** They are not
interchangeable:

| Purpose | Source | `parstart` | `parmin` | `parmax` |
| --- | --- | --- | --- | --- |
| Deciding `trained_1a`/`1b` | `ibllib` `compute_psychometric` | `[mean(c), 20, .05, .05]` | `[min(c), 0, 0, 0]` | `[max(c), 100, 1, 1]` |
| The published figure values | `paper_behavior_functions.fit_psychfunc` | `[0, 20, .05, .05]` | `[min(c), 5, 0, 0]` | `[max(c), 40, 1, 1]` |

The released routine seeds four of its five restarts from an **unseeded** uniform draw, so
the published pipeline is not reproducible run to run. This benchmark pins the seed and
measures the cost: `threshold_seed_sensitivity_pct` is **3.8 × 10⁻⁶**, so the pinned seed is
doing none of the reproduction's work.

## Cohort

Selection reads only protocol names and dates — never a choice, reward or accuracy — so it
cannot select for the outcome being replicated.

- 138 subjects whose earliest session in release `2021_Q1_IBL_et_al_Behaviour` runs
  `trainingChoiceWorld`; 3,058 training sessions; 1,744,819 trials; 143 MB.
- All **7** of the paper's institutions, recovered from the 9 released laboratory names by
  the IBL's own mapping (`hoferlab` + `mrsicflogellab` → SWC, `churchlandlab` + `zadorlab` →
  CSHL).
- 84 of those subjects then meet `trained_1a` or `trained_1b` on three consecutive sessions,
  using the criteria transcribed from the Methods and cross-checked against
  `brainbox.behavior.training`.

## Why the cohort claim fails

The paper analyses **140** mice that reached basic-task proficiency, from **206** that
entered training. This replication finds **84**, and the shortfall is a property of the
public release rather than of the analysis:

- The release exposes 170 subjects in total, of which only **139** begin with a
  `trainingChoiceWorld` session; the other 31 appear first in `biasedChoiceWorld`, their
  training history absent. One of the 139 carries no training trial table, leaving **138**
  reachable. At least 67 of the paper's 206 mice are therefore not in public at all.
- Of the 138 reachable mice, 84 meet the criterion — a proficiency rate of **60.9%**, close
  to the paper's own **68.0%** (140/206).

`result.json` carries the breakdown of the 54 that do not, so the failure documents itself:

| Why a reachable mouse is not counted proficient | Mice |
| --- | ---: |
| Never clears the behavioural gates — per-session trial count and easy-trial performance — in any three-session window | **41** |
| Clears the behavioural gates somewhere but fails the psychometric bounds | 10 |
| Has fewer than three sessions in the release | 3 |

Only the middle row is a case where a different psychometric implementation could plausibly
disagree. Even recruiting **all** of the borderline and short-history mice would give 97,
still far below 140. The shortfall is therefore data availability, not a criterion this
benchmark got wrong — and it is reported as `fail` rather than waived because the published
number genuinely is not recovered.

Its most important consequence is that the five passing claims are means over 84 mice, not
the paper's 140 — which is exactly why the tolerances were pinned to the paper's published
dispersion rather than to any quantity this cohort controls.

## A retained data-integrity finding

For **1 of 3,058** datasets (`4d7dab62-c777-4faa-9559-e91ab11b4609`, KS016 2019-08-13), the
release's `datasets` index records 47,770 bytes / MD5 `6509885f…` while the bucket serves
40,136 bytes / MD5 `17ea8854…`. **ONE's own `check_hash=True` does not raise on this.** The
manifest therefore pins both the index's `md5` and the served `content_md5`, and
`fetch_data.py` enforces the served digest once pinned. `test_ibl2021_psychometrics_benchmark.py`
asserts the mismatch count, so if the release is repaired the test will say so.

## Reproduce

```bash
uv run --extra ibl python -m benchmarks.ibl2021_psychometrics.fetch_data
uv run --extra ibl python -m benchmarks.ibl2021_psychometrics.benchmark \
  --output benchmarks/ibl2021_psychometrics/result.json
```

The fetch verifies 3,058 checksum-pinned trial tables (~143 MB, about two minutes) into the
Git-ignored cache. The benchmark then loads those exact dataset UUIDs through
`behavio.adapters.ibl_one`, which re-checks provenance per session and preserves IBL's
native `-1/0/+1` choice coding. The rightward-choice code is **derived from the data**, not
assumed, and recorded in `result.json` as `rightward_choice_code`.

The offline contract in [`published_claims.json`](published_claims.json) is walked by
`tests/test_published_parity.py` on every default test run. The re-run lives in
`tests/test_ibl2021_psychometrics_benchmark.py` under `@pytest.mark.slow` and skips cleanly
when the cache is absent.

## What this does not license

Reproducing these values shows the public release plus the published Methods suffice to
recover the published numbers. It does not validate the IBL's conclusions, does not
establish that standardisation *caused* the cross-laboratory agreement, and does not
reproduce the paper's attrition, since 68 of the 206 mice that entered training are not in
the release.

## Next extension

The waived **28.5%** bias shift at 0% contrast (Figure 4d,e). It needs `biasedChoiceWorld`
sessions, the `ready4ephysrig` cohort of 98 mice, and a decision about two apparent sign
errors in the published full-task criterion ("the bias above 5"; median reaction time "over
2 s" where `trained_1b` required *under* 2 s).
