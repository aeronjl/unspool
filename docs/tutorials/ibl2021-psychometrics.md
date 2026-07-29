# Replicating published IBL psychometrics

This chapter walks through the repository's first **published-parity attempt against another
group's printed numbers**: five values from International Brain Laboratory et al. (2021),
recomputed from the public behaviour release and compared under tolerances fixed before the
analysis ran.

It is worth being precise about why this is different from the other IBL studies here. The
[trajectory study](ibl2021-learning-trajectories.md) and the
[prospective study](ibl2021-prospective-selection.md) run Behavio's pipeline on IBL data and
report *their own* numbers. Nothing in them can disagree with the IBL. This chapter can, and
partly does.

> International Brain Laboratory, Aguillon-Rodriguez V., Angelaki D., Bayer H., Bonacchi N.,
> Carandini M., et al. (2021). Standardized and reproducible measurement of decision-making
> in mice. *eLife* 10:e63711. <https://doi.org/10.7554/eLife.63711>

## What you will get

| Claim | Published | Reproduced | Tolerance | Status |
| --- | ---: | ---: | ---: | --- |
| Contrast threshold σ during training | 17.8% | 16.28% | ±1.938 | pass |
| Easy-trial error rate at proficiency | 9.5% | 9.65% | ±2.667 | pass |
| Contrast threshold σ at proficiency | 14.3 | 14.17 | ±2.815 | pass |
| Training days to proficiency | 18.4 | 16.67 | ±2.153 | pass |
| Training kilotrials to proficiency | 10.8 | 10.07 | ±1.425 | pass |
| Mice reaching proficiency | 140 | 84 | ±5% | **fail** |

Five of six reproduce. The sixth does not, and the benchmark is therefore classified
`failed-parity`. Keeping that is the point of the exercise.

## Run it

Install the optional IBL client, fetch the pinned tables, then run the benchmark.

```bash
uv sync --extra ibl
uv run --extra ibl python -m benchmarks.ibl2021_psychometrics.fetch_data
uv run --extra ibl python -m benchmarks.ibl2021_psychometrics.benchmark
```

The fetch verifies 3,058 checksum-pinned trial tables — 138 mice, seven institutions,
about 143 MB — into a Git-ignored cache, and takes roughly two minutes. The benchmark takes
about ten, most of it in the psychometric fits.

To write the committed result:

```bash
uv run --extra ibl python -m benchmarks.ibl2021_psychometrics.benchmark \
  --output benchmarks/ibl2021_psychometrics/result.json
```

## Read the protocol first

`benchmarks/ibl2021_psychometrics/PROTOCOL.md` was frozen before any claim was computed. It
states the analysis, the cohort criteria, every tolerance and its justification, one dated
amendment made after retrieving the IBL's own analysis source, and a disclosure of the one
quantity the author had already seen. Read it before the code — a replication whose
acceptance criteria were chosen after seeing the answer is not a replication.

## Three things the paper does not say the way people repeat it

Working from the article rather than a summary changed the analysis three times.

**The "threshold" and the "slope" are one parameter.** The fitted psychometric

```
P(rightward) = γ + (1 − γ − λ) · [ erf( (c − µ) / σ ) + 1 ] / 2
```

has no slope distinct from σ. The Results text calls σ "the slope of the curves"; Figure 3d's
own legend calls the same panel "contrast threshold". The published 17.8% and 14.3 are the
same quantity measured over different session sets — all training sessions, versus the three
sessions at which the mouse reached proficiency.

**The published 9.5% "lapse rate" is not a fitted lapse.** The paper defines it in-line as
"the errors made in response to easy contrasts of 50% and 100%". It is an empirical error
rate, computed here from `feedbackType`. Averaging the fitted γ and λ will not reproduce it.

**The IBL uses two different fits.** Training status is decided with threshold bounded to
[0, 100]; the published figure values come from a fit with threshold clamped to [5, 40].
Using one where the other belongs changes the answer.

## How the code is arranged

```python
from benchmarks.ibl2021_psychometrics.benchmark import (
    load_study,
    rightward_choice_sign,
    summarize_sessions,
    summarize_subjects,
)

study = load_study()  # exact dataset UUIDs, hash-checked
sign = rightward_choice_sign(study)  # derived from data, never assumed
sessions = summarize_sessions(study, rightward_sign=sign)
subjects = summarize_subjects(sessions)  # applies trained_1a / trained_1b
```

`load_study` goes through `behavio.adapters.ibl_one`, which asks ONE for each pinned dataset
UUID with hash checking and stamps the Alyx origin, release tag, session UUID, dataset UUID,
path, size and MD5 onto every trial.

That adapter deliberately preserves IBL's native `-1 / 0 / +1` choice coding rather than
silently binarising it, which leaves you responsible for the convention. Rather than
hard-code it, `rightward_choice_sign` derives it: on rewarded, non-zero-contrast trials, it
counts which sign co-occurs with a right-side stimulus and refuses to proceed unless the
mapping holds for over 99% of them. The answer, `-1`, is recorded in `result.json`.

## Choosing a tolerance you cannot game

Every published value here is a cohort mean with a published dispersion, so a
rounding-width tolerance would be dishonest — it would guarantee failure for reasons
unrelated to whether the analysis was reproduced. The rule fixed in the protocol is:

> A claim passes if the reproduced mean falls inside the 95% interval that the paper's own
> reported dispersion places around its own mean: `1.96 × SD_published / √n_published`.

Three properties matter. It depends only on published quantities, so it **cannot be widened
by narrowing the reproduced cohort** — the usual way this kind of tolerance is abused. It is
tighter than the two-sample band that treating the cohorts as independent samples would
justify. And in relative terms it is narrow: ±2.15 days on 18.4 is ±11.7%, while a
misimplemented proficiency criterion moves training duration far more than that.

Where the paper is internally inconsistent — it prints "n = 7 laboratories" beside values
whose figure panels plot one point per mouse — the **more permissive** printed reading gates
the claim, and the stricter band is computed and reported as non-gating evidence. Both
Figure 3 claims clear the strict band too, landing within 0.15 and 0.13 of the published
values. The reproduced across-mice standard deviation for the proficiency threshold is
3.799 against a published 3.8, which is itself evidence that the printed "n = 7" annotation
is the error.

## Why one claim fails, and why it is kept

The paper analyses 140 mice that reached proficiency, out of 206 that entered training. This
replication finds 84. The reason is data availability, not analysis:

- Of 170 subjects in the release, only 139 begin with a training session at all; the other
  31 appear first in the biased task, their training history absent. At least 67 of the
  paper's mice are not public.
- Of the 138 reachable mice, 84 meet the criterion — a 60.9% proficiency rate against the
  paper's own 68.0%.

`result.json` records why each of the other 54 is not counted: **41** never clear the
behavioural gates (per-session trial count and easy-trial performance) in any three-session
window, 10 clear those gates but fail the psychometric bounds, and 3 have fewer than three
sessions. Only those 10 are cases a different psychometric implementation might plausibly
recruit, and even taking all 13 borderline mice would reach 97 — still far below 140.

The rate is about right; there are simply fewer mice. That is a genuine failure to recover a
published number, so it is recorded as `fail`, the gate `contract_passed` is false, and the
benchmark is classified `failed-parity` even though five of six values reproduce. The
tolerance was not widened and the cohort was not narrowed to move it.

## Checking it stayed true

`published_claims.json` is walked by `tests/test_published_parity.py` on every default test
run, offline and in milliseconds — a silent drift away from a published number cannot
survive it. The re-run lives in `tests/test_ibl2021_psychometrics_benchmark.py` behind
`@pytest.mark.slow`, so the nightly benchmark tier recomputes the science rather than reading
it back.

```bash
uv run pytest tests/test_published_parity.py     # offline contract
uv run pytest -m slow                            # nightly re-execution
```

## What it does not license

Recovering these values shows that the public release plus the published Methods suffice to
recompute the published numbers. It does not validate the IBL's conclusions, does not show
that standardisation *caused* the cross-laboratory agreement, and does not reproduce the
paper's attrition, since a third of the mice that entered training are not in the release.

One further finding is retained rather than smoothed over: for one of the 3,058 datasets the
release index's recorded MD5 and byte size disagree with the object the bucket actually
serves, and ONE's own `check_hash=True` does not raise. The manifest pins both digests so
that a change in either is visible.
