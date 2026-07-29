# Restless bandit: reward history versus incremental value learning

Chen et al. trained 32 mice—16 male and 16 female—for eight sessions on a restless
two-armed bandit, with each arm's reward probability drifting independently. Their open
data make a useful canonical reinforcement-learning recipe because choices, obtained
rewards, and the action-contingent reward environment are all explicit.

This is a **literature-shaped prospective analysis**, not a reproduction of the paper's
reported sex effect. The bounded question is: *after seven sessions from the same animals,
which standard behavioural account best forecasts their eighth session?*

<figure class="doc-figure doc-figure--wide" data-figure-kind="Literature-shaped">
  <img src="../../assets/chen2021-bandit.svg" alt="Four-panel restless-bandit evidence display: an example held-out session, animal-balanced model scores, paired animal differences, and a model-recovery matrix.">
  <figcaption><strong>Literature-shaped · reward-sensitive accounts improve the future-session forecast.</strong> Q-learning's point estimate is better than win–stay/lose–shift, but their paired animal interval crosses zero. This is a new prospective analysis rather than a reproduction of the paper's sex effect.<span class="doc-figure__meta"><strong>Unit:</strong> mouse · <strong>n:</strong> 32 mice, 25,279 trials · <strong>Estimand:</strong> animal-balanced session-8 log loss with animal-bootstrap uncertainty · <a href="../../reference/figure-provenance/">provenance</a></span></figcaption>
</figure>

## Source and runtime

- Paper: Chen et al. (2021), [“Sex differences in learning from
  exploration”](https://doi.org/10.7554/eLife.69748), *eLife* 10:e69748.
- Data: [Dryad `10.5061/dryad.z612jm6c0`](https://doi.org/10.5061/dryad.z612jm6c0),
  CC0; the fetcher uses its stable Zenodo mirror and verifies SHA-256
  `90f0f9fa843a16788d0dcd7b857f81db068e8d18b8dd4eabf20ccaee3b67db04`.
- Runtime: the source download is about 1.2 MB. The empirical fit is quick; the five-repeat
  exact-design recovery is the benchmark step and takes roughly half a minute on a current
  laptop. The committed result lets the documentation build without rerunning either.

```bash
uv run python -m benchmarks.chen2021_bandit.fetch_data
uv run python -m benchmarks.chen2021_bandit.benchmark
```

## Cohort and task

The independent uncertainty unit is the mouse, not the trial. All 32 mice and all eight
sessions enter. To keep recovery appropriate for routine documentation, the recipe retains
the first 100 source rows of each mouse-session. This cap depends only on source order and
was declared before reading choices or rewards. Short sessions remain short, giving 25,279
trials rather than an artificially complete 25,600-row panel.

The source choices 1/2 become canonical actions 0/1. Both drifting reward-probability
columns are preserved so generative models can sample the consequence of the action they
actually choose.

```python
from behavio import ChoiceSpec, RewardSpec, TaskSpec

task = TaskSpec(
    choice=ChoiceSpec(options=(0, 1)),
    reward=RewardSpec(minimum=0, maximum=1),
)
validation = task.validate(study)

assert validation.n_trials == 25_279
assert validation.n_observed_choices == 25_279
assert validation.n_omissions == 0
```

The released `state` column is retained as `source_state` for provenance only. It was
derived by the original authors' HMM and is therefore neither an observation nor a
ground-truth target for fitting or recovery.

## Declare the deployment boundary

One cohort fold fits every model jointly to sessions 1–7 and scores session 8. No held-out
choice or reward affects fitting. Filtered forecasts may update within session 8 after each
observed event, matching online use, but never inspect a later trial.

```python
from behavio import cohort_forward_session_splits

splits = cohort_forward_session_splits(study, min_train_sessions=7)
assert len(splits) == 1
```

## Compare common accounts

The set deliberately starts with simple explanations. Bias asks whether a constant action
preference suffices. Perseveration adds observable choice history. Win–stay/lose–shift adds
the last observed reward. Q-learning maintains session-reset action values and updates the
chosen value with its reward prediction error.

```python
from behavio import (
    BiasOnly,
    BinaryQLearning,
    Perseveration,
    WinStayLoseShift,
    compare_models,
)

models = {
    "bias": BiasOnly(l2=0.01),
    "perseveration": Perseveration(l2=0.01),
    "win-stay-lose-shift": WinStayLoseShift(l2=0.01),
    "q-learning": BinaryQLearning(n_restarts=2, random_seed=2401),
}
comparison = compare_models(
    models,
    study,
    splits,
    aggregation_column="subject",
    bootstrap_resamples=5_000,
    bootstrap_seed=2402,
)
```

| Candidate | Animal-balanced held-out log loss | 95% animal-bootstrap interval |
| --- | ---: | ---: |
| Bias | 0.7079 | [0.6956, 0.7199] |
| Perseveration | 0.6622 | [0.6367, 0.6855] |
| Win–stay/lose–shift | 0.6119 | [0.5810, 0.6430] |
| Q-learning | 0.6033 | [0.5713, 0.6336] |

All four fits pass the numerical audit. Reward-sensitive models improve substantially over
the static accounts. Q-learning's point loss is 0.00853 below win–stay/lose–shift, but the
paired animal-bootstrap interval for `WSLS − Q-learning` is [−0.01381, 0.03152]. The
direction is suggestive, not decisive.

## Recover the distinction, not the story

Recovery uses the exact 32-animal, eight-session environment and the same 7→8 split. It
compares only win–stay/lose–shift and Q-learning: unlike the generic static baselines, both
simulate coherent action-contingent rewards from the explicit environment.

Five fixed-seed repeats recover each of two declared regimes five times. Four Q-learning
fits to win–stay/lose–shift-generated data retain boundary warnings; these warnings are
part of the evidence rather than filtered away.

That diagonal matrix validates discrimination for these parameter values and this design.
It does not show that Q-learning generated the real animals, validate the source HMM's
states, test the paper's sex-effect claim, or establish generality beyond this cohort.

The complete executable source and frozen evidence are in
[`benchmarks/chen2021_bandit`](https://github.com/aeronjl/behavio/tree/main/benchmarks/chen2021_bandit).
