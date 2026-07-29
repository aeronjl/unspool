# Fixed-scale partial-pooling benchmark

This synthetic benchmark asks one bounded question: when subjects differ by Gaussian
deviations from a population Bernoulli history GLM, does the first Behavio hierarchical
fit improve recovery and future-session prediction relative to complete pooling or fitting
every subject independently?

```bash
uv run python -m benchmarks.hierarchical_glm.benchmark
```

The matched design contains 12 subjects, three 35-trial training sessions, and one held-out
future session. It repeats low (`0.1`), moderate (`0.5`), and high (`1.0`) subject
heterogeneity 20 times. Each generated dataset is fit with:

- **complete pooling:** one static coefficient vector for all subjects;
- **independent fitting:** one static fit per subject;
- **partial pooling:** population coefficients plus Gaussian-penalized subject deviations.

The partial-pooling fit is given the true generating subject scale. This deliberately
isolates whether the implemented MAP shrinkage works; it does not test variance-component
estimation. The held-out session belongs to subjects represented in training, so this also
does not establish performance for a new animal.

## Pinned result

In the committed 20-repetition result, partial pooling has the lowest mean individual-
coefficient RMSE and prospective log loss in all three regimes. At low heterogeneity it is
nearly equivalent to complete pooling. As heterogeneity increases, it avoids the bias of
complete pooling while improving on the variance of independent fits.

The exact outputs are retained in [`result.json`](result.json). They are evidence for this
design and parameter distribution, not a general guarantee that partial pooling wins on
every dataset. Learning the subject scale, propagating its uncertainty, and testing truly
unseen-animal prediction remain separate validation targets.
