# Cross-lab trajectory component recovery

This benchmark asks whether Unspool keeps four scientifically different statements about
aligned group trajectories separate:

- a constant difference in overall level;
- a difference in the amplitude of change;
- a difference in scale-free shape;
- no shape difference after removing level and amplitude.

```bash
uv run python -m benchmarks.trajectory_shapes.benchmark
```

Twenty matched repetitions simulate four labs with ten independent animals per lab on the
same nine-position learning clock. The reference lab has a linear rise. A second lab adds
a constant, a third doubles the centered amplitude, and a fourth follows a sinusoidal
trajectory. Subject intercepts, amplitudes, and observations vary independently. The
bootstrap resamples animals within each fixed lab.

The matched design also exercises four whole-lab holdouts. Every lab fold contains all
nine clock positions for each of its ten animals, training and test animals are disjoint,
and every row is tested exactly once. This validates the split geometry; predictive
transport still depends on the behavioural model fitted inside those folds.

The pinned contract requires every repetition to recover the generating decomposition:
the constant shift is the largest level contrast, the doubled rise is the largest
amplitude contrast, and all three linear trajectories are closer to each other in
scale-free shape than any is to the sinusoidal trajectory. The level and amplitude
bootstrap intervals must also exclude zero in every repetition.

The benchmark separately audits a nine-lab, one-animal-per-lab design. It is correctly
reported as unready for trajectory-shape inference. This mirrors the deliberate limit of
the compact IBL engineering panel: singleton labs can test provenance and split coverage,
but cannot separate an animal trajectory from a lab trajectory.

The exact aggregate is retained in [`result.json`](result.json). These intervals condition
on the four named labs; they do not generalize to a population of laboratories. The API
also performs no interpolation or time warping: callers must declare and construct a
scientifically meaningful common clock first.
