# SDR-0062: Implement normative belief updating clean-room, with declared conventions

- **Status:** Accepted
- **Date:** 2026-07-30
- **Related decisions:** [SDR-0059](0059-consume-movement-datasets-without-depending-on-movement.md)

## Context

An audit of the model catalogue found zero hits across `src/` for *ideal observer*, *belief
updating*, *HGF*, *predictive coding*, *active inference*, *POMDP* or *Kalman*. The whole
predictive-processing wing was absent, and with it the computational-psychiatry
constituency, for whom the Hierarchical Gaussian Filter is the model that decides whether a
library is usable at all.

Two things had to be settled before writing any code: whether to wrap the existing
implementation, and what to do about the fact that implementations of the HGF's update
equations disagree with each other in ways that are invisible in a fitted number.

### Wrapping `pyhgf`

`pyhgf` is the maintained reference implementation and the obvious candidate. It was
rejected on two grounds, neither of them about its quality.

**Its licence is contradictory.** PyPI states GPL-3.0; the repository states something else.
Behavio is not GPL, and a dependency whose licence cannot be determined from its own
metadata cannot be added to a package other people redistribute. This is the same test
SDR-0059 applied to `movement`, and it fails at the first step rather than the fourth.

**Its pin is incompatible with this package's other extras.** `pyhgf` requires
`jax<0.4.32`. The extras Behavio already offers resolve to newer JAX, so the two cannot
share an environment; a `belief` extra would be an extra nobody with the existing ones could
install.

### The update equations

Mathys et al. (2011) give the filter's equations; Mathys et al. (2014) restate them for the
binary case. Between the papers, the TAPAS MATLAB implementation, and the reimplementations
in the applied literature, at least four things are written differently, and each of them
produces a filter that runs, converges, and reports a number:

1. whether the binary first level contributes its **variance** or its **precision** to the
   second level's posterior precision;
2. whether the volatility coupling multiplies inside or outside the exponent, and whether it
   reads the previous trial's volatility estimate or the current one;
3. how the second level's volatility prediction error is formed -- which mean, which
   variance, and against which precision;
4. what happens when the third level's posterior precision goes negative.

A wrong choice in any of these is not detectable by fitting: the likelihood is smooth, the
optimizer converges, the standard errors are finite, and the parameter has the units the
paper says it has. Only a closed form can tell the difference.

## Decision

**Implement both families clean-room from the published equations, in
`src/behavio/models/belief.py`, and state every disputed convention in the module docstring
next to the equation it settles.**

`BetaBernoulliObserver` is a leaky Beta-Bernoulli ideal observer; `HierarchicalGaussianFilter`
is the binary HGF at two or three levels. `pyhgf` is not imported, vendored, or consulted for
numbers; the only external references are the two papers.

The conventions fixed, each of them asserted in `tests/test_belief.py`:

- **The binary first level contributes a variance.**
  `pi2 = pihat2 + muhat1 * (1 - muhat1)`, which is `pihat2 + 1 / pihat1`. Adding `pihat1`
  itself -- the reading the notation invites -- inverts the update's dependence on the
  observer's current confidence.
- **Volatility couples through the exponent, on the previous trial's estimate.**
  `nu2 = exp(kappa2 * mu3_previous + omega2)`.
- **The third level's prediction errors.** `delta2 = (sigma2 + (mu2 - muhat2)**2) * pihat2 - 1`,
  formed from the *posterior* mean and variance at level two against the *prediction*
  precision; the volatility weight is `w2 = nu2 * pihat2`; and
  `pi3 = pihat3 + 0.5 * kappa2**2 * w2 * (w2 + (2 * w2 - 1) * delta2)`,
  `mu3 += 0.5 * kappa2 * w2 * delta2 / pi3`.
- **No drift, unit trial spacing, declared initial variances.** `rho2 = rho3 = 0`, `t = 1`,
  and `sigma2(0)`, `sigma3(0)`, `mu3(0)` are constructor arguments that appear in the model
  signature.
- **A negative third-level precision is refused, never clipped.** See below.
- **Estimated coordinates are unconstrained.** `omega2` and `omega3` are already log
  variances and are estimated and reported unchanged; `kappa2` is estimated as its logarithm;
  the ideal observer's retention and prior mean are logits and its prior strength a
  logarithm. This is what
  `behavio.contracts.bounded.BoundedCoordinateEstimator` requires of a model that is to be
  made hierarchical or smooth.

**Validate against closed forms rather than against another implementation.** Four known
answers carry the file:

- the ideal observer at `retention=1` is the exact Beta-Bernoulli posterior mean
  `(alpha0 + n1) / (nu + n)`, asserted bitwise;
- its learning-rate recursion contains no observation and converges to
  `1 / (nu + 1 / (1 - rho))`, so a leaky ideal observer *is* a Rescorla-Wagner learner with a
  computable rate;
- the binary HGF's level-two update is exactly `mu2 += sigma2 * (u - muhat1)`, and with the
  volatility held constant `sigma2` converges to the root of `v s^2 + v c s - c = 0`
  (`c = exp(omega2)`, `v = muhat1 * (1 - muhat1)`) -- the **Rescorla-Wagner reduction**,
  asserted to twelve decimal places by driving the filter with an observation of one half,
  which makes the prediction error zero on every trial and lets the step-size recursion run
  alone;
- `kappa2 = 0` makes the third level inert, so a three-level filter's first two levels equal
  a two-level filter's exactly.

**Separate the perceptual model from the response model, as declared components.** A
`BeliefResponse` supplies a linear predictor in the belief and its two derivatives, and
nothing else; `BeliefSoftmax` is a softmax on the belief's value difference and
`UnitSquareSigmoid` is Mathys et al.'s rule, which is exactly
`expit(zeta * logit(belief))` and therefore the same shape of object. A response *lapse* is
not declared here: `behavio.compose.mix` already expresses it.

**Declare, rather than estimate, the parameters most designs cannot locate.** `kappa2 = 1`
and `mu2(0) = 0` for the filter, `Beta(1, 1)` for the observer -- the same defaults TAPAS and
the textbook use. Passing `None` estimates them instead. A declared parameter leaves the
model entirely: it is absent from `parameter_names`, from the box, from the restarts and from
every combinator, and its value is in the model signature.

**Measure identifiability instead of assuming it.** `belief_sensitivity(study)` displaces each
estimated perceptual parameter by one unit on its own scale and reports the Euclidean norm of
the change in the study's whole belief vector. A parameter the responses cannot see through
the belief is a `describe()` finding before anything is fitted.

## Consequences

**The third level's tonic volatility does not recover from binary responses on a
reversal-learning design, and this is now demonstrated rather than discovered.** Displacing
`omega3` by a factor of `e` moves the belief vector by under a tenth in norm on a 480-trial
sequence with `kappa2 = 1`; `describe()` reports `belief_insensitive_parameter`, and
`tests/test_belief.py` asserts both the finding and the failed recovery -- in the same run in
which `omega2` and the decision noise recover with a bias under 0.2. This is a property of
the model and the design, not of the implementation, and it is the single most useful thing
the module tells its constituency.

**A negative third-level posterior precision is surfaced three ways and clipped none.**
`hgf_beliefs` and every prediction, score and simulation path raise
`NegativePosteriorPrecision` naming the trial. Inside the objective the region scores
`VIOLATION_PENALTY` per row -- finite rather than infinite, because L-BFGS-B's first trial
point routinely lands there and an infinite value NaNs the line search into returning the
starting vector while reporting convergence -- and both `fit` and `fit_rows` refuse a result
that comes to rest above that wall. An admissible fit carries the margin it succeeded by as
the `minimum_volatility_precision` derived quantity and sets `boundary_estimate` when the
margin is small.

**All three combinators worked untouched, including `mix()`.** This is the second family
tested against the "a family costs 1x not 4x" claim and the first whose likelihood involves a
recursion. The recursion is driven by the task's *observations*, which are exogenous, so
unlike a value-updating agent each row does have a density of its own and a lapse mixture is
well defined -- and correct, because a lapsed response leaves the perceptual model untouched.

The consequence for `behavio.contracts.bounded.RowObjective` is that its single `row_blocks`
member conflates two questions that come apart here: *which rows have their own density*
(every one) and *which rows must share a coordinate* (all of a session, for the perceptual
parameters only). `row_blocks` reports the first, because that is what `mix()` asks it and
what a per-row mixture responsibility requires; the second is enforced by the model itself
against its own reset blocks and against the perceptual columns only. The result is finer
than the agents' rule rather than weaker: smoothing a tonic volatility over a within-session
clock is refused, and smoothing a decision noise over the same clock is fitted.

The contract was **not** changed. Splitting `row_blocks` into `density_blocks` and
`coordinate_blocks` is the obvious follow-up and is deliberately left for whoever has a second
family that wants it.

## Alternatives considered

**Wrap `pyhgf`.** Rejected on the licence and the JAX pin, above. Reconsider if the licence is
made consistent and the pin is lifted; even then the wrapper would have to reproduce the
closed forms this module is validated against, so the tests survive the change.

**Fix `kappa2` and `omega3` both, and offer only a two-level filter with a third level of
constants.** Rejected: the field means the three-level filter by "HGF", and a filter whose
third level has no estimated parameter at all is not it. The chosen middle -- estimate
`omega3`, declare `kappa2`, measure and report what the design can actually see -- matches
TAPAS's own default configuration.

**Estimate `omega3` under a declared Gaussian prior, which is what TAPAS does.** This would
make the fit a MAP fit rather than a maximum-likelihood one, and would return an `omega3`
estimate on every design. Rejected for now because a prior-dominated estimate is not a
recovery and reporting one as though it were is the failure this module exists to avoid.
`penalty_matrix()` is the member that would carry it, and it is already in the contract, so
the change is additive when someone wants it.

**Analytic forward-mode derivatives through the filter.** The belief's sensitivity to the
perceptual coordinate is central-differenced through the recursion, one extra pass per
coordinate per block, read per row. A hand-written Jacobian would be faster and would be a
second place for a sign to be wrong in exactly the arithmetic this record exists to pin down;
it would also have to be checked against the difference anyway.

**Report `row_blocks` as the session blocks, as the reinforcement-learning agents do.** That
would have refused `mix()`, which is the wrong answer: the mixture is well defined here and
the lapse belongs on the response. See the consequences above.

## Revisit trigger

- `pyhgf` publishes a consistent licence **and** drops the `jax<0.4.32` pin.
- A second bounded-coordinate family needs the density/coordinate block distinction, at which
  point `RowObjective` should carry both rather than one.
- Someone brings a design -- many short blocks, or an explicit stable/volatile manipulation
  with several transitions -- on which `belief_sensitivity` says `omega3` *is* identified. The
  finding is a measurement, so it will simply stop firing, and the negative recovery assertion
  in `tests/test_belief.py` should then be re-examined against that design rather than
  weakened.
- A continuous-input level one is added, at which point the Rescorla-Wagner reduction becomes
  exact for every trajectory rather than only where the belief is stationary, and the
  qualification in the module docstring should move.

## References

Mathys, C., Daunizeau, J., Friston, K. J., & Stephan, K. E. (2011). A Bayesian foundation for
individual learning under uncertainty. *Frontiers in Human Neuroscience, 5*, 39.

Mathys, C. D., Lomakina, E. I., Daunizeau, J., Iglesias, S., Brodersen, K. H., Friston, K. J.,
& Stephan, K. E. (2014). Uncertainty in perception and the Hierarchical Gaussian Filter.
*Frontiers in Human Neuroscience, 8*, 825.

Behrens, T. E. J., Woolrich, M. W., Walton, M. E., & Rushworth, M. F. S. (2007). Learning the
value of information in an uncertain world. *Nature Neuroscience, 10*(9), 1214-1221.
