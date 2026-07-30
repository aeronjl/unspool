# Normative belief updating

Two families, one shape. A **perceptual model** turns a sequence of binary observations into
a belief held *before* the next trial is seen; a **response model** turns that belief into
the probability of the subject's own binary response:

\[
\hat{\mu}_k = \text{perception}(u_{1:k-1};\ \vartheta), \qquad
P(y_k = 1) = \sigma\!\left(\eta(\hat{\mu}_k;\ \zeta)\right).
\]

The two are separate objects because that is the field's own distinction and because the
same belief trajectory is routinely read out through different decision rules. Both response
models here are a logistic function of a linear predictor in the belief, so the likelihood,
its gradient, the solver and all three combinators are written once and a new response rule
costs one linear predictor and its two derivatives.

```python
from behavio.models import (
    BeliefSoftmax,
    BetaBernoulliObserver,
    HierarchicalGaussianFilter,
    UnitSquareSigmoid,
)

observer = BetaBernoulliObserver(response=BeliefSoftmax())
filter_model = HierarchicalGaussianFilter(levels=3, response=UnitSquareSigmoid())
```

Each row carries an `observation` — what the world did — and an `outcome` — what the subject
did. Those are different columns and the difference is structural; see
[below](#the-observation-is-not-the-response).

## The Beta-Bernoulli ideal observer

The foundational normative model, and what most papers mean by "the ideal observer": a
conjugate Beta posterior over a binary outcome's rate, whose counts are discounted by a
retention \(\rho\) each trial so that the observer averages over an effective window of
\(1/(1-\rho)\) trials rather than over the whole session.

\[
\alpha_k = \alpha_0 + \sum_{s<k}\rho^{\,k-1-s} u_s, \qquad
\beta_k  = \beta_0  + \sum_{s<k}\rho^{\,k-1-s} (1 - u_s),
\qquad \hat{\mu}_k = \frac{\alpha_k}{\alpha_k + \beta_k}.
\]

At \(\rho = 1\) this is the **exact** Bayesian posterior mean for a static rate,
\((\alpha_0 + n_1)/(\nu + n)\), which is asserted bitwise rather than approximately. Below
one it is the exponential-forgetting approximation to a change-point process.

### A leaky ideal observer *is* a delta rule, with a number

Rearranged, the update is Rescorla-Wagner with a decay toward the prior mean:

\[
\mu_{k+1} = \mu_k + \frac{u_k - \mu_k}{n_{k+1}}
  + \frac{(1 - \rho)\nu\,(m - \mu_k)}{n_{k+1}}, \qquad
n_{k+1} = \rho n_k + (1 - \rho)\nu + 1.
\]

The count recursion contains no observation at all, so it converges to
\(n^{*} = \nu + 1/(1-\rho)\) whatever the data do, and the asymptotic learning rate is the
closed form \(1/n^{*}\). That is what makes "a normative observer is a delta rule" a
statement with a number in it rather than an analogy, and it is what
`BetaBernoulliParameters.asymptotic_learning_rate` returns.

```python
from behavio.models import beta_bernoulli_beliefs

trajectory = beta_bernoulli_beliefs([1, 1, 0, 1, 0, 0], retention=0.9)
trajectory.belief  # one-step-ahead P(u_k = 1 | u_{1:k-1})
trajectory.learning_rate  # 1 / n_k, the step the prediction error was multiplied by
```

The prior is **declared** at \(\text{Beta}(1,1)\) by default, because a block of more than a
few dozen trials has washed it out entirely and \(\nu\) and \(\rho\) both control how much
one observation moves the belief. Pass `prior_mean=None` or `prior_strength=None` to
estimate them — which is right for a design built to identify them, one with many short
blocks — and read `parameter_correlation()` on the result. A design of long blocks gets a
`washed_out_prior` finding instead.

## The Hierarchical Gaussian Filter

Two or three levels. A binary observation \(u\); a Gaussian belief about its tendency
\(x_2\) on the **logit** scale; and, at three levels, a Gaussian belief about that tendency's
log volatility \(x_3\). The level-two update is exactly a delta rule,
\(\mu_2^{(k)} = \mu_2^{(k-1)} + \sigma_2^{(k)}(u_k - \hat{\mu}_1^{(k)})\), whose learning
rate is the posterior variance — so unlike a Rescorla-Wagner agent the filter's step size
moves with how uncertain and how volatile it currently believes the world to be.

Hold the volatility constant — a two-level filter, or \(\kappa_2 = 0\) — and that variance
obeys a data-independent recursion with a unique attracting fixed point:

```python
from behavio.models import hgf_beliefs, hgf_fixed_learning_rate

hgf_fixed_learning_rate(tonic_volatility=-3.0)  # the Rescorla-Wagner rate it reduces to
hgf_beliefs([1, 1, 0, 1], tonic_volatility=-3.0)  # two levels; add the third with kwargs
```

The honest qualification, which the module states rather than glosses: a binary observation
makes that fixed point depend on the current belief through \(\hat{\mu}_1(1-\hat{\mu}_1)\),
so the reduction is exact where the belief is stationary and approximate elsewhere. It is
exactly Rescorla-Wagner for *every* trajectory only in the continuous-input HGF, which is
not this one.

### This is a clean-room implementation, and every disputed convention is declared

`pyhgf` is not wrapped, vendored, or consulted for numbers. Its licence is stated as GPL-3.0
on PyPI and differently in its repository, and it pins `jax<0.4.32`, which cannot share an
environment with the other extras this package offers. Everything here is written from
Mathys et al. (2011, 2014) and validated against closed forms.

That matters more than it usually would, because implementations of these equations disagree
with each other in ways **a fitted number cannot reveal**. A wrong choice in any of the four
places below produces a filter that runs, converges, reports finite standard errors, and
gives the parameter the units the paper says it has. The module docstring states each choice
next to the equation it settles:

| Convention | What this module does | What the alternative reading changes |
| --- | --- | --- |
| The binary first level's contribution | \(\pi_2 = \hat{\pi}_2 + \hat{\mu}_1(1-\hat{\mu}_1)\), a **variance** | adding \(\hat{\pi}_1\) inverts the update's dependence on the observer's current confidence |
| Volatility coupling | \(\nu_2 = \exp(\kappa_2\mu_3^{(k-1)} + \omega_2)\): inside the exponent, on the **previous** trial | a filter that is no longer one-step-ahead everywhere |
| The level-three prediction error | \(\delta_2 = (\sigma_2 + (\mu_2 - \hat{\mu}_2)^2)\hat{\pi}_2 - 1\), posterior moments against the prediction precision | a different weighting of volatility evidence |
| A negative \(\pi_3\) | refused, never clipped | a clipped filter reports numbers from outside its own assumptions |

Four known answers carry the file: the exact static Beta-Bernoulli posterior mean; the
closed-form asymptotic learning rate; the Rescorla-Wagner reduction, asserted to twelve
decimal places by driving the filter with a belief-neutral observation; and \(\kappa_2 = 0\)
making a three-level filter's first two levels *identical* to a two-level filter's.

### A negative third-level precision is refused, not clipped

\(\pi_3 = \hat{\pi}_3 + \tfrac{1}{2}\kappa_2^2 w_2\bigl(w_2 + (2w_2-1)\delta_2\bigr)\) has a
negative second term whenever \(w_2 < \tfrac{1}{2}\) and the volatility prediction error is
large enough, and for admissible parameter values the sum can cross zero. Past that point the
Gaussian approximation has no posterior to describe.

```python
from behavio.models import NegativePosteriorPrecision

filter_model.volatility_stability(study, parameters)  # ask without raising
```

`hgf_beliefs` and every prediction, score and simulation path raise
`NegativePosteriorPrecision` naming the trial. Inside the objective the region scores a
large **finite** penalty rather than an infinite one — L-BFGS-B's first trial point routinely
lands there, and an infinite value NaNs the line search into returning the starting vector
while reporting convergence. A fit is refused only when *every* restart comes to rest there;
one that succeeds carries the margin it succeeded by as the `minimum_volatility_precision`
derived quantity and sets `boundary_estimate` when that margin is small.

## The third level's volatility does not recover from binary responses

This is the single most useful thing the module tells its constituency, so it is stated
before the fit rather than discovered after publication.

On **every reversal design tested**, displacing \(\omega_3\) by a factor of \(e\) moves the
study's whole belief vector by less than 0.1 in Euclidean norm over 480 trials with
\(\kappa_2 = 1\). No set of binary responses can reveal a difference that small, because the
responses see the parameter *only* through the belief. `describe()` says so before anything
is fitted, and the committed recovery test asserts the failure — in the same run in which
\(\omega_2\) and the decision noise recover with a bias under 0.2.

```python
model = HierarchicalGaussianFilter(levels=3, response=UnitSquareSigmoid())
model.describe(study).findings
# [warning] belief_insensitive_parameter: displacing meta_volatility by 1 on its estimated
# scale moves this study's belief vector by 0.0031 in norm, below the 0.1 a response can be
# expected to reveal ...
```

The mechanism is `belief_sensitivity()`, and it is a **measurement rather than a heuristic**:
the filter is run at its own first restart, then again with one coordinate displaced in each
direction, and what is reported is the norm of the change in the study's whole belief vector.
It is design-specific by construction, and that is the point — whether \(\omega_3\) is
identified is not a fact about the HGF, it is a fact about how much volatility a particular
observation sequence contains. Bring a design that puts it to work and the finding simply
stops firing.

The statistic is a norm rather than a maximum on purpose: a parameter that moves three trials
a long way and four hundred not at all is not identified by four hundred and three trials.

### The other hazards, and where each is reported

| Finding | The design that produces it |
| --- | --- |
| `belief_insensitive_parameter` | any estimated perceptual parameter this study's belief cannot see |
| `stationary_observations` | the observation rate never changes, so there is no volatility to estimate and no change for a leak to be identified by |
| `short_blocks_for_volatility` | a three-level filter on blocks shorter than a volatility could have changed within |
| `coupled_volatility_scale` | both \(\kappa_2\) and \(\omega_3\) estimated: they set the third level's scale together |
| `degenerate_observations` | a block whose observations are all identical, so the belief saturates |
| `constant_response` | every response in the study is the same |
| `washed_out_prior` | the Beta prior is estimated from blocks long enough to have forgotten it |

None is an error, because each is a statement about a design only its author can change. The
post-fit half arrives through machinery that already exists: a coordinate resting on its box
sets `boundary_estimate`, and the correlation the coupling trade-off actually produces is
read straight off the fitted covariance by
`coupling_volatility_correlation(fit)`, exactly as the economic family's
`temperature_scale_correlation` does.

## Declared, rather than estimated against nothing

`kappa2 = 1`, `mu2(0) = 0` and \(\text{Beta}(1,1)\) are the defaults TAPAS and the textbook
use. Passing `None` estimates them instead:

```python
free_coupling = HierarchicalGaussianFilter(levels=3, volatility_coupling=None)
free_start = HierarchicalGaussianFilter(levels=3, initial_belief=None)
identified_prior = BetaBernoulliObserver(prior_mean=None, prior_strength=None)
```

A declared parameter **leaves the model entirely**: it is absent from `parameter_names`, from
the box, from the restarts and from every combinator, and its value goes into the model
signature so a fit can never be read without it. So do \(\sigma_2^{(0)}\),
\(\sigma_3^{(0)}\) and \(\mu_3^{(0)}\), which are constructor arguments and are never
estimated.

`omega3` is estimated under **no prior**. TAPAS estimates it under a declared Gaussian, which
would make the fit a MAP fit and would return an `omega3` on every design; a prior-dominated
estimate is not a recovery, and reporting one as though it were is the failure this module
exists to avoid.

## Two response models, and why neither is baked in

| | `BeliefSoftmax` | `UnitSquareSigmoid` |
| --- | --- | --- |
| Predictor | \(\eta = \beta(2\hat{\mu} - 1) + b\) | \(\eta = \zeta\,\text{logit}(\hat{\mu})\) |
| Coordinate | `inverse_temperature_log`, optional `choice_bias` | `decision_noise_log` |
| Use it when | the response is a choice between the two outcomes | the response is a bet read off the belief's log odds |

`UnitSquareSigmoid` is Mathys et al.'s rule and is *exactly* `expit(zeta * logit(belief))`,
which is why it is the same shape of object as a softmax rather than a special case.
\(\zeta = 1\) is exact probability matching. The practical difference: a belief of 0.99 is
4.6 log odds but only 0.98 value units, so the unit-square sigmoid keeps discriminating
between confident beliefs where a softmax has already saturated. Which is right is a claim
about the task.

`BeliefSoftmax` multiplies a difference of expected values, which is what makes its
\(\beta\) comparable with the inverse temperature of a reinforcement-learning policy or of a
value-based choice model.

## The observation is not the response

This is the structural fact that separates these models from the reinforcement-learning
families, and it decides what the combinators may do with them.

A Q-learning agent's value trace is written by **the action the agent took**, so the
likelihood is a genuine recursion and row \(k\) has no density of its own. A normative
observer's belief trajectory is written by **the task's observations**, which are exogenous,
so conditional on the parameters the rows *are* independent and each one does have a density.

That is why `mix()` may put a lapse on an observer and may not put one on an agent, and it is
scientifically the right answer: a lapse on the response leaves the perceptual model
untouched, because the perceptual model never saw the response.

```python
from behavio.compose import UniformChoiceGuess, hierarchical, mix, smooth

lapsing = mix(observer, UniformChoiceGuess(), weight_bounds=(0.0, 0.3))
per_animal = hierarchical(filter_model, over="subject", parameters=("tonic_volatility",), scale=0.5)
drifting = smooth(
    filter_model, over="session_order", knots=(0.0, 4.0), parameters=("decision_noise_log",)
)
```

All three combinators worked untouched, and the three stack.

### What survives is the other half of the block rule

A *perceptual* parameter that changed part-way through a session would leave the belief
trajectory unable to say which of its values produced which part. So the perceptual
coordinate must be constant within each subject/session block **even though every row is its
own density**, and those two statements are different.

`row_blocks` reports the density statement — `arange(n_rows)` — because that is what `mix()`
asks it and what a per-row mixture responsibility requires. The coordinate statement is
enforced by the model itself, against the recursion's own blocks and against the perceptual
columns only. The result is finer than the reinforcement-learning families' rule rather than
weaker:

```python
smooth(filter_model, over="trial", knots=(0.0, 200.0), parameters=("tonic_volatility",))
# refused: a within-session clock on a perceptual parameter

smooth(filter_model, over="trial", knots=(0.0, 200.0), parameters=("decision_noise_log",))
# fitted: a decision noise may drift within a session
```

## What a fitted HGF does not establish

A tonic volatility is a property of a fitted filter on a declared observation sequence, not a
trait of a subject. It moves with the initial variances, with the coupling, with the response
rule, and with how much volatility the sequence actually contained.

**A three-level fit is not evidence that the subject tracked volatility.** On a
reversal-learning design the third level's \(\omega_3\) is not identified by binary
responses at all, and a number reported for it comes from the restart and the box rather than
from the data. `describe()` names it before the fit; a two-level filter is the honest model
for such a design, and the finding should be read rather than filtered.

**A recovered \(\omega_2\) is not a "learning rate" without its reduction.** It is a log step
variance; `hgf_fixed_learning_rate` is what turns it into a rate, and only where the belief
is stationary.

**Agreement with another implementation is not validation.** Four published implementations
disagree about the equations above and all four converge. That is why this one is checked
against closed forms.

## References

- Mathys, C., Daunizeau, J., Friston, K. J., & Stephan, K. E. (2011). A Bayesian foundation
  for individual learning under uncertainty. *Frontiers in Human Neuroscience, 5*, 39.
- Mathys, C. D., Lomakina, E. I., Daunizeau, J., Iglesias, S., Brodersen, K. H., Friston,
  K. J., & Stephan, K. E. (2014). Uncertainty in perception and the Hierarchical Gaussian
  Filter. *Frontiers in Human Neuroscience, 8*, 825.
- Behrens, T. E. J., Woolrich, M. W., Walton, M. E., & Rushworth, M. F. S. (2007). Learning
  the value of information in an uncertain world. *Nature Neuroscience, 10*(9), 1214-1221.
- Rescorla, R. A., & Wagner, A. R. (1972). A theory of Pavlovian conditioning. In A. H. Black
  & W. F. Prokasy (Eds.), *Classical Conditioning II* (pp. 64-99). Appleton-Century-Crofts.

The clean-room decision, the licence and pin analysis, and every convention above are
recorded in
[SDR-0062](decisions/0062-implement-normative-belief-updating-clean-room.md).
