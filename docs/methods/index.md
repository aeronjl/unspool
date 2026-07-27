# Methods catalog

Unspool groups methods by the explanation they offer for change. Availability is not the
same as validation: consult the [capability matrix](capability-matrix.md) and each model's
recovery evidence.

<figure class="doc-figure doc-figure--wide">
  <img src="../assets/model-atlas.svg" alt="Five conceptual trajectories comparing a static GLM, smooth drift, GLM-HMM regimes, Q-learning value updating, and a drift-diffusion tendency across early-to-late learning.">
  <figcaption><strong>Model-family atlas.</strong> Each family encodes a different explanation of behavioural change. The curves are conceptual signatures, not fitted data or evidence that any family is correct.</figcaption>
</figure>

## Stable choice structure

The Bernoulli history GLM estimates stimulus and choice-history effects that remain fixed
through the study. Hierarchical variants share information across animals without treating
trials as independent subjects.

## Smooth longitudinal change

Fixed-knot GLMs and drift-diffusion models allow declared parameters to vary over an
explicit clock. Hierarchical smooth models separate a population trajectory from shrunken
individual deviations.

## Discrete latent regimes

The GLM-HMM represents switching among discrete emission states. Filtered prediction is
kept separate from smoothed description, and latent-state labels are aligned only through
permutation-invariant recovery.

## Reward-driven learning

The binary Q-learning agent represents trial-by-trial value updating. It competes under the
same pointwise predictive contract as the GLM and GLM-HMM families.

## Choice and response time

Wiener drift-diffusion models jointly score choice and response time. Static, smooth,
hierarchical, and explicit-contaminant variants share physical-unit and fit-audit
contracts.

## Comparison is part of the method

A model family becomes interpretable only after specifying its alternatives, prospective
target, aggregation unit, fit diagnostics, and design-specific recovery. See
[model comparison](../comparison.md) and [model recovery](../model-recovery.md).
