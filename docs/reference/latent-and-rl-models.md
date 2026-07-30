# `behavio.models`: latent-state and reinforcement-learning API

Latent states and learned values need observable competitors, prospective predictions,
state or parameter recovery, and explicit reset semantics.

## GLM-HMM

::: behavio.models.glm_hmm
    options:
      members_order: source
      show_root_heading: false
      show_source: false

## Q-learning

::: behavio.models.q_learning
    options:
      members_order: source
      show_root_heading: false
      show_source: false

## Composable RL agents

::: behavio.models.rl
    options:
      members_order: source
      show_root_heading: false
      show_source: false

## Normative belief updating

The belief is a latent state and the response model reads it, so this family belongs beside
the value-learning agents rather than with the observable-choice models — but the recursion
is written by the task's observations rather than by the subject's response, which is what
lets `mix()` reach it. The conventions this module fixes, and the closed forms it is
validated against, are in
[SDR-0062](../decisions/0062-implement-normative-belief-updating-clean-room.md).

::: behavio.models.belief
    options:
      members_order: source
      show_root_heading: false
      show_source: false
