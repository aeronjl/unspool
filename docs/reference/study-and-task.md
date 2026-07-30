# `behavio.trials` and `behavio.task` API

Use these objects to preserve longitudinal identity and declare the meaning of observed
trial columns before choosing a model.

## Longitudinal studies

::: behavio.trials
    options:
      members_order: source
      show_root_heading: false
      show_source: false

## Task observations

::: behavio.task.spec
    options:
      members_order: source
      show_root_heading: false
      show_source: false

## Response times

::: behavio.task.response_times
    options:
      members_order: source
      show_root_heading: false
      show_source: false

## Controlled vocabulary

The closed sets a declaration is written in. See
[the task ontology](../task-ontology.md) for how the named layer produces the structural
one above.

::: behavio.task.vocabulary
    options:
      members_order: source
      show_root_heading: false
      show_source: false

## Task families, protocols, and canonical trials

::: behavio.task.ontology
    options:
      members_order: source
      show_root_heading: false
      show_source: false

## Emitted JSON Schema

::: behavio.task.jsonschema
    options:
      members_order: source
      show_root_heading: false
      show_source: false
