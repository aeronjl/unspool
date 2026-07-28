# Scientific figure and replication standard

Unspool figures are evidence interfaces. A reader should be able to tell what kind of
display they are seeing, what entered it, which quantity it estimates, and how far its
claim reaches without searching the surrounding chapter.

This standard applies to every figure in the documentation. It is deliberately stricter
for empirical and literature-linked displays than for conceptual diagrams.

## Typography and visual grammar

All generated figures use **sans-serif type**. The canonical, redistributable font is
DejaVu Sans so local builds and Linux documentation CI render the same glyphs. Serif
families, decorative display faces, and mixed font families are not permitted.

- Body, axis, tick, and annotation text must remain searchable SVG text rather than paths.
- Text must be at least 7 pt at the exported size; 9 pt is the normal baseline.
- Mathematical text uses a matching sans-serif glyph set.
- Panel letters are bold sans-serif and stay in a consistent upper-left position.
- Colour supplements position, shape, line style, or direct labels; it never carries the
  only distinction.
- Axes state physical units or scoring direction. Captions state the inferential unit.
- White figure grounds are retained in dark mode so scientific colours and contrasts do
  not silently change with the site theme.

The shared generator contract lives in
[`scripts/figure_style.py`](https://github.com/aeronjl/unspool/blob/main/scripts/figure_style.py).
Standalone copy-and-run examples repeat the minimal font settings explicitly so they do
not depend on an internal helper.

## Classification is visible

Every display carries one of these labels in the figure card and in its caption:

| Classification | Meaning |
| --- | --- |
| Exact reproduction | Released inputs and the released specification reconstruct a reported display or result. |
| Independent reproduction | The published scientific quantity is recomputed from source observations. |
| Literature-shaped analysis | A new bounded question uses a published task, dataset, or model structure. |
| Released result | A checksum-pinned released result is summarized without claiming an independent refit. |
| Mixed evidence | Panels combine clearly identified evidence classes. |
| Synthetic benchmark | Known generators test recovery, calibration, or software behaviour under a declared design. |
| Demonstration | Small simulated data teach an API without an empirical claim. |
| Conceptual | Values and geometry are schematic rather than study estimates. |

“Replication” is not a generic badge. A display that changes the outcome, cohort,
candidate set, validation boundary, or estimand is literature-shaped even when it uses the
paper's public data.

## Evidence-card anatomy

Use the following markup for a new display:

```html
<figure class="doc-figure doc-figure--wide"
        data-figure-kind="Independent reproduction">
  <img src="../../assets/example.svg"
       alt="Conclusion-bearing description that does not rely on colour.">
  <figcaption>
    <strong>Independent reproduction · concise result.</strong>
    Interpret the visible pattern and its uncertainty.
    <span class="doc-figure__meta">
      <strong>Unit:</strong> animal · <strong>n:</strong> 30 ·
      <strong>Estimand:</strong> animal-level correlation ·
      <a href="../../reference/figure-provenance/">provenance</a>
    </span>
  </figcaption>
</figure>
```

The card must expose:

1. **classification** — the relationship to prior work or simulation;
2. **source** — public observations, released artifact, frozen benchmark, or schematic;
3. **unit and denominator** — animals, sessions, laboratories, participants, or trials;
4. **estimand** — the plotted quantity and aggregation rule;
5. **uncertainty** — interval type and resampling unit, when applicable;
6. **supported claim** — what the display establishes; and
7. **claim limit** — the nearest tempting interpretation it does not establish.

The versioned [figure manifest](figure-manifest.json) records these fields for every asset.
The [provenance register](figure-provenance.md) is the reader-facing index.

## Literature correspondence

Every literature recipe with a reproduction claim includes a correspondence table before
its first result figure:

| Unspool display | Published target | Relationship | Preserved | Changed or unavailable |
| --- | --- | --- | --- | --- |
| `example.svg`, panel A | Paper Figure X, panel Y | Independent reproduction | cohort rule, outcome, unit | plotting geometry |

Use exact paper panel identifiers only after checking the paper and released analysis.
When panel identity is uncertain, name the reported quantity and say that exact panel
correspondence is unresolved. Never imply like-for-like replication from visual
resemblance alone.

The table should make changes to cohort, preprocessing, model specification, validation,
or uncertainty impossible to miss. A composite Unspool figure may map different panels to
different evidence classes; each panel receives its own row.

## Caption order

A caption reads from result to boundary:

1. classification and one-sentence result;
2. unit, denominator, estimand, and uncertainty;
3. source and panel correspondence;
4. closest claim that remains unsupported.

Do not spend the first sentence describing colours or marks. Those belong in alternative
text when they help a screen-reader user reconstruct the display.

## Reproducibility contract

Generated SVGs must be deterministic, versioned, and reviewable:

```bash
uv run python -m benchmarks.cell2025.fetch_data
uv run python -m benchmarks.cell2025_flagship.fetch_released_artifacts
uv run --group docs python -m scripts.plot_documentation_figures
uv run --group docs pytest tests/test_documentation.py
uv run --group docs mkdocs build --strict
```

The documentation tests reject unregistered assets, serif typography, converted text
paths, missing alternative text, unknown classification labels, incomplete provenance
records, and figures without captions. Expensive numerical evidence remains frozen in its
benchmark artifact; the plotting layer must not silently create new scientific results.

## Review questions

Before approving a figure, ask:

- Would a reader classify the evidence correctly from the card alone?
- Is the denominator the scientific unit rather than merely the row count?
- Does the visual aggregation match the stated estimand and uncertainty unit?
- Can every panel be traced to an artifact or identified as schematic?
- Does the alternative text convey the conclusion without colour?
- Is the nearest overclaim stated explicitly?
- Is every label legible at the rendered documentation width on mobile and desktop?
