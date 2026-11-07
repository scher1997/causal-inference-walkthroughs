# Causal Inference

Quarto book of six causal-inference worked examples (DiD, synthetic control, event studies, RDD, propensity scores, geo/switchback tests) on simulated data with known true effects. Converted from Jupyter notebooks.

## Structure

- `_quarto.yml` — book config; edit the `format:` block to change output (html active).
- `index.qmd`, `01_…`–`06_…qmd`, `appendix_causalsimkit.qmd` — the content. `_setup.qmd` is the shared setup cell.
- `src/causalsimkit/` — estimators; `tests/` — truth-recovery tests.

## How to run

```bash
uv sync --extra dev --extra render
uv run pytest -q
uv run quarto render        # only when asked; a full render takes ~3 min
```

## Notes

- The `.qmd` files are the source of truth; the original notebooks are retired.
- `_freeze/` and `_book/` are generated and git-ignored.
