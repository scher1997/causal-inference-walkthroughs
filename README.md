# Causal inference: Quarto book

Six worked causal-inference examples on simulated data with known effects, as a Quarto book. See `index.qmd` for the overview and `CLAUDE.md` for the project layout.

## Render

One-time setup (creates `.venv` and installs `causalkit`):

```bash
uv sync --extra dev --extra render
```

Then, from the project root:

```bash
uv run quarto render                 # every format active in _quarto.yml (HTML by default)
uv run quarto render --to docx       # one format, ignoring the YAML
uv run quarto render 01_difference_in_differences.qmd   # one chapter
```

- Always use `uv run`. Plain `quarto render` can't find a Python with Jupyter, and the Windows `python` command is a Microsoft Store stub.
- Output goes to `_book/`; open `_book/index.html`.
- The first full render takes about 3 minutes because every simulation runs. Results are cached in `_freeze/`, so later renders only re-run chapters whose source changed. Delete `_freeze/` to force a full re-run.
- To change the output, edit the `format:` block in `_quarto.yml`. HTML is active; `pdf`, `docx` and `epub` are commented out.

## Known issues

- **Typst fails.** `quarto render --to typst` stopped on the index page with `ERROR: error: expected content, found array` / `Typst compilation failed`. Cause not diagnosed (the index's chapter table, which uses `@sec-...` cross-references in its cells, is the first thing to check). Typst is therefore not listed in `_quarto.yml`.
- **Tested formats.** HTML and Word (`docx`) rendered successfully. `pdf` (needs LaTeX: `quarto install tinytex`) and `epub` have not been tried.
- **Harmless console noise.** Each chapter prints a `DeprecationWarning` about `asyncio.WindowsSelectorEventLoopPolicy` and a Jupyter "Kernel is running over TCP without encryption" warning. Neither affects the output.
