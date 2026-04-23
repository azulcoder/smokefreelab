# reports/

Executive-facing artifacts that get forwarded to hiring managers.

## Contents

| File | Format | Purpose |
|---|---|---|
| `_derivation.ipynb` | Jupyter | Arithmetic-only notebook that reproduces every headline IDR number. Single source of truth. |
| `executive_onepager.md` | Markdown | 1-page executive summary (English). Source for the PDF. |
| `executive_onepager.pdf` | PDF | Rendered output of the markdown above. **Build step below.** |
| `smokefreelab_deck.pptx` | PowerPoint | 10-slide McKinsey-style deck. **Build step below.** |
| `deck_outline.md` | Markdown | Slide-by-slide outline for the PPTX (narrative + speaker notes). |
| `cover_letter_sampoerna.md` | Markdown | Tailored cover letter (source). |
| `cover_letter_sampoerna.pdf` | PDF | Rendered 2-page cover letter. **Build step below.** |

## Build steps

Both artifacts render from their markdown sources via two standalone Python
scripts — **no pandoc, quarto, or LaTeX required**. The scripts ship in
`scripts/` and are the canonical build path for this repo.

### PDF one-pager

```bash
python3 scripts/render_onepager.py
# → writes reports/executive_onepager.pdf  (single-page A4, reportlab)
```

### PPTX deck

```bash
python3 scripts/render_deck.py
# → writes reports/smokefreelab_deck.pptx  (10 slides, 16:9, python-pptx)
```

### Tailored cover letter

```bash
python3 scripts/render_cover_letter.py
# → writes reports/cover_letter_sampoerna.pdf  (2-page A4, reportlab)
```

### Dependencies (one-time)

```bash
pip3 install --user reportlab fpdf2 python-pptx
```

`python-pptx` is already declared in `pyproject.toml`; `reportlab` and
`fpdf2` are shipped as build-only deps for the rendering scripts.

### Optional: pandoc fallback

If you have pandoc + a LaTeX engine installed locally, these one-liners
also work and produce near-identical output:

```bash
pandoc reports/executive_onepager.md -o reports/executive_onepager.pdf \
    --pdf-engine=pdflatex -V geometry:margin=1in -V fontsize=11pt
pandoc reports/deck_outline.md -o reports/smokefreelab_deck.pptx
```

## Update protocol

If any IDR number changes:

1. Edit `_derivation.ipynb` cells first.
2. Re-run top-to-bottom, confirm all numbers still derive.
3. Propagate to:
   - `README.md` hero table
   - `docs/ringkasan_eksekutif.md` five-number table
   - `executive_onepager.md`
   - `deck_outline.md` (slides 7 + 10)
4. Rebuild PDF and PPTX.

Drift between `_derivation.ipynb` and the rendered artifacts is a bug,
not a variant.
