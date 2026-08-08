---
name: pymc-artifact-style
description: Apply PyMC Labs' house style to every artifact you produce — reports, PDFs, charts, images, notebooks. Use whenever you generate something a person will look at, before you deliver it.
---

# PyMC artifact style

Everything you hand a user is a PyMC Labs deliverable and should look like one.
A client skims a report; polish is what makes the content land. This is PyMC's
own style — never substitute a client's brand colors unless the user explicitly
asks for a client-branded artifact.

**The canonical implementation is a real repo, not a description.**
`pymc-labs/pymc-labs-report-template` holds the Typst report class, the brand
fonts, the cover art, and a matplotlib package that matches it. When a document
is the deliverable, **render through that repo**. Everything below the first
section is the fallback for when you genuinely cannot reach it.

## Reports and PDFs — use the template repo

```
https://github.com/pymc-labs/pymc-labs-report-template
```

It is **private**. Cloning it needs a repo binding on this agent — if you get a
404, that is the cause, not a missing repo. Ask the user to bind it via
`request_repo_binding` (or `/agent-setup` → GitHub) and retry.

```bash
git clone https://github.com/pymc-labs/pymc-labs-report-template
cd pymc-labs-report-template
pip install -e .                      # installs `pymclabsreport`
python -m pymclabsreport.report_figures   # regenerate example figures
make fmt                              # wrap python code blocks to the column
./build.sh path/to/your.typ out.pdf   # ALWAYS build via build.sh
```

`build.sh` is the only supported way to compile — it wires `--root .` plus the
three font paths. Compiling with bare `typst` gets you wrong weights or a hard
import error.

Read `examples/report.typ` before writing your own; it is the worked example and
the visual regression target.

### Writing the document

```typ
#import "../lib/pymc-report.typ": *

#show: pymc-report.with(
  title:    [Report title],
  subtitle: [One-line description],
  client:   [Client name],
  author:   [PyMC Labs],
  date:     [August 2026],
  status:   "Confidential",   // cover + footer; none to hide
  paper:    "a4",             // or "us-letter"
  cover-background: 4,        // approved cover art: 4 or 9
  draft: false,               // true → DRAFT on cover + footer
  abstract: [ Executive summary … ],
  outline-depth: 2,           // 0 disables the TOC
  number-headings: false,     // unnumbered headings are the house default
)

= Section heading
Body text …
```

### Layout helpers — the Tufte margin is the whole point

The template is a ⅔ text column + ⅓ margin. Figures, captions, sidenotes and key
numbers live in that margin, beside the prose that discusses them. Using it as a
plain single-column document throws away the design.

| Helper | Use |
|---|---|
| `#sidenote[…]` | numbered margin note — citations, sourcing, asides |
| `#marginnote[…]` | un-numbered margin commentary |
| `#dtable(…)` | table with Butterick rules (heavy top/bottom, light header rule, **no verticals**) |
| `#flowfigure(x, caption: […], label: <id>)` | figure/table in the text column, caption in the margin |
| `#marginfigure(…)` | small figure entirely in the margin |
| `#widefigure(…)` | spans text + margin |
| `#fullpagefigure(…)` | its own page |
| `#widetable(dtable(…), caption: […])` | full-width, page-breakable — for tables taller than a page |
| `#keyfigure[value][label]` | large key-number callout in the margin |
| `#executive-summary[…]` | the summary panel |
| `#callout[…]` | in-flow emphasis box |
| `#quotebox(by: [Name])[…]` | pull quote, colored spine |
| `#appendix("A", [Title])` | appendix divider |

Cross-reference with `@label`, and attach labels via the helper's `label:`
argument — never a trailing `<label>`, which errors with `cannot reference
context`.

## Brand palette — these exact hexes

| Token | Hex | Use |
|---|---|---|
| navy | `#0C1F40` | body text, headings, first series |
| periwinkle | `#9FAAE2` | accent, series |
| aqua | `#B4E7DD` | rules, spines, series |
| peach | `#F6AE72` | single status accent — use sparingly |
| soft-white | `#F7F7F7` | code slabs, fills |

Variants — light: navy `#798496`, peri `#CAD0EF`, aqua `#D6F2EC`, peach `#FAD2B1`.
Dark: navy `#08142A`, peri `#676E93`, aqua `#759690`, peach `#A0714A`.

Navy text on white. Color earns its place (Butterick) — the accent punctuates,
it does not decorate.

## Typography

| Role | Face |
|---|---|
| Body | **Inter** (static `Inter 18pt`) |
| Headings | **Archivo Expanded** (`Archivo` at `stretch: 125%`) |
| Math | **Fira Math**, weight 300 |
| Code | **Fira Mono** on soft-white |

Headings are unnumbered by default — hierarchy is typographic, the register is
editorial.

## Charts — `pymclabsreport` does this for you

```python
import pymclabsreport   # activates the style on import
```

That one import applies the brand matplotlibrc, patches `fill_between` to a
clean edge, and exposes `PALETTE`, `PALETTE_LIGHT`, `PALETTE_DARK` and
`add_axis_end_tick_caps`. ArviZ draws through matplotlib, so `az.plot_posterior`,
`az.plot_trace` and friends inherit it.

Save figures as **vector PDF** and embed with `image("fig.pdf")`. Requires
matplotlib ≥ 3.10.1. Do not use SVG path output — the Inter statics share a
PostScript name and SVG path-mode collides bold/regular glyphs.

If you cannot install the package, this is the rc it applies:

```
font.family:       sans-serif
font.sans-serif:   Inter, Fira Sans, DejaVu Sans
font.size:         8.0

axes.prop_cycle: cycler('color', ['0C1F40', 'F6AE72', '9FAAE2', '759690', '798496', 'E0886A', '676E93', 'B4E7DD'])

text.color: 333333
axes.edgecolor: 333333
axes.labelcolor: 333333
xtick.color: 333333
ytick.color: 333333
axes.linewidth: 0.8

axes.titlecolor: 0C1F40
axes.titleweight: bold
axes.titlesize: medium
axes.titlelocation: left

axes.spines.top: False
axes.spines.right: False

lines.linewidth: 1.5
lines.markersize: 5.0
lines.markeredgecolor: white
lines.markeredgewidth: 0.6
scatter.edgecolors: white

legend.frameon: False
legend.fontsize: 7
grid.color: 333333
grid.alpha: 0.12
grid.linewidth: 0.6

figure.figsize: 4.8, 3.0     # 122 mm = the Typst text column
figure.facecolor: white
axes.facecolor: white
savefig.facecolor: white
savefig.dpi: 300
pdf.fonttype: 42
```

Note the figure width: 4.8 in matches the text column exactly, so figures land
at native scale rather than being resampled.

## Chart rules that matter more than color

- **Assign series colors in the cycle's fixed order.** Never reorder by rank — a
  filter that drops a series must not repaint the survivors.
- **Never a dual y-axis.** Two measures of different scale become two charts,
  small multiples, or an index to a common base.
- **Sequential: one hue, light→dark. Diverging: two hues, neutral midpoint.**
  Never a rainbow, never a hue at a diverging midpoint.
- **A legend whenever there are 2+ series.** With ≤ 4, label directly too.
  Identity must never rest on color alone.
- **Beyond the cycle, do not invent another color.** Fold the tail into "Other",
  or use small multiples.
- **Label the axes and name the units.** The title states the finding, not the
  variable names.

## When the template repo is unreachable

Match the palette and typography by hand: navy body and headings, aqua rules,
peach for a single status accent, soft-white code slabs, Inter (or the closest
humanist sans available). Tables get horizontal rules only. Say plainly in the
delivery that this is an unbranded fallback and the binding is missing — do not
quietly ship something that looks nothing like a PyMC report.

There is no Word or Google Docs template. For a Docs deliverable, apply the
palette manually, or export a PDF from the Typst template and attach that — the
PDF path is the one with a real template behind it.

## Sales collateral is a different standard

Proposals and sales material follow `teams/sales/` in `daimon-memory` (navy
`#1e3a5f`, Calibri), which deliberately differs from the report style above. Do
not mix them. If unsure which applies, ask — a modeling report and a proposal are
not interchangeable.
