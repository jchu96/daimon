---
name: pymc-artifact-style
description: Apply PyMC Labs' house style to every artifact you produce — reports, PDFs, charts, images, notebooks. Use whenever you generate something a person will look at, before you deliver it.
---

# PyMC artifact style

Everything you hand a user is a PyMC Labs deliverable and should look like one.
A client skims a report; polish is what makes the content land. This is PyMC's
own style — never substitute a client's brand colors unless the user explicitly
asks for a client-branded artifact.

**The house style is a real Typst template, not a description of one.** It
comes from `pymc-labs/pymc-labs-report-template`, and the parts you need — the
report class, the brand fonts, the logo, the cover art, the matplotlib style —
are **bundled inside this skill**. Build through them. Do not improvise a
"corporate report" look: numbered navy section bars, stat-tile rows and
generic blue/orange charts are what this style exists to replace.

## Reports and PDFs — everything you need is bundled here

The Typst report class, the brand fonts, the logo and the cover art all ship
**inside this skill**. You do not need to clone anything and you do not need
network access to the brand repos.

```
typst/pymc-report.typ      the report class — import this
typst/starter.typ          minimal working report; copy it and replace content
typst/report-example.typ   the full worked example — read it for the helpers
fonts/                     Inter, Archivo Expanded, Fira Math, Fira Mono
assets/pymc-labs-logo.png  the cover logo (already wired as the default)
assets/cover-4.png, -9.png the two approved cover backgrounds
mpl/                       matplotlibrc + plotstyle.py + axes.py
```

### Build

The `typst` CLI is not installed. Use the Python package — it ships the
compiler as a wheel and installs in seconds:

```bash
pip install typst
```

```python
import typst
typst.compile(
    "starter.typ",
    output="report.pdf",
    root="..",              # the skill dir, so ../assets and ../fonts resolve
    font_paths=["../fonts"],  # REQUIRED — without it headings silently go serif
)
```

`font_paths` is not optional. Typst has no access to the bundled faces without
it, and the failure is silent: the document still compiles, but every heading
falls back to a serif and the result stops looking like a PyMC report.

### Two font traps, both verified the hard way

- **Headings are `font: "Archivo"` with `stretch: 125%`** — *not*
  `font: "Archivo Expanded"`. Typst folds the OS/2 width class into the stretch
  axis, so the family registers as plain `Archivo`; asking for
  "Archivo Expanded" resolves to nothing and falls back to serif.
- **Body is `font: "Inter"`.** The bundled statics register under family
  `Inter`. Do not ask for `Inter 18pt` — that is the Google Fonts optical-size
  packaging, which is not what ships here.

Both are already set correctly in `typst/pymc-report.typ`. Do not "fix" them.

### Writing the document

```typ
#import "pymc-report.typ": *

#show: pymc-report.with(
  title:    [Report title],
  subtitle: [One-line description],
  client:   [Client name],
  author:   [PyMC Labs],
  date:     [August 2026],
  status:   "Confidential",   // cover + footer; none to hide
  paper:    "a4",             // or "us-letter"
  cover-background: 4,        // approved cover art: 4 or 9
  draft: false,
  abstract: [ Executive summary … ],
  outline-depth: 2,
  number-headings: false,     // unnumbered headings are the house default
)

= Section heading
Body text …
```

**Heading numbering is a parameter, not a decision to improvise.** Short and
presentational documents use unnumbered headings (the default) — hierarchy is
typographic and the register is editorial. Long technical reports that need
cross-references set `number-headings: true` and get `1`, `1.1`. Either is
correct; hand-typed "01"/"02" prefixes are not.

### Layout helpers — the Tufte margin is the whole point

The page is a ⅔ text column plus a ⅓ margin. Figures, captions, sidenotes and
key numbers live in that margin, beside the prose that discusses them. Setting
the content as a plain single column throws away the design.

| Helper | Use |
|---|---|
| `#sidenote[…]` | numbered margin note — citations, sourcing, asides |
| `#marginnote[…]` | un-numbered margin commentary |
| `#dtable(…)` | table with Butterick rules (heavy top/bottom, light header rule, **no verticals**) |
| `#flowfigure(x, caption: […], label: <id>)` | figure/table in the text column, caption in the margin |
| `#marginfigure(…)` | small figure entirely in the margin |
| `#widefigure(…)` | spans text + margin |
| `#fullpagefigure(…)` | its own page |
| `#widetable(dtable(…), caption: […])` | full-width, page-breakable — tables taller than a page |
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
| Body | **Inter** — ask for family `Inter` |
| Headings | **Archivo Expanded** — ask for `Archivo` at `stretch: 125%` |
| Math | **Fira Math**, weight 300 |
| Code | **Fira Mono** on soft-white |

Headings are unnumbered by default — hierarchy is typographic, the register is
editorial.

## Charts — use the bundled matplotlib style

```python
import matplotlib as mpl
mpl.rc_file("mpl/matplotlibrc")     # brand palette, Inter, navy bold titles
```

`mpl/plotstyle.py` carries `PALETTE`, `PALETTE_LIGHT`, `PALETTE_DARK` and a
`fill_between` patch that gives clean edges; `mpl/axes.py` adds
`add_axis_end_tick_caps`. Import them alongside the rc if you want the helpers.

ArviZ draws through matplotlib, so `az.plot_posterior`, `az.plot_trace` and
friends inherit the style once the rc is loaded.

Save figures as **vector PDF** and embed with `image("fig.pdf")`. The rc sets
`figure.figsize: 4.8, 3.0` — 122 mm, exactly the Typst text column — so figures
land at native scale instead of being resampled. Needs matplotlib ≥ 3.10.1; do
not switch to SVG path output (the Inter statics share a PostScript name and
SVG path-mode collides bold/regular glyphs).

The palette the rc cycles, if you need it by hand:

```
0C1F40  navy        F6AE72  peach       9FAAE2  periwinkle   759690  aqua-dark
798496  navy-light  E0886A  peach-mid   676E93  peri-dark    B4E7DD  aqua
```

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
