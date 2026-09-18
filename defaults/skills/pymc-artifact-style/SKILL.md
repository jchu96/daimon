---
name: pymc-artifact-style
description: Apply PyMC Labs' house style to every artifact you produce — reports, PDFs, slide decks, charts, images, notebooks. Use whenever you generate something a person will look at, before you deliver it.
---

# PyMC Labs artifact style

Everything you hand a user is a PyMC Labs deliverable and should look like one.
A client skims a report; polish is what makes the content land. This is PyMC's
own style — never substitute a client's brand colors unless the user explicitly
asks for a client-branded artifact.

Use the current [PyMC Labs website](https://www.pymc-labs.com/) identity:
Inter, navy text, pale accents, and the dark/light PyMC Labs wordmarks.
The bundled report class adapts the existing Tufte layout to that identity.
Read [references/website-brand.md](references/website-brand.md) for source
provenance, exact website tokens, and the distinction between website styling
and document conventions.

Keep titles and labels short, positioning broad, and copy sparse. Reports still
need enough evidence, units, and uncertainty to support their conclusions.
Build layout, text, and functional controls in code. Reuse the bundled official
logos; if new logos or decorative artwork are requested, use image generation.

## Reports and PDFs — everything you need is bundled here

The Typst report class, the brand fonts, and the website logos all ship
**inside this skill**. You do not need to clone anything and you do not need
network access to the brand repos.

```
typst/pymc-report.typ      the report class — import this
typst/starter.typ          minimal working report; copy it and replace content
typst/report-example.typ   the full worked example — read it for the helpers
fonts/                     Inter, JetBrains Mono, Fira Math
assets/pymc-labs-logo-dark.png   dark wordmark for light backgrounds
assets/pymc-labs-logo-light.png  light wordmark for dark backgrounds
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

### Fonts

Use `Inter` for body and headings, with normal stretch (`100%`). Use
`JetBrains Mono` for code. Fira Math remains the document math face; the
website does not specify a math font. All are bundled. Archivo and Fira Mono
remain available for legacy sources, but are no longer the defaults.

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
  cover-background: none,     // plain; pass a path for supplied artwork
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
| navy-deep | `#07142A` | dark surfaces |
| violet | `#C8B4E7` | optional website accent |

Chart variants retained from the report style (not website tokens) — light: navy `#798496`, peri `#CAD0EF`, aqua `#D6F2EC`, peach `#FAD2B1`.
Dark: navy `#08142A`, peri `#676E93`, aqua `#759690`, peach `#A0714A`.

Navy text on white. Color earns its place (Butterick) — the accent punctuates,
it does not decorate.

## Typography

| Role | Face |
|---|---|
| Body | **Inter** regular — ask for family `Inter` |
| Headings | **Inter** bold/semibold, normal stretch |
| Math | **Fira Math**, weight 300 |
| Code | **JetBrains Mono** on soft-white |

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

## Slide decks (.pptx) — render the slides, do not typeset them

There is no PowerPoint template. Build a deck by typesetting each slide in
Typst at slide dimensions, rendering it to PNG, and placing one full-bleed
image per slide. The house style then survives the trip, because the fonts are
baked into the pixels.

```python
import typst
from pptx import Presentation
from pptx.util import Inches

# 1. One Typst file per slide, 16:9 with no margin:
#      #set page(width: 13.333in, height: 7.5in, margin: 0pt)
typst.compile("slide1.typ", output="slide1.png", root="..",
              font_paths=["../fonts"], format="png", ppi=144)  # 144 → 1920x1080

# 2. Blank layout, picture at full bleed.
prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
for png in slides:
    s = prs.slides.add_slide(prs.slide_layouts[6])   # 6 is the blank layout
    s.shapes.add_picture(png, 0, 0, width=prs.slide_width, height=prs.slide_height)
prs.save("deck.pptx")
```

**Do not build slides out of pptx text boxes.** A text run stores the font by
*name*, and the bundled fonts may not be installed on the machines that
open the file — PowerPoint and Google Slides silently substitute a generic
sans and the deck stops looking like PyMC. This is the same failure as the
`font_paths` trap above, one layer further out: the PDF survives because Typst
embeds the faces, and a pptx has nothing to embed.

The cost is real: rendered slides are images, so the text is no longer
editable and does not reflow. Slide order, deletion, and speaker notes still
work. If the recipient needs to edit the words, say so and hand over the
Typst sources — do not quietly ship a deck that cannot be edited.

### Use the matching website logo

Use `assets/pymc-labs-logo-dark.png` on white or soft-white, and
`assets/pymc-labs-logo-light.png` on navy or navy-deep. Preserve aspect ratio
and clear space. Both are official website assets with real transparency;
do not recolor or regenerate them.

The report defaults to the dark logo and a plain cover. The older logo files
and `cover-4.png` / `cover-9.png` remain for existing documents only; do not
use them for new artifacts unless requested.

## When the bundled template cannot be used

Match the palette and typography by hand: navy body and headings, aqua rules,
peach for a single status accent, soft-white code slabs, Inter (or the closest
humanist sans available). Tables get horizontal rules only. Say plainly in the
delivery that this is an unbranded fallback and the binding is missing — do not
quietly ship something that looks nothing like a PyMC report.

There is no Word or Google Docs template. For a Docs deliverable, apply the
palette manually, or export a PDF from the Typst template and attach that — the
PDF path is the one with a real template behind it.

## Explicit template requests

Honor an explicitly requested client or legacy sales template. Otherwise use
this website-aligned identity for new artifacts, including sales collateral.
Do not silently revert to the older Calibri/navy proposal style.
