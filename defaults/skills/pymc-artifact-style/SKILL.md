---
name: pymc-artifact-style
description: Apply PyMC Labs' house style to every artifact you produce — charts, images, PDFs, reports, notebooks. Use whenever you generate something a person will look at, before you deliver it.
---

# PyMC artifact style

Everything you hand a user is a PyMC Labs deliverable. It should look like one.
This is the house style — PyMC's own, not any client's. Never substitute a
client's brand colors unless the user explicitly asks for a client-branded
artifact.

Apply it to charts, plots, diagrams, generated images, PDFs, and notebooks. If
you are about to render something and you have not applied this, stop and apply it.

## Brand palette

| Token | Hex | Use |
|---|---|---|
| brand-blue | `#006FFF` | primary; first series; links and accents |
| brand-cyan | `#1AD5FF` | highlight; never as body text |
| brand-navy | `#0A3A7E` | headings, rules, dark accents |
| ink | `#111311` | body text |
| mist | `#F2F2F2` | grid lines, table banding, fills |

These five are the identity. The chart palette below is derived from them.

## Charts — write this file, then use it

The brand five are not directly usable as a categorical chart palette: navy and
cyan sit outside the readable lightness band and cyan falls under 3:1 contrast on
a light surface. The set below keeps brand-blue as the first series and re-steps
the rest so that all six pass a lightness band of OKLCH L 0.43–0.77, a chroma
floor, colorblind separation on adjacent pairs, and 3:1 contrast against a light
background.

Write this to `pymc.mplstyle` and load it before plotting:

```
# pymc.mplstyle — PyMC Labs house chart style
axes.prop_cycle: cycler('color', ['006FFF', 'E8590C', '0CA678', '7048E8', '0090B3', 'C2255C'])

figure.facecolor: FCFCFB
axes.facecolor:   FCFCFB
savefig.facecolor: FCFCFB
figure.dpi: 150
savefig.dpi: 150
savefig.bbox: tight

font.family: sans-serif
font.sans-serif: Liberation Sans, DejaVu Sans, Arial, sans-serif
font.size: 10.5
text.color: 111311

axes.edgecolor: 111311
axes.labelcolor: 111311
axes.titlecolor: 0A3A7E
axes.titlesize: 13
axes.titleweight: semibold
axes.titlelocation: left
axes.titlepad: 12
axes.labelsize: 10.5
axes.linewidth: 0.8
axes.spines.top: False
axes.spines.right: False

axes.grid: True
axes.grid.axis: y
grid.color: F2F2F2
grid.linewidth: 0.8
grid.alpha: 1.0
axes.axisbelow: True

xtick.color: 111311
ytick.color: 111311
xtick.labelsize: 9.5
ytick.labelsize: 9.5
xtick.direction: out
ytick.direction: out

lines.linewidth: 2.0
lines.markersize: 8
lines.solid_capstyle: round

legend.frameon: False
legend.fontsize: 9.5
```

```python
import matplotlib.pyplot as plt
plt.style.use("pymc.mplstyle")
```

ArviZ draws through matplotlib, so the same style applies to `az.plot_posterior`,
`az.plot_trace`, and friends — load it first and they inherit it.

## Chart rules that matter more than color

- **Assign series colors in the cycle's fixed order.** Never reorder by rank — a
  filter that drops a series must not repaint the survivors.
- **Never a dual y-axis.** Two measures of different scale become two charts, small
  multiples, or an index to a common base.
- **Sequential data: one hue, light→dark.** Diverging: two hues with a neutral gray
  midpoint. Never a rainbow, never a hue at a diverging midpoint.
- **A legend whenever there are 2+ series.** With ≤ 4, also label them directly.
  Identity must never rest on color alone.
- **Beyond 6 series, do not invent a 7th color.** Fold the tail into "Other", or
  use small multiples.
- **Label the axes, and name the units.** Title states the finding, not the
  variable names.

## Documents and PDFs

The canonical EAP document format lives in the private repo
`pymc-labs/daimon-memory` under `eaps/_general/document-template/` — `README.md`
is the spec, `eap-template.typ` the Typst template, `_brand.yml` the Quarto brand
file, plus the logo and a worked example. When that repo is available in your
workspace, render through it and do not ship an unbranded export.

When it is not available, match this palette by hand: brand-navy headings,
brand-blue accents, ink body text, mist rules and table banding, Liberation Sans
throughout.

There is no Word or Google Docs template. For a Docs deliverable, apply the
palette and typography manually, or export a PDF from Typst/Quarto and attach that
instead — the PDF path is the one with a real template behind it.

## Sales collateral is a different standard

Proposals and sales material follow `teams/sales/` in `daimon-memory` (navy
`#1e3a5f`, Calibri), which deliberately differs from the EAP style above. Do not
mix them. If you are unsure which applies, ask — an EAP deliverable and a
proposal are not interchangeable.
