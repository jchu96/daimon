# Website identity

Source: https://www.pymc-labs.com/, inspected 2026-09-14. Website CSS:
https://www.pymc-labs.com/_astro/MarketingLayout.BlM-pfUk.css.
The hashed stylesheet is a provenance snapshot; consult the homepage's linked
stylesheet when checking future changes.

The website defines navy `#0C1F40`, navy-deep `#07142A`, aqua `#B4E7DD`,
periwinkle `#9FAAE2`, peach `#F6AE72`, violet `#C8B4E7`, soft-white `#F7F7F7`,
and white `#FFFFFF`. Its headline and body families are Inter; its monospace
family is JetBrains Mono. Use navy on light surfaces and soft-white on dark
surfaces. Pale accents are fills or highlights, not small text on white.

Official logo sources, downloaded as PNG through Cloudinary's `f_png`
delivery transform without redrawing or recoloring:

- Dark: https://res.cloudinary.com/dx3t8udaw/image/upload/v1787796517/website/pymc_labs_logo_dark.webp
- Light: https://res.cloudinary.com/dx3t8udaw/image/upload/v1787796594/website/pymc_labs_logo_light.webp

JetBrains Mono regular and medium are bundled from the Google Fonts stylesheet
requested by the website, with their SIL Open Font License in
`../fonts/JetBrainsMono-OFL.txt`. Inter is already bundled.

The report's Tufte margin layout, Fira Math, accessible chart cycle and its
light/dark variants are retained document conventions, not website requirements.
In particular, chart navy-dark `#08142A` is distinct from website navy-deep
`#07142A`. New reports use Inter headings, navy text, the dark website logo,
and a plain cover. Legacy cover artwork and fonts remain for old sources.

For HTML and notebooks, apply these colors and type families in frontend code.
Use spacious layouts and short labels; do not copy the site's navigation,
marketing sections, or claims into an analysis. Preserve meaningful legends,
units, uncertainty, and sourcing when reducing copy.
