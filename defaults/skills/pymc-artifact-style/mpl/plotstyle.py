"""PyMC Labs matplotlib style helpers.

The global look (brand palette + variants, Inter font, thin white marker edges,
text-column-width default figure) lives in ``pymclabsreport/matplotlibrc`` (and the
selectable ``pymclabs`` style). One thing matplotlibrc cannot express is
``fill_between``'s default edge: passing ``color=`` makes the filled band's edge
the same color as its face, leaving a faint hairline. Importing this module
patches ``Axes.fill_between`` / ``fill_betweenx`` to default ``edgecolor="none"``
(explicit ``edgecolor`` still wins), so filled bands are clean by default.

    import pymclabsreport.plotstyle  # noqa: F401  — applies the patch on import

(``import pymclabsreport`` does this for you and also activates the rc.)
It also exposes the palette as Python dicts for convenient reference.
"""

import functools

import matplotlib.axes as _maxes

# Brand palette + variants (hex), mirroring the matplotlibrc header.
PALETTE = {
    "navy": "#0C1F40",
    "periwinkle": "#9FAAE2",
    "aqua": "#B4E7DD",
    "peach": "#F6AE72",
    "soft_white": "#F7F7F7",
}
PALETTE_LIGHT = {
    "navy": "#798496",
    "periwinkle": "#CAD0EF",
    "aqua": "#D6F2EC",
    "peach": "#FAD2B1",
}
PALETTE_DARK = {
    "navy": "#08142A",
    "periwinkle": "#676E93",
    "aqua": "#759690",
    "peach": "#A0714A",
}


def _patch_fill_between():
    for name in ("fill_between", "fill_betweenx"):
        orig = getattr(_maxes.Axes, name)
        if getattr(orig, "_pymclabs_patched", False):
            continue

        def _wrap(orig):
            @functools.wraps(orig)
            def wrapper(self, *args, **kwargs):
                kwargs.setdefault("edgecolor", "none")
                return orig(self, *args, **kwargs)

            wrapper._pymclabs_patched = True
            return wrapper

        setattr(_maxes.Axes, name, _wrap(orig))


_patch_fill_between()


def demo(path="pymclabs_style_demo.png"):
    """Render a swatch + sample plot using the global pymclabs style."""
    import matplotlib.pyplot as plt
    import numpy as np

    keys = ["navy", "periwinkle", "aqua", "peach"]
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(9.6, 3.0))
    for r, (lab, pal) in enumerate(
        [("base", PALETTE), ("light", PALETTE_LIGHT), ("dark", PALETTE_DARK)]
    ):
        for c, k in enumerate(keys):
            a0.add_patch(plt.Rectangle((c, -r), 0.92, 0.92, color=pal[k]))
        a0.text(-0.15, -r + 0.46, lab, ha="right", va="center", fontsize=8)
    for c, k in enumerate(keys):
        a0.text(c + 0.46, 1.12, k, ha="center", fontsize=8)
    a0.set_xlim(-1.3, 4)
    a0.set_ylim(-2.2, 1.5)
    a0.axis("off")
    a0.set_title("palette  (base / light / dark)")

    x = np.linspace(0, 10, 60)
    for i in range(6):
        a1.plot(x, np.sin(x + i * 0.5) + i * 0.4, marker="o", ms=4, markevery=7)
    a1.fill_between(x, -1.4, -0.9 + 0.25 * np.sin(x), alpha=0.55, color=PALETTE["aqua"])
    a1.set_title("cycle + markers (white edge) + clean fill_between")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    demo()
