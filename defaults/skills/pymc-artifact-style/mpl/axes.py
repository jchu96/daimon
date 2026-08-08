"""Axis helpers for the PyMC Labs plot style.

``add_axis_end_tick_caps`` draws a small cap at the *open* end of each spine that
visually matches the real ticks, so a spine that stops between labeled ticks
still reads as "bounded" — without adding a real tick, label, or gridline.

It follows the spine layout, so it works for right-side y-axes (the margin-figure
convention):
  * the x cap sits at the end of the x-axis *away from the y-axis* — the right end
    for a normal left y-axis, the left end when the y-axis is on the right;
  * the y cap sits at the *top* of whichever side (left or right) the y-axis is on.

Call it after the limits, ticks and spines are set (it also re-syncs on limit and
draw events).
"""

import numpy as np
from matplotlib.lines import Line2D


def add_axis_end_tick_caps(ax, x=True, y=True, which="major"):
    """Add endpoint caps matching the ticks at each spine's open end.

    Returns the list of cap ``Line2D`` artists. Pass ``y=False`` for hidden or
    categorical y-axes where a top cap would not help.
    """
    tick_getter = {
        "major": lambda axis: axis.get_major_ticks(),
        "minor": lambda axis: axis.get_minor_ticks(),
    }[which]

    # Spine layout: which side each axis lives on.
    y_right = ax.spines["right"].get_visible() and not ax.spines["left"].get_visible()
    x_top = ax.spines["top"].get_visible() and not ax.spines["bottom"].get_visible()
    x_attr = "tick2line" if x_top else "tick1line"      # bottom vs top tick glyph
    y_attr = "tick2line" if y_right else "tick1line"    # left vs right tick glyph
    x_which = "tick2" if x_top else "tick1"
    y_which = "tick2" if y_right else "tick1"
    x_frac = 1.0 if x_top else 0.0      # axes-fraction position of the x spine
    y_frac = 1.0 if y_right else 0.0    # axes-fraction position of the y spine

    def tickline(axis, attr):
        for tick in tick_getter(axis):
            line = getattr(tick, attr)
            if line.get_visible():
                return line
        ticks = tick_getter(axis)
        return getattr(ticks[0], attr) if ticks else None

    def style_cap(dst, src):
        dst.update_from(src)
        dst.set_clip_on(False)
        dst.set_label("_nolegend_")

    def has_tick(axis_obj, bound):
        locs = (axis_obj.get_majorticklocs() if which == "major"
                else axis_obj.get_minorticklocs())
        return np.any(np.isclose(locs, bound))

    xcap = Line2D([], [])
    ycap = Line2D([], [])
    caps = []
    if x:
        ax.add_artist(xcap)
        caps.append(xcap)
    if y:
        ax.add_artist(ycap)
        caps.append(ycap)

    def sync(_=None):
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()

        if x:
            # open end of the x-axis: the end away from the y-axis
            x_end = xmin if y_right else xmax
            if has_tick(ax.xaxis, x_end):
                xcap.set_visible(False)
            else:
                src = tickline(ax.xaxis, x_attr)
                if src is not None:
                    style_cap(xcap, src)
                    xcap.set_data([x_end], [x_frac])
                    xcap.set_transform(ax.get_xaxis_transform(which=x_which))
                    xcap.set_visible(True)

        if y:
            # top of the y-axis, on whichever side it is drawn
            if has_tick(ax.yaxis, ymax):
                ycap.set_visible(False)
            else:
                src = tickline(ax.yaxis, y_attr)
                if src is not None:
                    style_cap(ycap, src)
                    ycap.set_data([y_frac], [ymax])
                    ycap.set_transform(ax.get_yaxis_transform(which=y_which))
                    ycap.set_visible(True)

    sync()

    ax.callbacks.connect("xlim_changed", sync)
    ax.callbacks.connect("ylim_changed", sync)
    ax.figure.canvas.mpl_connect("draw_event", sync)

    return caps
