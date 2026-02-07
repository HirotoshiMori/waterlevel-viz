"""
水位・降雨・堤体内水位のグラフ描画。
"""
from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt

_LEGEND_PARTS_ORDER = ["front", "shoulder", "slope", "toe"]
_LEGEND_DISPLAY_NAMES = {
    "front": "River slope",
    "shoulder": "Back shoulder",
    "slope": "Back slope",
    "toe": "Back toe",
}


def _plt_wl_rf_single(
    ax_rain,
    ax_wl,
    stacked1,
    stacked2,
    stacked3,
    stacked4,
    offset: float,
    prefix: str,
    width2: float,
    start_date,
    end_date,
    miny: float,
    maxy: float,
    rain_miny: float | None,
    rain_maxy: float | None,
    out: Any = None,
    fig_name: str | None = None,
) -> None:
    """単一のグラフを描画。上: 降雨のみ、下: 水位＋堤体内水位。"""
    if stacked2 is not None and not stacked2.empty:
        ax_rain.bar(stacked2.index, stacked2["値"].values, width=width2, color="blue")
        if rain_miny is not None and rain_maxy is not None:
            ax_rain.set_ylim(rain_miny, rain_maxy)
        else:
            ax_rain.set_ylim(ax_rain.get_ylim()[::-1])
    ax_rain.set_ylabel("Rainfall (mm/h)")
    ax_rain.set_xlim(start_date, end_date)
    ax_rain.grid(True)
    ax_rain.tick_params(labelbottom=False)

    if stacked1 is not None and not stacked1.empty:
        ax_wl.plot(
            stacked1.index,
            stacked1["値"] + offset,
            label=f"{prefix} River water",
            color="black",
        )
    for part in _LEGEND_PARTS_ORDER:
        col_name = f"{prefix} {part}"
        for src in (stacked3, stacked4):
            if src is not None and not src.empty and col_name in src.columns:
                ax_wl.plot(
                    src.index,
                    src[col_name],
                    label=f"{prefix} {_LEGEND_DISPLAY_NAMES[part]}",
                )
                break
    ax_wl.set_ylabel("Water level (m)")
    ax_wl.set_xlabel("Date")
    ax_wl.set_xlim(start_date, end_date)
    ax_wl.set_ylim(miny, maxy)
    ax_wl.legend()
    ax_wl.grid(True)

    fig = ax_rain.get_figure()
    fig.tight_layout()
    if out is not None and fig_name is not None:
        fig.savefig(out.fig_path(fig_name))
        plt.close(fig)
    else:
        plt.show()


def plt_wl_rf(
    stacked1=None,
    stacked2=None,
    stacked3=None,
    stacked4=None,
    start_date=None,
    end_date=None,
    miny=None,
    maxy=None,
    rain_miny=None,
    rain_maxy=None,
    offset1=0,
    offset2=0,
    width2=0.042,
    out=None,
    fig_name_7_8k="waterlevel-7.8k.png",
    fig_name_8_0k="waterlevel-8.0k.png",
):
    """指定したデータでグラフを描画する。out に OutputManager を渡すと figures/ に保存する。"""
    fig_a, (ax_rain_a, ax_wl_a) = plt.subplots(2, 1, height_ratios=[1, 2], sharex=True)
    _plt_wl_rf_single(
        ax_rain_a, ax_wl_a,
        stacked1, stacked2, stacked3, stacked4,
        offset1, "7.8k", width2,
        start_date, end_date, miny, maxy, rain_miny, rain_maxy,
        out, fig_name_7_8k,
    )
    fig_b, (ax_rain_b, ax_wl_b) = plt.subplots(2, 1, height_ratios=[1, 2], sharex=True)
    _plt_wl_rf_single(
        ax_rain_b, ax_wl_b,
        stacked1, stacked2, stacked3, stacked4,
        offset2, "8.0k", width2,
        start_date, end_date, miny, maxy, rain_miny, rain_maxy,
        out, fig_name_8_0k,
    )
