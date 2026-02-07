"""
ノートブック用: アーティファクト・フィルタ・表出力・図化の共通処理。
処理内容は変更せず、呼び出し側の重複を減らすためのヘルパー。
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import numpy as np

try:
    from scipy.signal import savgol_filter
except ImportError:
    savgol_filter = None

from data_oyo import plt_wl_rf

if TYPE_CHECKING:
    from output_io import OutputManager


ARTIFACT_NAMES = ("riverlevel.pkl", "rainfall.pkl", "groundwater.pkl")


def artifact_cache_available(out: "OutputManager") -> bool:
    """artifacts に図化用 DataFrame が揃っているか判定する。"""
    return all(out.artifact_path(name).exists() for name in ARTIFACT_NAMES)


def load_artifacts(out: "OutputManager"):
    """artifacts から DataFrame を読み込む。戻り値: (stacked1, stacked2, stacked3)。"""
    s1 = pd.read_pickle(out.artifact_path(ARTIFACT_NAMES[0]))
    s2 = pd.read_pickle(out.artifact_path(ARTIFACT_NAMES[1]))
    s3 = pd.read_pickle(out.artifact_path(ARTIFACT_NAMES[2]))
    return s1, s2, s3


def save_artifacts(
    out: "OutputManager",
    stacked1: pd.DataFrame | None,
    stacked2: pd.DataFrame | None,
    stacked3: pd.DataFrame | None,
) -> None:
    """図化前の DataFrame を artifacts に保存する。"""
    (pd.DataFrame() if stacked1 is None else stacked1).to_pickle(
        out.artifact_path(ARTIFACT_NAMES[0])
    )
    (pd.DataFrame() if stacked2 is None else stacked2).to_pickle(
        out.artifact_path(ARTIFACT_NAMES[1])
    )
    (pd.DataFrame() if stacked3 is None else stacked3).to_pickle(
        out.artifact_path(ARTIFACT_NAMES[2])
    )


def apply_plot_filters(
    df: pd.DataFrame | None,
    median_window: int | None = None,
    sg_window: int | None = None,
    sg_poly: int = 3,
) -> pd.DataFrame | None:
    """
    堤体内水位 DataFrame に median / SG フィルタを適用したコピーを返す。図化用。
    NaN は補間してフィルタ後復元。
    """
    if df is None or df.empty:
        return df
    median_window = int(median_window) if median_window and int(median_window) >= 2 else None
    sg_window = int(sg_window) if sg_window and int(sg_window) >= 2 else None
    sg_poly = int(sg_poly) if sg_poly is not None else 3
    if not median_window and not (sg_window and sg_poly is not None and sg_window > sg_poly):
        return df.copy()
    out = pd.DataFrame(index=df.index)
    for col in df.columns:
        s = df[col].astype(float)
        if median_window:
            s = s.rolling(window=median_window, center=True, min_periods=1).median()
        if sg_window and sg_poly is not None and sg_window > sg_poly and savgol_filter is not None:
            valid = s.notna()
            n_valid = valid.sum()
            if n_valid < sg_window:
                out[col] = s
                continue
            y = s.interpolate(method="time").bfill().ffill()
            filtered = savgol_filter(y.values, sg_window, sg_poly, mode="nearest")
            out[col] = filtered
            out.loc[~valid, col] = np.nan
        else:
            out[col] = s
    return out


def gwl_for_plot(
    stacked3: pd.DataFrame | None,
    use_median_filter: bool,
    median_window: int | None,
    use_sg_filter: bool,
    sg_window: int | None,
    sg_poly: int,
) -> pd.DataFrame | None:
    """
    フィルタ設定に応じて図化用の堤体内水位 DataFrame を返す（artifacts はフィルタ前のまま）。
    """
    if not use_median_filter and not use_sg_filter:
        return stacked3
    if stacked3 is None or stacked3.empty:
        return stacked3
    median_w = median_window if use_median_filter else None
    sg_w = sg_window if use_sg_filter else None
    sg_p = sg_poly if use_sg_filter else 3
    return apply_plot_filters(stacked3, median_window=median_w, sg_window=sg_w, sg_poly=sg_p)


def write_case_tables(
    out: "OutputManager",
    stacked1: pd.DataFrame | None,
    stacked2: pd.DataFrame | None,
    stacked3: pd.DataFrame | None,
) -> None:
    """河川水位・降雨・堤体内水位の CSV を output/{case_id}/tables/ に保存する。"""
    if stacked1 is not None and not stacked1.empty:
        stacked1.to_csv(out.table_path("riverlevel.csv"))
    if stacked2 is not None and not stacked2.empty:
        stacked2.to_csv(out.table_path("rainfall.csv"))
    if stacked3 is not None and not stacked3.empty:
        stacked3.to_csv(out.table_path("groundwater.csv"))


def plot_waterlevel_figures(
    out: "OutputManager",
    cfg: dict,
    stacked1: pd.DataFrame | None,
    stacked2: pd.DataFrame | None,
    stacked3: pd.DataFrame | None,
    stacked3_plot: pd.DataFrame | None,
    start_date,
    end_date,
    offset1: float,
    offset2: float,
    use_median_filter: bool,
    median_window: int | None,
    use_sg_filter: bool,
    sg_window: int | None,
    sg_poly: int,
    save_both_filter_figures: bool,
) -> None:
    """
    水位・降雨・堤体内水位のグラフを描画し figures/ に保存する。
    フィルタ有効かつ save_both_filter_figures のとき、適用前（unfiltered）と適用後の両方を保存。
    """
    graph_params = cfg["graph_params"]
    filter_on = use_median_filter or use_sg_filter
    plot_kw = dict(
        stacked1=stacked1,
        stacked2=stacked2,
        start_date=start_date,
        end_date=end_date,
        miny=graph_params["miny"],
        maxy=graph_params["maxy"],
        rain_miny=graph_params.get("rain_miny"),
        rain_maxy=graph_params.get("rain_maxy"),
        offset1=offset1,
        offset2=offset2,
        out=out,
    )
    if filter_on and save_both_filter_figures:
        plt_wl_rf(
            stacked3=stacked3,
            fig_name_7_8k="waterlevel-7.8k-unfiltered.png",
            fig_name_8_0k="waterlevel-8.0k-unfiltered.png",
            **plot_kw,
        )
    plt_wl_rf(stacked3=stacked3_plot, **plot_kw)
