"""
ノートブック用: セットアップ・ケース実行のオーケストレーション。
処理内容は変更せず、 notebook の重複をモジュールに集約する。
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


@dataclass
class RunConfig:
    """実行設定（params-sample から読み込んだもの）"""
    use_cached_artifacts: bool
    use_median_filter: bool
    median_window: int | None
    use_sg_filter: bool
    sg_window: int | None
    sg_poly: int
    save_both_filter_figures: bool
    base_dir: str
    data_folder: str
    module_path: str
    verbose: bool = False


def _resolve_project_paths() -> tuple[Path, Path]:
    """params_dir と src のパスを解決する。params-sample 優先、params にフォールバック。"""
    for name in ("params-sample", "params"):
        params_dir = Path(name) if Path(name).exists() else Path("..") / name
        if params_dir.exists():
            break
    else:
        raise FileNotFoundError("params-sample または params フォルダが見つかりません")
    # Colab 等で cwd が data/params/output 専用フォルダのときは環境変数で src を指定可能
    env_src = os.environ.get("WATERLEVEL_VIZ_SRC")
    if env_src and Path(env_src).exists():
        src_path = Path(env_src).resolve()
    else:
        src_path = Path("src") if Path("src").exists() else Path("../src")
    if not src_path.exists():
        raise FileNotFoundError("src フォルダが見つかりません")
    return params_dir, src_path


def bootstrap(config_files: list[str] | None = None) -> tuple[RunConfig, list[Path], Any, Path]:
    """
    環境セットアップを行い、RunConfig と params_list を返す。
    config_files: 図化する設定ファイルのリスト。None または [] ならフォルダ内の全ファイルを処理。
    Returns: (run_config, params_list, base_cfg, params_dir)
    """
    params_dir, src_path = _resolve_project_paths()
    sys.path.insert(0, os.path.abspath(src_path))

    from params_loader import load_base_config, load_params_with_base, resolve_params_list

    base_cfg = load_base_config(params_dir)
    rp = base_cfg.get("run_params", {})
    fp = base_cfg.get("filter_params", {})

    # ノートブックから渡されなければ base の run_params を参照
    if config_files is None and rp:
        cfg_list = rp.get("config_files")
        if cfg_list is not None:
            config_files = cfg_list if isinstance(cfg_list, list) else [cfg_list]
    params_list = resolve_params_list(params_dir, config_files)

    _cfg0 = load_params_with_base(params_list[0], base_cfg=base_cfg, params_dir=params_dir)
    if "paths" not in _cfg0 and base_cfg:
        _cfg0 = {**base_cfg, **_cfg0}
    if "paths" not in _cfg0:
        raise KeyError("パラメータに 'paths' がありません")
    paths = _cfg0["paths"]
    base_dir = paths["base_dir"]
    module_path = paths["module_path"]
    # Colab 等で cwd が waterlevel_analysis のときは params の ./src が存在しないため環境変数で上書き
    env_src = os.environ.get("WATERLEVEL_VIZ_SRC")
    if env_src and os.path.isdir(env_src):
        module_path = env_src

    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"指定されたディレクトリは存在しません: {base_dir}")
    os.chdir(base_dir)
    if not os.path.isdir(module_path):
        raise FileNotFoundError(f"モジュールパスが存在しません: {module_path}")
    sys.path.insert(0, os.path.abspath(module_path))

    run_config = RunConfig(
        use_cached_artifacts=rp.get("use_cached_artifacts", False),
        use_median_filter=fp.get("use_median_filter", False),
        median_window=fp.get("median_window", 5),
        use_sg_filter=fp.get("use_sg_filter", False),
        sg_window=fp.get("sg_window", 11),
        sg_poly=fp.get("sg_poly", 3),
        save_both_filter_figures=fp.get("save_both_filter_figures", False),
        base_dir=base_dir,
        data_folder=paths["data_folder"],
        module_path=module_path,
        verbose=False,
    )
    return run_config, params_list, base_cfg, params_dir


def run(
    run_config: RunConfig,
    params_list: list[Path],
    base_cfg: dict[str, Any],
    params_dir: Path,
) -> None:
    """全ケースまたは単一ケースを実行する。"""
    from params_loader import load_params_with_base
    from output_io import OutputManager
    from data_oyo import process_data, process_data_oyo
    from waterlevel_utils import (
        artifact_cache_available,
        load_artifacts,
        save_artifacts,
        gwl_for_plot,
        write_case_tables,
        plot_waterlevel_figures,
    )

    cfg = run_config
    verbose = cfg.verbose

    def _run_one_case(params_path: Path, shared_data: tuple | None = None) -> None:
        params_path = params_path.resolve()
        case_cfg = load_params_with_base(
            params_path, base_cfg=base_cfg, params_dir=params_path.parent
        )
        paths = case_cfg["paths"]
        obs_params = case_cfg["obs_params"]
        rwl_params = case_cfg["rwl_params"]
        gwl_params = case_cfg["gwl_params"]
        period = case_cfg["period"]
        start_date = datetime.strptime(period["start_date"], "%Y-%m-%d")
        end_date = datetime.strptime(period["end_date"], "%Y-%m-%d")
        offset1 = rwl_params["obs_elev"] + rwl_params["slide"][0]
        offset2 = rwl_params["obs_elev"] + rwl_params["slide"][1]
        data_folder = paths["data_folder"]

        fig_cfg = case_cfg.get("figure", {})
        ratio = fig_cfg.get("graph_size_ratio")
        base_w = fig_cfg.get("base_width")
        if ratio and len(ratio) >= 2 and base_w is not None:
            w_inch = float(base_w)
            figsize = (w_inch, w_inch * float(ratio[1]) / float(ratio[0]))
        else:
            figsize = (fig_cfg.get("width", 15), fig_cfg.get("height", 4))
        font_cfg = fig_cfg.get("font") or {}
        default_font = font_cfg.get("size") if isinstance(font_cfg, dict) else fig_cfg.get("font_size", 12)
        def _font(key: str):
            if not isinstance(font_cfg, dict):
                return default_font
            v = font_cfg.get(key)
            return default_font if v is None else v
        plt.rcParams.update({
            "figure.figsize": figsize,
            "figure.dpi": fig_cfg.get("dpi", 100),
            "font.size": default_font,
            "legend.fontsize": _font("legend"),
            "axes.labelsize": _font("axis_label"),
            "xtick.labelsize": _font("tick"),
            "ytick.labelsize": _font("tick"),
            "axes.titlesize": _font("title"),
            "lines.linewidth": fig_cfg.get("linewidth", 1.5),
        })

        out = OutputManager.from_params(str(params_path))
        out.start_run()

        if shared_data is not None:
            stacked1, stacked2, stacked3_shared = shared_data
            # groundwater は年ごとにフォルダが分かれているため、ケースごとにその年のデータを読む
            if stacked3_shared is None:
                stacked3, _ = process_data_oyo(
                    obs_params["GWL"],
                    os.path.join(data_folder, "groundwater", str(start_date.year)),
                    gwl_params["parts"],
                    gwl_params["SNs"],
                    gwl_params["gw_elevs"],
                    gwl_params["ND"],
                    verbose=verbose,
                )
            else:
                stacked3 = stacked3_shared
            save_artifacts(out, stacked1, stacked2, stacked3)
        elif cfg.use_cached_artifacts and artifact_cache_available(out):
            stacked1, stacked2, stacked3 = load_artifacts(out)
            print(f"  [cache] {out.case_id}")
        else:
            stacked1 = process_data(
                obs_params["RWL"],
                os.path.join(data_folder, "riverlevel"),
                verbose=verbose,
            )
            stacked2 = process_data(
                obs_params["RF"],
                os.path.join(data_folder, "rainfall"),
                verbose=verbose,
            )
            stacked3, _ = process_data_oyo(
                obs_params["GWL"],
                os.path.join(data_folder, "groundwater", str(start_date.year)),
                gwl_params["parts"],
                gwl_params["SNs"],
                gwl_params["gw_elevs"],
                gwl_params["ND"],
                verbose=verbose,
            )
            save_artifacts(out, stacked1, stacked2, stacked3)

        write_case_tables(out, stacked1, stacked2, stacked3)
        if obs_params["RWL"] or obs_params["RF"] or obs_params["GWL"]:
            stacked3_plot = gwl_for_plot(
                stacked3,
                cfg.use_median_filter,
                cfg.median_window,
                cfg.use_sg_filter,
                cfg.sg_window,
                cfg.sg_poly,
            )
            plot_waterlevel_figures(
                out,
                case_cfg,
                stacked1,
                stacked2,
                stacked3,
                stacked3_plot,
                start_date,
                end_date,
                offset1,
                offset2,
                cfg.use_median_filter,
                cfg.median_window,
                cfg.use_sg_filter,
                cfg.sg_window,
                cfg.sg_poly,
                cfg.save_both_filter_figures,
            )
        out.finish_run()
        print(f"  done: {out.case_id}")

    if len(params_list) > 1:
        print(f"実行: {len(params_list)} 件")
        shared_data = None
        if not cfg.use_cached_artifacts:
            cfg0 = load_params_with_base(
                params_list[0], base_cfg=base_cfg, params_dir=params_list[0].parent
            )
            data_folder0 = cfg0["paths"]["data_folder"]
            obs0 = cfg0["obs_params"]
            gwl0 = cfg0["gwl_params"]
            stacked1 = process_data(
                obs0["RWL"],
                os.path.join(data_folder0, "riverlevel"),
                verbose=verbose,
            )
            stacked2 = process_data(
                obs0["RF"],
                os.path.join(data_folder0, "rainfall"),
                verbose=verbose,
            )
            # groundwater は各ケースの年フォルダで読むため共有しない（None を渡し _run_one_case 内で読む）
            shared_data = (stacked1, stacked2, None)
        for p in params_list:
            _run_one_case(p, shared_data=shared_data)
        print("完了.")
    else:  # 単一ケース
        params_path = params_list[0].resolve()
        case_cfg = load_params_with_base(
            params_path, base_cfg=base_cfg, params_dir=params_dir
        )
        out = OutputManager.from_params(str(params_path))
        out.start_run()
        obs_params = case_cfg["obs_params"]
        rwl_params = case_cfg["rwl_params"]
        gwl_params = case_cfg["gwl_params"]
        period = case_cfg["period"]
        start_date = datetime.strptime(period["start_date"], "%Y-%m-%d")
        end_date = datetime.strptime(period["end_date"], "%Y-%m-%d")
        print(f"実行: 1 件 ({params_path.name})")

        if cfg.use_cached_artifacts and artifact_cache_available(out):
            stacked1, stacked2, stacked3 = load_artifacts(out)
            print(f"  [cache] {out.case_id}")
        else:
            stacked1 = process_data(
                obs_params["RWL"],
                os.path.join(cfg.data_folder, "riverlevel"),
                verbose=verbose,
            )
            stacked2 = process_data(
                obs_params["RF"],
                os.path.join(cfg.data_folder, "rainfall"),
                verbose=verbose,
            )
            stacked3, _ = process_data_oyo(
                obs_params["GWL"],
                os.path.join(cfg.data_folder, "groundwater", str(start_date.year)),
                gwl_params["parts"],
                gwl_params["SNs"],
                gwl_params["gw_elevs"],
                gwl_params["ND"],
                verbose=verbose,
            )
            save_artifacts(out, stacked1, stacked2, stacked3)

        write_case_tables(out, stacked1, stacked2, stacked3)
        offset1 = rwl_params["obs_elev"] + rwl_params["slide"][0]
        offset2 = rwl_params["obs_elev"] + rwl_params["slide"][1]
        if obs_params["RWL"] or obs_params["RF"] or obs_params["GWL"]:
            stacked3_plot = gwl_for_plot(
                stacked3,
                cfg.use_median_filter,
                cfg.median_window,
                cfg.use_sg_filter,
                cfg.sg_window,
                cfg.sg_poly,
            )
            plot_waterlevel_figures(
                out,
                case_cfg,
                stacked1,
                stacked2,
                stacked3,
                stacked3_plot,
                start_date,
                end_date,
                offset1,
                offset2,
                cfg.use_median_filter,
                cfg.median_window,
                cfg.use_sg_filter,
                cfg.sg_window,
                cfg.sg_poly,
                cfg.save_both_filter_figures,
            )
        out.finish_run()
        print("完了.")
