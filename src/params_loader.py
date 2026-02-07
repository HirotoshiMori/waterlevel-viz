"""
パラメータ YAML の読み込みとケース一覧の解決。
base とケースファイルのディープマージを提供する。
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def load_base_config(params_dir: Path) -> dict[str, Any]:
    """params_dir 内の sabagawa-base.yaml を読み込み、辞書で返す。無ければ空辞書。"""
    base_path = params_dir / "sabagawa-base.yaml"
    if not base_path.exists():
        return {}
    with open(base_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> None:
    """base を破壊的に更新。override の値で上書き。ネストした dict は再帰マージ。"""
    for key, value in override.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, dict)
        ):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def load_params_with_base(
    params_path: Path,
    base_cfg: dict[str, Any] | None = None,
    params_dir: Path | None = None,
) -> dict[str, Any]:
    """
    ケース用 YAML を読み、sabagawa-* の場合は base とディープマージして返す。
    base_cfg が None のときは params_dir から sabagawa-base.yaml を読む（params_dir 必須）。
    """
    params_path = Path(params_path)
    with open(params_path, encoding="utf-8") as f:
        case_cfg = yaml.safe_load(f) or {}

    if params_path.name.startswith("sabagawa-") and params_path.name != "sabagawa-base.yaml":
        if base_cfg is None:
            if params_dir is None:
                params_dir = params_path.parent
            base_cfg = load_base_config(params_dir)
        if base_cfg:
            cfg = copy.deepcopy(base_cfg)
            _deep_merge(cfg, case_cfg)
            return cfg
    return case_cfg


def resolve_params_list(
    params_dir: Path,
    config_files: list[str] | None = None,
) -> list[Path]:
    """
    実行対象のパラメータファイルのリストを返す。
    config_files=None または [] なら params_dir 内の全 yaml/yml（base 除く）。
    config_files にファイル名を指定したら、そのファイルのみ。
    """
    if not config_files:
        files = sorted(
            list(params_dir.glob("*.yaml")) + list(params_dir.glob("*.yml"))
        )
        out = [p for p in files if p.name != "sabagawa-base.yaml"]
        if not out:
            raise FileNotFoundError(
                f"{params_dir} 内に .yaml / .yml がありません"
            )
        return out

    out = []
    for name in config_files:
        p = Path(name)
        cand = (
            params_dir / p.name
            if p.suffix in (".yaml", ".yml")
            else params_dir / (p.stem + ".yaml")
        )
        if not cand.exists():
            cand = params_dir / (p.stem + ".yml")
        if not cand.exists():
            cand = params_dir / p.name
        if cand.exists():
            out.append(cand)
        else:
            raise FileNotFoundError(f"パラメータファイルが見つかりません: {name}")
    return out
