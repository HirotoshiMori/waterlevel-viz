"""
OutputManager: ケース別出力フォルダの作成とパス生成・メタデータ保存を一元管理する。
"""
from __future__ import annotations

import json
import logging
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


class OutputManager:
    """
    解析の出力先を output/{case_id}/ に集約し、
    figures / tables / logs / artifacts のパス生成と params コピー・metadata 保存を提供する。
    """

    SUBDIRS = ("figures", "tables", "logs", "artifacts")

    def __init__(
        self,
        case_dir: Path,
        case_id: str,
        params_path: Path,
        params_cfg: dict[str, Any] | None = None,
    ):
        self.case_dir = Path(case_dir)
        self.case_id = case_id
        self.params_path = Path(params_path)
        self.params_cfg = params_cfg or {}
        self._run_started_at: datetime | None = None
        self._log_handler: logging.FileHandler | None = None

    @classmethod
    def from_params(cls, params_path: str | Path) -> OutputManager:
        """
        params_path の YAML を読み、case_id を決定し出力ディレクトリを用意して返す。
        case_id は YAML 内の case_id があればそれ、なければファイル名の stem（例: case01.yml → case01）。
        """
        params_path = Path(params_path)
        if not params_path.is_absolute():
            # カレント基準で探す。notebook からは ../params の可能性あり
            for base in (Path.cwd(), Path.cwd().parent):
                cand = base / params_path
                if cand.exists():
                    params_path = cand.resolve()
                    break
            else:
                params_path = (Path.cwd() / params_path).resolve()
        if not params_path.exists():
            raise FileNotFoundError(f"パラメータファイルが見つかりません: {params_path}")

        with open(params_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        case_id = cfg.get("case_id") or params_path.stem
        # 出力ルートは「params の親の親」をプロジェクトルートとする
        project_root = params_path.resolve().parent.parent
        output_root = project_root / "output"
        case_dir = output_root / case_id

        for sub in cls.SUBDIRS:
            (case_dir / sub).mkdir(parents=True, exist_ok=True)

        # 実行時点のパラメータをそのままコピー（params.yml）
        dest_params = case_dir / "params.yml"
        shutil.copy2(params_path, dest_params)

        return cls(
            case_dir=case_dir,
            case_id=case_id,
            params_path=params_path,
            params_cfg=cfg,
        )

    def fig_path(self, name: str) -> Path:
        """output/{case_id}/figures/{name} のパスを返す。"""
        return self.case_dir / "figures" / name

    def table_path(self, name: str) -> Path:
        """output/{case_id}/tables/{name} のパスを返す。"""
        return self.case_dir / "tables" / name

    def log_path(self, name: str) -> Path:
        """output/{case_id}/logs/{name} のパスを返す。"""
        return self.case_dir / "logs" / name

    def artifact_path(self, name: str) -> Path:
        """output/{case_id}/artifacts/{name} のパスを返す。"""
        return self.case_dir / "artifacts" / name

    def start_run(self) -> None:
        """実行開始時刻を記録し、run.log に出力するロギングを有効にする。"""
        self._run_started_at = datetime.now(timezone.utc)
        log_file = self.log_path("run.log")
        self._log_handler = logging.FileHandler(log_file, encoding="utf-8")
        self._log_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        )
        root = logging.getLogger()
        root.addHandler(self._log_handler)
        root.setLevel(logging.DEBUG)
        logging.info("Run started at %s (case_id=%s)", self._run_started_at.isoformat(), self.case_id)

    def finish_run(self) -> None:
        """実行終了時刻を記録し、metadata.json を書き、ログハンドラを外す。"""
        finished_at = datetime.now(timezone.utc)
        if self._log_handler:
            root = logging.getLogger()
            root.removeHandler(self._log_handler)
            self._log_handler.close()
            self._log_handler = None

        elapsed_sec = None
        if self._run_started_at:
            elapsed_sec = (finished_at - self._run_started_at).total_seconds()

        self.save_metadata(
            run_started_at=self._run_started_at.isoformat() if self._run_started_at else None,
            run_finished_at=finished_at.isoformat(),
            elapsed_sec=elapsed_sec,
        )

    def save_metadata(self, **overrides: Any) -> None:
        """
        output/{case_id}/metadata.json を生成する。
        必須項目を埋め、overrides で上書き・追加できる。
        """
        meta: dict[str, Any] = {
            "case_id": self.case_id,
            "params_path": str(self.params_path),
            "run_started_at": overrides.get("run_started_at"),
            "run_finished_at": overrides.get("run_finished_at"),
            "elapsed_sec": overrides.get("elapsed_sec"),
            "python_version": sys.version.split()[0],
            "platform": {
                "system": platform.system(),
                "machine": platform.machine(),
                "node": platform.node(),
            },
            "git_commit": _get_git_commit(),
            "seed": self.params_cfg.get("seed"),
        }
        meta.update(overrides)

        path = self.case_dir / "metadata.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)


def _get_git_commit() -> str | None:
    """カレントディレクトリの git HEAD を返す。取得失敗時は None。"""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            cwd=Path.cwd(),
        )
        if out.returncode == 0 and out.stdout:
            return out.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None
