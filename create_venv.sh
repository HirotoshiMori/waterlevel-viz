#!/bin/sh
set -e

# スクリプトがあるディレクトリをプロジェクトルートとする（sh  compatible）
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
PROJECT_DIR="$(pwd)"
PROJECT_NAME="$(basename "$PROJECT_DIR")"

# プロジェクト直下に .venv を作成（ノートブック・IDE が認識しやすい）
VENV_DIR="$PROJECT_DIR/.venv"
PYTHON_BIN="$VENV_DIR/bin/python"

VSCODE_DIR="$PROJECT_DIR/.vscode"
SETTINGS_FILE="$VSCODE_DIR/settings.json"

if [ ! -f "pyproject.toml" ]; then
    echo "エラー: pyproject.toml が見つかりません。プロジェクトルートで実行してください。"
    exit 1
fi

echo "==============================================="
echo ">>> プロジェクト名: $PROJECT_NAME"
echo ">>> 仮想環境: $VENV_DIR"
echo ">>> Python: $PYTHON_BIN"
echo "==============================================="

echo ">>> [1/4] uv lock で pyproject.toml を uv.lock に反映..."
uv lock

echo ">>> [2/4] uv sync で仮想環境作成＆パッケージ同期..."
uv sync

echo ">>> [3/4] VS Code / Cursor 設定を書き込み..."
mkdir -p "$VSCODE_DIR"

cat > "$SETTINGS_FILE" <<EOF
{
  "python.defaultInterpreterPath": "$PYTHON_BIN",
  "python.terminal.activateEnvironment": true,
  "jupyter.jupyterServerType": "local",
  "jupyter.notebookFileRoot": "\${workspaceFolder}"
}
EOF

echo ">>> [4/4] 完了"
echo ">>> 利用する Python: $PYTHON_BIN"
echo "==============================================="