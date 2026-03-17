# AROS Research Agent

LangGraph を使って、論文調査から実験コード生成・評価までのループを構築するためのプロジェクトです。  
この段階では、基盤ディレクトリとローカル実行の準備を進めています。

## セットアップ

1. `research_agent` に移動

```bash
cd /Users/yudai/Documents/WebApp/AROS/research_agent
```

2. 仮想環境を作成して有効化

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. 依存ライブラリをインストール

```bash
pip install -r requirements.txt
```

4. 環境変数ファイルを用意

```bash
cp .env.example .env
```

`OPENAI_API_KEY` など必要な値を `.env` に設定してください。

## 現在の主要構成

- `graph/`: LangGraph の state / nodes / edges
- `schema/`: Pydantic モデル
- `tools/`: 外部副作用を持つ処理群
- `prompts/`: プロンプト定義
- `storage/`: ローカル保存データ・ログ
- `tests/`: テストコード

## 実行とテスト

- 実行エントリーポイント（`main.py`）はこの後のロードマップで実装予定です。
- テストは `pytest` を想定しています。
