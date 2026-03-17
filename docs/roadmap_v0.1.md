# AROS 開発ロードマップ v0.1: 基盤アーキテクチャとローカル実行ループの確立

このフェーズでは、システム全体の骨組みとなるディレクトリ構造、型定義、およびLangGraphによる最小限のループ（モックノードを含む）を構築します。外部APIやGPU環境にはまだ依存せず、単一のローカル開発環境で状態遷移が正しく行われることを保証します。

## 1. プロジェクト構造と環境構築
- [x] `research_agent/` およびサブディレクトリの作成（`02_directly.md` に基づく）
- [x] 仮想環境（venvまたはPoetry/uv）のセットアップと依存ライブラリ（`langgraph`, `langchain`, `pydantic`, `pytest` 等）のインストール
- [x] 環境変数管理（`.env`）のセットアップ（LLM APIキー等）
- [x] `README.md` の初期作成（プロジェクト概要、セットアップ手順）

## 2. スキーマと状態管理の定義 (Schema & State)
- [x] `schema/task.py`: Pydanticを用いた `Task`, `ExperimentResult` モデルの定義
  - [x] バリデーションルールの実装（型チェック、必須フィールの設定）
- [x] `graph/state.py`: `AgentState` の定義（TypedDict または Pydantic）
  - [x] 必要なキーの定義（例: `task`, `messages`, `current_step`, `code`, `logs`, `status`）
  - 📝 **詳細設計**: [v0.1/01_schema_design.md](v0.1/01_schema_design.md) を参照して実装すること。

## 3. モックノードとエッジの実装 (Graph Definitions)
- [ ] `graph/nodes/planner.py`: ダミーのタスク分割ロジック（モック）実装
- [ ] `graph/nodes/researcher.py`: ダミーの論文検索結果を返すロジック（モック）実装
- [ ] `graph/nodes/coder.py`: 単純な「Hello World」Pythonコードを生成するロジック実装
- [ ] `graph/nodes/evaluator.py`: コードの実行結果（モック）を評価し、成功・失敗を判定するロジック実装
- [ ] `graph/edges.py`: ノード間の条件分岐（例：エラーがあれば `coder` に戻る）の実装
- [ ] `graph/__init__.py`: 上記ノードとエッジを結合し、StateGraphをコンパイルするロジック実装

## 4. 実行エントリーポイントとロギング
- [ ] `main.py`: CLIからLangGraphを呼び出して初期Stateを渡し、実行を開始する機能の実装
- [ ] 基本的なロギング設定（標準出力およびローカルファイルへの実行ログ保存）
- [ ] `storage/` ディレクトリを用いた簡易的な状態記録（JSON等）の仕組み作成

## 5. テストと動作検証
- [ ] `tests/test_schema.py`: Pydanticモデルのバリデーションテスト
- [ ] `tests/test_graph.py`: グラフの遷移（Planner -> Researcher -> Coder -> Evaluator）が想定通りに行われるかの単体テスト
- [ ] （チェックポイント）モック環境で全体のループがエラーなく終了することを確認
