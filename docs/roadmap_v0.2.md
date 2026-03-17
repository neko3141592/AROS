# ARL 開発ロードマップ v0.2: ツール統合とLLM連携（Local実行フェーズ）

このフェーズでは、モックとして実装した各ノードに実際のLLM（Claude/GPT）を組み込み、実際の外部API（論文検索）を利用するようにします。また、生成されたコードをローカル環境で安全に実行・評価する基本パイプラインを構築します。

## 1. プロンプト管理システムの構築
- [ ] `prompts/system_planner.yaml`: プランナー用のシステムプロンプト作成（タスク分解の指示）
- [ ] `prompts/system_coder.yaml`: コーダー用のシステムプロンプト作成（PyTorch等を用いた安全で実行可能なコード生成の指示）
- [ ] プロンプト読み込み・適用ユーティリティ関数の実装

## 2. 実機能ツールの実装 (Tools)
- [ ] `tools/file_io.py`: 生成コードの保存、実行ログの読み書き機能の実装
  - [ ] 実行ごとのタイムスタンプ付きディレクトリ作成機能（バージョニング）
- [ ] `tools/paper_search.py`: Arxiv API または Semantic Scholar API との連携ツール実装
  - [ ] 検索クエリの構築とAPI呼び出し
  - [ ] 結果のパースとLLMプロンプト向けフォーマット整形

## 3. LLMノードの実装 (LLM Integration)
- [ ] `graph/nodes/planner.py`: LLMを用いてユーザーの入力をサブタスクに分解する処理の実装
- [ ] `graph/nodes/researcher.py`: `paper_search.py` ツールを呼び出し、必要なコンテキストを収集・要約する処理の実装
- [ ] `graph/nodes/coder.py`: 収集したコンテキストとタスクに基づき、実際のPythonコード（ローカル実行用）をLLMで生成する処理の実装

## 4. ローカル実行と自己修正ループ (Local Execution & Evaluation)
- [ ] `graph/nodes/evaluator.py` にローカル実行エンジンを一時的に統合
  - [ ] `subprocess` などを用いて生成されたPythonコードをサンドボックス/仮想ローカル環境で実行
  - [ ] 標準出力と標準エラー出力をキャプチャし、AgentState に保存
- [ ] LLMを用いた実行エラーの解析と、改善案（エラー文に基づく修正指示）の生成処理を Evaluator に実装
- [ ] Coder -> Local Run -> Evaluator のフィードバックループのテスト

## 5. テストと動作検証
- [ ] `tests/test_tools.py`: 各ツール（API連携、ファイルIO）の単体テスト
- [ ] 実際のLLMバックエンドを使用したエンドツーエンドの簡単な機械学習タスク（ローカルCPUで動く軽量なもの）の実行テスト
- [ ] 意図的にエラーのあるコードを生成させ、自己修正ループが機能して修正されるかのテスト（Self-Correction テスト）
