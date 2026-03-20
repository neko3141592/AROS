# ARL 開発ロードマップ v0.2a: LLM統合（アーキテクチャ維持）

このフェーズでは、既存のノード構造（Planner -> Researcher -> Coder -> Evaluator）を保ったまま、LLM連携とローカル実行の基盤を成立させます。  
この段階のCoderは、従来どおり「コード全文を一括生成する方式」のまま運用します。

## 1. プロンプト管理システム
- [x] `prompts/system_planner.yaml`: プランナー用システムプロンプト作成
- [x] `prompts/system_coder.yaml`: コーダー用システムプロンプト作成（一括生成方式）
- [x] プロンプト読み込み・適用ユーティリティの実装

## 2. 実機能ツールの実装
- [x] `tools/file_io.py`: 生成コード保存、実行ログ読み書き
  - [x] タイムスタンプ付き実行ディレクトリ作成
- [x] `tools/paper_search.py`: ArXiv / Semantic Scholar 連携
  - [x] 検索クエリ構築とAPI呼び出し
  - [x] 結果パースとLLM向け整形

## 3. LLMノード統合（構造は維持）
- [x] `graph/nodes/planner.py`: LLMでサブタスク分解
- [x] `graph/nodes/researcher.py`: 論文検索 + 要約生成
- [x] `graph/nodes/coder.py`: タスクと調査結果からコード全文を生成
- [x] Coderは一括生成（JSON files返却）方式を維持

## 4. ローカル実行エンジンの基本統合
- [x] `graph/nodes/evaluator.py` のモック実行を実行エンジンへ置換
  - [x] `subprocess` で `workspace_dir` 上のコードを実行
  - [x] stdout / stderr / return code を取得して `AgentState` とログへ保存
  - [x] 成否判定をランダムではなく実行結果ベースに変更

## 5. テストと動作確認
- [x] `tests/test_tools.py`: `file_io` / `paper_search` の単体テスト整備
- [x] LLMバックエンドを使った軽量E2Eタスクの実行確認

## 完了条件
1. Planner/Researcher/Coder/Evaluator が実LLM + 実データで一巡できる。  
2. Evaluatorがモックではなく、ローカル実行結果で評価する。  
3. Coderはまだ一括生成方式のままで動作する（ReAct移行はv0.2bで実施）。
