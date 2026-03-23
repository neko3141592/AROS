# ARL 開発ロードマップ v0.2b: Coderアーキテクチャ移行（Tool Calling）

このフェーズでは、Coderを「1回の全文生成」から「Workspaceを探索し、必要箇所だけを編集する」方式へ移行します。  
v0.2aの動作基準点を維持しながら、段階的に差し替えてリグレッションを検出しやすくします。

## 1. Workspace操作ツール
- [x] `tools/workspace_tools.py` を新規実装
  - [x] `list_files`: Workspace配下のファイル一覧取得
  - [x] `read_file`: 任意ファイルの内容読取
  - [x] `edit_file` / `replace_string`: 部分修正(検索)
  - [x] `create_file`: 新規作成
  - [x] パス安全性（`workspace_dir`外アクセス、絶対パス、`..`）の防止

## 2. CoderノードのReAct化
- [x] Coderに `workspace_tools.py` をバインドし、Tool Calling可能にする
- [x] `prompts/system_coder.yaml` を「全文生成」から「探索 + 差分修正」前提に刷新
- [x] 1回のCoder実行内で複数回のツール反復を許可
  - [x] 例: `list_files -> read_file -> edit_file -> read_file -> ...`
- [x] Coder内部に反復上限（例: `max_tool_steps`）と終了条件を導入

## 3. State管理の正本移行
- [x] 前段階として `generated_files` 対応を導入済み
- [x] `AgentState` の正本を `workspace_dir` ベースへ移行
- [x] `generated_code` / `generated_files` 依存をクリティカルパスから外す
- [x] Evaluator の評価対象を `AgentState["generated_code"]` ではなく `workspace_dir` の物理ファイル参照へ完全切り替え
- [x] 既存入出力（ログ保存、状態保存）との互換を保ちつつ段階移行

## 4. プロジェクト単位実行管理（基盤）
- [x] `project_id` を導入し、`run` をプロジェクト配下で管理
- [x] ストレージ構成を project/run 階層へ整理（例: `storage/projects/{project_id}/runs/{run_id}/workspace`）
- [x] `run` 作成時に `parent_run_id` を指定して前回Workspaceを引き継げる仕組みを実装（破壊的上書きではなくコピー/スナップショット方式）
- [x] `meta.json` に `project_id`, `parent_run_id`, `workspace_source_run_id` を保存
- [x] 既存の単発実行フロー（project未指定時）との後方互換を維持

## 5. 回帰防止テスト
- [x] `tests/test_tools.py` に `workspace_tools.py` の単体テスト追加
- [x] v0.2aの既存ケースが破壊されないことを確認する回帰テスト
- [x] Coderが最小差分で修正できることの統合テスト
- [x] 同一 `project_id` で複数runを連続実行し、Workspace継承が正しく働くテスト
- [x] `parent_run_id` 不正時の安全なフォールバック（新規Workspace作成）テスト

## 完了条件
1. Coderがワークスペースを探索し、必要部分だけを複数ステップで編集できる。  
2. `workspace_dir` がコード状態の正本として機能し、状態管理が安定している。  
3. プロジェクト単位でrun履歴とWorkspace継承を管理できる。  
4. v0.2aのベースライン機能に対して重大な回帰がない。
