# ARL 開発ロードマップ v0.2c: 自己修正ループの確立

このフェーズでは、CoderとEvaluatorの反復フィードバックループを本番運用可能な形に仕上げます。  
特に、停止条件とフェイルセーフを明確化し、無限ループや無駄な再試行を防ぎます。

## 1. 反復フィードバックループの実装
- [x] 基本の `Coder -> Evaluator -> Coder/END` 分岐骨格は存在
- [x] Evaluatorが `stdout` / `stderr` / 失敗要因をそのままCoderへ返す
- [x] Coderがエラー内容を元に `read_file` / `edit_file` / `replace_string` で再修正
- [x] 複数回の反復で成功に到達できる実行パスを整備
- [x] 動的ルーティングを導入し、Evaluatorが「修正不能・知識不足」と判断した場合はCoderではなくResearcherへ戻す条件付きエッジを実装
- [x] 同一 `project_id` の反復では、直前runのWorkspaceを継承して修正を続行する（ループ中の文脈を保持）

## 2. 停止条件とフェイルセーフ
- [x] `max_retry` 到達時の強制停止
- [x] 同一エラーの反復検知（エラーフィンガープリント）と停止
- [x] 実行タイムアウト（1試行あたり / 全体）の停止制御
- [x] 停止理由を `AgentState` / `result` に明示的に記録
  - `AgentState.stop_reason` と `ExperimentResult.stop_reason` の両方へ反映
  - 実行時間は `last_execution_duration_sec` / `total_execution_duration_sec` で追跡

## 3. Evaluatorの改善
- [x] LLMで実行エラーを解析し、具体的な修正指示を生成
- [x] Coderへ渡すフィードバック形式を標準化（ログ、要約、次アクション）
- [x] 実行ログを追跡しやすい形で保存（run単位で連続記録）
- [x] Coder のシステムプロンプトに「必要に応じて `run_shell_command` で検索・探索を行う」指示を追加
  - `run_shell_command` は allowlist 付きの read-only 探索用ツールとして実装

## 4. テストと動作検証
- [x] Self-Correctionテスト（初回失敗コードが2回目以降で修正される）
- [x] 同一エラー反復時に停止できることのテスト
- [x] タイムアウト停止のテスト
- [x] 停止理由 (`max_retry` / `repeated_error` / `total_timeout`) の記録テスト
- [x] Evaluator LLM解析の成功時上書き / 失敗時フォールバックのテスト
- [x] `run_shell_command` の許可・拒否条件のテスト
- [x] `system_evaluator.yaml` / `system_coder.yaml` のロード・利用テスト
- [x] 反復修正ループの統合テスト（収束性と回帰検証）
- [ ] 「修正不能・知識不足」判定時に `Evaluator -> Researcher -> Coder` へ遷移する動的ルーティングのテスト
- [ ] 同一プロジェクト内で `run_n -> run_{n+1}` のWorkspace継承が維持されることの統合テスト
- [ ] 別 `project_id` 間でWorkspaceが混在しないことの分離テスト

## 完了条件
1. CoderとEvaluatorが複数回反復し、成功または定義済み停止条件で終了できる。  
2. 停止理由が追跡可能で、運用時のデバッグ性が確保されている。  
3. Self-Correctionを示す統合テストが安定して通る。
