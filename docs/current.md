1. AgentState に自己修正用フィールドを追加する。  
    編集: state.py  
    追加候補: failure_summary, failure_fingerprint, stop_reason, evaluator_feedback。  
    完了条件: Evaluatorが次回Coderに渡したい情報を state に保持できる。
    進捗: 完了。`evaluator_feedback`, `error_signature`, `same_error_count`, `stop_reason` を保持できる。
    
2. Evaluator に失敗解析ロジックを追加する。  
    編集: evaluator.py  
    実装関数例: _summarize_failure(stderr, returncode), _fingerprint_failure(stderr, returncode)。  
    分類方針例: ModuleNotFoundError, SyntaxError, AssertionError, Timeout, RuntimeError。  
    完了条件: 実行失敗時に failure_summary と failure_fingerprint が必ず生成される。
    進捗: 完了。失敗分類とエラーフィンガープリント生成が実装済み。
    
3. Evaluator の返却 payload に上記を含める。  
    編集: evaluator.py  
    return {...} に failure_summary, failure_fingerprint, evaluator_feedback, stop_reason を追加。  
    完了条件: Coderノードが state から失敗要約を読める。
    進捗: 完了。`evaluator_feedback`, `execution_stderr`, `error_signature`, `stop_reason` を state に返却している。
    
4. Coder の user prompt 構築に失敗コンテキストを注入する。  
    編集: coder.py  
    _build_user_prompt(...) または追加ヘルパーで、前回 execution_stderr と failure_summary を必ず入れる。  
    完了条件: 2回目以降のCoderが「前回どこで失敗したか」を明示的に認識して修正する。
    進捗: 完了。retry時は `summary / likely_cause / suggested_fixes / stderr` を prompt 先頭に注入する。
    
5. Coder システムプロンプトを自己修正向けに追記する。  
    編集: system_coder.yaml  
    追記内容例: 「前回の失敗要因を優先して修正」「無関係な変更を避ける」「必要ならツールで再確認」。  
    完了条件: プロンプト仕様と実装（state注入）が一致する。
    進捗: 一部未完。自己修正ルールは `coder.py` の user prompt 側で補完済みだが、`system_coder.yaml` 本体の整理は未反映。
    
6. 停止条件の最小版を入れる。  
    編集: evaluator.py, edges.py  
    まずは max_retry と「同一 fingerprint の連続回数」で停止。  
    完了条件: 同じ失敗を無限に繰り返さない。
    進捗: 一部未完。`max_retry` と fingerprint 検知はあるが、同一 fingerprint 反復での停止条件は未実装。
    
7. テストを追加する。  
    編集: test_evaluator.py, test_nodes_llm.py  
    追加ケース: 失敗解析がstateへ入る、同一エラー反復で停止、2回目で修正成功。  
    完了条件: self-correction の最小統合テストが通る。
    進捗: 一部完了。失敗解析・retry prompt・2回目で修正成功のテストは追加済み。同一エラー反復停止テストは未完。
    
8. v0.2b の未更新チェックを整理してから v0.2c に移る。  
    編集: roadmap_v0.2b.md, roadmap_v0.2c.md  
    v0.2b-5 は「テスト追加済み」の項目を更新。  
    完了条件: ロードマップと実装状態が一致する。
    進捗: 進行中。v0.2c 側は今回の自己修正ループ進捗を反映、v0.2b 側の未完テスト項目は継続管理。
