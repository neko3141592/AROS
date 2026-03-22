**1. 停止理由を AgentState / result に明示記録**  
編集先:  
task.py  
state.py  
evaluator.py

やること:

- ExperimentResult に stop_reason: Optional[str] を追加する
- AgentState に必要なら
    - last_execution_duration_sec
    - total_execution_duration_sec  
        を追加する
- evaluator_node() の result = ExperimentResult(...) に stop_reason=decision.stop_reason を入れる
- state return にも
    - "stop_reason": decision.stop_reason
    - "last_execution_duration_sec": execution.duration_sec
    - "total_execution_duration_sec": next_total_execution_duration_sec  
        を返す

ポイント:

- stop_reason は None | "max_retry" | "repeated_error" | "total_timeout" くらいで十分です
- 成功時は None のままでOKです

**2. timeout 制御を仕上げる**  
編集先:  
local_executor.py  
evaluator.py  
main.py

やること:

- LocalExecutionResult に duration_sec: float を追加する
- run_workspace_python() で time.monotonic() を使って実行時間を測る
- evaluator.py に定数を置く
    - DEFAULT_EXECUTION_TIMEOUT_SEC = 60.0
    - DEFAULT_MAX_TOTAL_EXECUTION_TIME_SEC = 180.0
- env から読めるようにする
    - EXECUTION_TIMEOUT_SEC
    - MAX_TOTAL_EXECUTION_TIME_SEC
- run_workspace_python(... timeout_sec=per_try_timeout) に差し替える
- state["total_execution_duration_sec"] を累積し、閾値超過なら failed/done/stop_reason="total_timeout" にする

実装の形:

- 1試行 timeout は subprocess.run(..., timeout=...) で止める
- 全体 timeout は Evaluator 側で累積時間を見て止める

判定順はこれが安全です:

1. success
2. total_timeout
3. repeated_error
4. max_retry
5. needs_research
6. coder retry

**3. Evaluator の LLM解析を入れる**  
編集先:  
evaluator.py  
llm_outputs.py  
prompt_manager.py  
新規:  
system_evaluator.yaml

おすすめ実装:

- まず今の _classify_failure() を残す
- その結果をベースに、失敗時だけ LLM で
    - likely_cause
    - suggested_fixes
    - can_self_fix
    - needs_research  
        を補強する
- LLM が失敗したら必ず今のヒューリスティックへフォールバックする

流れ:

1. _classify_failure() で base を作る
2. _analyze_failure_with_llm(...) を呼ぶ
3. JSON を parse できたら base に merge
4. 失敗したら base をそのまま返す

おすすめのガード:

- OPENAI_API_KEY が無いときは LLM解析しない
- ENABLE_EVALUATOR_LLM_ANALYSIS=0 でも無効化できるようにする

プロンプト入力はこれで十分です:

- task title
- task description
- return code
- stdout
- stderr
- base summary
- base likely cause
- base suggested fixes

**4. run_shell_command を追加して Coder プロンプトを合わせる**  
編集先:  
workspace_tools.py  
coder.py  
system_coder.yaml

おすすめ実装:

- workspace_tools.py に run_shell_command(run_paths, command, timeout_sec=5.0) を追加
- shell=True は使わない
- shlex.split() して allowlist command のみ許可する
- cwd=run_paths.workspace_dir で実行する
- 読み取り系だけ許可する
    - rg
    - ls
    - find
    - cat
    - head
    - tail
    - wc
    - pwd
    - sed
- 拒否するもの
    - 絶対パス
    - ..
    - |
    - &&
    - ;
    - >
    - <

coder.py では:

- TOOL_SCHEMAS に run_shell_command を追加
- TOOL_FUNCTIONS にも追加

system_coder.yaml では:

- まず built-in tools を優先
- 必要なら run_shell_command で read-only に探索
- 変更操作には使わない  
    を明記すれば十分です

**5. 先に足すべきテスト**  
編集先:  
test_evaluator.py  
test_tools.py  
test_prompt_manager.py  
test_nodes_llm.py

最低限ほしいケース:

- ExperimentResult.stop_reason が max_retry / repeated_error / total_timeout で埋まる
- total_execution_duration_sec 超過で停止する
- LLM解析成功時に suggested_fixes が上書きされる
- LLM解析失敗時にヒューリスティックへフォールバックする
- run_shell_command が rg などを実行できる
- run_shell_command が ../, absolute path, pipe/redirect を拒否する
- system_evaluator.yaml が load/render できる
- system_coder.yaml に run_shell_command が入っている

**6. 最後に roadmap を更新**  
編集先:  
roadmap_v0.2c.md

更新する項目:

- 実行タイムアウト（1試行あたり / 全体）の停止制御
- 停止理由を AgentState / result に明示的に記録
- LLMで実行エラーを解析し、具体的な修正指示を生成
- Coder のシステムプロンプトに ... run_shell_command ...
- 関連テスト項目