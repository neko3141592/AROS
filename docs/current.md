**1. 先に仕様を固定する（コードを書く前）**

1. project_id 未指定時は**従来どおり** storage/runs/{run_id} を使う。
2. project_id 指定時は storage/projects/{project_id}/runs/{run_id} を使う。
3. parent_run_id は「同じ project の過去 run」を参照する。
4. parent_run_id が不正・不存在なら**失敗にせず**新規 workspace 作成にフォールバック。
5. workspace 継承は「コピー（snapshot）」で実施し、親を破壊しない。
6. meta.json に project_id, parent_run_id, workspace_source_run_id を必ず記録する（未使用時は null）。

---

**2. RunPaths を拡張する**  
編集: file_io.py (line 12)

RunPaths に次を追加します。

1. project_id: str | None
2. parent_run_id: str | None
3. workspace_source_run_id: str | None
4. （任意）runs_root_dir: Path を持たせるとデバッグしやすい。

この時点では既存コードが壊れないよう、追加フィールドは None 許容にします。

---

**3. パス解決ヘルパーを追加する**  
編集: file_io.py

必要な内部関数を先に作ると安全です。

1. _sanitize_id(value: str) -> str（/, .., 空白などを安全化）
2. _get_runs_root(base_dir, project_id)
3. _resolve_parent_workspace(runs_root, parent_run_id)（同一ルート内のみ許可）
4. _copy_workspace_snapshot(src, dst)（shutil.copytree ベース）

---

**4. create_run_paths を project 対応にする**  
編集: file_io.py (line 25)

シグネチャを次のように拡張します。

`def create_run_paths( task_id: str, base_dir: Path | None = None, project_id: str | None = None, parent_run_id: str | None = None, ) -> RunPaths: ...`

実装順は次です。

1. project_id がある場合は storage/projects/{project_id}/runs を root にする。
2. project_id がない場合は従来の storage/runs を root にする。
3. run_id 生成は現行ロジックを流用。
4. run_dir 作成。
5. parent_run_id が有効なら親 workspace を新 run の workspace へコピー。
6. 無効なら空 workspace を作成。
7. workspace_source_run_id は「実際にコピー元として使えた run_id」を記録。使えなければ None。

---

**5. meta.json 拡張**  
編集: file_io.py (line 129)

write_meta の payload を拡張します。

1. project_id
2. parent_run_id
3. workspace_source_run_id

実装は paths から読む形にすると呼び出し側変更が最小です。  
（write_meta(paths, task_id, files) の呼び出しを維持できる）

---

**6. 実行エントリ側を対応させる**  
編集: main.py (line 77)

run_aros を次のように拡張します。

1. run_aros(..., project_id: str | None = None, parent_run_id: str | None = None)
2. create_run_paths(task_id=..., project_id=project_id, parent_run_id=parent_run_id) を呼ぶ。
3. _serialize_run_paths に新フィールドも含める。
4. 既存呼び出し（引数2つ）でも動くようデフォルト値維持。

---

**7. Evaluator はそのままでも良いが、メタ記録だけ確認**  
確認対象: evaluator.py (line 92)

すでに write_meta(run_paths, ...) を呼んでいるので、write_meta を拡張すれば自動で新キーが保存されます。  
追加変更は基本不要です。

---

**8. テストを増やす（ここが超重要）**  
編集: test_tools.py (line 33)

この順で追加すると安全です。

1. project_id ありで storage/projects/{id}/runs/{run_id} が作られる。
2. project_id なしで従来どおり storage/runs/{run_id} になる。
3. 有効 parent_run_id で workspace がコピーされる。
4. 無効 parent_run_id で空 workspace にフォールバックする。
5. meta.json に3キーが入る。
6. 従来テスト（create_run_paths(task_id, base_dir)）がそのまま通る。

---

**9. 後方互換のチェック項目**

1. 既存 API 呼び出しが1行も変更不要で動くこと。
2. 既存の storage/runs 形式の run が読めること。
3. project_id=None 時の挙動が完全一致なこと。
4. meta.json の新キーは追加のみで、既存キーを壊さないこと。

---

**10. 実装を小さく分ける（おすすめコミット単位）**

1. RunPaths 拡張 + create_run_paths project 対応
2. parent_run_id workspace snapshot 実装
3. write_meta 拡張 + main.py シリアライズ更新
4. テスト追加
5. roadmap チェック更新

---

必要なら次の返答で、上記をそのまま実装できるように  
file_io.py の具体的な差分設計（擬似コードではなく、ほぼ貼り付け可能な関数単位）まで出します。