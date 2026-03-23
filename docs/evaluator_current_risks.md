# Evaluator 現状設計の危険点メモ

最終更新: 2026-03-23

このドキュメントは、現時点の `research_agent/graph/nodes/evaluator.py` 周辺実装について、
運用上・設計上の危険点を整理したものです。

目的は次の2つです。

1. いま何が危ないのかを、実装者が短時間で把握できるようにする
2. どこから順に直すべきかを、優先度つきで判断できるようにする

---

## 1. 現在の Evaluator のざっくりした挙動

現在の Evaluator は、概ね次の順で動きます。

1. `workspace/main.py` をローカル実行する
2. 成功なら `success` を返す
3. 失敗なら `_classify_failure()` でヒューリスティック分類する
4. LLM 解析が有効なら `_analyze_failure_with_llm()` で補強する
5. `can_self_fix` / `needs_research` / `retry_count` / `same_error_count` / `total_execution_duration_sec` を使って次の遷移先を決める

この構造自体は悪くありません。
ただし、LLM を失敗全般に対して使うようになったことで、従来よりも「判断の揺れ」「情報漏えい」「タイムアウトの取りこぼし」が起きやすくなっています。

---

## 2. 危険点の一覧

### 2.1 高優先: `stdout` / `stderr` を LLM にそのまま送っている [完了済み]

#### 対応状況

- 完了済み
- `stdout` / `stderr` の LLM送信用整形関数を導入済み
- 機密っぽい値のマスキング、行長制限、全体長制限、`stderr` の例外周辺優先抽出を実装済み

#### 何が起きているか

失敗時、Evaluator は `stdout` と `stderr` をそのまま `system_evaluator` の入力に渡しています。
このとき、長さ制限・秘匿化・マスキングが入っていません。

#### なぜ危険か

- APIキー、トークン、パス、個人情報、社内URLなどがログに混ざると、そのまま外部 LLM に送信される
- 出力が長いとトークンコストが急増する
- 長すぎるログで、本当に重要な例外行よりノイズが支配的になる
- 将来的にバイナリ断片や巨大 traceback が混ざると、JSON 解析以前に品質が崩れる

#### 具体例

- `stderr` に `.env` 読み込み失敗時の接続文字列が出る
- `stdout` にテスト用のダミーデータではなく本番寄りの情報が出る
- 何百行もの stack trace をそのまま送り、LLM が本質でない情報に引っ張られる

#### 先にやるべき対策

- `stdout` / `stderr` の最大文字数を制限する
- APIキーっぽい文字列、Bearer token、長いURL、ホームディレクトリ配下の絶対パスをマスクする
- 例外行とその前後数行を優先し、全文送信はしない

---

### 2.2 高優先: LLM が `can_self_fix` / `needs_research` をそのまま上書きしている [完了済み]

#### 対応状況

- 完了済み
- `likely_cause` / `suggested_fixes` は引き続き LLM で補強
- `can_self_fix` / `needs_research` は無条件上書きを廃止
- `missing_module` / `syntax_error` / `name_or_type_error` / `timeout` はヒューリスティック由来の制御フラグを維持
- それ以外のカテゴリでも、LLM は保守的な方向にしか制御フラグを動かせないよう変更済み

#### 何が起きているか

現在はヒューリスティックで作った `base_feedback` に対し、LLM の返した
`likely_cause` / `suggested_fixes` / `can_self_fix` / `needs_research`
をそのままマージしています。

#### なぜ危険か

この2つは単なる説明文ではなく、グラフの遷移先を決める制御信号です。
つまり、LLM が少し判断を誤るだけで、次のような誤動作が起こります。

- 本来は Coder で直せるのに Researcher に飛ぶ
- 本来は Researcher が必要なのに Coder に戻して無駄にループする
- 停止条件に入る前の試行が無駄に増える

#### 具体例

- `NameError` なのに `needs_research=true` になり、明らかな typo 修正が外部調査タスクへ化ける
- `ModuleNotFoundError` なのに `can_self_fix=true` になり、Coder が依存導入不能なまま編集ループを回す

#### 先にやるべき対策

- `summary` はヒューリスティック優先で固定する
- `can_self_fix` / `needs_research` は全面上書きではなく、制約付きで採用する
- たとえば `missing_module` は原則 `needs_research=True` を維持する
- LLM は説明の具体化に寄せ、制御フラグは保守的に扱う

---

### 2.3 高優先: 全体 timeout が「超過前に止める」仕組みになっていない [完了済み]

#### 対応状況

- 完了済み
- 実行前に残り予算を計算し、`min(per_try_timeout, remaining_budget)` を適用するよう変更済み
- 残り予算が 0 以下の場合は、実行せず `total_timeout` で停止するよう変更済み

#### 何が起きているか

現在の `total_timeout` は、各試行の実行後に `total_execution_duration_sec` を加算してから判定しています。

#### なぜ危険か

この設計だと、予算を超えそうでもその試行自体は最後まで走ります。
つまり、厳密には「上限を守る制御」ではなく「超えたことを後から検知する制御」です。

#### 具体例

- 全体上限が 180 秒
- すでに 179 秒使っている
- 次の試行が 60 秒走る
- 実際には 239 秒まで進んだ後で `total_timeout` になる

#### 先にやるべき対策

- 実行前に残り予算を計算し、`min(per_try_timeout, remaining_budget)` を試行 timeout に使う
- 残り予算が 0 以下なら、その場で実行せず停止する

---

### 2.4 中優先: LLM 解析時間が累積 timeout に入っていない

#### 何が起きているか

`total_execution_duration_sec` は `run_workspace_python()` の実行時間しか積算していません。
Evaluator の LLM 呼び出し待ち時間はこの合計に含まれていません。

#### なぜ危険か

システム全体としては時間を使っているのに、停止判定上は「まだ余裕がある」ように見えます。
とくに API が遅いと、実時間と state の累積時間が大きくズレます。

#### 具体例

- Python 実行は毎回 2 秒
- LLM 解析が毎回 15 秒
- state 上は 10 秒しか使っていないのに、壁時計では 85 秒経っている

#### 先にやるべき対策

- `execution_duration_sec` と別に `evaluation_duration_sec` を測る
- `total_execution_duration_sec` を「実行のみ」ではなく「Evaluator 全体予算」にするか、
  もしくは `total_loop_duration_sec` のような別指標を導入する

---

### 2.5 中優先: LLM 解析の失敗が観測しづらい

#### 何が起きているか

LLM 呼び出し失敗、JSON 解析失敗、ValidationError などは catch され、
静かにヒューリスティック結果へフォールバックします。

#### なぜ危険か

安全側の実装ではありますが、運用中に次のことが見えません。

- そもそも LLM 解析が呼ばれたのか
- 呼ばれたが失敗したのか
- どの理由でフォールバックしたのか
- フォールバック率が上がっていないか

これが見えないと、品質劣化や API 障害に気づきにくくなります。

#### 先にやるべき対策

- `execution_log` に `llm_analysis_used: true/false` を残す
- 失敗時は `llm_analysis_fallback_reason` を短く記録する
- state にまで持たせるかは別として、最低限ログには残す

---

### 2.6 中優先: `MODEL_NAME` を Evaluator と他ノードで共有している [完了済み]

#### 対応状況

- 完了済み
- `EVALUATOR_MODEL_NAME` を分離済み
- 未設定時は `MODEL_NAME`、さらに未設定時は既定値へフォールバックする実装に変更済み

#### 何が起きているか

Evaluator の LLM 解析モデルは `MODEL_NAME` 環境変数をそのまま参照しています。

#### なぜ危険か

この設計だと、Coder 用に選んだモデル設定がそのまま Evaluator にも適用されます。
しかし、Evaluator に求める性質は Coder と少し違います。

- Coder: 生成能力や tool use が重要
- Evaluator: 安定した短い JSON と判断一貫性が重要

同じモデル設定を共有すると、無駄に高コストになったり、JSON 安定性が崩れたりします。

#### 先にやるべき対策

- `EVALUATOR_MODEL_NAME` を分離する
- 未設定時だけ `MODEL_NAME` や既定値へフォールバックする

---

### 2.7 中優先: エラーフィンガープリントが誤判定する可能性がある

#### 何が起きているか

同一エラー判定は、`stderr` の最終例外行をある程度正規化してハッシュ化しています。

#### なぜ危険か

この方式は軽くて良い一方、情報をかなり落としているため、
本質的に違う失敗を同じと見なすことがあります。
逆に、本質的には同じ失敗でも例外行だけ少し変わると別物扱いされます。

#### 具体例

- `ValueError: invalid value 10`
- `ValueError: invalid value 20`

数字が正規化されるため、別入力起因の問題が同一エラー扱いになる可能性があります。

また、例外の最終行が同じでも traceback 上の発生箇所が違うことがあります。

#### 先にやるべき対策

- 例外行だけでなく traceback の末尾数行も材料にする
- `summary` と組み合わせる
- それでも曖昧なら repeated stop 前に「最終確認用の1回」を入れる

---

### 2.8 中優先: 環境変数の不正値で Evaluator 全体が落ちる

#### 何が起きているか

`EXECUTION_TIMEOUT_SEC` や `MAX_TOTAL_EXECUTION_TIME_SEC` が不正値だと、
`_read_positive_float_env()` が `ValueError` を投げます。

#### なぜ危険か

設定ミス1つで、評価処理全体がハードに失敗します。
本番運用では「厳密に落とす」より「警告して既定値に戻す」方が扱いやすい場面も多いです。

#### 先にやるべき対策

- 開発中は例外でもよい
- 運用を想定するなら、warning を出して default へフォールバックする方が安全

---

### 2.9 低〜中優先: `suggested_fixes` の品質保証が弱い

#### 何が起きているか

LLM が返した `suggested_fixes` は、型が正しければそのまま採用されます。

#### なぜ危険か

- 抽象的すぎる修正案が混ざる
- ワークスペース内で実行不能な修正案が混ざる
- Coder にとって行動に落ちない表現でも通ってしまう

#### 具体例

- "Review the architecture carefully"
- "Investigate the dependency tree deeply"
- "Refactor the system for better robustness"

これらは人間にはわかっても、即座の自己修正指示としては弱いです。

#### 先にやるべき対策

- 1項目の最大長を制限する
- 「workspace 内の編集または確認行動に落ちる表現のみ許可」に寄せる
- 必要なら post-process で曖昧すぎる文を落とす

---

## 3. いちばん危ない組み合わせ

単独の問題より、組み合わさると危険になるパターンがあります。

### パターンA: 長大ログ + LLM判断上書き

1. `stderr` が長すぎる
2. LLM がノイズに引っ張られる
3. `needs_research` を誤って返す
4. 本来 Coder で直る問題が Researcher へ流れる

結果:

- ループが遠回りになる
- 外部検索コストが増える
- 原因分析の一貫性が落ちる

### パターンB: 残り時間わずか + 試行後 timeout 判定

1. すでに総予算のほとんどを消費している
2. 次の試行をそのまま開始する
3. 実行と LLM 解析で大きく時間超過する
4. その後にようやく停止する

結果:

- 「timeout があるのに時間を守れない」状態になる
- 実運用で最もストレスになるタイプの不具合になる

### パターンC: LLM fallback 多発 + ログに痕跡なし

1. LLM 解析が JSON 失敗や API エラーで落ちる
2. 毎回ヒューリスティックへ silently fallback する
3. 見た目は動いているが、改善品質だけが落ちる

結果:

- 開発者が品質低下に気づきにくい
- 原因がモデル側なのかプロンプト側なのか切り分けにくい

---

## 4. 修正優先順位

### 優先度 A: すぐやる

1. `stdout` / `stderr` の長さ制限とマスキング
2. LLM に制御フラグを全面委譲しない
3. 残り予算ベースの timeout 制御に変える

### 優先度 B: 次にやる

1. LLM 解析の使用有無と fallback 理由をログに残す
2. Evaluator 専用モデル名を分離する
3. LLM 解析対象を `runtime_error_unknown` 中心に絞る

### 優先度 C: 必要になったらやる

1. フィンガープリント精度の改善
2. `suggested_fixes` の後処理
3. timeout 指標を `execution` と `evaluation` に分ける

---

## 5. 当面の安全な運用方針

すぐに大きく作り変えない場合は、次の運用が安全です。

- LLM 解析はデフォルト ON にしてもよいが、まずログ短縮を入れる
- `needs_research` はヒューリスティック優先にする
- `runtime_error_unknown` 以外では LLM は説明補強だけに使う
- 総時間制限は「試行後判定」から「試行前残予算判定」へ寄せる
- fallback の発生回数を可視化する

---

## 6. まとめ

現状の Evaluator は、基本動作としては十分前進しています。
特に、

- 停止理由の保持
- 1試行 timeout
- 累積 timeout
- LLM による失敗解析補強

までは形になっています。

一方で、現在の危険点は「壊れる」よりも「静かに判断品質や安全性が落ちる」タイプが中心です。
とくに重要なのは次の3点です。

1. ログをそのまま LLM に送っていること
2. LLM が routing 用フラグまで上書きしていること
3. 全体 timeout が厳密制御ではなく事後検知になっていること

この3つを先に潰せば、現状設計の不安定さはかなり減らせます。
