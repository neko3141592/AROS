# このプロジェクトは以下のディレクトリで作成する。

```text
research_agent/
├── main.py                # 実行エントリーポイント（CLI/Worker）
├── graph/                 # LangGraph関連
│   ├── __init__.py        # グラフの構築・定義
│   ├── state.py           # AgentStateの定義（タスク情報、RunPod ID等）
│   ├── nodes/             # 各エージェントの思考ロジック
│   │   ├── planner.py     # タスク作成・分解
│   │   ├── researcher.py  # 論文検索・情報抽出
│   │   ├── coder.py       # 実験コード生成
│   │   └── evaluator.py   # 結果分析・改善案提示
│   └── edges.py           # 条件分岐（成功なら終了、失敗なら再実装など）
├── tools/                 # エージェントが使う道具（純粋なPython関数）
│   ├── paper_search.py    # Arxiv/Semantic Scholar連携
│   ├── runpod_client.py   # RunPodの起動・コード転送・実行指示
│   └── file_io.py         # 実験結果の保存・読み込み
├── docker/                # RunPodで動かす環境定義
│   ├── base.Dockerfile    # PyTorch等の基本環境
│   └── setup.sh           # 初期設定スクリプト
├── schema/                # データの型定義
│   └── task.py            # 「実験タスク」のPydanticモデル
├── prompts/               # プロンプト管理
│   ├── system_planner.yaml
│   └── system_coder.yaml
├── storage/               # ローカルでの一時データ・ログ
└── tests/                 # ツール単体のテストコード
```