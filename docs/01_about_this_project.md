# プロジェクト名：Autonomous Research Loop (ARL)
**〜 論文調査からGPU実験、自己改善までを完結させる自律型研究エージェント 〜**

## 1. プロジェクトのビジョン
研究者が「仮説」を立てることに集中できるよう、それ以降の「文献調査」「実装」「GPUリソース確保」「実験実行」「結果分析・改善」のサイクルを、LangGraphを用いた自律型エージェントにより完全に自動化・高速化する。

## 2. 解決する課題
1.  **実装のオーバーヘッド**: 論文からコードへの落とし込みにかかる時間の削減。
2.  **インフラ管理の手間**: RunPod等のGPU環境の起動・設定・シャットダウンの自動化。
3.  **試行錯誤のコスト**: エラー修正やパラメータ調整のループをAIが肩代わりし、人間は高レベルな指示に専念。
4.  **コードの複雑化（崩壊）**: 疎結合なアーキテクチャにより、機能拡張時のバグやコードのスパゲッティ化を防止。

## 3. 主要機能（Core Features）
- **Autonomous Planner**: 実験タスク（人間入力/AI生成）を具体的な実行ステップに分解。
- **Doc-to-Code Pipeline**: Arxiv等から最新知見を取得し、即座に実行可能なPythonコードを生成。
- **Remote Execution Engine**: Docker + RunPod SDKによる、スケーラブルなGPU実行環境の自動制御。
- **Self-Correction Loop**: 実行結果（Log/Metrics）を評価し、コードのバグ修正やアルゴリズムの改善案を提示・再実行。
- **Human-in-the-loop Interface**: 重要な判断（高額な計算資源の使用、最終結果の承認）における人間の介入。

## 4. 技術スタック（Technical Stack）
- **Orchestration**: LangGraph (State management & Cyclical workflow)
- **Intelligence**: Claude 3.5 Sonnet / GPT-4o (Reasoning & Coding)
- **Execution Environment**: Docker, RunPod Python SDK
- **Data/Task Management**: Pydantic (Strict typing), PostgreSQL (Task persistence)
- **External APIs**: Arxiv API, Semantic Scholar API
- **UI/Frontend**: Streamlit (Dashboard for monitoring)

## 5. システムアーキテクチャ（設計方針）
「脳（思考）」「記憶（状態）」「手足（ツール）」を完全に分離する。
- **State-Centric**: すべてのノードは共通の `AgentState` 型（Pydantic）を介してのみ通信する。
- **Tool-Based Execution**: OS操作やネットワーク通信は、LLMが直接行うのではなく、定義された「Tool」を介してのみ行う。
- **Immutable Trace**: 実験ログと生成コードは、上書きせずすべてバージョン管理（experiments/フォルダ）に保存。