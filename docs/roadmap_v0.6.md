# Roadmap v0.6: Web API Integration (FastAPI)

## 目的
AROSをバックエンドサービスとして独立させ、フロントエンド（React/Next.js等）から非同期にリサーチジョブを投入・監視・操作できるようにする。

## 1. アーキテクチャ構成
- **API Framework**: FastAPI
- **Worker**: Webサーバーとは別に、リサーチ実行用のバックグラウンドワーカーを用意（当初は FastAPI の `BackgroundTasks` でも可）
- **Communication**: WebSocket（実行ログのリアルタイムストリーミング）、REST API（ステータス取得、タスク投入）
- **Storage**: PostgreSQL (v0.4で導入済み) + Redis (ジョブキュー用)

## 2. 実装チェックリスト

### API 設計 (Phase 0.6.1)
- [ ] `POST /jobs`: リサーチタスクを新規作成し、バックグラウンドでの実行を開始。
- [ ] `GET /jobs/{job_id}`: 現在のタスクの進行ステータス、生成コード、実験結果を取得。
- [ ] `GET /jobs/{job_id}/logs`: リサーチエージェントの詳細な内部ログを取得。
- [ ] `WS /jobs/{job_id}/stream`: WebSocketによるリアルタイムな状態アップデート。

### 非同期実行基盤 (Phase 0.6.2)
- [ ] LangGraph の `invoke` を非同期 (`ainvoke`) に対応させる。
- [ ] ジョブの並列実行管理（同時実行数の制限など）。
- [ ] リサーチ中断機能 (Interrupt handling) の実装。

### 認証・セキュリティ (Phase 0.6.3)
- [ ] API Key または JWT による認証。
- [ ] ユーザーごとのジョブ分離。
- [ ] リソース使用量の制限（RunPod/LLM API コスト管理）。

## 3. 次のステップ
- Phase 0.5 の自律型ループ完成後、この v0.6 構成に移行する。
- フロントエンドプロジェクト（AROS-Web等）を立ち上げ、Vercelなどのプラットフォームにデプロイを検討する。
