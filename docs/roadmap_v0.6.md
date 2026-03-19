# Roadmap v0.6: Local API Integration (FastAPI in Desktop Runtime)

## 目的
AROSをデスクトップアプリ内で動作するローカルAPIとして構築し、Next.js UI（およびElectron）から非同期にリサーチジョブを投入・監視・操作できるようにする。

## 1. アーキテクチャ構成
- **API Framework**: FastAPI
- **Runtime**: Electron起動時に FastAPI をローカルプロセスとして起動（例: `127.0.0.1`）
- **Worker**: 当初は FastAPI `BackgroundTasks` またはアプリ内ワーカースレッドで実行
- **Communication**: ローカルループバック上の WebSocket（実行ログストリーミング） + REST API（ステータス取得、タスク投入）
- **Storage**: SQLite (v0.4で導入済み)

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

### ローカルセキュリティ・運用 (Phase 0.6.3)
- [ ] FastAPI のバインド先を `127.0.0.1` に固定（外部公開しない）
- [ ] CORS/Origin をローカルUI起点のみに制限
- [ ] リソース使用量の制限（CPUメモリ上限、RunPod/LLM API コスト管理）

## 3. 次のステップ
- Phase 0.5 の自律型ループ完成後、この v0.6 構成に移行する。
- Electron 側で FastAPI の起動/停止ライフサイクルを管理する。
- v0.7 以降（Next.js UI + Electron Desktop + v1.0 Release）の計画は `roadmap_v0.7_to_v1.0.md` を参照。
