# AROS v0.1: Context Optimization & Summary Logic

大規模な実験（多数の論文調査）において、LangGraph の `AgentState` が肥大化し、LLM のコンテキスト上限（Context Window）を突破したり、重要な情報を忘却したりすることを防ぐための実装指針です。

## 1. 原則: 「全文保存」から「実装要約保存」へ

`researcher_node` は収集した論文の全文を `AgentState` に格納するのではなく、後続の `coder_node` が実装に必要とする**最小限のエッセンス**のみを抽出・要約して格納します。

### 要約に含めるべき要素 (Implementation-Critical Information)
1.  **Core Algorithm**: アルゴリズムの核となる数式やロジックの断片。
2.  **Hyperparameters**: 論文で推奨されているデフォルトのパラメータ値。
3.  **IO Structure**: 入力データと出力データの形状やデータ型。
4.  **Common Pitfalls**: 実装時にハマりやすいポイントや、論文著者が言及している注意点。

## 2. データ構造の最適化 (Future-Proofing)

現在は `research_context: str` という単純な文字列ですが、将来的なベクトル検索 (RAG) 導入を見据え、要約プロセスを独立した関数として実装します。

```python
def summarize_for_implementation(raw_paper_content: str) -> str:
    """
    収集した生の論文情報から、実装に直結する情報のみを抽出・要約する。
    v0.1 では固定の要約テキストを返すが、v0.2 以降は LLM がこの役割を担う。
    """
    # TODO: v0.2 で LLM を用いた抽出ロジックに差し替え
    return extracted_summary
```

## 3. スケーラビリティへの道筋 (Roadmap to RAG)

*   **v0.1 - v0.2**: プロンプトによる抽出・要約。State の肥大化を抑制。
*   **v0.3**: RunPod 側での環境構築履歴も含めたコンテキスト管理。
*   **v0.4**: ベクトルデータベース (Chroma/FAISS) の導入。
    *   `AgentState` には「現在アクティブな要約」のみを保持。
    *   過去の全データはベクトルDBから必要に応じて `retriever` ノードが取得する構成へ移行。

## 4. 実行時のフロー
1.  **Paper Search**: 外部APIから情報を取得（raw）。
2.  **Summarize**: `raw` を `implementation-ready summary` に変換。
3.  **State Update**: 要約のみを `AgentState["research_context"]` にマージ。
4.  **Clear Raw**: 重い生データはノード終了時に破棄し、グラフのメモリを節約。
