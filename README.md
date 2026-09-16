# RAG(B2B)

GraphRAG cho bán lẻ mẹ & bé: router 5 nhánh trên Neo4j (graph) + Qdrant (vector hybrid) +
LightGBM (dự đoán sản phẩm) + Tavily (web) + Composio/Cloudflare (email).

**Tiến độ & thiết kế chi tiết: [`docs/README.md`](docs/README.md)** (nguồn chính thức) và `docs/phase-1..4`.

## Cài đặt

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e . --no-deps      # để "import rag_b2b.*" chạy từ eval/, scripts/, test
cp .env.example .env                        # rồi điền khoá (xem docs/README.md §Phụ thuộc mới)
```

## Chạy

```bash
streamlit run src/rag_b2b/app.py                       # UI chat
python -m rag_b2b.indexer verify                       # kiểm tra kết nối + số liệu store
python -c "from rag_b2b.pipeline import agent_pipeline; \
          print(agent_pipeline.invoke({'question': 'Khách 10001 sẽ mua gì tiếp theo?'}))"
```

## Bố cục

| Thư mục | Nội dung |
|---|---|
| `src/rag_b2b/` | package: `config` (clients dùng chung) · `pipeline` (router) · `indexer` (ETL) · `app` (UI) · `tools/` (5 nhánh) |
| `scripts/` | vận hành: `run_athena.py`, `wipe_graph.py`, `train_ranker.py`, `convert_2025_pickle.py`, `sql/` (CTAS Athena) |
| `eval/` | harness đánh giá; `eval/results/` = artefact sinh ra |
| `data/` | `test/` ground-truth · `cache/` model tải từ S3 |
| `docs/` | kế hoạch + tiến độ theo phase |
