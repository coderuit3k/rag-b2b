# RAG(B2B)

![CI](https://github.com/coderuit3k/rag-b2b/actions/workflows/ci.yml/badge.svg)

GraphRAG cho bán lẻ mẹ & bé: router 5 nhánh trên Neo4j (graph) + Qdrant (vector hybrid) +
LightGBM (dự đoán sản phẩm) + Tavily (web) + Composio/Cloudflare (email). Nhánh graph/vector/web
có Corrective RAG (CRAG): chấm điểm ngữ cảnh truy hồi trước khi trả lời, không liên quan thì tự
viết lại câu hỏi và tìm lại thay vì trả lời liều hoặc nói "không biết".

**Tiến độ & thiết kế chi tiết: [`docs/README.md`](docs/README.md)** (nguồn chính thức) và `docs/phase-1..4`.

## Cài đặt

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e . --no-deps      # để "import rag_b2b.*" chạy từ eval/, scripts/, test
cp .env.example .env                        # rồi điền khoá (xem docs/README.md §Phụ thuộc mới)
cd frontend && npm install && cd ..         # phụ thuộc frontend Next.js
```

## Chạy

```bash
.venv/bin/uvicorn rag_b2b.api:app --reload --port 8000   # backend (cold start ~30-35s: Neo4j/Qdrant/embeddings)
cd frontend && npm run dev                                # frontend Next.js, http://localhost:3000
```

Kiểm tra rời (không cần chạy cả app):

```bash
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
| `.github/workflows/` | CI (`ci.yml`): compile-check backend + lint/build frontend trên mỗi push/PR, không cần secret |
