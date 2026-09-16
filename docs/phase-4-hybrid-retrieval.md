# Phase 4 — Qdrant hybrid retrieval (dense + sparse)

## Trạng thái: 6/6 task — 100%  (bộ 5000-sample, 20k doc; A/B đo trên bộ 2000/10k doc — kết luận không đổi)

### Kết quả A/B — hybrid THẮNG dense-only rõ rệt (2026-09-06)

`eval/compare_retrieval.py --n 400 --k 10` — known-item retrieval trên cùng 1 collection
`b2b_customers_openai_hybrid` (10.000 doc, có cả vector `dense` + `sparse` BM25), chỉ khác chế độ.
2 kiểu query từ `rag_context` của khách: `full` (lát cắt ~paraphrase) và `rare` (token hiếm nhất).

| query | mode | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|
| full | dense  | 0.235 | 0.368 | 0.433 | 0.295 |
| full | **hybrid** | **0.500** | **0.725** | **0.753** | **0.605** |
| rare | dense  | 0.178 | 0.293 | 0.353 | 0.227 |
| rare | **hybrid** | **0.438** | **0.848** | **0.873** | **0.627** |

Δ hybrid − dense: **full +0.32 R@10 / +0.31 MRR**, **rare +0.52 R@10 / +0.40 MRR**.
Sparse BM25 giúp mạnh nhất ở query token hiếm (tên hàng/brand/tỉnh) mà dense làm nhoè — đúng kỳ vọng.
Ngay cả query "ngữ nghĩa" cũng gần gấp đôi mọi metric.

→ `eval/results/retrieval_compare_20260906_235710.json`, `eval/results/retrieval_compare.csv`.
**Kết luận: chuyển nhánh `vector` sang hybrid.** `src/rag_b2b/tools/vector.py` đã trỏ `get_hybrid_store()`.

### Chi tiết thực hiện

- **4.1** `pip install fastembed` (model `Qdrant/bm25`, tải 18 file HF lần đầu).
- **4.2** `src/rag_b2b/config.py::get_hybrid_store()` (lazy) — collection `b2b_customers_openai_hybrid`,
  `vectors_config={"dense": 1536/COSINE}` + `sparse_vectors_config={"sparse"}`,
  `QdrantVectorStore(retrieval_mode=HYBRID, vector_name="dense", sparse_vector_name="sparse")`.
  Thêm `timeout=60` cho `QdrantClient`.
- **4.3** `python -m rag_b2b.indexer hybrid_sample` — nạp `final_data_sample` (10k). Batch 1000 timeout ghi
  Qdrant Cloud → hạ **batch 200**, OK (50 batch). Deterministic UUIDv5 như Phase 1.
- **4.4** `src/rag_b2b/tools/vector.py` → `get_hybrid_store()` (lazy cache). Smoke: "khách nữ ở Hà Nội…" +
  "khách nào hay mua Aptamil NZ?" → trả lời tiếng Việt hợp lý qua `agent_pipeline`.

### Checklist gốc

**Mục tiêu**: nhánh `vector` truy hồi bằng **hybrid** — kết hợp dense (ngữ nghĩa,
`text-embedding-3-small`) và sparse (từ khoá / token hiếm, BM25). Giúp bắt đúng mã SKU,
tên brand, thuật ngữ hiếm mà embedding dense hay bỏ lỡ.

**File đụng tới**: `src/rag_b2b/config.py`, `src/rag_b2b/tools/vector.py`, `src/rag_b2b/indexer.py`, `requirements.txt`.

**Độc lập** với Phase 1–3 — làm lúc nào cũng được, nhưng nên sau Phase 3 để có số liệu so sánh.
Tạo collection mới `b2b_customers_openai_hybrid`, không đụng collection dense hiện tại (rollback dễ).

---

## Bước 4.1: Cài `fastembed`

- [x] **4.1.1** — `pip install fastembed`; thêm dòng vào `requirements.txt` + bảng "Phụ thuộc mới" trong `docs/README.md`.

### Kiểm tra
```bash
"/home/thanh/projects/RAG(B2B)/.venv/bin/pip" show fastembed | grep -E "Name|Version"
```

---

## Bước 4.2: `src/rag_b2b/config.py` — thêm sparse embedding + vector_db hybrid

- [x] **4.2.1** — Khởi tạo sparse embedding:
  ```python
  from langchain_qdrant import FastEmbedSparse, RetrievalMode
  sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
  ```
- [x] **4.2.2** — `vector_db` hybrid (collection mới, named vectors):
  ```python
  vector_db = QdrantVectorStore(
      client=qdrant_client,
      collection_name="b2b_customers_openai_hybrid",
      embedding=embeddings,
      sparse_embedding=sparse_embeddings,
      retrieval_mode=RetrievalMode.HYBRID,
      vector_name="dense",
      sparse_vector_name="sparse",
  )
  ```
  Lần `add_documents` đầu tiên `QdrantVectorStore` tự tạo collection với đúng named dense + sparse vector.

### Kiểm tra
```bash
cd "/home/thanh/projects/RAG(B2B)/src"
../.venv/bin/python -c "
from config import qdrant_client
print(qdrant_client.get_collection('b2b_customers_openai_hybrid').config.params)"
```
→ có cả vector `dense` và named sparse vector `sparse`.

---

## Bước 4.3: Nạp lại collection hybrid

- [x] **4.3.1** — Chạy `src/rag_b2b/indexer.py` `process_vector_data(S3_FINAL_DATA)` (đã có deterministic UUIDv5
  từ Phase 1 bước 1.5.2) để nạp `final_data_v2` vào collection hybrid.

### Kiểm tra
```bash
cd "/home/thanh/projects/RAG(B2B)/src"
../.venv/bin/python -c "
from config import qdrant_client, vector_db
print(qdrant_client.count('b2b_customers_openai_hybrid', exact=True))            # ≈ số khách
print([d.metadata.get('customer_id') for d in vector_db.similarity_search('mã SKU 0020010000305', k=3)])
"
```
→ `count` ≈ số khách trong `final_data_v2`; query token hiếm trả 3 kết quả.

---

## Bước 4.4: `src/rag_b2b/tools/vector.py` dùng store hybrid

- [x] **4.4.1** — `vector.py` import `vector_db` từ `config` như cũ (đã trỏ hybrid) — không đổi logic
  `similarity_search(question, k=3)`.
- [x] **4.4.2** — A/B tay: so kết quả nhánh vector giữa dense-only (collection cũ) và hybrid trên
  vài câu chứa mã SKU / tên brand / thuật ngữ hiếm; ghi nhận xét vào file này.

### Kiểm tra
```bash
cd "/home/thanh/projects/RAG(B2B)"
.venv/bin/python -c "
from rag_b2b.pipeline import agent_pipeline
print(agent_pipeline.invoke({'question':'Khách nữ ở Hà Nội thường mua gì?'}))     # vẫn trả lời hợp lý
print(agent_pipeline.invoke({'question':'Có khách nào mua mã SKU 0020010000305 không?'}))
"
```
→ cả 2 câu trả lời tiếng Việt hợp lý; câu có mã SKU truy hồi tốt hơn dense thuần.

---

## Bước 4.5: Rerank nhánh `vector` (bổ sung 2026-09-10, "nhóm 2")

`vector.py` giờ over-fetch **k=12** rồi để `llm_router` (Haiku) chọn **5 hồ sơ liên quan nhất**
(`_rerank()` → prompt "trả về các số"; parse lỗi/thiếu → fallback thứ tự truy hồi gốc). Không tải
cross-encoder cục bộ vì đĩa `/` chỉ còn ~700 MB. `vector_contexts()` (ragas) dùng chung `_search()`
nên đo đúng cái pipeline dùng.

| ragas (`vector`) | trước (k=3) | sau (k=12→rerank 5) |
|---|---|---|
| context_precision | 1.00 | 0.91 (do đo @5 thay vì @3) |
| **faithfulness** | 0.57 | **0.785** ✅ câu trả lời bám hồ sơ hơn hẳn |
| answer_relevancy | 0.42 | 0.41 (không đổi) |

**Kết luận:** rerank kéo `faithfulness` lên rõ. `answer_relevancy` không nhích — trần của *chỉ số ragas*
trên câu hỏi chân dung mở, không phải lỗi truy hồi. Không tối ưu thêm chỉ số này.

---

## Ghi chú

- Giữ collection dense cũ `b2b_customers_openai` cho tới khi hybrid được xác nhận tốt hơn — rollback = đổi
  `collection_name` lại trong `config.py`.
- Nếu muốn đo định lượng: chạy lại `eval/evaluate_recommender.py` không đổi (đo nhánh predict), còn nhánh
  vector thì dùng `eval/ragas_eval.py` (Phase 3.2) trước/sau để so `context_precision` / `context_recall`.
