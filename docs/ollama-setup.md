# Chạy LLM local qua Ollama

Tuỳ chọn thay `llm_router` + `llm_main` (Anthropic Haiku + OpenAI gpt-4o-mini) bằng model local.
**`cypher_llm` luôn giữ `gpt-4o`** (Cypher sai = số sai âm thầm) và **embeddings vẫn OpenAI**
(đổi = phải re-index Qdrant).

Gate ở `src/rag_b2b/config.py`: khi `LLM_BACKEND=ollama`, `llm_router` + `llm_main` thành
`langchain_ollama.ChatOllama` (<https://docs.langchain.com/oss/python/integrations/chat/ollama>),
điều khiển qua `OLLAMA_BASE_URL` / `OLLAMA_MODEL` / `OLLAMA_NUM_CTX` trong `.env`.

Giả định: **Ollama chạy trên máy Windows có GPU (vd RTX 3060 6GB)**, project trên **Ubuntu** gọi qua
mạng. Ubuntu KHÔNG cần cài Ollama server — chỉ cần package `langchain-ollama` (đã có trong `.venv`).

---

## A. Trên máy Windows (GPU)

### 1. Cài Ollama
Tải `OllamaSetup.exe` từ <https://ollama.com/download> → cài. Ollama chạy nền (icon khay hệ thống),
lắng nghe `127.0.0.1:11434`.

### 2. Tải model
```powershell
ollama pull qwen2.5:7b-instruct-q4_K_M      # ~4.7 GB — khớp OLLAMA_MODEL mặc định trong config
# nếu 7B chậm / thiếu VRAM:
# ollama pull qwen2.5:3b-instruct-q4_K_M    # ~2 GB
```

### 3. Test local + kiểm tra chạy GPU
```powershell
ollama run qwen2.5:7b-instruct-q4_K_M "Chào, trả lời 1 câu tiếng Việt."
ollama ps      # cột PROCESSOR phải là "100% GPU" (không phải "CPU")
```
Nếu ra "CPU" hoặc chia "xx%/xx%" → 6GB không đủ cho context lớn; xuống model 3B hoặc giảm `num_ctx`
(xem cuối phần A).

### 4. Cho phép Ubuntu kết nối + tối ưu VRAM 6GB — set biến môi trường rồi khởi động lại Ollama
PowerShell mở bằng **Run as Administrator**:
```powershell
[System.Environment]::SetEnvironmentVariable('OLLAMA_HOST','0.0.0.0','Machine')
[System.Environment]::SetEnvironmentVariable('OLLAMA_KEEP_ALIVE','30m','Machine')
[System.Environment]::SetEnvironmentVariable('OLLAMA_KV_CACHE_TYPE','q8_0','Machine')
```
`OLLAMA_KV_CACHE_TYPE=q8_0` **bắt buộc thực tế** với 6GB VRAM + `qwen2.5:7b`: không set thì model
tràn sang CPU (18%/82% CPU/GPU đã đo được), 1 lời gọi mất ~12,7s; set rồi còn **0,3s** (xem bảng
Trục trặc). Rồi: chuột phải icon Ollama ở khay → **Quit**, mở lại Ollama từ Start Menu.

### 5. Mở firewall cổng 11434
PowerShell admin:
```powershell
New-NetFirewallRule -DisplayName "Ollama LAN" -Direction Inbound -LocalPort 11434 -Protocol TCP -Action Allow -Profile Private
```

### 6. Lấy IP LAN của Windows
```powershell
ipconfig    # ghi lại "IPv4 Address" của card đang dùng, vd 192.168.1.50
```

### 7. Xác nhận Ollama nghe đúng
```powershell
curl http://localhost:11434/api/tags        # phải thấy qwen2.5 trong danh sách
```

### (Tuỳ chọn) Nếu bước 4 (`q8_0`) vẫn chưa đủ → hạ thêm context
`OLLAMA_NUM_CTX=4096` đã đặt sẵn ở `.env` bên Ubuntu (bước B2) — `config.py` truyền thẳng vào
`ChatOllama(num_ctx=...)`, không cần tạo model mới. Thực tế `q8_0` một mình đã đủ (0,3s), bước
này chỉ cần nếu dùng model to hơn hoặc context dài hơn.

---

## B. Trên Ubuntu (project)

### 1. Kiểm tra thông mạng tới Windows
```bash
curl http://192.168.1.50:11434/api/tags       # thay bằng IP bước A6
```
Không ra gì → firewall Windows chưa mở, hoặc `OLLAMA_HOST` chưa `0.0.0.0` (quên restart Ollama),
hoặc 2 máy khác subnet.

### 2. Sửa `.env` — thêm vào cuối, không đụng các khoá cũ
```dotenv
LLM_BACKEND=ollama
OLLAMA_BASE_URL=http://192.168.1.50:11434
OLLAMA_MODEL=qwen2.5:7b-instruct-q4_K_M
# OLLAMA_NUM_CTX=4096      # hạ context nếu VRAM 6GB chật (mặc định: theo model)
```

### 3. Xác nhận `langchain-ollama` đã cài
```bash
cd "/home/thanh/projects/RAG(B2B)"
.venv/bin/python -c "from langchain_ollama import ChatOllama; print('ok')"
# nếu thiếu: .venv/bin/pip install langchain-ollama
```

---

## C. Verify + đo chất lượng

### 1. Smoke test 1 lời gọi
Lần đầu chậm 5–15s (nạp model vào VRAM), sau đó nhanh.
```bash
cd "/home/thanh/projects/RAG(B2B)"
.venv/bin/python -c "
from rag_b2b.config import llm_router
print(llm_router.invoke('Phân loại: \"Khách 10001 sẽ mua gì?\" -> chỉ trả 1 từ').content)
"
```

### 2. Chạy end-to-end 1 câu
```bash
.venv/bin/python -c "
from rag_b2b.pipeline import agent_pipeline
print(agent_pipeline.invoke({'question':'Khách nữ ở Hà Nội thường mua gì?'}))
"
```

### 3. Đo so với cloud (trước khi tin dùng thật)
```bash
.venv/bin/python eval/test_routing_100.py      # so với 99/100 của Haiku
.venv/bin/python eval/ragas_eval.py            # so với faithfulness 0.84 / relevancy 0.45
.venv/bin/python eval/smoke_graph.py           # phải vẫn 15/15 (Cypher vẫn gpt-4o)
```

### Quay lại cloud
Đổi `.env` → `LLM_BACKEND=cloud` (hoặc xoá dòng đó).

### Kết quả đã đo thật (2026-09-11, RTX 3060 6GB, `qwen2.5:7b-instruct-q4_K_M`, `q8_0` KV cache)

**Routing** (`eval/test_routing_100.py`): **99/100**, 100 câu chỉ **12s** (nhanh hơn cloud vì LAN thay Internet).
e2e mẫu 14/14 OK — routing local dùng tốt.

**Chất lượng câu trả lời** (`eval/ragas_eval.py`, **judge luôn cố định `gpt-4o-mini`** — không theo
`LLM_BACKEND`, để so sánh đúng nghĩa; xem comment đầu file):

| Metric | Cloud (Haiku + gpt-4o-mini) | Ollama (qwen2.5:7b) | Δ |
|---|---|---|---|
| faithfulness | 0,841 (v 0,785 / w 0,933) | 0,755 (v 0,682 / w 0,877) | −0,09 |
| answer_relevancy | 0,446 (v 0,411 / w 0,504) | 0,357 (v 0,309 / w 0,453) | −0,09 |
| context_precision | 0,943 (v 0,909 / w 1,000) | 0,900 (v 0,840 / w 1,000) | −0,04 |

⚠️ **Bẫy đã gặp**: lần đầu quên cố định judge → `context_precision` tụt giả xuống 0,454 (model local
vừa sinh vừa tự chấm điểm, nhiễu nặng). Luôn kiểm tra `ragas_eval.py` dùng `ChatOpenAI` riêng cho judge,
không phải `config.llm_main`.

**Kết luận**: `llm_router` (routing + rerank) dùng ollama tốt — nhanh, chính xác tương đương cloud.
`llm_main` nhánh `web` kém cloud ít (~5-6 điểm, task đơn giản: diễn đạt lại tóm tắt Tavily); nhánh `vector`
kém rõ hơn (~9-10 điểm, task tổng hợp nhiều hồ sơ khó hơn với 7B) — từng thấy cụ thể: qwen trả lời
*"Thủ đô Nhật Bản không phải là Tokyo"* cho câu hỏi web đơn giản. Cân nhắc giữ `llm_main` = cloud cho
`vector`, chỉ chuyển `llm_router` + nhánh `web`/`predict`/`graph` sang ollama nếu cần tối ưu chi phí.

---

## Trục trặc thường gặp

| Triệu chứng | Xử lý |
|---|---|
| Ubuntu `curl` timeout | `OLLAMA_HOST=0.0.0.0` chưa áp (quên restart Ollama) · firewall · khác subnet/VPN |
| `ollama ps` báo CPU, trả lời rất chậm | Set `OLLAMA_KV_CACHE_TYPE=q8_0` bên Windows (biến môi trường Machine, quit+mở lại Ollama — như bước A4). *Đã gặp thật*: `qwen2.5:7b` 5,1GB @ ctx=4096 → `18%/82% CPU/GPU`, 1 lời gọi router mất **12,7s**. Sau khi set `q8_0`: **0,3s** (~42×, nhanh hơn cả cloud ~1-2s). Nếu vẫn chậm sau `q8_0`: xuống model 3B hoặc giảm `OLLAMA_NUM_CTX`. |
| Lần gọi đầu mỗi vài phút lại chậm | model bị unload → `OLLAMA_KEEP_ALIVE=30m` (A4), hoặc `-1` giữ mãi |
| Routing / ragas tụt nhiều | bình thường với 7B; cân nhắc chỉ `llm_router` = ollama, `llm_main` = cloud (sửa `config.py`) |
| Ubuntu là VM cloud, GPU ở nhà | **KHÔNG** hở cổng 11434 ra Internet — dùng Tailscale (cài 2 máy) → `OLLAMA_BASE_URL=http://100.x.x.x:11434` |

---

## Các topology khác

- **Ubuntu là VM VMware trên chính máy Windows đó** (đã xác nhận đúng cho setup này —
  `ip addr` cho adapter `ens33`, `ip route` default qua `192.168.23.2`): dùng
  **IP host của adapter VMnet8** (`192.168.23.1` trong VM này — lấy bằng
  `ip route | grep default | awk '{print $3}'` rồi trừ 1 ở octet cuối, hoặc thử lần lượt
  `192.168.<x>.1` với `x` = octet 3 của IP VM). IP này **ổn định** dù laptop đổi Wi-Fi
  (khác với IP Wi-Fi `192.168.0.x` của Windows — đổi theo mạng). Đã verify: cả IP Wi-Fi lẫn
  IP VMnet8/VMnet1 của host đều `curl` thông từ trong VM (VMware bridge sẵn) — chọn VMnet8/VMnet1
  cho ổn định lâu dài.
- **Ubuntu là WSL2 trên chính máy Windows đó**: cài Ollama native trên Windows. Win11 *mirrored networking*
  → `OLLAMA_BASE_URL=http://localhost:11434` chạy luôn. Không thì lấy IP host:
  `ip route show default | awk '{print $3}'` → `http://<ip đó>:11434` (vẫn cần A4 + A5).
- **2 máy khác mạng / qua Internet**: Tailscale (miễn phí) — cài trên cả 2, dùng IP `100.x.x.x`.
  Không mở 11434 ra public.
