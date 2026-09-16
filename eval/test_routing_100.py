"""100 câu test cho 5 nhánh router (predict/graph/vector/web/email).

- Kiểm phân luồng: chạy router_chain cho cả 100 câu, so nhãn dự kiến.
- Kiểm kết quả: chạy agent_pipeline end-to-end cho N câu/nhánh (mặc định 3; email 2 để hạn chế gửi thật),
  sanity theo nhánh (không rỗng, không traceback, có dấu hiệu đúng nhánh).
- Xuất eval/routing_test_<ts>.json.

Chạy:  python eval/test_routing_100.py [--full-per-branch 3]
"""
import argparse
import json
import os
import re
import time
from datetime import datetime


import polars as pl  # noqa: E402
from rag_b2b.pipeline import router_chain, agent_pipeline  # noqa: E402

S3_CUST = "s3://rag-b2b-data-2024/transaction_2024/_sample/sample_cust/"


def load_cids(n: int):
    df = pl.read_parquet(S3_CUST).select(pl.col("customer_id").cast(pl.Utf8))
    return df.get_column("customer_id").unique().sort().to_list()[:n]


def build_cases(cids):
    c = (cids + cids * 20)[:80]  # đủ id, lặp lại nếu sample nhỏ
    predict = [
        f"Khách hàng {c[0]} sẽ mua gì tiếp theo?",
        f"Gợi ý sản phẩm cho khách {c[1]}",
        f"Dự đoán đơn hàng kế tiếp của khách {c[2]}",
        f"Khách {c[3]} có khả năng mua sản phẩm nào?",
        f"Nên gợi ý gì cho khách hàng {c[4]}?",
        f"Sản phẩm tiếp theo khách {c[5]} sẽ quan tâm là gì?",
        f"Recommend sản phẩm cho customer {c[6]}",
        f"Khách {c[7]} sắp tới mua gì?",
        f"Đề xuất 10 sản phẩm cho khách {c[8]}",
        f"Khách hàng {c[9]} thường mua lại món gì?",
        f"Dự báo nhu cầu mua sắm của khách {c[10]}",
        f"Món hàng nào phù hợp để bán thêm cho khách {c[11]}?",
        f"Khách {c[12]} sẽ cần mua gì trong tháng tới?",
        f"Gợi ý giỏ hàng tiếp theo cho {c[13]}",
        f"Cross-sell gì cho khách hàng {c[14]}?",
        f"Khách {c[15]} có xu hướng mua sản phẩm nào kế tiếp?",
        f"Dự đoán sản phẩm cho mã khách {c[16]}",
        f"Khách {c[17]} nên được giới thiệu sản phẩm gì?",
        f"Next best offer cho khách {c[18]} là gì?",
        f"Khách hàng {c[19]} sẽ mua thêm gì?",
    ]
    graph = [
        f"Tổng tiền khách hàng {c[0]} đã chi là bao nhiêu?",
        f"Khách {c[1]} đã mua tất cả bao nhiêu sản phẩm?",
        f"Đếm số giao dịch của khách hàng {c[2]}",
        f"Khách {c[3]} đã mua những danh mục nào?",
        f"Có bao nhiêu khách hàng ở tỉnh Hồ Chí Minh?",
        f"Khách {c[4]} mua bao nhiêu sản phẩm khác nhau?",
        f"Tổng số lượng hàng khách {c[5]} đã mua?",
        f"Khách hàng {c[6]} chi nhiều nhất cho danh mục nào?",
        f"Số tiền trung bình mỗi đơn của khách {c[7]}?",
        f"Khách {c[8]} mua lần gần nhất vào ngày nào?",
        f"Có bao nhiêu khách giới tính Nữ trong hệ thống?",
        f"Khách {c[9]} đã giao dịch bao nhiêu lần từ 2025-01-01?",
        f"Liệt kê 5 sản phẩm khách {c[10]} mua nhiều nhất",
        f"Tổng chi tiêu của khách {c[11]} cho danh mục sữa?",
        f"Khách {c[12]} mua hàng ở bao nhiêu cửa hàng khác nhau?",
        f"Đếm số khách hàng ở tỉnh Bến Tre",
        f"Khách {c[13]} có bao nhiêu đơn được giảm giá?",
        f"Sản phẩm nào khách {c[14]} mua với số lượng lớn nhất?",
        f"Tổng doanh thu từ khách {c[15]} trong năm 2024?",
        f"Khách {c[16]} mua sản phẩm đầu tiên khi nào?",
    ]
    vector = [
        "Khách hàng nữ ở Hà Nội thường mua những gì?",
        "Nhóm khách hay mua bỉm sữa có đặc điểm chung gì?",
        "Mô tả hành vi mua sắm của khách ở Đà Nẵng",
        "Khách mua nhiều sữa công thức thường mua kèm sản phẩm nào?",
        "Chân dung khách hàng trung thành của cửa hàng mẹ và bé",
        "Khách ở khu vực miền Tây có thói quen mua sắm ra sao?",
        "Những khách hay mua khăn ướt thường quan tâm điều gì?",
        "Đặc điểm khách hàng chi tiêu cao là gì?",
        "Khách mới thường mua sản phẩm nào đầu tiên?",
        "Phân khúc khách hàng mua đồ chơi trẻ em trông như thế nào?",
        "Khách nam mua hàng cho con thường chọn gì?",
        "Thói quen mua tã bỉm của khách ở thành phố lớn",
        "Khách hàng tại Bình Dương thường mua nhóm hàng nào?",
        "Mô tả khách hay mua sữa chua và phô mai",
        "Khách mua hàng theo mùa có hành vi gì đặc biệt?",
        "Nhóm khách nhạy cảm về giá thường mua gì?",
        "Khách hàng mua nhiều quần áo trẻ em quan tâm yếu tố nào?",
        "Chân dung khách mua hàng online so với tại cửa hàng",
        "Khách ở tỉnh lẻ thường mua sản phẩm thương hiệu nào?",
        "Hành vi mua lặp lại của khách mua bỉm Merries",
    ]
    web = [
        "Thủ đô của nước Pháp là gì?",
        "Giá vàng SJC hôm nay khoảng bao nhiêu?",
        "Python decorator là gì?",
        "Cách nấu phở bò truyền thống?",
        "Tỷ giá USD sang VND hôm nay?",
        "Ai là tổng thống Mỹ hiện tại?",
        "Công thức tính diện tích hình tròn?",
        "Thời tiết Hà Nội ngày mai thế nào?",
        "GDP Việt Nam năm 2024 là bao nhiêu?",
        "Sự khác nhau giữa HTTP và HTTPS?",
        "Chiều cao của núi Everest?",
        "Khi nào World Cup tiếp theo diễn ra?",
        "Lãi suất ngân hàng Vietcombank hiện nay?",
        "React hook useEffect dùng để làm gì?",
        "Dân số thế giới hiện nay khoảng bao nhiêu?",
        "Cách đổi mật khẩu Gmail?",
        "Bitcoin đang có giá bao nhiêu USD?",
        "Nguyên nhân gây ra biến đổi khí hậu?",
        "Chuyến bay từ Hà Nội đi Tokyo mất bao lâu?",
        "Định nghĩa lạm phát trong kinh tế học?",
    ]
    email = [
        f"Gửi email cho tôi dự đoán sản phẩm của khách hàng {c[0]}",
        f"Mail kết quả tổng tiền khách {c[1]} đã chi cho tôi",
        f"Email cho tôi danh sách gợi ý sản phẩm cho khách {c[2]}",
        f"Gửi qua gmail giúp tôi dự đoán đơn kế tiếp của khách {c[3]}",
        f"Cho tôi nhận qua email số giao dịch của khách {c[4]}",
        f"Gửi mail báo cáo chi tiêu khách {c[5]}",
        f"Email tôi các sản phẩm nên bán thêm cho khách {c[6]}",
        f"Gửi kết quả dự đoán khách {c[7]} vào hòm thư của tôi",
        f"Mail cho tôi những danh mục khách {c[8]} đã mua",
        f"Gửi email tổng hợp gợi ý sản phẩm khách {c[9]}",
        f"Cho tôi email dự đoán mua sắm của khách {c[10]}",
        f"Gửi qua email giúp tôi thống kê đơn hàng khách {c[11]}",
        f"Email báo cáo khách hàng {c[12]} cho tôi",
        f"Gửi mail danh sách sản phẩm tiếp theo cho khách {c[13]}",
        f"Nhờ gửi email kết quả phân tích khách {c[14]}",
        f"Gửi cho tôi qua gmail dự đoán của khách {c[15]}",
        f"Mail tôi tổng chi tiêu năm 2024 của khách {c[16]}",
        f"Email giúp tôi gợi ý cross-sell cho khách {c[17]}",
        f"Gửi kết quả recommend khách {c[18]} tới email của tôi",
        f"Cho tôi nhận email dự đoán sản phẩm khách {c[19]}",
    ]
    out = []
    for label, qs in [("predict", predict), ("graph", graph), ("vector", vector),
                      ("web", web), ("email", email)]:
        out += [(label, q) for q in qs]
    return out


_MARKERS = {
    "predict": lambda a: bool(re.search(r"\d{6,}|điểm|sản phẩm", a)),
    "graph":   lambda a: len(a) > 0 and "Traceback" not in a,
    "vector":  lambda a: len(a) > 20 and "Traceback" not in a,
    "web":     lambda a: len(a) > 20 and "Traceback" not in a,
    "email":   lambda a: ("Đã gửi" in a) or ("❌" in a),  # gửi ok hoặc lỗi rõ ràng (không crash)
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-per-branch", type=int, default=3)
    ap.add_argument("--email-full", type=int, default=2)
    args = ap.parse_args()

    cids = load_cids(40)
    cases = build_cases(cids)
    print(f"{len(cases)} câu, {len(cids)} customer_id thật\n")

    routed, conf = [], {}
    t0 = time.time()
    for i, (exp, q) in enumerate(cases, 1):
        got = router_chain.invoke({"question": q}).strip().lower()
        got = next((k for k in _MARKERS if k in got), got)
        ok = got == exp
        routed.append({"exp": exp, "got": got, "ok": ok, "q": q})
        conf[(exp, got)] = conf.get((exp, got), 0) + 1
        if i % 20 == 0:
            print(f"  routed {i}/{len(cases)}  ({time.time()-t0:.0f}s)", flush=True)

    by = {}
    for r in routed:
        d = by.setdefault(r["exp"], [0, 0])
        d[1] += 1
        d[0] += r["ok"]
    print("\n=== PHÂN LUỒNG ===")
    for b, (ok, tot) in by.items():
        print(f"  {b:8} {ok}/{tot}")
    acc = sum(r["ok"] for r in routed) / len(routed)
    print(f"  TỔNG    {sum(r['ok'] for r in routed)}/{len(routed)}  ({acc:.1%})")
    miss = [r for r in routed if not r["ok"]]
    if miss:
        print("  sai:")
        for r in miss:
            print(f"    [{r['exp']}→{r['got']}] {r['q']}")

    print("\n=== KẾT QUẢ end-to-end (mẫu) ===")
    exec_res = []
    seen = {}
    for exp, q in cases:
        lim = args.email_full if exp == "email" else args.full_per_branch
        if seen.get(exp, 0) >= lim:
            continue
        seen[exp] = seen.get(exp, 0) + 1
        try:
            ans = str(agent_pipeline.invoke({"question": q}))
            good = _MARKERS[exp](ans)
        except Exception as e:  # noqa: BLE001
            ans, good = f"EXC: {e}", False
        exec_res.append({"branch": exp, "q": q, "ok": good, "answer": ans[:300]})
        print(f"  [{'OK ' if good else 'FAIL'}] {exp:8} {q[:55]}")
        print(f"        {ans[:140].replace(chr(10),' ')}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    _out = os.path.join(os.path.dirname(__file__), "results"); os.makedirs(_out, exist_ok=True)
    path = os.path.join(_out, f"routing_test_{ts}.json")
    json.dump({
        "generated_at": datetime.now().isoformat(),
        "routing_accuracy": acc,
        "routing_by_branch": {b: f"{ok}/{tot}" for b, (ok, tot) in by.items()},
        "confusion": {f"{k[0]}->{k[1]}": v for k, v in conf.items()},
        "routed": routed,
        "exec_sample": exec_res,
    }, open(path, "w"), ensure_ascii=False, indent=2)
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
