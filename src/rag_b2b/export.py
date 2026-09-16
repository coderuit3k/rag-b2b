"""Xuất lịch sử hỏi-đáp ra .csv/.xlsx/.pdf/.docx — dùng cho nút tải xuống trong app.py."""
import io
import logging
import os

import pandas as pd

_log = logging.getLogger(__name__)

# Font Unicode cho PDF (core font mặc định của fpdf2 không có dấu tiếng Việt). Đường dẫn chuẩn
# Ubuntu/Debian (gói fonts-dejavu-core) — dự án chỉ target Ubuntu, xem docs/README.md.
_VN_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def _dataframe(qa_pairs: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(qa_pairs, columns=["Câu hỏi", "Trả lời"])


def to_csv_bytes(qa_pairs: list[tuple[str, str]]) -> bytes:
    # utf-8-sig: kèm BOM để Excel mở đúng dấu tiếng Việt (Excel mặc định đoán ANSI nếu không có BOM).
    return _dataframe(qa_pairs).to_csv(index=False).encode("utf-8-sig")


def to_xlsx_bytes(qa_pairs: list[tuple[str, str]]) -> bytes:
    buf = io.BytesIO()
    _dataframe(qa_pairs).to_excel(buf, index=False, engine="openpyxl")
    return buf.getvalue()


def to_pdf_bytes(qa_pairs: list[tuple[str, str]]) -> bytes:
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos

    pdf = FPDF()
    pdf.add_page()
    if os.path.exists(_VN_FONT):
        pdf.add_font("DejaVu", "", _VN_FONT)
        pdf.set_font("DejaVu", size=12)
    else:
        _log.warning("thiếu font %s -> PDF dùng font mặc định, có thể mất dấu tiếng Việt", _VN_FONT)
        pdf.set_font("Helvetica", size=12)
    for q, a in qa_pairs:
        # new_x/new_y=LMARGIN/NEXT: về đầu dòng mới sau mỗi đoạn (mặc định fpdf2 để con trỏ cuối
        # dòng vừa in -> đoạn sau tràn lề phải, lỗi "Not enough horizontal space").
        pdf.multi_cell(0, 8, f"Hỏi: {q}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.multi_cell(0, 8, f"Đáp: {a}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(4)
    return bytes(pdf.output())


def to_docx_bytes(qa_pairs: list[tuple[str, str]]) -> bytes:
    from docx import Document

    doc = Document()
    for q, a in qa_pairs:
        doc.add_paragraph(f"Hỏi: {q}", style="Heading 3")
        doc.add_paragraph(f"Đáp: {a}")
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


if __name__ == "__main__":
    demo = [("Khách hàng nữ ở Hà Nội thường mua gì?", "Sữa bột và tã, theo hồ sơ mua sắm.")]
    assert to_csv_bytes(demo).startswith(b"\xef\xbb\xbf")
    assert to_xlsx_bytes(demo)[:2] == b"PK"  # xlsx = zip
    assert to_pdf_bytes(demo)[:4] == b"%PDF"
    assert to_docx_bytes(demo)[:2] == b"PK"  # docx = zip
    print("OK: cả 4 định dạng xuất đúng magic bytes")
