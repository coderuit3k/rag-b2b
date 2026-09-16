"""Chạy 1 file .sql trên Athena, tuần tự từng statement, poll tới khi xong.

    python notebooks/run_athena.py notebooks/tier12_sample.sql

workgroup=primary, output=s3://rag-b2b-data-2024/_athena/, db=transaction_2024, region ap-southeast-2.
Tách statement bằng ';' + xuống dòng (không vỡ chuỗi literal '; ' trong ARRAY_JOIN).
"""
import re
import sys
import time

import boto3

DB = "transaction_2024"
OUTPUT = "s3://rag-b2b-data-2024/_athena/"
WG = "primary"
REGION = "ap-southeast-2"


def _statements(sql: str):
    for raw in re.split(r";\s*\n", sql):
        s = "\n".join(l for l in raw.splitlines() if not l.strip().startswith("--"))
        if s.strip():
            yield s.strip()


def main(path: str):
    ath = boto3.client("athena", region_name=REGION)
    sql = open(path).read()
    stmts = list(_statements(sql))
    print(f"{path}: {len(stmts)} statement\n")
    for i, q in enumerate(stmts, 1):
        head = " ".join(q.split())[:90]
        print(f"[{i}/{len(stmts)}] {head}...")
        qid = ath.start_query_execution(
            QueryString=q,
            QueryExecutionContext={"Database": DB},
            ResultConfiguration={"OutputLocation": OUTPUT},
            WorkGroup=WG,
        )["QueryExecutionId"]
        while True:
            ex = ath.get_query_execution(QueryExecutionId=qid)["QueryExecution"]
            st = ex["Status"]["State"]
            if st in ("SUCCEEDED", "FAILED", "CANCELLED"):
                break
            time.sleep(3)
        if st != "SUCCEEDED":
            sys.exit(f"  ✗ {st}: {ex['Status'].get('StateChangeReason', '')}")
        scanned = ex["Statistics"].get("DataScannedInBytes", 0) / 1e6
        print(f"  ✓ {st}  ({scanned:.1f} MB scanned)")
        # in kết quả nếu là SELECT nhỏ
        if q.lstrip().upper().startswith("SELECT"):
            rs = ath.get_query_results(QueryExecutionId=qid, MaxResults=20)["ResultSet"]["Rows"]
            for r in rs:
                print("   ", [c.get("VarCharValue", "") for c in r["Data"]])
    print("\n✅ xong")


if __name__ == "__main__":
    main(sys.argv[1])
