from datetime import date

import duckdb
import pytest

from spiritops.ingestion import ingest_files


def test_overlapping_order_exports_preserve_lines_and_restore_fractional_litres(tmp_path):
    header = (
        "invoice_id,ordered_on,item_no,category_code,sales_bottles,sales_liters,"
        "sales_dollars,store_no,bottle_volume_ml\n"
    )
    seed = "INV-00000100001,2025-01-06,42,100,10,7.5,100,010270,750\n"
    conflict = "INV-00000200001,2025-11-03,42,100,10,7.5,100,010270,750\n"
    conflicting_copy = conflict.replace(",10,7.5,100,", ",12,9,120,")
    rounded_purchase = "INV-882097,2025-11-03,42,100,5,0,36,010270,50\n"
    second_product = "INV-882097,2025-11-03,43,100,6,4,157,010270,750\n"
    invalid = "INV-882097,2025-11-03,42,100,1,-1,10,010270,750\n"
    returned = "RINV-058970,2025-11-03,42,100,-3,-2,-30,010270,750\n"
    first, overlap = tmp_path / "part1.csv", tmp_path / "part2.csv"
    first.write_text(
        header + seed + conflict + rounded_purchase + second_product + invalid + returned
    )
    overlap.write_text(header + conflicting_copy + rounded_purchase * 2 + second_product + returned)

    manifest = ingest_files(tmp_path, [first, overlap], date(2025, 1, 6), date(2025, 11, 9))
    assert manifest["outcomes"] == {
        "accepted": 5,
        "duplicate": 3,
        "conflicting invoice": 2,
        "inconsistent signs": 1,
    }
    assert manifest["reconstructed_volume_lines"] == 4
    assert manifest["return_lines"] == 1
    assert manifest["returned_litres"] == pytest.approx(2.25)
    with duckdb.connect() as con:
        actual = con.execute(
            "SELECT volume, returns, purchases FROM read_parquet(?) "
            "WHERE week=DATE '2025-11-03' AND product='42'",
            [str(tmp_path / manifest["path"])],
        ).fetchone()
    assert actual == pytest.approx((0.5, 2.25, 2))
