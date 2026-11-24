"""One-step, strictly as-of features shared by fitting and serving."""

from __future__ import annotations

import json
import math
import zlib
from dataclasses import dataclass
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

import duckdb
import numpy as np

from spiritops.ingestion.sources import checksum

SCHEMA_VERSION = "iowa-weekly-v1"
FEATURE_NAMES = (
    "product_code", "category", "lag1", "lag4", "seasonal_naive", "mean4", "std4",
    "mean13", "std13", "trend4", "purchase_frequency", "mean_orders4", "historical_price",
    "price_observed", "week_sin", "week_cos", "month", "weeks_observed",
)
WEEK = timedelta(weeks=1)


def monday(value: str) -> date:
    day = date.fromisoformat(value)
    if day.isoformat()!=value or day.weekday()!=0:
        raise ValueError("week must be an ISO Monday date")
    return day


@dataclass(frozen=True)
class Series:
    volume: np.ndarray
    category: np.ndarray
    price: np.ndarray
    priced: np.ndarray
    purchases: np.ndarray


@dataclass(frozen=True)
class Panel:
    first: date
    last: date
    selection_end: date
    series: dict[str, Series]
    sha256: str

    def features(self, product: str, day: date) -> dict[str, float]:
        if product not in self.series:
            raise ValueError("unknown product")
        if day.weekday()!=0 or not self.selection_end < day <= self.last + WEEK:
            raise ValueError("week outside supported one-step history")
        s = self.series[product]
        i = (day-self.first).days // 7
        history = s.volume[:i]
        recent, season = history[-4:], history[-13:]
        angle = 2 * math.pi * (day.isocalendar().week - 1) / 52.1775
        try:
            code = float(int(product))
        except ValueError:
            code = float(zlib.crc32(product.encode()))
        values = (
            code, s.category[i-1], history[-1], history[-4],
            history[-52] if i>=52 else history[-1],
            recent.mean(), recent.std(), season.mean(), season.std(),
            recent.mean()-history[-8:-4].mean(), np.mean(season>0),
            s.purchases[i-4:i].mean(), s.price[i-1], s.priced[i-1],
            math.sin(angle), math.cos(angle), day.month, i,
        )
        return dict(zip(FEATURE_NAMES, map(float,values), strict=True))

    def examples(self, start: date, end: date) -> tuple[list[dict[str,float]], np.ndarray, list[str]]:
        if start.weekday()!=0 or end.weekday()!=0 or not self.selection_end < start <= end <= self.last:
            raise ValueError("invalid chronological example window")
        rows, targets, weeks = [], [], []
        day = start
        while day<=end:
            index = (day-self.first).days//7
            for product, series in self.series.items():
                rows.append(self.features(product,day))
                targets.append(series.volume[index])
                weeks.append(day.isoformat())
            day += WEEK
        return rows, np.asarray(targets), weeks


@lru_cache(maxsize=4)
def _read(path: Path, digest: str, first: date, last: date, selection_end: date,
          products: tuple[str,...], modified: int) -> Panel:
    if checksum(path)!=digest:
        raise ValueError("weekly Parquet checksum mismatch")
    length = (last-first).days//7 + 1
    if len(products)>100 or length>5200 or length<9 or selection_end<first+7*WEEK:
        raise ValueError("unsupported data manifest")
    with duckdb.connect(config={"threads":"1"}) as con:
        table = con.read_parquet(str(path))
        if table.count("*").fetchone() != (len(products)*length,):
            raise ValueError("incomplete product-week panel")
        orders = "purchases" if "purchases" in table.columns else "CAST(volume>0 AS INTEGER)"
        rows = table.project(f"product, week, volume, category, price, {orders}").order("product, week").fetchall()
    series = {}
    for group, product in enumerate(products):
        volume, categories, prices, priced, purchases = [], [], [], [], []
        cat, price, observed = 0.0, 0.0, 0.0
        for index, (p,w,vol,cost_cat,cost,n) in enumerate(rows[group*length:(group+1)*length]):
            if str(p)!=product or str(w)[:10]!=(first+index*WEEK).isoformat():
                raise ValueError("unordered or incomplete weekly data")
            if cost_cat is not None:
                try:
                    cat = float(cost_cat)
                except ValueError:
                    cat = float(zlib.crc32(str(cost_cat).encode()))
            if cost is not None:
                price,observed = float(cost),1.0
            if any(not math.isfinite(v) or v<0 for v in [float(vol),price,float(n),cat]):
                raise ValueError("invalid curated measurement")
            volume.append(float(vol))
            categories.append(cat)
            prices.append(price)
            priced.append(observed)
            purchases.append(float(n))
        series[product] = Series(*(np.asarray(x) for x in (volume,categories,prices,priced,purchases)))
    return Panel(first,last,selection_end,series,digest)


def panel(home: Path) -> Panel:
    info: dict[str, Any] = json.loads((home / "data/current.json").read_text())
    path = (home / info["path"]).resolve()
    if not path.is_relative_to((home / "data").resolve()):
        raise ValueError("curated dataset must reside under data/")
    return _read(path, info["sha256"], monday(info["first_week"]), monday(info["last_week"]),
                 monday(info["product_selection_end"]), tuple(sorted(info["products"])),
                 path.stat().st_mtime_ns)


def features_for(home: Path, product: str, week: str) -> dict[str,float]:
    return panel(home).features(product, monday(week))
