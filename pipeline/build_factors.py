"""C 部分：把 TEJ 原始資料轉成「每月因子分數表」。

輸入  股價資料.csv、融資資料.csv（TEJ 日資料）
輸出  factor_scores.csv，每列 = 一檔股票在某個月底訊號日的因子值與分數

兩個因子
    margin_chg  融資餘額變化率（籌碼類）
                = 本月底融資餘額 / 上月底融資餘額 - 1
    drawdown    股價回撤幅度（價格風險類）
                = 目前報酬指數 / 近 N 個交易日最高報酬指數 - 1  （<= 0，越負代表跌越深）

防止偷看未來（point-in-time）
    * 訊號日 t = 每個月最後一個交易日。t 日的因子只用到 t 日（含）以前的資料。
    * 因子分數在 hold_month（t 的下一個月）才拿來持股，回測時請在 t 之後才交易。
    * 橫斷面標準化只在同一個訊號日內做，不跨月，所以不會用到其他月份的資訊。

用法
    python -m src.build_factors --price 股價資料.csv --margin 融資資料.csv \
        --out output/factor_scores.csv
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Config:
    """所有可以調整的參數都放這裡，改參數不用動邏輯。"""

    margin_lookback: int = 1          # 融資變化率：和幾個月前的月底比
    margin_min_base: float = 100.0    # 前期融資餘額(張)低於此值 -> 基期太小，變化率設為 NaN
    ffill_limit: int = 5              # 月底當天融資缺值時，最多沿用前幾個交易日的值
    dd_window: int = 60               # 回撤：回看幾個交易日的高點（資料 2014/10 起，故用 60 日）
    liq_window: int = 20              # 流動性：近幾個交易日的平均成交值
    min_avg_value: float = 10_000.0   # 20 日平均成交值門檻（千元）= 1,000 萬元
    winsor: float = 0.01              # 標準化前，每月兩端各縮尾的比例
    markets: tuple[str, ...] = ("TSE", "OTC")  # 只留上市、上櫃
    w_margin_chg: float = 0.5         # baseline 綜合分數的固定權重（兩者相加為 1）
    w_drawdown: float = 0.5


PRICE_COLS = {
    "證券代碼": "stock",
    "年月日": "date",
    "收盤價(元)": "close",
    "成交量(千股)": "volume",
    "成交值(千元)": "value",
    "報酬率％": "ret_pct",
    "市場別": "market",
}
MARGIN_COLS = {
    "證券代碼": "stock",
    "年月日": "date",
    "融資餘額(張)": "margin_bal",
}


# --------------------------------------------------------------------------
# 讀檔
# --------------------------------------------------------------------------
def _read(path: str | Path, cols: dict[str, str], numeric: list[str]) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        encoding="utf-8-sig",
        usecols=list(cols),
        dtype=str,
    ).rename(columns=cols)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    for c in numeric:
        # TEJ 用 "." 代表缺值，轉成 NaN
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_prices(path: str | Path) -> pd.DataFrame:
    return _read(path, PRICE_COLS, ["close", "volume", "value", "ret_pct"])


def load_margin(path: str | Path) -> pd.DataFrame:
    return _read(path, MARGIN_COLS, ["margin_bal"])


# --------------------------------------------------------------------------
# 因子計算
# --------------------------------------------------------------------------
def _wide(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """長表轉寬表：列 = 日期，欄 = 股票。"""
    return df.pivot(index="date", columns="stock", values=col).sort_index()


def month_end_dates(dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """每個月最後一個交易日（以資料裡實際出現的交易日為準）。"""
    s = pd.Series(dates, index=dates)
    return pd.DatetimeIndex(s.groupby(dates.to_period("M")).max().values)


def compute_drawdown(ret_pct: pd.DataFrame, close: pd.DataFrame, window: int) -> pd.DataFrame:
    """股價回撤。用「報酬率」連乘出報酬指數，而不是直接用收盤價。

    原因：收盤價沒有還原，除權息當天會假跌。報酬率有含股利，不會有這個問題。
    """
    listed = close.notna()
    tri = (1.0 + ret_pct.fillna(0.0) / 100.0).cumprod().where(listed)
    roll_max = tri.rolling(window, min_periods=window).max()
    return tri / roll_max - 1.0


def compute_margin_change(
    margin_bal: pd.DataFrame,
    signal_dates: pd.DatetimeIndex,
    lookback: int,
    min_base: float,
    ffill_limit: int,
) -> pd.DataFrame:
    """融資餘額變化率：月底餘額相對 lookback 個月前月底餘額的變化。"""
    bal = margin_bal.ffill(limit=ffill_limit)
    at_signal = bal.loc[signal_dates]
    base = at_signal.shift(lookback)
    chg = at_signal / base - 1.0
    # 基期為 0 或太小 -> 變化率沒有意義（例如 5 張變 50 張 = +900%）
    return chg.where(base >= min_base)


def build_raw_factors(prices: pd.DataFrame, margin: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """回傳長表：每列 = (訊號日, 股票)，含兩個原始因子與過濾用欄位。"""
    close = _wide(prices, "close")
    ret_pct = _wide(prices, "ret_pct")
    value = _wide(prices, "value")
    volume = _wide(prices, "volume")
    dates = close.index
    signal_dates = month_end_dates(dates)

    mbal = _wide(margin, "margin_bal").reindex(dates)

    drawdown = compute_drawdown(ret_pct, close, cfg.dd_window).loc[signal_dates]
    margin_chg = compute_margin_change(
        mbal, signal_dates, cfg.margin_lookback, cfg.margin_min_base, cfg.ffill_limit
    )
    avg_value = (
        value.rolling(cfg.liq_window, min_periods=cfg.liq_window).mean().loc[signal_dates]
    )
    traded = (volume.loc[signal_dates] > 0)

    def to_long(wide: pd.DataFrame, name: str) -> pd.DataFrame:
        w = wide.copy()
        w.index.name = "signal_date"
        w.columns.name = "stock"
        return w.reset_index().melt(id_vars="signal_date", var_name="stock", value_name=name)

    out = to_long(margin_chg, "margin_chg")
    for wide, name in [(drawdown, "drawdown"), (avg_value, "avg_value_20d"), (traded.astype(float), "traded")]:
        out = out.merge(to_long(wide, name), on=["signal_date", "stock"], how="left")

    meta = prices.loc[prices["date"].isin(signal_dates), ["stock", "date", "market"]].rename(
        columns={"date": "signal_date"}
    )
    out = out.merge(meta, on=["signal_date", "stock"], how="inner")  # 該日有股價的才保留
    out["traded"] = out["traded"].fillna(0).astype(bool)
    return out


# --------------------------------------------------------------------------
# 標準化
# --------------------------------------------------------------------------
def _winsorize_z(x: pd.Series, p: float) -> tuple[pd.Series, pd.Series]:
    lo, hi = x.quantile(p), x.quantile(1 - p)
    xw = x.clip(lo, hi)
    std = xw.std(ddof=0)
    z = (xw - xw.mean()) / std if std > 0 else xw * 0.0
    return z, xw.rank(pct=True)


def standardize(raw: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """每個訊號日、在「可選股票」內做縮尾 + z-score + 百分位排名。

    可選股票 eligible = 上市櫃 & 訊號日有成交 & 20 日平均成交值達門檻 & 兩個因子都算得出來。
    只有 eligible 的列會有分數，其他列的分數是 NaN。
    """
    df = raw.sort_values(["signal_date", "stock"]).reset_index(drop=True)  # 固定順序，同分時排名才穩定
    df["in_universe"] = (
        df["market"].isin(cfg.markets)
        & df["traded"]
        & (df["avg_value_20d"] >= cfg.min_avg_value)
    )
    df["eligible"] = df["in_universe"] & df["margin_chg"].notna() & df["drawdown"].notna()

    for f in ("margin_chg", "drawdown"):
        df[f"z_{f}"] = np.nan
        df[f"rk_{f}"] = np.nan

    for _, idx in df[df["eligible"]].groupby("signal_date").groups.items():
        for f in ("margin_chg", "drawdown"):
            z, rk = _winsorize_z(df.loc[idx, f], cfg.winsor)
            df.loc[idx, f"z_{f}"] = z
            df.loc[idx, f"rk_{f}"] = rk

    # 逆向策略方向（研究假設）：融資減少越多、股價回撤越深 -> 分數越高。
    # rk_ 是「原始數值越大、排名越高」，而這兩個因子的「越小／越負」才是我們要的，所以用 1 - rk 反轉。
    df["score_margin_chg"] = 1.0 - df["rk_margin_chg"]
    df["score_drawdown"] = 1.0 - df["rk_drawdown"]
    df["composite_score"] = (
        cfg.w_margin_chg * df["score_margin_chg"] + cfg.w_drawdown * df["score_drawdown"]
    )
    # 每月由高到低排名，1 = 最想買；只有 eligible 的股票有值
    df["composite_rank"] = (
        df.groupby("signal_date")["composite_score"].rank(ascending=False, method="first")
    )
    return df


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def build_factor_table(price_path, margin_path, cfg: Config = Config()) -> pd.DataFrame:
    prices = load_prices(price_path)
    prices = prices[prices["market"].isin(cfg.markets)]  # 先縮小資料量，只留上市櫃
    margin = load_margin(margin_path)
    margin = margin[margin["stock"].isin(prices["stock"].unique())]

    scored = standardize(build_raw_factors(prices, margin, cfg), cfg)
    scored["stock_code"] = scored["stock"].str.split(" ").str[0]
    scored["stock_name"] = scored["stock"].str.split(" ", n=1).str[1]
    scored["hold_month"] = (scored["signal_date"].dt.to_period("M") + 1).astype(str)

    cols = [
        "signal_date", "hold_month", "stock_code", "stock_name", "market",
        "margin_chg", "drawdown", "avg_value_20d",
        "in_universe", "eligible",
        "z_margin_chg", "z_drawdown", "rk_margin_chg", "rk_drawdown",
        "score_margin_chg", "score_drawdown", "composite_score", "composite_rank",
    ]
    return scored[cols].sort_values(["signal_date", "stock_code"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="建立每月因子分數表")
    ap.add_argument("--price", required=True, help="股價資料.csv")
    ap.add_argument("--margin", required=True, help="融資資料.csv")
    ap.add_argument("--out", default="output/factor_scores.csv")
    a = ap.parse_args()

    table = build_factor_table(a.price, a.margin)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(a.out, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")

    n_months = table["signal_date"].nunique()
    per_month = table.groupby("signal_date")["eligible"].sum()
    print(f"輸出 {a.out}：{len(table):,} 列，{n_months} 個月")
    print(f"可選股票數／月：最少 {per_month.min()}，中位數 {int(per_month.median())}，最多 {per_month.max()}")


if __name__ == "__main__":
    main()
