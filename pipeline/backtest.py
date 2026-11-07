"""D 部分：選股與回測（月調倉、整張交易、逐筆扣成本）。

輸入  C 產出的 factor_scores.csv + 股價資料.csv
輸出  月報酬表、交易明細、績效摘要

規則（依組內約定）
    本金           1,000 萬元
    選股           每月最後交易日（訊號日）收盤後，eligible 股票依 composite_rank 取前 10 檔
    權重           等權重 10%，每月調回
    成交           訊號日的下一個交易日收盤價（回測假設）
    整張交易       買進以 1,000 股為單位，用不完的錢留現金
    成本           手續費 0.1425%（買賣各一次）、證交稅 0.3%（僅賣出）、滑價 0.05%（買賣各一次）
                   依每筆實際成交金額計算，不是每月固定扣

持股價值怎麼算（重要）
    收盤價沒有還原，除權息、減資、分割當天會跳動。所以持股價值用「報酬率」連乘出的報酬指數（TRI）
    推進：價值 = 股數 × 參考價 × TRI_今天 / TRI_參考日。股利視為再投入。
    實際收盤價只用在「決定買幾張」。每次調倉時，會把參考價與 TRI 重設為當天的值（價值不變）。

不偷看未來
    * 訊號日 t 的分數只用 t 以前的資料（C 部分已測試）。
    * 成交日一定晚於訊號日。
    * 唯一用到成交日資訊的地方：成交日當天沒有成交量（停牌）的股票買不到，會跳過、由下一名遞補。
      這在現實中也是下單當天才知道的事。
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from pipeline.build_factors import load_prices
except ImportError:
    from build_factors import load_prices


@dataclass(frozen=True)
class BTConfig:
    capital: float = 10_000_000.0
    n_hold: int = 10
    fee: float = 0.001425       # 手續費，買賣各收一次（無折扣牌告價）
    tax: float = 0.003          # 證交稅，只在賣出收
    slippage: float = 0.0005    # 滑價，買賣各一次（模擬假設，非法定費率）
    lot: int = 1000             # 一張 = 1,000 股
    rank_col: str = "composite_rank"   # 用哪個欄位排名（1 = 最想買）
    delist_haircut: float = 0.0  # 持股下市時，以最後價格強制賣出；此為額外折價比例（0 = 樂觀，1 = 全賠）
    start: str = "2015-01-01"   # 回測起日（樣本內）
    end: str = "2022-12-31"     # 回測迄日（樣本內）


# --------------------------------------------------------------------------
# 市場資料
# --------------------------------------------------------------------------
class Market:
    """把股價長表轉成 numpy 矩陣（列 = 交易日，欄 = 股票），回測迴圈比較快。"""

    def __init__(self, prices: pd.DataFrame):
        close = prices.pivot(index="date", columns="stock", values="close").sort_index()
        ret = prices.pivot(index="date", columns="stock", values="ret_pct").reindex(close.index)
        vol = prices.pivot(index="date", columns="stock", values="volume").reindex(close.index)
        # 報酬指數：用含息報酬率連乘；下市後沿用最後一個值（部位凍結在最後價值）
        tri = (1.0 + ret.fillna(0.0) / 100.0).cumprod().where(close.notna()).ffill()
        self.dates = close.index
        self.cols = list(close.columns)
        self.idx = {s: i for i, s in enumerate(self.cols)}
        self.close = close.to_numpy(float)
        self.volume = vol.to_numpy(float)
        self.tri = tri.to_numpy(float)

    def tradable(self, di: int, ci: int) -> bool:
        return bool(self.volume[di, ci] > 0 and not np.isnan(self.close[di, ci]))


# --------------------------------------------------------------------------
# 回測引擎
# --------------------------------------------------------------------------
def run_backtest(factors: pd.DataFrame, mkt: Market, cfg: BTConfig = BTConfig()):
    """回傳 (monthly, trades, daily)。"""
    f = factors[factors["eligible"] & factors[cfg.rank_col].notna()].copy()
    f["stock"] = f["stock_code"] + " " + f["stock_name"]
    ranked = {
        s: g.sort_values(cfg.rank_col)["stock"].tolist() for s, g in f.groupby("signal_date")
    }

    start, end = pd.Timestamp(cfg.start), pd.Timestamp(cfg.end)
    dates = mkt.dates
    reb: dict[pd.Timestamp, pd.Timestamp] = {}  # 成交日 -> 訊號日
    for s in sorted(ranked):
        i = dates.searchsorted(s, side="right")  # 訊號日之後的第一個交易日
        if i < len(dates) and start <= dates[i] <= end:
            reb[dates[i]] = s
    if not reb:
        raise ValueError("沒有任何可回測的調倉日")
    first_trade = min(reb)

    c_buy = cfg.fee + cfg.slippage
    cash = cfg.capital
    pos: dict[int, list[float]] = {}  # 欄位索引 -> [股數, 參考價, 參考TRI]
    trades: list[dict] = []
    rebs: list[dict] = []
    daily: list[tuple[pd.Timestamp, float]] = []

    def value_of(di: int) -> float:
        return sum(sh * pref * mkt.tri[di, c] / tref for c, (sh, pref, tref) in pos.items())

    # 每檔股票最後一個有收盤價的日期索引；超過這天就代表下市（資料不再有價格）
    valid = ~np.isnan(mkt.close)
    last_idx = np.where(valid.any(axis=0), valid.shape[0] - 1 - np.argmax(valid[::-1], axis=0), -1)

    acc = dict(buy=0.0, sell=0.0, fee=0.0, tax=0.0, slip=0.0)

    def record(c, side, sh, px, d, sig, forced=False):
        nonlocal cash
        val = sh * px
        fee, slip = val * cfg.fee, val * cfg.slippage
        tax = val * cfg.tax if side == "sell" else 0.0
        if side == "sell":
            cash += val - fee - tax - slip
            acc["sell"] += val
        else:
            cash -= val + fee + slip
            acc["buy"] += val
        acc["fee"] += fee
        acc["tax"] += tax
        acc["slip"] += slip
        trades.append(dict(
            trade_date=d, signal_date=sig, stock=mkt.cols[c], side=side,
            shares=sh, price=px, value=val, fee=fee, tax=tax, slippage=slip, forced=forced,
        ))

    def flush_forced_costs():
        """上一期調倉之後、下一次調倉之前發生的強制賣出成本，算在上一期。"""
        if rebs:
            rebs[-1]["forced_sale_costs"] = rebs[-1].get("forced_sale_costs", 0.0) + acc["fee"] + acc["tax"] + acc["slip"]
        for k in acc:
            acc[k] = 0.0

    for di, d in enumerate(dates):
        if d < first_trade or d > end:
            continue
        # 持股下市：以最後價值強制賣出（含賣出成本）
        for c in list(pos):
            if last_idx[c] < di:
                sh, pref, tref = pos[c]
                val = sh * pref * mkt.tri[di, c] / tref * (1 - cfg.delist_haircut)
                record(c, "sell", sh, val / sh, d, rebs[-1]["signal_date"] if rebs else pd.NaT, forced=True)
                del pos[c]
        if d in reb:
            flush_forced_costs()
            sig = reb[d]
            # 1) 價值不變地重設參考點（股利視為再投入）
            for c, p in pos.items():
                if mkt.tradable(di, c):
                    v = p[0] * p[1] * mkt.tri[di, c] / p[2]
                    p[0], p[1], p[2] = v / mkt.close[di, c], mkt.close[di, c], mkt.tri[di, c]
            e_pre = cash + value_of(di)

            # 2) 從排名由前往後挑出「成交日買得到」的前 n 檔
            targets, skipped = [], 0
            for s in ranked[sig]:
                c = mkt.idx.get(s)
                if c is None or not mkt.tradable(di, c):
                    skipped += 1
                    continue
                targets.append(c)
                if len(targets) == cfg.n_hold:
                    break
            tset = set(targets)
            target_value = e_pre / cfg.n_hold

            def target_shares(c: int) -> float:
                px = mkt.close[di, c]
                return math.floor(target_value / (px * (1 + c_buy)) / cfg.lot) * cfg.lot

            # 3) 先賣：不在名單內的全部賣出；名單內超過目標的賣到整張
            for c in list(pos):
                if not mkt.tradable(di, c):
                    continue  # 停牌賣不掉，續抱
                px = mkt.close[di, c]
                sh = pos[c][0]
                if c not in tset:
                    record(c, "sell", sh, px, d, sig)
                    del pos[c]
                else:
                    excess = sh - target_shares(c)
                    if excess >= cfg.lot:
                        sell_sh = math.floor(excess / cfg.lot) * cfg.lot
                        record(c, "sell", sell_sh, px, d, sig)
                        pos[c][0] -= sell_sh

            # 4) 再買：依排名補到目標張數，錢不夠就少買
            for c in targets:
                px = mkt.close[di, c]
                need = target_shares(c) - (pos[c][0] if c in pos else 0.0)
                if need < cfg.lot:
                    continue
                buy_sh = math.floor(need / cfg.lot) * cfg.lot
                affordable = math.floor(max(cash, 0.0) / (px * (1 + c_buy)) / cfg.lot) * cfg.lot
                buy_sh = min(buy_sh, affordable)
                if buy_sh < cfg.lot:
                    continue
                record(c, "buy", buy_sh, px, d, sig)
                if c in pos:
                    pos[c][0] += buy_sh
                else:
                    pos[c] = [float(buy_sh), px, mkt.tri[di, c]]

            e_post = cash + value_of(di)
            costs = acc["fee"] + acc["tax"] + acc["slip"]
            rebs.append(dict(
                trade_date=d, signal_date=sig, equity_pre=e_pre, equity_post=e_post,
                costs=costs, fee=acc["fee"], tax=acc["tax"], slippage=acc["slip"],
                buy_value=acc["buy"], sell_value=acc["sell"],
                turnover=(acc["buy"] + acc["sell"]) / 2 / e_pre,
                n_positions=len(pos), n_targets=len(targets),
                n_unfunded=sum(1 for c in targets if c not in pos),
                cash_pct=cash / e_post, skipped_untradable=skipped,
                forced_sale_costs=0.0,
            ))
            for k in acc:
                acc[k] = 0.0
        daily.append((d, cash + value_of(di)))

    flush_forced_costs()
    daily_s = pd.Series(dict(daily), name="equity")
    monthly = pd.DataFrame(rebs)
    # 每個月的期末權益 = 下一次調倉前的權益；最後一期用回測最後一天
    nxt = monthly["equity_pre"].shift(-1)
    nxt.iloc[-1] = daily_s.iloc[-1]
    monthly["equity_end"] = nxt
    monthly["net_return"] = monthly["equity_end"] / monthly["equity_pre"] - 1      # 含當月調倉成本
    monthly["gross_return"] = monthly["equity_end"] / monthly["equity_post"] - 1   # 持有期間價格變動
    monthly["cost_pct"] = (monthly["costs"] + monthly["forced_sale_costs"]) / monthly["equity_pre"]
    monthly["cum_return"] = monthly["equity_end"] / cfg.capital - 1
    monthly["hold_month"] = monthly["trade_date"].dt.to_period("M").astype(str)

    # 參考基準：該月股票池（eligible）等權、同樣期間、不扣成本
    period_end = list(monthly["trade_date"].iloc[1:]) + [daily_s.index[-1]]
    ref = []
    for (sig, d0), d1 in zip(monthly[["signal_date", "trade_date"]].itertuples(index=False), period_end):
        cs = [mkt.idx[s] for s in ranked[sig] if s in mkt.idx]
        i0, i1 = dates.get_loc(d0), dates.get_loc(d1)
        r = mkt.tri[i1, cs] / mkt.tri[i0, cs] - 1
        ref.append(np.nanmean(r))
    monthly["universe_ew_return"] = ref
    monthly["period_end"] = period_end

    return monthly, pd.DataFrame(trades), daily_s


# --------------------------------------------------------------------------
# 績效指標
# --------------------------------------------------------------------------
def performance(returns: pd.Series, equity_daily: pd.Series | None, capital: float, rf: float = 0.0) -> dict:
    """returns: 月報酬。equity_daily: 日權益（算最大回撤用，沒有就用月資料）。"""
    r = returns.astype(float)
    n = len(r)
    cum = (1 + r).prod()
    years = n / 12
    sd = r.std(ddof=1)
    if equity_daily is not None:
        eq = pd.concat([pd.Series([capital]), equity_daily.reset_index(drop=True)])
    else:
        eq = pd.concat([pd.Series([1.0]), (1 + r).cumprod().reset_index(drop=True)])
    return {
        "累積報酬": cum - 1,
        "年化報酬": cum ** (1 / years) - 1,
        "年化波動": sd * math.sqrt(12),
        "Sharpe": (r.mean() - rf / 12) / sd * math.sqrt(12) if sd > 0 else np.nan,
        "最大回撤": float((eq / eq.cummax() - 1).min()),
        "月勝率": float((r > 0).mean()),
        "月數": n,
    }


def summarize(monthly: pd.DataFrame, daily: pd.Series, cfg: BTConfig) -> dict:
    out = performance(monthly["net_return"], daily, cfg.capital)
    out["平均月換手率"] = monthly["turnover"].mean()
    out["年化換手率"] = monthly["turnover"].mean() * 12
    out["年化成本拖累"] = monthly["cost_pct"].mean() * 12
    out["平均持股檔數"] = monthly["n_positions"].mean()
    out["平均現金比例"] = monthly["cash_pct"].mean()
    out["最終資產"] = float(daily.iloc[-1])
    return out


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="月調倉回測")
    ap.add_argument("--factors", default="output/factor_scores.csv")
    ap.add_argument("--price", required=True)
    ap.add_argument("--outdir", default="output")
    a = ap.parse_args()

    factors = pd.read_csv(
        a.factors, parse_dates=["signal_date"], dtype={"stock_code": str}
    )
    prices = load_prices(a.price)
    prices = prices[prices["market"].isin(["TSE", "OTC"])]
    mkt = Market(prices)

    base = BTConfig()
    scenarios = {
        "主要結果（扣成本）": base,
        "不扣成本": replace(base, fee=0.0, tax=0.0, slippage=0.0),
        "滑價 0.1% 情境": replace(base, slippage=0.001),
        "下市股全賠（極端壓力）": replace(base, delist_haircut=1.0),
    }
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)

    rows, curves = {}, {}
    for name, cfg in scenarios.items():
        monthly, trades, daily = run_backtest(factors, mkt, cfg)
        rows[name] = summarize(monthly, daily, cfg)
        curves[name] = pd.Series((1 + monthly["net_return"]).cumprod().values, index=monthly["period_end"])
        if name.startswith("主要"):
            main_monthly, main_trades, main_daily = monthly, trades, daily
    ref_name = "參考：股票池等權（未扣成本，非加權指數）"
    curves[ref_name] = pd.Series(
        (1 + main_monthly["universe_ew_return"]).cumprod().values, index=main_monthly["period_end"]
    )
    start_row = pd.DataFrame([[1.0] * len(curves)], columns=list(curves), index=[main_monthly["trade_date"].iloc[0]])
    pd.concat([start_row, pd.DataFrame(curves)]).rename_axis("date").to_csv(
        out / "backtest_curves.csv", encoding="utf-8-sig", date_format="%Y-%m-%d"
    )
    ref = performance(main_monthly["universe_ew_return"], None, base.capital)
    rows["參考：股票池等權（未扣成本，非加權指數）"] = ref

    summary = pd.DataFrame(rows).T
    summary.to_csv(out / "backtest_summary.csv", encoding="utf-8-sig")
    main_monthly.to_csv(out / "backtest_monthly.csv", index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    main_trades.to_csv(out / "backtest_trades.csv", index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    main_daily.rename_axis("date").to_csv(out / "backtest_daily_equity.csv", encoding="utf-8-sig", date_format="%Y-%m-%d")

    pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
    print(summary.drop(columns=["最終資產"]).to_string())
    print("\n最終資產：", {k: f"{v['最終資產']:,.0f}" for k, v in rows.items() if "最終資產" in v})


if __name__ == "__main__":
    main()
