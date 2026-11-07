"""回測引擎的自動測試（合成資料）。"""
import numpy as np
import pandas as pd
import pytest

try:
    from pipeline.backtest import BTConfig, Market, run_backtest
except ImportError:
    from backtest import BTConfig, Market, run_backtest

DATES = pd.bdate_range("2020-01-01", "2020-04-30")


def make_prices(close_by_stock: dict, ret_by_stock: dict | None = None):
    rows = []
    for s, closes in close_by_stock.items():
        rets = (ret_by_stock or {}).get(s, np.zeros(len(DATES)))
        for d, c, r in zip(DATES, closes, rets):
            rows.append(dict(stock=s, date=d, close=float(c), volume=1000.0, value=1e5, ret_pct=float(r), market="TSE"))
    return pd.DataFrame(rows)


def make_factors(order: list[str], signals: list[str]):
    """每個訊號日，order 內的股票依序 rank 1..n，全部 eligible。"""
    rows = []
    for sd in signals:
        for rank, s in enumerate(order, 1):
            code, name = s.split(" ")
            rows.append(dict(signal_date=pd.Timestamp(sd), stock_code=code, stock_name=name,
                             eligible=True, composite_rank=float(rank)))
    return pd.DataFrame(rows)


SIGNALS = ["2020-01-31", "2020-02-28", "2020-03-31"]
CFG = BTConfig(capital=1_000_000, n_hold=1, start="2020-01-01", end="2020-04-30")


def flat(price=100.0):
    return np.full(len(DATES), price)


def test_buy_cost_and_lot_sizing():
    mkt = Market(make_prices({"1001 甲": flat(100)}))
    m, t, d = run_backtest(make_factors(["1001 甲"], SIGNALS), mkt, CFG)
    first = t.iloc[0]
    # 1,000,000 / (100 × 1.001925) = 9980.8 股 -> 9 張 = 9,000 股
    assert first["side"] == "buy" and first["shares"] == 9000
    assert first["value"] == pytest.approx(900_000)
    assert first["fee"] == pytest.approx(900_000 * 0.001425)
    assert first["slippage"] == pytest.approx(900_000 * 0.0005)
    assert first["tax"] == 0


def test_sell_pays_tax_and_fees():
    mkt = Market(make_prices({"1001 甲": flat(100), "1002 乙": flat(50)}))
    f = pd.concat([make_factors(["1001 甲", "1002 乙"], SIGNALS[:1]),
                   make_factors(["1002 乙", "1001 甲"], SIGNALS[1:])])
    _, t, _ = run_backtest(f, mkt, CFG)
    sell = t[(t.side == "sell") & (t.stock == "1001 甲")].iloc[0]
    assert sell["tax"] == pytest.approx(sell["value"] * 0.003)
    assert sell["fee"] == pytest.approx(sell["value"] * 0.001425)
    assert (t[t.side == "buy"]["tax"] == 0).all()


def test_equity_identity_post_equals_pre_minus_costs():
    mkt = Market(make_prices({"1001 甲": flat(100), "1002 乙": flat(50)}))
    f = pd.concat([make_factors(["1001 甲", "1002 乙"], SIGNALS[:1]),
                   make_factors(["1002 乙", "1001 甲"], SIGNALS[1:])])
    m, _, _ = run_backtest(f, mkt, CFG)
    assert (m["equity_post"] - (m["equity_pre"] - m["costs"])).abs().max() < 1e-6


def test_trade_date_strictly_after_signal_date():
    mkt = Market(make_prices({"1001 甲": flat(100)}))
    m, t, _ = run_backtest(make_factors(["1001 甲"], SIGNALS), mkt, CFG)
    assert (m["trade_date"] > m["signal_date"]).all()
    assert (t["trade_date"] > t["signal_date"]).all()


def test_cash_never_negative_and_buys_are_whole_lots():
    stocks = {f"{1000 + i} S{i}": flat(20 + 37 * i) for i in range(12)}
    mkt = Market(make_prices(stocks))
    cfg = BTConfig(capital=1_000_000, n_hold=10, start="2020-01-01", end="2020-04-30")
    m, t, d = run_backtest(make_factors(list(stocks), SIGNALS), mkt, cfg)
    assert (m["cash_pct"] >= -1e-9).all()
    assert (t[t.side == "buy"]["shares"] % 1000 == 0).all()
    assert (m["n_positions"] <= 10).all()


def test_ex_dividend_drop_is_not_a_loss():
    """除息日收盤價掉 10%，但含息報酬率為 0 -> 權益不應該虧損。"""
    close = flat(100).copy()
    close[40:] = 90.0
    mkt = Market(make_prices({"1001 甲": close}))  # ret_pct 全為 0
    m, _, d = run_backtest(make_factors(["1001 甲"], SIGNALS), mkt, CFG)
    # 除息前後權益只差成本，不應差 10%
    before, after = d.iloc[20], d.iloc[-1]
    assert after / before > 0.99


def test_price_gain_flows_into_equity():
    """股價（含息報酬）上漲 10% -> 該期持股價值增加約 10%。"""
    close = flat(100).copy()
    rets = np.zeros(len(DATES))
    j = list(DATES).index(pd.Timestamp("2020-02-10"))
    rets[j] = 10.0
    close[j:] = 110.0
    mkt = Market(make_prices({"1001 甲": close}, {"1001 甲": rets}))
    m, _, _ = run_backtest(make_factors(["1001 甲"], SIGNALS), mkt, CFG)
    feb = m[m.hold_month == "2020-02"].iloc[0]
    assert feb["gross_return"] == pytest.approx(0.9 * 0.10, rel=0.02)  # 9 成資金在股票上（1 成是整張剩下的現金）


def test_untradable_stock_is_skipped_and_next_rank_used():
    prices = make_prices({"1001 甲": flat(100), "1002 乙": flat(50)})
    first_trade = DATES[DATES > pd.Timestamp("2020-01-31")][0]
    prices.loc[(prices.stock == "1001 甲") & (prices.date == first_trade), "volume"] = 0.0
    mkt = Market(prices)
    f = make_factors(["1001 甲", "1002 乙"], SIGNALS)
    m, t, _ = run_backtest(f, mkt, CFG)
    first = t[t.trade_date == first_trade]
    assert list(first.stock) == ["1002 乙"]
    assert m.iloc[0]["skipped_untradable"] == 1
