"""因子計算的自動測試。用小型合成資料，幾秒內跑完。

執行：pytest -q
"""
import numpy as np
import pandas as pd
import pytest

try:
    from pipeline.build_factors import (
except ImportError:
    from build_factors import (
    Config,
    build_raw_factors,
    compute_drawdown,
    month_end_dates,
    standardize,
)


def make_data(n_days=130, stocks=("1001 甲", "1002 乙", "1003 丙"), seed=0):
    """合成資料：每檔股票有日報酬、收盤價、成交值、融資餘額。"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    price_rows, margin_rows = [], []
    for k, s in enumerate(stocks):
        ret = rng.normal(0, 1.5, n_days)
        close = 50 * np.cumprod(1 + ret / 100)
        mb = 1000 + 200 * k + np.cumsum(rng.integers(-20, 25, n_days))
        for i, d in enumerate(dates):
            price_rows.append(
                dict(stock=s, date=d, close=close[i], volume=1000, value=50_000.0,
                     ret_pct=ret[i], market="TSE")
            )
            margin_rows.append(dict(stock=s, date=d, margin_bal=float(mb[i])))
    return pd.DataFrame(price_rows), pd.DataFrame(margin_rows)


CFG = Config(dd_window=20, liq_window=10, min_avg_value=1_000.0)


def test_month_end_is_last_trading_day():
    dates = pd.bdate_range("2020-01-01", "2020-03-31")
    me = month_end_dates(dates)
    assert list(me) == [pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-28"), pd.Timestamp("2020-03-31")]


def test_margin_change_matches_hand_calc():
    prices, margin = make_data()
    raw = build_raw_factors(prices, margin, CFG)
    s, t_now, t_prev = "1001 甲", pd.Timestamp("2020-03-31"), pd.Timestamp("2020-02-28")
    mb = margin[margin.stock == s].set_index("date")["margin_bal"]
    expected = mb[t_now] / mb[t_prev] - 1
    got = raw[(raw.stock == s) & (raw.signal_date == t_now)]["margin_chg"].iloc[0]
    assert got == pytest.approx(expected)


def test_margin_change_nan_when_base_too_small():
    prices, margin = make_data()
    # 讓上個月底的融資餘額只有 5 張（低於門檻 100）
    margin.loc[(margin.stock == "1001 甲") & (margin.date == "2020-02-28"), "margin_bal"] = 5.0
    raw = build_raw_factors(prices, margin, CFG)
    v = raw[(raw.stock == "1001 甲") & (raw.signal_date == "2020-03-31")]["margin_chg"].iloc[0]
    assert np.isnan(v)


def test_drawdown_ignores_ex_dividend_drop():
    """除息日收盤價掉 5%，但報酬率（含股利）是 0 -> 回撤應該是 0，不是 -5%。"""
    dates = pd.bdate_range("2020-01-01", periods=30)
    ret = pd.DataFrame({"X": np.zeros(30)}, index=dates)
    close = pd.DataFrame({"X": np.full(30, 100.0)}, index=dates)
    close.iloc[15:] = 95.0  # 除息後收盤價下降，但報酬率為 0
    dd = compute_drawdown(ret, close, window=20)
    assert dd["X"].iloc[-1] == pytest.approx(0.0)


def test_drawdown_value_and_sign():
    dates = pd.bdate_range("2020-01-01", periods=25)
    r = np.zeros(25)
    r[10] = 20.0   # 漲 20%
    r[24] = -10.0  # 最後一天跌 10%
    ret = pd.DataFrame({"X": r}, index=dates)
    close = pd.DataFrame({"X": np.full(25, 100.0)}, index=dates)
    dd = compute_drawdown(ret, close, window=20)
    assert dd["X"].iloc[-1] == pytest.approx(0.9 - 1)  # 從高點 1.2 降到 1.08：1.08/1.2-1 = -10%
    assert (dd["X"].dropna() <= 1e-12).all()


def test_drawdown_needs_full_window():
    prices, margin = make_data(n_days=130)
    raw = build_raw_factors(prices, margin, Config(dd_window=60, liq_window=10, min_avg_value=1_000.0))
    first = raw[raw.signal_date == raw.signal_date.min()]
    assert first["drawdown"].isna().all()  # 第一個月底資料不到 60 天


def test_no_lookahead():
    """把訊號日之後的資料全部刪掉，該訊號日的因子值與分數不能改變。"""
    prices, margin = make_data()
    t = pd.Timestamp("2020-04-30")
    full = standardize(build_raw_factors(prices, margin, CFG), CFG)
    cut = standardize(
        build_raw_factors(prices[prices.date <= t], margin[margin.date <= t], CFG), CFG
    )
    cols = ["stock", "margin_chg", "drawdown", "z_margin_chg", "z_drawdown"]
    a = full[full.signal_date == t][cols].sort_values("stock").reset_index(drop=True)
    b = cut[cut.signal_date == t][cols].sort_values("stock").reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_zscore_mean_zero_std_one_per_month():
    stocks = tuple(f"{1000 + i} S{i}" for i in range(40))
    prices, margin = make_data(stocks=stocks)
    out = standardize(build_raw_factors(prices, margin, CFG), CFG)
    e = out[out.eligible]
    assert len(e) > 0
    for f in ("z_margin_chg", "z_drawdown"):
        g = e.groupby("signal_date")[f]
        assert g.mean().abs().max() < 1e-9
        assert (g.std(ddof=0) - 1).abs().max() < 1e-9


def test_illiquid_stock_gets_no_score():
    prices, margin = make_data()
    prices.loc[prices.stock == "1002 乙", "value"] = 1.0  # 成交值極低
    out = standardize(build_raw_factors(prices, margin, CFG), CFG)
    assert not out[out.stock == "1002 乙"]["eligible"].any()
    assert out[out.stock == "1002 乙"]["z_drawdown"].isna().all()


def _scored(n=40):
    stocks = tuple(f"{1000 + i} S{i}" for i in range(n))
    prices, margin = make_data(stocks=stocks)
    return standardize(build_raw_factors(prices, margin, CFG), CFG)


def test_direction_is_reversed_for_contrarian_strategy():
    """融資減少越多、回撤越深 -> 單因子分數越高。"""
    out = _scored()
    for _, g in out[out.eligible].groupby("signal_date"):
        assert g.loc[g.margin_chg.idxmin(), "score_margin_chg"] == g.score_margin_chg.max()
        assert g.loc[g.drawdown.idxmin(), "score_drawdown"] == g.score_drawdown.max()
        # 原始 rk_ 欄位維持「數值越大排名越高」
        assert g.loc[g.margin_chg.idxmax(), "rk_margin_chg"] == g.rk_margin_chg.max()


def test_composite_score_formula():
    out = _scored()
    e = out[out.eligible]
    expected = 0.5 * (1 - e.rk_margin_chg) + 0.5 * (1 - e.rk_drawdown)
    pd.testing.assert_series_equal(e.composite_score, expected, check_names=False)
    assert e.composite_score.between(0, 1).all()


def test_composite_only_for_eligible_and_rank_is_1_to_n():
    out = _scored()
    assert out.loc[~out.eligible, "composite_score"].isna().all()
    assert out.loc[~out.eligible, "composite_rank"].isna().all()
    for _, g in out[out.eligible].groupby("signal_date"):
        assert sorted(g.composite_rank) == list(range(1, len(g) + 1))
        best = g.loc[g.composite_score.idxmax()]
        assert best.composite_rank == 1


def test_composite_no_lookahead():
    prices, margin = make_data(stocks=tuple(f"{1000 + i} S{i}" for i in range(40)))
    t = pd.Timestamp("2020-04-30")
    full = standardize(build_raw_factors(prices, margin, CFG), CFG)
    cut = standardize(build_raw_factors(prices[prices.date <= t], margin[margin.date <= t], CFG), CFG)
    cols = ["stock", "composite_score", "composite_rank"]
    a = full[full.signal_date == t][cols].sort_values("stock").reset_index(drop=True)
    b = cut[cut.signal_date == t][cols].sort_values("stock").reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)
