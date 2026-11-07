import math
from behave import given, when, then

# ==================== 定義專屬的錯誤類別 ====================
class DataMissingAnomaly(Exception):
    pass

# ==================== 模擬被測模組 ====================
class MockDataEngine:
    def __init__(self):
        self.financial_reports = {}

    def publish_report(self, date_str, period, value):
        self.financial_reports[period] = {"pub_date": date_str, "value": value}

    def get_latest_available_report(self, current_date, period):
        info = self.financial_reports.get(period)
        if info and info["pub_date"] <= current_date:
            return info["value"]
        return None

    def clean_missing_prices(self, price_series, method):
        if method == "bfill":
            raise ValueError("DataLeakError: backward fill is strictly prohibited.")
        return True

    def check_continuity(self, missing_count, threshold=3):
        if missing_count >= threshold:
            # 改為拋出我們自訂的錯誤類別
            raise DataMissingAnomaly("連續缺失特徵超過容忍上限")

class MockPortfolio:
    def __init__(self, cash=100000):
        self.cash = cash
        self.positions = {}
        self.fee_rate = 0.001425
        self.tax_rate = 0.003
        self.min_fee = 20

    def buy(self, symbol, price, shares):
        trade_value = price * shares
        # 加入 round() 修正浮點數誤差，避免 170.999 變成 170
        fee = max(self.min_fee, math.floor(round(trade_value * self.fee_rate, 4)))
        total_cost = trade_value + fee
        self.cash -= total_cost
        self.positions[symbol] = self.positions.get(symbol, 0) + shares
        return total_cost

    def sell(self, symbol, price, shares):
        trade_value = price * shares
        # 加入 round() 修正浮點數誤差，確保手續費與稅金計算完全準確
        fee = max(self.min_fee, math.floor(round(trade_value * self.fee_rate, 4)))
        tax = math.floor(round(trade_value * self.tax_rate, 4))
        net_revenue = trade_value - fee - tax
        self.cash += net_revenue
        self.positions[symbol] -= shares
        return net_revenue

# ==================== Behave Steps 實作 ====================

# --- 財報時序 ---
@given('某公司 2024 年 Q3 財報於 "{pub_date}" 正式公告')
def step_pub_report(context, pub_date):
    context.engine = MockDataEngine()
    context.engine.publish_report(pub_date, "2024Q3", 500)

@when('回測系統在 "{query_date}" 執行因子計算')
def step_query_factor(context, query_date):
    context.fetched_val = context.engine.get_latest_available_report(query_date, "2024Q3")

@then('系統取得的 Q3 營收數據應為空值或上一期數值，不得讀取到 Q3 實際財報數值')
def step_assert_no_lookahead(context):
    assert context.fetched_val is None, f"偵測到未來資訊外洩！取得數值: {context.fetched_val}"

@given('當前模擬交易日為 "{sim_date}"')
def step_set_sim_date(context, sim_date):
    context.sim_date = sim_date
    context.engine = MockDataEngine()

@given('2023 年 Q4 年報於 "{q4_pub}" 公告，2023 年 Q3 財報於 "{q3_pub}" 公告')
def step_set_multi_reports(context, q4_pub, q3_pub):
    context.engine.publish_report(q4_pub, "2023Q4", {"eps": 10.0})
    context.engine.publish_report(q3_pub, "2023Q3", {"eps": 7.5})

@when('策略引擎計算最新本益比 (PE)')
def step_calc_pe(context):
    q4_eps = context.engine.get_latest_available_report(context.sim_date, "2023Q4")
    q3_eps = context.engine.get_latest_available_report(context.sim_date, "2023Q3")
    context.selected_period = "2023Q4" if q4_eps else ("2023Q3" if q3_eps else None)

@then('運算所依賴的每股盈餘 (EPS) 基準期必須為 2023 年 Q3')
def step_assert_period(context):
    assert context.selected_period == "2023Q3", f"取樣期錯誤：{context.selected_period}"

# --- 缺失值清洗 ---
@given('股票 "{symbol}" 於 "{date_str}" 暫停交易且開高低收均無撮合成交')
def step_stock_missing(context, symbol, date_str):
    context.engine = MockDataEngine()
    context.cleaning_method = "bfill" 

@when('資料前處理模組補齊缺失值')
def step_clean_missing(context):
    context.error_captured = None
    try:
        context.engine.clean_missing_prices(None, method=context.cleaning_method)
    except Exception as e:
        context.error_captured = e

@then('處理方式不得採用後續交易日的數值填充（禁止 bfill）')
def step_assert_no_bfill(context):
    assert context.error_captured is not None, "未攔截到禁止 bfill 的安全限制！"
    assert "backward fill is strictly prohibited" in str(context.error_captured)

@given('股票 "{symbol}" 的收盤價連續 {count:d} 個交易日皆為缺失值')
def step_stock_cont_missing(context, symbol, count):
    context.engine = MockDataEngine()
    context.missing_count = count

@when('資料前處理檢查資料品質')
def step_check_quality(context):
    context.exception_raised = None
    try:
        context.engine.check_continuity(context.missing_count, threshold=3)
    except Exception as e:
        context.exception_raised = type(e).__name__

@then('系統應拋出 "{expected_exception}" 例外並將該標的排除於當日選股清單')
def step_assert_anomaly(context, expected_exception):
    assert context.exception_raised == expected_exception, f"預期 {expected_exception}，但收到 {context.exception_raised}"

# --- 交易成本與滑價 ---
@given('帳戶初始現金為 {cash:d} 元')
def step_init_cash(context, cash):
    context.portfolio = MockPortfolio(cash=cash)

@given('券商手續費率為 {fee_rate:f}%，最低手續費為 {min_fee:d} 元')
def step_fee_config(context, fee_rate, min_fee):
    context.portfolio.fee_rate = fee_rate / 100.0
    context.portfolio.min_fee = min_fee

@when('以每股 {price:d} 元買入 "{symbol}" 共 {shares:d} 股')
def step_buy_stock(context, price, symbol, shares):
    context.deducted = context.portfolio.buy(symbol, price, shares)

@then('帳戶扣除的總金額必須為 {expected_total:d} 元')
def step_assert_buy_cost(context, expected_total):
    assert context.deducted == expected_total, f"扣除金額不符：預期 {expected_total}，實扣 {context.deducted}"

@given('帳戶持有每股 {buy_price:d} 元買入的 "{symbol}" 共 {shares:d} 股')
def step_hold_stock(context, buy_price, symbol, shares):
    context.portfolio = MockPortfolio(cash=0)
    context.portfolio.positions[symbol] = shares

@given('券商手續費率為 {fee_rate:f}%，證券交易稅率為 {tax_rate:f}%')
def step_fee_and_tax(context, fee_rate, tax_rate):
    context.portfolio.fee_rate = fee_rate / 100.0
    context.portfolio.tax_rate = tax_rate / 100.0

@when('以每股 {price:d} 元全數賣出')
def step_sell_all(context, price):
    symbol = list(context.portfolio.positions.keys())[0]
    shares = context.portfolio.positions[symbol]
    context.net_revenue = context.portfolio.sell(symbol, price, shares)

@then('帳戶實際入帳金額必須為 {expected_net:d} 元')
def step_assert_sell_net(context, expected_net):
    assert context.net_revenue == expected_net, f"淨入帳不符：預期 {expected_net}，實際 {context.net_revenue}"