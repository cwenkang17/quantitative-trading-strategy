# 智慧量化交易專題成果庫：融資減碼與股價回撤逆向選股策略
> 逢甲大學 115-1 智慧量化交易課程 ｜ 第五組期末成果專案

本專案依據作業繳交規範，整合為**單一完整程式庫**，嚴格劃分 `spec/`、`pipeline/`、`tests/`、`ai_log/` 與 `README.md` 五大核心模組。

---

## 一、專案模組架構導覽

```text
quantitative_trading_final/
├── spec/                          # 策略規格與需求定義
│   ├── strategy_specification.md  # 策略數學模型、Universe 門檻與回測規則規格書
│   └── quant_rules.feature        # BDD (Gherkin) 業務規則與合規規格定義
├── pipeline/                      # 核心運算管線與執行模組
│   ├── __init__.py
│   ├── build_factors.py           # 因子工程：融資變化率與回撤計算、Point-in-Time 截面標準化
│   ├── backtest.py                # 回測引擎：整張離散交易、逐筆成本扣除、下市強平處理
│   └── plot_results.py            # 繪圖腳本：繪製資產成長走勢與回撤雙圖表
├── tests/                         # 測試驗證模組
│   ├── __init__.py
│   ├── test_factors.py            # 因子單元測試 (含防前視偏誤 test_no_lookahead)
│   ├── test_backtest.py           # 回測單元測試 (資金恆等式、成本扣除、整張買入)
│   ├── steps/
│   │   └── quant_rules_steps.py   # Behave BDD step definitions 與 Mock 資料引擎
│   └── behave_test_passed.png     # Behave 測試 6 大情境 21 個步驟全數通過之執行截圖
├── ai_log/                        # AI 協同開發與除錯日誌
│   └── AI_COLLABORATION_LOG.md    # 6位組員實際 Prompts、3 筆深度協作案例、IEEE 754 精度錯誤除錯紀錄
├── output/                        # 回測成果輸出檔
│   ├── backtest_summary.csv       # 策略各情境績效總覽表
│   └── backtest_chart.png         # 策略累積收益與最大回撤視覺化圖表
├── requirements.txt               # 專案環境相依套件清單
└── README.md                      # 專案主說明文件 (本檔案)
```

---

## 二、團隊成員與分工職掌 (A～F)

| 分工代號 | 學號 | 姓名 | 負責模組 | 核心工作內容 |
| :---: | :---: | :---: | :---: | :--- |
| **A** | D1339769 | 陳廷軒 | `spec/` | 提出「融資減碼＋股價回撤」逆向投資假說，制定 1,000 萬流動性篩選規則與選股邏輯 |
| **B** | D1338890 | 張廷瑋 | `pipeline/` (Data) | TEJ 原始資料清洗、除權息日報酬率提取、Y9997 加權報酬指數基準對齊 |
| **C** | D1339520 | 徐靖洋 | `pipeline/` (Factor) | 實作 `build_factors.py`，設計 Point-in-Time 截面標準化演算法，產出因子分數表 |
| **D** | D1240089 | 羅珮岑 | `pipeline/` (Backtest) | 實作 `backtest.py` 與 `plot_results.py`，精確扣除手續費/稅/滑價，產出績效圖表 |
| **E** | D1222382 | 彭禹喬 | `tests/` | 撰寫 Behave BDD 規格與 steps 實作，進行防 bfill 補值、連續缺值與浮點數精度測試 |
| **F** | D1299218 | 陳紋康 | 成果整合 & `ai_log/` | 統一專案架構、建立 GitHub 共同管理機制、彙整 AI 協作日誌與期末簡報大綱 |

---

## 三、環境建置與套件安裝

本專案支援 **Python 3.10+** 環境，建議使用虛擬環境隔離依賴：

```bash
# 1. 建立虛擬環境
python3 -m venv .venv

# 2. 啟動虛擬環境
# macOS / Linux:
source .venv/bin/activate
# Windows:
.venv\Scripts\Activate.ps1

# 3. 安裝依賴套件
pip install -r requirements.txt
```

*相依套件清單 (`requirements.txt`)：*
```text
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
pytest>=7.3.0
behave>=1.2.6
openpyxl>=3.1.0
```

---

## 四、如何執行 Pipeline（策略管線）

### 步驟 1：計算因子特徵與產出分數表 (`pipeline.build_factors`)
讀取 TEJ 原始資料，計算每月底融資變化率與 60 日回撤，並完成橫截面標準化與排序：
```bash
python -m pipeline.build_factors --price data/tej/還原股價.csv --margin data/tej/還原股價.csv --out data/factors/factor_scores.csv
```

### 步驟 2：執行月調倉回測 (`pipeline.backtest`)
依據因子分數表執行 2015/01～2022/12 樣本內回測，考慮整張買入與逐筆交易摩擦成本：
```bash
python -m pipeline.backtest --price data/tej/還原股價.csv --factors data/factors/factor_scores.csv --outdir output
```

### 步驟 3：產出績效雙圖表 (`pipeline.plot_results`)
產出資產成長曲線與水深回撤圖，自動保存至 `output/backtest_chart.png`：
```bash
python -m pipeline.plot_results --outdir output --capital 10000000
```

---

## 五、如何執行 Tests（測試驗證）

本專案具備完整且自動化的測試套件：

### 1. 執行 Pytest 單元測試（21 項測試）
驗證因子計算正確性、Point-in-Time 防偷看未來、除權息校正與回測資金恆等式：
```bash
pytest -q tests/
```
*主要測試涵蓋：*
* `test_no_lookahead`：切斷訊號日後資料，檢驗因子數值與排名 100% 相同（誤差 $< 10^{-15}$）。
* `test_drawdown_ignores_ex_dividend_drop`：確認除息日收盤價跳空不計為損失。
* `test_equity_identity_post_equals_pre_minus_costs`：驗證調倉後淨值完全等於調倉前扣除摩擦成本。

### 2. 執行 Behave BDD 行為驅動合規測試（6 大情境，21 個步驟）
驗證量化交易系統之時序公告隔離、缺失值清洗禁止 bfill、連續缺值排除與交易成本取整計算：
```bash
behave spec/quant_rules.feature
```
*測試執行結果全數通過（詳見 `tests/behave_test_passed.png`）：*
* **1 feature passed, 0 failed, 0 skipped**
* **6 scenarios passed, 0 failed, 0 skipped**
* **21 steps passed, 0 failed, 0 skipped**

---

## 六、回測實證結果與檢討

* **回測期間**：2015-01-05 ～ 2022-12-30（96 個月）
* **起始本金**：新臺幣 1,000 萬元
* **核心績效總覽**：
  * **策略（扣完整成本）**：累積報酬率 **-57.52%** ｜ 年化報酬率 **-10.15%** ｜ 夏普比率 **-0.22** ｜ 最大回撤 **-75.24%** ｜ 期末資產 **424.8 萬元**
  * **策略（不扣成本）**：累積報酬率 **-25.42%** ｜ 年化報酬率 **-3.60%** ｜ 夏普比率 **0.02** ｜ 最大回撤 **-64.42%** ｜ 期末資產 **745.8 萬元**
  * **參考基準（股票池等權）**：累積報酬率 **+82.38%** ｜ 年化報酬率 **+7.80%** ｜ 夏普比率 **0.47** ｜ 最大回撤 **-24.44%** ｜ 期末資產 **1,823.8 萬元**

### 實證檢討重點
1. **交易成本拖累（年化高達 7.08%）**：每月選前 10 檔等權，單邊平均月換手率達 86.20%（年化換手逾 10 倍），摩擦成本累計吃掉 32% 以上的淨值。
2. **逆向選股的價值陷阱**：融資斷頭與暴跌股票往往伴隨公司基本面永久損害，回測期間踩到 3 檔下市股（佳營、齊民、華美）。
3. **優化建議**：引入動能與財務健全度濾網，並增設換手緩衝區（Buffer Zone）以壓低調倉頻率。
