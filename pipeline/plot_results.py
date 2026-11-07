"""把回測結果畫成一張圖：上圖＝資產成長，下圖＝回撤。

用法：python -m src.plot_results --outdir output
資料來源：backtest_curves.csv（月資料，起點 = 1,000 萬元）
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager

# 驗證過的配色（藍、橘、青綠；前三格在 all-pairs 下通過 CVD 與一般視力門檻）
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

SERIES = [  # (欄位名稱, 顯示名稱, 顏色, 線型)
    ("主要結果（扣成本）", "策略（扣成本）", BLUE, "-"),
    ("不扣成本", "策略（不扣成本）", ORANGE, "-"),
    ("參考：股票池等權（未扣成本，非加權指數）", "參考：股票池等權（非加權指數）", AQUA, "-"),
]


def pick_font() -> str:
    for name in ("Noto Sans CJK TC", "Noto Sans CJK JP", "Microsoft JhengHei", "PingFang TC", "Arial Unicode MS"):
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            return name
    return "sans-serif"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="output")
    ap.add_argument("--capital", type=float, default=10_000_000)
    a = ap.parse_args()
    out = Path(a.outdir)

    c = pd.read_csv(out / "backtest_curves.csv", index_col="date", parse_dates=True)
    plt.rcParams["font.family"] = pick_font()
    plt.rcParams["axes.unicode_minus"] = False

    fig, (ax, bx) = plt.subplots(
        2, 1, figsize=(11, 7.2), dpi=200, sharex=True,
        gridspec_kw=dict(height_ratios=[3, 1.6], hspace=0.12), facecolor=SURFACE,
    )
    for axis in (ax, bx):
        axis.set_facecolor(SURFACE)
        for s in ("top", "right", "left"):
            axis.spines[s].set_visible(False)
        axis.spines["bottom"].set_color(GRID)
        axis.grid(axis="y", color=GRID, linewidth=0.8)
        axis.tick_params(colors=MUTED, length=0, labelsize=10)

    cap_wan = a.capital / 10_000  # 萬元
    ax.axhline(cap_wan, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(pd.Timestamp("2021-02-01"), cap_wan * 0.98, "本金 1,000 萬", color=MUTED, fontsize=9, va="top")

    ends = []
    for col, label, color, ls in SERIES:
        v = c[col] * cap_wan
        ax.plot(v.index, v.values, color=color, linewidth=2, linestyle=ls, label=label, solid_capstyle="round")
        dd = c[col] / c[col].cummax() - 1
        bx.plot(dd.index, dd.values * 100, color=color, linewidth=2, linestyle=ls)
        ends.append((v.iloc[-1], label, color, dd.min() * 100))

    # 直接標示線尾（避免文字重疊：依數值排序後拉開間距）
    ends.sort(key=lambda x: x[0])
    ymin, ymax = ax.get_ylim()
    gap = (ymax - ymin) * 0.06
    placed = []
    for val, label, color, _ in ends:
        y = val if not placed else max(val, placed[-1] + gap)
        placed.append(y)
        ax.annotate(f"{label}  {val:,.0f} 萬", xy=(c.index[-1], val), xytext=(8, 0), textcoords="offset points",
                    color=INK, fontsize=9.5, va="center", annotation_clip=False)
        ax.plot([c.index[-1]], [val], marker="o", markersize=6, color=color, markeredgecolor=SURFACE, markeredgewidth=1.5, clip_on=False)

    ax.set_ylabel("資產價值（萬元）", color=MUTED, fontsize=10)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    bx.set_ylabel("自高點回撤（%）", color=MUTED, fontsize=10)
    bx.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    bx.axhline(0, color=GRID, linewidth=1)

    leg = ax.legend(loc="upper left", frameon=False, fontsize=10, labelcolor=INK)
    fig.suptitle("融資減碼 + 股價回撤 逆向選股：樣本內回測（2015–2022，月資料）",
                 x=0.06, ha="left", fontsize=14, color=INK, fontweight="bold", y=0.975)
    fig.text(0.06, 0.915, "每月選綜合分數前 10 檔等權；扣手續費、證交稅、滑價。虛線＝本金。參考線為股票池等權，不是臺灣加權報酬指數。",
             fontsize=9.5, color=MUTED, ha="left")
    fig.subplots_adjust(left=0.08, right=0.74, top=0.88, bottom=0.07)
    fig.savefig(out / "backtest_chart.png", facecolor=SURFACE)
    print("saved", out / "backtest_chart.png")


if __name__ == "__main__":
    main()
