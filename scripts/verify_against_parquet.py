#!/usr/bin/env python3
"""对比检验脚本: 读取 factor_win_parquet 产出的 win.parquet, 打印各核心因子的统计指标,
便于与 FluxCompute (xlt_factor_example) 输出的控制台统计表直接对账。

用法:
    python3 scripts/verify_against_parquet.py <path/to/win.parquet>
"""
import sys
import os

try:
    import polars as pl
except ImportError:
    print("需要安装 polars: pip install polars", file=sys.stderr)
    sys.exit(1)


def main():
    if len(sys.argv) < 2:
        print("用法: python3 scripts/verify_against_parquet.py <path/to/win.parquet>")
        sys.exit(1)

    pq_path = sys.argv[1]
    if not os.path.exists(pq_path):
        print(f"找不到文件: {pq_path}", file=sys.stderr)
        sys.exit(1)

    print("=" * 80)
    print(f"  Polars 对账检验: {pq_path}")
    print("=" * 80)

    df = pl.read_parquet(pq_path)
    total_rows = df.height
    print(f"  总行数: {total_rows}, 列数: {len(df.columns)}")

    cols = [
        "trd_amt",
        "large_net_inflow",
        "net_inflow_ratio",
        "act_buy_share",
        "ord_imb",
        "cxl_ratio",
        "pct_to_limit",
    ]

    print("  ----------------------------------------------------------------------------")
    print(f"  {'Factor Name':<20}{'Valid Count':>12}{'NaN %':>10}{'Mean':>14}{'Min':>14}{'Max':>14}")
    print("  ----------------------------------------------------------------------------")

    for col in cols:
        if col not in df.columns:
            print(f"  {col:<20} [MISSING]")
            continue
        series = df[col].cast(pl.Float64)
        valid_cnt = series.is_not_null().sum()
        nan_cnt = total_rows - valid_cnt
        nan_pct = (nan_cnt / total_rows * 100.0) if total_rows > 0 else 0.0
        
        valid_s = series.drop_nulls()
        mean_v = valid_s.mean() if valid_cnt > 0 else 0.0
        min_v = valid_s.min() if valid_cnt > 0 else 0.0
        max_v = valid_s.max() if valid_cnt > 0 else 0.0

        print(f"  {col:<20}{valid_cnt:>12}{nan_pct:>9.1f}%{mean_v:>14.4f}{min_v:>14.4f}{max_v:>14.4f}")

    print("  ----------------------------------------------------------------------------\n")


if __name__ == "__main__":
    main()
