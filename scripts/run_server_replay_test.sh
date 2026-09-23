#!/usr/bin/env bash
# ==============================================================================
# FluxCompute 端到端全量回放与因子计算测试脚本
# 流程: 盛立 .dat 行情 -> xlt_replay 逐笔环 -> xlt_factor 窗口聚合 dump -> FluxCompute 张量计算
# ==============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"

DAT_FILE="${1:-/dev/shm/ult_efh_20260629.dat}"
DUMP_BIN="${2:-/dev/shm/win_raw.bin}"
T0="${3:-092500}"
T1="${4:-150000}"

QABROKER_DIR="/home/knight/cpp/qabroker_xlight/qamdbroker"
FACTOR_BUILD="$QABROKER_DIR/app/factor/build"
FLUX_BIN="$HERE/build/examples/xlt_factor/xlt_factor_example"

echo "=========================================================================="
echo "  FluxCompute: 服务器端 .dat 回放与高频因子计算测试流水线"
echo "=========================================================================="

# 1. 确保 FluxCompute 编译完成
if [ ! -x "$FLUX_BIN" ]; then
    echo "[Step 1] 正在编译 FluxCompute xlt_factor_example..."
    cmake -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build -j"$(nproc 2>/dev/null || echo 4)" --target xlt_factor_example
fi
echo "[Step 1] FluxCompute 因子计算引擎就绪: $FLUX_BIN"

# 2. 判断是否已有现成的 win_raw.bin
if [ -f "$DAT_FILE" ] && [[ "$DAT_FILE" == *.bin ]]; then
    echo "[Step 2] 输入参数直接为 win_raw.bin 归档 ($DAT_FILE)，跳过行情回放直接计算..."
    DUMP_BIN="$DAT_FILE"
elif [ -f "$DUMP_BIN" ] && [ "${SKIP_REPLAY:-0}" = "1" ]; then
    echo "[Step 2] 环境变量 SKIP_REPLAY=1，直接复用已有归档: $DUMP_BIN"
else
    echo "[Step 2] 检查 .dat 行情文件与回放引擎..."
    if [ ! -f "$DAT_FILE" ]; then
        echo "  [提示] 找不到行情文件: $DAT_FILE"
        echo "  若已单独生成 win_raw.bin，可直接运行:"
        echo "    $0 /path/to/win_raw.bin"
        echo "    或 $FLUX_BIN --bin /path/to/win_raw.bin"
        echo "  在无数据环境下，直接运行仿真校验:"
        "$FLUX_BIN"
        exit 0
    fi

    # 检查 xlt_replay 和 xlt_factor 可执行文件
    if [ ! -x "$FACTOR_BUILD/xlt_replay" ] || [ ! -x "$FACTOR_BUILD/xlt_factor" ]; then
        echo "  正在编译 qamdbroker factor app..."
        cmake -S "$QABROKER_DIR/app/factor" -B "$FACTOR_BUILD" >/dev/null 2>&1 || true
        cmake --build "$FACTOR_BUILD" -j >/dev/null 2>&1 || true
    fi

    echo "  清理历史共享内存与残留 dump: $DUMP_BIN..."
    rm -f "$DUMP_BIN" /dev/shm/xlt_stock_tick /dev/shm/xlt_factor /dev/shm/xlt_stock_md /dev/shm/xlt_index_md

    echo "  启动 xlt_replay 回放: $DAT_FILE ($T0 -> $T1)..."
    "$FACTOR_BUILD/xlt_replay" "$DAT_FILE" \
        --snap xlt_stock_md --idx xlt_index_md --tick xlt_stock_tick \
        -t0 "$T0" -t1 "$T1" -s 1 -b 22 >/tmp/xlt_replay.log 2>&1 &
    REP_PID=$!

    # 等待回放就绪
    for i in $(seq 1 60); do
        sleep 1
        grep -q '就绪' /tmp/xlt_replay.log 2>/dev/null && break
        kill -0 $REP_PID 2>/dev/null || { echo "  回放进程退出，日志:"; tail -5 /tmp/xlt_replay.log; exit 1; }
    done

    echo "  启动 xlt_factor 窗口聚合引擎并导出二进制归档 (--dump $DUMP_BIN)..."
    DAY_NUM="$(date +%Y%m%d)"
    "$FACTOR_BUILD/xlt_factor" xlt_stock_tick xlt_factor \
        --from-start --snap xlt_stock_md --dump "$DUMP_BIN" --day "$DAY_NUM" --exit-idle 5 >/tmp/xlt_factor.log 2>&1 &
    FAC_PID=$!

    wait $FAC_PID 2>/dev/null || true
    kill -TERM $REP_PID 2>/dev/null || true

    echo "  回放与窗口降采样归档完成: $(ls -lh "$DUMP_BIN" 2>/dev/null || echo '未生成')"
fi

# 3. 执行 FluxCompute 计算图流水线
echo ""
echo "[Step 3] 启动 FluxCompute 张量计算引擎处理真实回放数据..."
"$FLUX_BIN" --bin "$DUMP_BIN" --cores 2,3,4,5

echo ""
echo "=========================================================================="
echo "  测试流水线完成!"
echo "=========================================================================="
