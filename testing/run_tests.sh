#!/bin/bash

# LogicConeMiner 測試腳本
# 用於執行標準測試和 CSV 元件庫測試

echo "=== LogicConeMiner 測試腳本 ==="
echo

# 設定測試參數
CONE_FINDER="../cone_finder.py"
CELL_LIBRARY="../cell_library.csv"

# 檢查必要檔案存在
if [ ! -f "$CONE_FINDER" ]; then
    echo "錯誤：找不到 cone_finder.py"
    exit 1
fi

if [ ! -f "$CELL_LIBRARY" ]; then
    echo "錯誤：找不到 cell_library.csv"
    exit 1
fi

echo "1. 執行標準測試電路（不使用 CSV 元件庫）"
echo "指令：python3 $CONE_FINDER --netlist test_circuit.v --n_in 3 --n_out 2 --n_depth 5 --out_dir test_results"
echo

python3 "$CONE_FINDER" \
  --netlist test_circuit.v \
  --n_in 3 --n_out 2 --n_depth 5 \
  --out_dir test_results

if [ $? -eq 0 ]; then
    echo "✅ 標準測試完成"
else
    echo "❌ 標準測試失敗"
    exit 1
fi

echo
echo "2. 執行 Macro 測試電路（使用 CSV 元件庫）"
echo "指令：python3 $CONE_FINDER --netlist test_with_macro.v --cell_library $CELL_LIBRARY --n_in 4 --n_out 2 --n_depth 10 --out_dir test_macro_results"
echo

python3 "$CONE_FINDER" \
  --netlist test_with_macro.v \
  --cell_library "$CELL_LIBRARY" \
  --n_in 4 --n_out 2 --n_depth 10 \
  --out_dir test_macro_results

if [ $? -eq 0 ]; then
    echo "✅ Macro 測試完成"
else
    echo "❌ Macro 測試失敗"
    exit 1
fi

echo
echo "=== 所有測試完成 ==="
echo

# 顯示結果摘要
echo "測試結果摘要："
echo "- 標準測試結果：test_results/"
echo "- Macro 測試結果：test_macro_results/"
echo

if [ -f "test_results/summary.json" ]; then
    echo "標準測試發現的邏輯錐數量："
    grep "total_cones" test_results/summary.json
fi

if [ -f "test_macro_results/summary.json" ]; then
    echo "Macro 測試發現的邏輯錐數量："
    grep "total_cones" test_macro_results/summary.json
fi