#!/usr/bin/env python3
"""
CSV 標準元件庫解析器
使用 CSV 檔案定義標準元件資訊，避免直接使用機密的 Liberty 檔案
"""

import csv
import os
from typing import Dict, List, Set, Optional
from dataclasses import dataclass

@dataclass
class CellDefinition:
    """標準元件定義"""
    name: str
    cell_type: str  # "combinational" or "sequential"
    input_pins: List[str]
    output_pins: List[str]
    is_sequential: bool
    clock_pin: Optional[str] = None
    data_pin: Optional[str] = None

class CSVCellLibrary:
    """CSV 標準元件庫解析器"""

    def __init__(self, csv_file: str = None):
        self.cells: Dict[str, CellDefinition] = {}

        if csv_file and os.path.exists(csv_file):
            self.load_from_csv(csv_file)
        else:
            # 如果沒有 CSV 檔案，載入內建的基本定義
            self._load_builtin_definitions()

    def load_from_csv(self, csv_file: str) -> None:
        """從 CSV 檔案載入元件定義"""
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)

                for row in reader:
                    cell_name = row['cell_name'].strip()
                    cell_type = row['cell_type'].strip()

                    # 解析輸入埠（逗號分隔）
                    input_pins = [pin.strip() for pin in row['input_pins'].split(',') if pin.strip()]

                    # 解析輸出埠（逗號分隔）
                    output_pins = [pin.strip() for pin in row['output_pins'].split(',') if pin.strip()]

                    # 解析是否為時序元件
                    is_sequential = row['is_sequential'].strip().lower() == 'true'

                    # 解析時鐘埠和資料埠
                    clock_pin = row.get('clock_pin', '').strip() or None
                    data_pin = row.get('data_pin', '').strip() or None

                    cell_def = CellDefinition(
                        name=cell_name,
                        cell_type=cell_type,
                        input_pins=input_pins,
                        output_pins=output_pins,
                        is_sequential=is_sequential,
                        clock_pin=clock_pin,
                        data_pin=data_pin
                    )

                    self.cells[cell_name] = cell_def

            print(f"從 {csv_file} 載入 {len(self.cells)} 個標準元件定義")

        except Exception as e:
            print(f"載入 CSV 檔案失敗: {e}")
            print("使用內建的基本元件定義")
            self._load_builtin_definitions()

    def _load_builtin_definitions(self) -> None:
        """載入內建的基本元件定義（降級方案）"""
        builtin_cells = [
            # 基本組合邏輯閘
            CellDefinition("AND2", "combinational", ["A", "B"], ["Y"], False),
            CellDefinition("OR2", "combinational", ["A", "B"], ["Y"], False),
            CellDefinition("NAND2", "combinational", ["A", "B"], ["Y"], False),
            CellDefinition("NOR2", "combinational", ["A", "B"], ["Y"], False),
            CellDefinition("XOR2", "combinational", ["A", "B"], ["Y"], False),
            CellDefinition("NOT", "combinational", ["A"], ["Y"], False),
            CellDefinition("BUF", "combinational", ["A"], ["Y"], False),

            # 時序元件
            CellDefinition("DFF", "sequential", ["D", "CLK"], ["Q"], True, "CLK", "D"),
            CellDefinition("DFFR", "sequential", ["D", "CLK", "R"], ["Q"], True, "CLK", "D"),
            CellDefinition("DLAT", "sequential", ["D", "E"], ["Q"], True, "E", "D"),
        ]

        for cell in builtin_cells:
            self.cells[cell.name] = cell

        print(f"載入 {len(self.cells)} 個內建標準元件定義")

    def is_standard_cell(self, cell_name: str) -> bool:
        """檢查是否為標準元件"""
        return cell_name in self.cells

    def get_cell_definition(self, cell_name: str) -> Optional[CellDefinition]:
        """取得元件定義"""
        return self.cells.get(cell_name, None)

    def is_sequential(self, cell_name: str) -> bool:
        """檢查是否為時序元件"""
        cell_def = self.get_cell_definition(cell_name)
        return cell_def.is_sequential if cell_def else False

    def get_input_pins(self, cell_name: str) -> List[str]:
        """取得輸入埠列表"""
        cell_def = self.get_cell_definition(cell_name)
        return cell_def.input_pins if cell_def else []

    def get_output_pins(self, cell_name: str) -> List[str]:
        """取得輸出埠列表"""
        cell_def = self.get_cell_definition(cell_name)
        return cell_def.output_pins if cell_def else []

    def get_clock_pin(self, cell_name: str) -> Optional[str]:
        """取得時鐘埠名稱"""
        cell_def = self.get_cell_definition(cell_name)
        return cell_def.clock_pin if cell_def else None

    def get_data_pin(self, cell_name: str) -> Optional[str]:
        """取得資料埠名稱"""
        cell_def = self.get_cell_definition(cell_name)
        return cell_def.data_pin if cell_def else None

    def list_all_cells(self) -> List[str]:
        """列出所有已定義的標準元件"""
        return list(self.cells.keys())

    def export_template_csv(self, output_file: str) -> None:
        """匯出 CSV 範本檔案"""
        header = ['cell_name', 'cell_type', 'input_pins', 'output_pins',
                 'is_sequential', 'clock_pin', 'data_pin']

        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)

            # 寫入一些範例
            examples = [
                ['AND2', 'combinational', 'A,B', 'Y', 'false', '', ''],
                ['DFF', 'sequential', 'D,CLK', 'Q', 'true', 'CLK', 'D'],
                ['MUX2', 'combinational', 'A,B,S', 'Y', 'false', '', ''],
            ]

            for example in examples:
                writer.writerow(example)

        print(f"CSV 範本已匯出至 {output_file}")

class MacroHandler:
    """Macro（非標準元件）處理器"""

    def __init__(self, cell_library: CSVCellLibrary):
        self.cell_library = cell_library
        self.detected_macros: Set[str] = set()

    def is_macro(self, cell_name: str) -> bool:
        """判斷是否為 Macro（非標準元件）"""
        return not self.cell_library.is_standard_cell(cell_name)

    def handle_macro_instance(self, instance_name: str, cell_type: str, connections: Dict) -> Dict:
        """處理 Macro 實例，將其視為斷點"""
        if cell_type not in self.detected_macros:
            self.detected_macros.add(cell_type)
            print(f"檢測到 Macro: {cell_type} (實例: {instance_name})")

        # 將 Macro 的所有連接視為斷點
        # 輸入連接 → 視為 PO（從前級邏輯的角度）
        # 輸出連接 → 視為 PI（對後級邏輯的角度）

        macro_boundaries = {
            'inputs_as_po': [],   # Macro 輸入視為 PO
            'outputs_as_pi': []   # Macro 輸出視為 PI
        }

        # 分析連接，推斷輸入輸出
        for port, net in connections.items():
            # 簡單啟發式：常見的輸出埠名稱
            if port.upper() in ['Y', 'Z', 'Q', 'OUT', 'OUTPUT']:
                macro_boundaries['outputs_as_pi'].append({
                    'node_id': f"{instance_name}.{port}",
                    'net': net,
                    'type': 'MACRO_OUTPUT_AS_PI'
                })
            else:
                macro_boundaries['inputs_as_po'].append({
                    'node_id': f"{instance_name}.{port}",
                    'net': net,
                    'type': 'MACRO_INPUT_AS_PO'
                })

        return macro_boundaries

    def get_detected_macros(self) -> Set[str]:
        """取得所有檢測到的 Macro 類型"""
        return self.detected_macros.copy()

# 測試和使用範例
if __name__ == "__main__":
    # 測試 CSV 元件庫
    print("=== 測試 CSV 標準元件庫 ===")

    # 使用 CSV 檔案
    lib = CSVCellLibrary("cell_library.csv")

    # 測試一些功能
    print("\n標準元件檢查:")
    test_cells = ["AND2", "DFF", "MY_CUSTOM_BLOCK"]
    for cell in test_cells:
        is_std = lib.is_standard_cell(cell)
        print(f"  {cell}: {'標準元件' if is_std else 'Macro'}")

        if is_std:
            cell_def = lib.get_cell_definition(cell)
            print(f"    輸入埠: {cell_def.input_pins}")
            print(f"    輸出埠: {cell_def.output_pins}")
            print(f"    時序: {'是' if cell_def.is_sequential else '否'}")

    # 測試 Macro 處理
    print("\n=== 測試 Macro 處理 ===")
    macro_handler = MacroHandler(lib)

    # 模擬一個 Macro 實例
    macro_connections = {
        'data_in': 'net1',
        'clock': 'clk',
        'enable': 'en',
        'data_out': 'net2'
    }

    boundaries = macro_handler.handle_macro_instance(
        "cpu_core_inst", "CPU_CORE", macro_connections
    )

    print("Macro 邊界處理結果:")
    print(f"  輸入作為 PO: {boundaries['inputs_as_po']}")
    print(f"  輸出作為 PI: {boundaries['outputs_as_pi']}")

    # 匯出範本
    lib.export_template_csv("cell_library_template.csv")