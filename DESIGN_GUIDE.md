# LogicConeMiner 設計指南

## 總覽

LogicConeMiner 是一個專門用於門級 Verilog netlists 邏輯錐發現的工具。本文件詳細說明系統架構、模組設計及實作細節。

## 架構更新歷史

### v3.0 - 模組化重構 (最新版本)

**主要改進：**
- ✅ 完全模組化架構，將單體 `cone_finder.py` 拆分為專門模組
- ✅ 增強的 Verilog 解析器，支援複雜識別字和多模組
- ✅ Black Box 模組自動檢測與處理
- ✅ 改進的 CSV 元件庫與 Macro 處理機制
- ✅ 更強的錯誤處理和安全性驗證

**檔案結構：**
```
LogicConeMiner/
├── cone_finder.py          # 主程式入口
├── verilog_parser.py       # Verilog 解析模組
├── graph_builder.py        # 電路圖建構模組
├── cone_enumerator.py      # 錐枚舉演算法模組
├── output_writer.py        # 輸出格式化模組
├── data_structures.py      # 核心資料結構
├── csv_cell_library.py     # CSV 元件庫處理
└── __init__.py            # Python 套件檔案
```

---

## 核心模組設計

### 1. VerilogParser 模組 (`verilog_parser.py`)

**責任：** Verilog 檔案解析、多模組處理、Black Box 檢測

#### 1.1 類別設計

```python
class VerilogParser:
    """增強的 Verilog 解析器，支援多模組和 CSV 元件庫"""

    # 識別字模式：支援一般識別字和轉義識別字
    IDENTIFIER_PATTERN = r'(?:\\[^\\]+\s|[\w$]+)'

    def __init__(self, csv_library: CSVCellLibrary = None)
    def parse_file(self, filename: str) -> None
    def is_sequential(self, cell_type: str) -> bool
    def is_module_instantiation(self, cell_type: str) -> bool
    def is_black_box_module(self, cell_type: str) -> bool
```

#### 1.2 識別字支援

**支援的 Verilog 識別字類型：**

1. **一般識別字**：`[\w$]+`
   - 範例：`clk`, `data_bus`, `result$out`
   - 字母、數字、底線、$ 字元組合

2. **轉義識別字**：`\\[^\\]+\s`
   - 範例：`\clock-signal `, `\data bus `, `\result.out `
   - 反斜線開頭，可包含空格和特殊字元，以空格結尾

**正規化處理：**
```python
def _normalize_identifier(self, identifier: str) -> str:
    """正規化轉義識別字，移除前導反斜線和尾隨空格"""
    if identifier.startswith('\\'):
        return identifier[1:].rstrip()
    return identifier
```

#### 1.3 多模組處理

**模組分類策略：**

- **標準模組**：包含標準元件或混合元件的模組
- **Macro 定義**：完全由非標準元件組成的模組
- **Black Box 模組**：被實例化但未定義的模組

```python
def _is_module_macro(self, module_def: ModuleDefinition) -> bool:
    """判斷模組是否應被視為 Macro"""
    total_cells = len(module_def.cells)
    if total_cells == 0:
        return False

    non_standard_cells = 0
    for _, (cell_type, _) in module_def.cells.items():
        if not self.csv_library or not self.csv_library.is_standard_cell(cell_type):
            non_standard_cells += 1

    # 全部都是非標準元件才視為 Macro
    return non_standard_cells == total_cells
```

#### 1.4 Black Box 檢測

**自動檢測機制：**
1. 收集所有被實例化的模組名稱
2. 檢查哪些模組未被定義
3. 自動標記為 Black Box 並發出警告
4. 推斷 Black Box 模組的 port 方向

```python
def detect_black_box_modules(self) -> None:
    """檢測 black box 模組並顯示警告"""
    all_instantiated_modules = set()

    # 收集所有實例化的模組
    for module_def in self.modules.values():
        for _, (cell_type, _) in module_def.cells.items():
            if not self.csv_library or not self.csv_library.is_standard_cell(cell_type):
                all_instantiated_modules.add(cell_type)

    # 檢查未定義的模組
    for module_name in all_instantiated_modules:
        if (module_name not in self.modules and
            module_name not in self.macro_definitions):
            self.black_box_modules.add(module_name)
            logger.warning(f"⚠️  ALERT: Module '{module_name}' is undefined, will be treated as Black Box")
```

#### 1.5 Port 方向推斷

**Black Box 模組 Port 方向推斷邏輯：**

```python
def infer_black_box_port_direction(self, instance_name: str, cell_type: str,
                                 connections: Dict[str, str],
                                 module_def: ModuleDefinition) -> Dict[str, str]:
    """推斷 black box 模組的 port 方向"""
    port_directions = {}

    for port, net in connections.items():
        has_other_driver = False
        is_driven_by_pi = False

        # 檢查其他實例是否驅動此 net
        for other_instance, (other_cell_type, other_connections) in module_def.cells.items():
            if other_instance == instance_name:
                continue

            if self.csv_library and self.csv_library.is_standard_cell(other_cell_type):
                output_ports = self.csv_library.get_output_pins(other_cell_type)
                for out_port, out_net in other_connections.items():
                    if out_port in output_ports and out_net == net:
                        has_other_driver = True
                        break

        # 檢查是否由主要輸入驅動
        if net in module_def.input_ports:
            is_driven_by_pi = True

        # 推斷方向
        if has_other_driver or is_driven_by_pi:
            port_directions[port] = 'input'
        else:
            port_directions[port] = 'output'

    return port_directions
```

#### 1.6 解析流程

**主要解析流程：**

1. **檔案前處理**
   - 移除註解 (`//` 和 `/* */`)
   - 處理編碼問題 (UTF-8/Latin-1)

2. **模組提取**
   - 使用正規表達式提取所有 `module...endmodule` 區塊
   - 支援複雜的模組名稱（包含轉義識別字）

3. **Port 宣告解析**
   - **從模組標頭解析**：`module name(input clk, output data);`
   - **從模組內容解析**：`input [7:0] addr; output result;`
   - 自動去重並保持順序

4. **實例化解析**
   - 解析元件實例：`cell_type instance_name ( connections );`
   - 解析連線：`.port(net)` 格式
   - 支援所有類型的識別字

5. **模組分類**
   - 根據元件組成判斷是否為 Macro
   - 檢測並處理 Black Box 模組

#### 1.7 錯誤處理

**健全性檢查：**

- **檔案存在性檢查**
- **編碼容錯處理**：優先 UTF-8，降級至 Latin-1
- **CSV 元件庫可選性**：未提供時自動降級
- **語法容錯性**：跳過無法解析的語法結構
- **模組依賴檢查**：檢測循環依賴和未定義模組

---

### 2. GraphBuilder 模組 (`graph_builder.py`)

**責任：** 電路圖建構、時序切割、區塊分割

#### 2.1 核心功能

- **節點建立**：為 PI/PO、元件輸出建立圖節點
- **連線建構**：根據 net 連線建立有向邊
- **時序切割**：自動在 FF/Latch 邊界切斷路徑
- **區塊分割**：找出組合邏輯連通元件
- **拓撲層級化**：為每個區塊內的節點分配層級

#### 2.2 多模組支援

**模組圖建構器：**
```python
class ModuleGraphBuilder(GraphBuilder):
    """支援多模組的圖建構器"""

    def build_graphs_for_all_modules(self) -> Dict[str, 'ModuleGraphBuilder']:
        """為所有模組建構獨立的圖"""
        module_graphs = {}

        for module_name, module_def in self.parser.modules.items():
            if not self._should_process_module(module_def):
                continue

            graph = ModuleGraphBuilder(self.parser, module_name)
            graph.build_module_graph(module_def)
            module_graphs[module_name] = graph

        return module_graphs
```

---

### 3. ConeEnumerator 模組 (`cone_enumerator.py`)

**責任：** 錐枚舉演算法、約束驗證、多輸出處理

#### 3.1 演算法支援

- **單輸出錐**：k-feasible cuts 動態規劃演算法
- **多輸出錐**：支撐共享群組化策略
- **約束檢查**：輸入數量、輸出數量、深度約束
- **連通性驗證**：確保錐形成單一連通元件

#### 3.2 多模組錐枚舉

```python
class ModuleConeEnumerator(ConeEnumerator):
    """支援多模組的錐枚舉器"""

    def enumerate_cones_for_module(self, module_name: str,
                                 graph: 'ModuleGraphBuilder') -> List[Cone]:
        """為特定模組枚舉錐"""
        logger.info(f"開始為模組 '{module_name}' 枚舉錐")

        all_cones = []
        for block_id, block in enumerate(graph.blocks):
            # 單輸出錐
            self.enumerate_single_root_cones(block_id, block, module_name)

            # 多輸出錐
            if self.config['n_out'] > 1:
                self.enumerate_multi_root_cones(block_id, block, module_name)

        return self.discovered_cones
```

---

### 4. OutputWriter 模組 (`output_writer.py`)

**責任：** 格式化輸出、統計生成、視覺化支援

#### 4.1 輸出格式

**JSONL 格式** (`cones.jsonl`)：
```json
{"cone_id": "abc123", "module_name": "cpu_core", "block_id": 0, "roots": ["n1"], "leaves": ["n5", "n6"], "depth": 3, "num_nodes": 8, "num_edges": 12, "connected": true, "signature": "sha256hash"}
```

**摘要格式** (`summary.json`)：
```json
{
    "total_cones": 1250,
    "modules": {
        "cpu_core": {"cones": 800, "blocks": 15},
        "cache_ctrl": {"cones": 450, "blocks": 8}
    },
    "constraint_distributions": {
        "by_inputs": {"1": 300, "2": 450, "3": 350, "4": 150},
        "by_outputs": {"1": 1000, "2": 250},
        "by_depth": {"1-5": 600, "6-10": 450, "11-15": 200}
    }
}
```

#### 4.2 多模組支援

```python
class ConeOutputWriter:
    """錐輸出格式化器，支援多模組"""

    def write_cones_by_module(self, module_cones: Dict[str, List[Cone]]) -> None:
        """按模組寫入錐資料"""
        for module_name, cones in module_cones.items():
            module_file = self.out_dir / f"cones_{module_name}.jsonl"
            self._write_cones_to_file(cones, module_file)
```

---

### 5. 資料結構 (`data_structures.py`)

#### 5.1 核心資料結構

**ModuleDefinition**：
```python
@dataclass
class ModuleDefinition:
    name: str
    input_ports: List[str] = field(default_factory=list)
    output_ports: List[str] = field(default_factory=list)
    cells: Dict[str, Tuple[str, Dict]] = field(default_factory=dict)
    # instance_name -> (cell_type, connections)
```

**Node**：
```python
@dataclass
class Node:
    id: str
    type: str
    fanin: List[str] = field(default_factory=list)
    fanout: List[str] = field(default_factory=list)
    is_seq: bool = False
    is_pi: bool = False
    is_po: bool = False
    level: int = -1
    module_name: str = ""  # 新增：所屬模組
```

**Cone**：
```python
@dataclass
class Cone:
    cone_id: str
    block_id: int
    roots: List[str]
    leaves: List[str]
    depth: int
    num_nodes: int
    num_edges: int
    connected: bool
    signature: str
    module_name: str = ""  # 新增：所屬模組
    nodes: Set[str] = field(default_factory=set)
```

---

## 設計原則

### 1. 模組化設計
- **單一責任原則**：每個模組負責明確定義的功能
- **低耦合**：模組間透過明確介面互動
- **高內聚**：相關功能集中在同一模組

### 2. 可擴展性
- **多模組支援**：可處理複雜的多模組設計
- **演算法擴展**：易於新增新的錐枚舉演算法
- **輸出格式擴展**：支援多種輸出格式

### 3. 健全性
- **錯誤容錯**：各級錯誤處理和降級機制
- **輸入驗證**：嚴格的輸入參數和檔案格式檢查
- **記錄機制**：詳細的執行記錄和警告資訊

### 4. 效能考量
- **記憶體控制**：透過 `max_cuts_per_node` 控制記憶體使用
- **平行處理**：模組間可獨立處理，支援平行化
- **演算法最佳化**：使用高效的動態規劃和圖演算法

---

## 使用範例

### 基本多模組處理

```python
from verilog_parser import VerilogParser
from graph_builder import ModuleGraphBuilder
from cone_enumerator import ModuleConeEnumerator
from csv_cell_library import CSVCellLibrary

# 設定 CSV 元件庫
csv_lib = CSVCellLibrary("standard_cells.csv")

# 解析 Verilog 檔案
parser = VerilogParser(csv_lib)
parser.parse_file("design.v")

# 為每個模組建構圖並枚舉錐
config = {'n_in': 4, 'n_out': 2, 'n_depth': 8}
enumerator = ModuleConeEnumerator(None, config)

all_cones = {}
for module_name, module_def in parser.modules.items():
    graph = ModuleGraphBuilder(parser, module_name)
    graph.build_module_graph(module_def)
    cones = enumerator.enumerate_cones_for_module(module_name, graph)
    all_cones[module_name] = cones

# 輸出結果
writer = ConeOutputWriter("output/")
writer.write_cones_by_module(all_cones)
```

---

## 效能指標

### 支援規模
- **設計大小**：≥100k 節點
- **模組數量**：無限制（記憶體允許範圍內）
- **錐數量**：數萬個錐的高效處理

### 記憶體最佳化
- **Cut 控制**：`max_cuts_per_node` 參數
- **分塊處理**：按模組和區塊分別處理
- **垃圾回收**：適時釋放不需要的資料結構

### 執行時間
- **大型設計**：通常在分鐘級別完成
- **平行處理**：模組間可平行處理
- **進度回報**：詳細的進度和效能記錄

---

## 後續發展

### 短期目標
- **GUI 介面**：視覺化設計和結果檢視
- **更多輸出格式**：支援 XML、CSV 等格式
- **效能最佳化**：進一步減少記憶體使用和執行時間

### 長期目標
- **時序分析**：加入時序錐分析功能
- **功率分析**：錐級功率估計
- **可測試性分析**：基於錐的可測試性分析

---

*最後更新：2025年1月*