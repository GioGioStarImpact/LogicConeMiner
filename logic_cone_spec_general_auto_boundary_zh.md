# 門級 Netlist 的 Logic Cone 探勘規格書
**模式：** General　•　**邊界：** 自動（以時序切割）　•　**Leaf Scope：** `any_in_block`  
**對象：** 實作工程師（C++/Rust/Python）

---

## 0) 目標與範圍
- **輸入：** 結構化 Verilog 的門級 netlist，包含標準邏輯閘、常數、觸發器（FF）、閘鎖（Latch）、頂層 I/O。
- **一般化 I/O：** Logic Cone 的 **輸出根（roots）** 與 **輸入葉/前緣（leaves/frontier）** 可為 **分析區塊內的任意節點**，不侷限於 PI/PO/FF 腳位。
- **自動邊界：** 不需手動提供邊界。以 **FF／CK-Latch／PI／PO** 為障壁進行 **時序切割**，產生組合邏輯的 **區塊（blocks / islands）**。所有 cone 的搜尋 **僅在單一 block 內** 進行，**不得跨越** FF／CK-Latch／PI／PO。
- **約束：**
  - `n_In`：cone **輸入葉**（frontier）在區塊內的數量。
  - `n_Out`：cone **輸出根** 的數量。
  - `n_Depth`：cone 子圖中，任一葉到任一根的 **最長路徑長度**（以 gate 數計）。
  - **不可拆解：** cone 的誘導子圖在 **無向視角** 下必須是 **單一連通元件**。
- **Leaf Scope：** `any_in_block` —— 葉可為 **區塊內任意節點**；但在該 cone 子圖內，該節點 **沒有前驅**（即為該 cone 的 **frontier**）。

---

## 1) 名詞與形式定義
- **Block（自動邊界）：** 在 netlist 上以 **FF／CK-Latch／PI／PO** 進行時序切割。移除所有跨越時序元件的組合邊後，對剩餘的 **組合節點** 以 **無向連通性** 分群；每個連通成分即為一個 **block**。所有 cone 的枚舉與檢查 **逐 block** 進行。
- **Roots \(R\)：** block 內任意節點的集合。可包含內部節點、FF.D、Latch.D，或通往 PO 的節點。
- **TFI\(_B\)(R)：** 在 block 所誘導之子圖 \(G[B]\) 中，**遞移扇入**：所有能沿著前向邊（逆向遍歷）到達任一 `R` 節點的上游節點集合。
- **Cone 子圖 \(Sub\)：** 對根集合 \(R\)，定義 \(Sub = \mathrm{TFI}_B(R)\)。
- **Frontier / Leaves \(L\)：** \(Sub\) 中 **在 \(Sub\) 內無前驅** 的節點集合。於 `any_in_block` 模式，這些葉可為 **內部節點**（不僅是 PI/FF.Q/Const）。
- **Depth（深度）：** 在 \(Sub\) 內，任一 \(l\in L\) 到任一 \(r\in R\) 的最長路徑長度。每個邏輯閘成本 = 1（是否將 `inv/buf` 計入可由參數控制）。
- **不可拆解（連通）：** 將 \(Sub\) 視為無向圖，其連通元件數必為 **1**。

---

## 2) 輸入與輸出

### 2.1 參數
- `n_In` *(int, 必填)* —— 葉數門檻。
- `n_Out` *(int, 必填)* —— 根數門檻。
- `n_Depth` *(int, 必填)* —— 最長路徑門檻。
- `cmp_in`, `cmp_out`, `cmp_depth` ∈ `{<=, ==}` *(預設：`<=`)*。
- `count_inverters_in_depth` *(bool, 預設：`true`)* —— 是否將 `inv/buf` 納入深度成本。
- `max_cuts_per_node = M` *(int, 預設：100–200)* —— 每節點最多保留的 cut 數（控制記憶體/時間）。
- `max_roots_per_block` *(int, 選配)* —— 每個 block 的根候選上限（擴充性旋鈕）。
- `max_grouping_degree` *(int, 預設：`n_Out`)* —— 多輸出群組 \(R\) 的最大大小。

> **無需** 手動邊界；邊界由時序切割自動推導。

### 2.2 檔案
- **輸入：** `netlist.v`
- **輸出：**
  - `cones.jsonl` —— 一行一個 cone：
    ```json
    {
      "cone_id": "hash",
      "block_id": 7,
      "roots": ["n123", "n456"],
      "leaves": ["n12", "n34"],
      "depth": 9,
      "num_nodes": 37,
      "num_edges": 41,
      "connected": true,
      "signature": "sig128"
    }
    ```
  - `summary.json` —— 數量與分佈（依 #leaves/#roots/depth），並含每個 block 的統計。
  - （選）`viz/*.dot` —— 每個 cone 的 Graphviz 圖（葉=菱形 gold、根=雙圈 lightblue、內部=方框 lightgray）。

---

## 3) 解析與建圖

1) **Verilog 解析**  
   - 支援結構化實例、net、常數連線；不強制解析行為式 `always`。
   - 依 Liberty 或白名單辨識 FF/Latch（如 `DFF*`, `SDFF*`, `DLAT*`）。

2) **節點模型**  
   - `Node { id, type, fanin[], fanout[], is_seq?, is_pi?, is_po? }`  
   - 每個 cell 的 **輸出腳位** 對應一個圖節點；依 net 建立 driver→sinks 邊。

3) **時序切割（自動邊界）**  
   - **FF.Q / Latch.Q** 視為 **來源（Pseudo-PI）**，**FF.D / Latch.D** 視為 **匯點（Pseudo-PO）**；**不穿越** 時序元件。
   - **PI** 視為來源；**PO** 視為匯點。
   - 移除跨越 FF/Latch 的組合邊（斷開 D→Q 的路徑）。
   - 非同步 set/reset：預設排除於資料路徑（可設為包含）。

4) **Block 分群**  
   - 在 **時序切割後的組合圖** 上，對組合節點做 **無向連通性**，得到多個 **block**。
   - 後續運算 **逐 block** 進行（不得跨 block）。

5) **層級與反向圖**  
   - 在每個 block 內，從 **block 來源**（在 block 內沒有前驅的節點）做拓撲分層：  
     `level(src)=0`；一般閘 `level(v)=1+max(level(fanin))`（若 `inv/buf` 不計成本則為 0）。
   - 建立 **反向圖** \(R\) 以便 TFI 走訪。

---

## 4) 單輸出 Cone：k-Feasible Cuts（`leaf_scope = any_in_block`）
- 對每個 **根候選** `r`（block 內任一節點），枚舉大小 ≤ `K = n_In` 的 **cut**；**葉可為任意內部節點**。
- **Cut 狀態**
  - `leaves`：葉集合，`|leaves| ≤ K`。
  - `depth`：在由 `leaves` 到 `r` 的 cone 子圖內的最長路徑。
  - **支配裁剪：** 若 `C1.leaves ⊆ C2.leaves` 且 `C1.depth ≤ C2.depth`，捨棄 `C2`。
- **DP 合併**
  - 依拓撲序掃描：  
    - 若 `v` 在 block 內 **無前驅**，`Cuts[v] = { {leaves={v}, depth=0} }`。  
    - 否則合併其 fanins 的 cuts（笛卡兒積），檢查 `|leaves| ≤ K`、`depth ≤ n_Depth`（若比較子為 `<=`），套用支配裁剪並以 `M` 截頂。
- **輸出（單根）**
  - 若 `|leaves| cmp_in n_In` 且 `depth cmp_depth n_Depth`，建立 `Sub` 為 **在 block 內** 所有 `leaves→r` 路徑的聯集。
  - 單根 cone 天生連通（路徑皆匯向 `r`）。
  - 產出 `{roots={r}, leaves, depth, ...}` 並進入去重。

> **早期剪枝：** 合併時計算暫時深度，超過 `n_Depth`（在 `<=` 模式）即丟；優先保留葉數較少／深度較淺的 cuts 抑制組合爆炸。

---

## 5) 多輸出 Cone 擴展（|R| ≤ `n_Out`）
為控制組合爆炸，先做 **支撐共享** 的預篩：

1) **根候選與共享篩選**  
   - 每個 block 內挑選根候選（預設為 **所有節點**，也可依扇出/層級/距離 PO 或 FF.D 進行限量）。  
   - 對每個根 `r`，估算其 **支撐 bitset**（例如 `TFI_B(r)` 的節點或來源 bitset）。僅當兩根的支撐 **有交集** 時，保留該 pair；再擴成較大群組，直到 `|R| ≤ n_Out`，同時維持交集啟發式。

2) **評估群組 R**  
   - `Sub = ⋃_{r∈R} TFI_B(r)`（僅在本 block）。  
   - `L = frontier(Sub)` —— 在 `Sub` 內無前驅的節點（**可能為內部點**）。  
   - `depth = max_{l∈L, r∈R} dist(l→r)`（只在 `Sub`）。  
   - **連通性：** `undirected(Sub)` 的連通元件數必為 **1**。  
   - **過濾：** 檢查 `|L| cmp_in n_In`、`|R| cmp_out n_Out`、`depth cmp_depth n_Depth`。  
   - 通過即輸出 `{roots=R, leaves=L, depth, ...}`。

> 即便葉可為任意內部點，在 `Sub` 內它們必須是 **最前緣**（在 `Sub` 內沒有前驅）。

---

## 6) 去重與簽名（Dedup）
**動機：** 相同的 cone 可能經由不同的枚舉路徑或不同的根群組被重複找到。  
**等價定義：** 若兩個 cone 的 **節點集合（Sub）** 與 **根集合（R）** 完全一致，視為同一 cone。

**簽名計算：**
- 對 `Nodes(Sub)` 排序 → `H_nodes = hash(list(Nodes))`。  
- 對 `Roots(R)` 排序 → `H_roots = hash(list(R))`。  
- `signature = H_nodes ⊕ rotate(H_roots, k)`（建議 128-bit；可選擇再加入排序後的邊集合雜湊）。  
- 生成 cone 時即計算簽名，首次出現則收錄；極少數碰撞時以精確集合比較確認。

---

## 7) 正確性與測試
- **封閉性：** 任一 `Sub` 內節點至任一 `r∈R` 的所有路徑 **皆留在 Sub**；任一 `l∈L` 在 `Sub` 內 **無前驅**。  
- **深度驗證：** 隨機抽樣 cone，以 DAG **最長路徑** 重新計算對照 DP 值。  
- **不可拆解：** `undirected(Sub)` 僅有 **1** 個連通元件。  
- **邊界：** `Sub` 之所有節點皆屬於 **同一 block**。  
- **回歸：** 大型設計（≥ 1e5 nodes）在不同 `M` 下時間/記憶體曲線穩定。

---

## 8) 複雜度與效能
- 主要成本：**cut 枚舉** 與 **多根群組**。  
- **控制手段：**  
  - `M` —— 每節點 cut 上限（100–200 為常用起點）。  
  - 根候選過濾 —— 依扇出、層級、至 PO/FF.D 距離等。  
  - 支撐交集預篩 —— 僅在支撐有交集時才合併。  
- **早期剪枝：** 在合併階段即檢查 `K`、`depth`、與比較子。  
- **資料結構：** 支撐與 `Sub` 採壓縮 **bitset**；葉集合用小型排序向量；cut 以物件池管理。  
- **平行化：** block 之間可並行；單一 block 內可對不同根（單輸出）與不同根群組（多輸出）並行。

---

## 9) 邊界政策與邊角案例
- **Latch：** 預設視為時序障壁（`Q`=來源、`D`=匯點），不建模時間借用。（可選：標記特定 latch 為透明，但有工程風險。）  
- **常數：** 為有效來源；是否成為最終 leaves 取決於 `Sub` frontier。  
- **多驅動／tri-state：** 先正規化（插入仲裁）或拒收此類 netlist。  
- **組合迴圈：** 偵測到則斷邊成 DAG 並回報斷點。  
- **匯流排（bus）：** 以 bit 為單位建圖；多輸出 cone 可跨多個 bit（同一 block）。  
- **非同步 set/reset：** 預設排除；可選納入為來源。

---

## 10) 產物與可視化
- `cones.jsonl`、`summary.json` 如 §2。  
- （選）每個 cone 的 Graphviz：  
  - 葉：`shape=diamond, fill=gold`  
  - 內部：`shape=box, fill=lightgray`  
  - 根：`shape=doublecircle, fill=lightblue`  
  - 僅繪 `Sub`；高亮 `depth/#leaves/#roots`。

---

## 11) 偽碼（高階）
```text
BUILD_GRAPH(netlist):
  parse verilog -> Cells, Nets
  tag sequential cells (FF/Latch) via liberty/whitelist
  build node graph: driver(net) -> sinks(net)

SEQ_CUT_AND_BLOCKS():
  // 自動邊界
  cut across sequential:
    treat FF.Q/Latch.Q as sources; FF.D/Latch.D as sinks
    remove edges that cross sequential elements
  blocks = undirected_connected_components(combinational_nodes)

LEVEL_AND_REVERSE(block):
  topo level inside block (nodes without predecessors in block have level 0)
  build reverse graph R for TFI

ENUM_SINGLE_ROOT_CONES(block, K=n_In, D=n_Depth):
  for each node r in block as root candidate:
    Cuts = {} per node
    for v in topo order:
      if v has no predecessors inside block:
         Cuts[v] = { {leaves={v}, depth=0} }
      else
         Cuts[v] = merge_cuts(Cuts[fanin*], K, D, M, count_inv)
    for each C in Cuts[r]:
      if cmp(|C.leaves|, n_In, cmp_in) and cmp(C.depth, n_Depth, cmp_depth):
         Sub = induce_paths(C.leaves -> r) within block
         EMIT_CONE(block_id, roots={r}, leaves=C.leaves, depth=C.depth, Sub)

ENUM_MULTI_ROOT_CONES(block, n_Out):
  roots_pool = pick_root_candidates(block)
  pairs = { (r1,r2) | support(r1) ∩ support(r2) ≠ ∅ }
  groups = expand_pairs_to_groups(pairs, max_size=n_Out, with_support_check)
  for R in groups:
    Sub = union_TFI(R) within block
    L   = frontier(Sub)
    depth = longest_path(Sub, L, R, count_inv)
    if connected_components(undirected(Sub))==1 and
       cmp(|L|, n_In, cmp_in) and cmp(|R|, n_Out, cmp_out) and
       cmp(depth, n_Depth, cmp_depth):
         EMIT_CONE(block_id, roots=R, leaves=L, depth, Sub)

EMIT_CONE(...):
  sig = SIGNATURE(Sub_nodes, roots)
  if sig not in seen:
    seen.add(sig)
    write cones.jsonl record
```

---

## 12) 預設參數
- `cmp_in = cmp_out = cmp_depth = "<="`  
- `count_inverters_in_depth = true`  
- `max_cuts_per_node = 150`（依設計大小調整）  
- `max_grouping_degree = n_Out`  
- `roots_pool`：預設為 block 內 **所有節點**；必要時依扇出/層級/與 PO/FF.D 之距離過濾。

---

## 13) 一頁帶走
- **邊界：** 由時序切割自動產生 block；探索永不跨越 FF／Latch／PI／PO。  
- **一般化 I/O：** 根與葉皆可為 **內部節點**；葉以 cone 子圖的 **frontier** 定義。  
- **演算法：** 單根用 **k-feasible cuts**（`K=n_In`）；多根以 **支撐共享** 分組，再檢查 `|L|/|R|/depth/連通性`。  
- **不可拆解：** cone 的誘導子圖在無向視角下必須為 **單一連通元件**。  
- **去重：** 以（節點集合＋根集合）雜湊為簽名；僅保留首次出現。

---

## 14) CLI 使用（快速上手）

### 範例
```bash
cone_finder \
  --netlist netlist.v \
  --n_in 4 --n_out 2 --n_depth 10 \
  --cmp_in "<=" --cmp_out "<=" --cmp_depth "<=" \
  --count_inverters_in_depth true \
  --max_cuts_per_node 150 \
  --max_grouping_degree 2 \
  --emit-dot --dot-topk 50 \
  --out-dir results/
```

### 主要參數
- `--netlist <path>`：結構化 Verilog 輸入檔。
- `--n_in <int>` / `--n_out <int>` / `--n_depth <int>`：cone 約束條件。
- `--cmp_in|--cmp_out|--cmp_depth {<=,==}`：比較子（預設 `<=`）。
- `--count_inverters_in_depth {true,false}`：深度是否計入 `inv/buf`（預設 `true`）。
- `--max_cuts_per_node <int>`：每節點 cut 上限（預設 150）。
- `--max_grouping_degree <int>`：多輸出群組的最大根數（預設 `n_Out`）。
- `--emit-dot`：輸出 Graphviz `.dot`（可搭配 `--dot-topk`）。
- `--out-dir <dir>`：JSON 與視覺化輸出目錄。

### 輸出
- `results/cones.jsonl` —— 每列一個 cone（見 §2.2）。
- `results/summary.json` —— 全域與逐 block 統計。
- `results/viz/*.dot` —— （選）cone 視覺化。

### 結束碼（建議）
- `0`：成功。
- `1`：解析／建圖錯誤。
- `2`：偵測到組合迴圈且未處理。
- `3`：輸出寫入錯誤。
