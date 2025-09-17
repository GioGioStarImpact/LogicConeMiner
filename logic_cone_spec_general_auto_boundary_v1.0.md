# Logic Cone Mining in Gate-Level Netlists
**Mode:** General • **Boundary:** Auto (by sequential cut) • **Leaf Scope:** `any_in_block`  
**Audience:** Implementation engineers (C++/Rust/Python)

---

## 0) Goals & Scope
- **Input:** Gate-level, structural Verilog netlist containing standard logic gates, constants, flip-flops (FFs), latches, and top-level I/O.
- **General I/O:** Both **roots (outputs of a cone)** and **leaves/frontier (inputs of a cone)** may be **any nodes inside an analysis region**, not restricted to PI/PO/FF pins.
- **Automatic Boundary:** No manual boundary is required. The netlist is **cut at sequential elements and top-level I/O**, producing combinational **blocks/islands**. All cone discovery happens **inside one block**, never crossing FF/CK-latch/PI/PO boundaries.
- **Constraints:**
  - `n_In`: number of cone **leaves** (frontier) within the block.
  - `n_Out`: number of cone **roots**.
  - `n_Depth`: cone **longest path length** (in gate count) from any leaf to any root (within the cone subgraph).
  - **Indivisibility:** The cone’s induced subgraph must be **single connected component** when treated as an undirected graph.
- **Leaf Scope:** `any_in_block` — leaves may be **any nodes** in the block, as long as, **within the cone subgraph**, they have **no predecessors** (i.e., they are the **frontier** of that cone).

---

## 1) Terminology & Formal Definitions
- **Block (Automatic Boundary):** The netlist is cut at **FFs/CK-Latches/PI/PO**. After removing all combinational edges that cross sequential elements, we compute **undirected connected components** of the remaining combinational nodes. Each component is a **block**. All cones are discovered **per block**.
- **Roots \(R\):** Any set of nodes within a block. May include internal nodes, FF.D, Latch.D, or nodes on the way to POs.
- **TFI\(_B\)(R):** In the block-induced subgraph \(G[B]\), the **Transitive Fan-In** of roots \(R\): all nodes that can reach any node in \(R\) along forward edges (equivalently, along reverse edges when traversing backward).
- **Cone Subgraph \(Sub\):** For roots \(R\), \(Sub = \mathrm{TFI}_B(R)\).
- **Frontier/Leaves \(L\):** Nodes in \(Sub\) with **no predecessors inside \(Sub\)**. Under `any_in_block`, these can be **internal nodes** (not only PI/FF.Q/Const).
- **Depth:** In \(Sub\), the longest (gate-count) path from any \(l\in L\) to any \(r\in R\). Each gate contributes cost 1 (optionally exclude inv/buf per parameter).
- **Indivisible (Connected):** The undirected view of \(Sub\) must have exactly **one connected component**.

---

## 2) Inputs & Outputs

### 2.1 Parameters
- `n_In` *(int, required)* — leaf count threshold.
- `n_Out` *(int, required)* — root count threshold.
- `n_Depth` *(int, required)* — longest-path threshold.
- `cmp_in`, `cmp_out`, `cmp_depth` ∈ `{<=, ==}` *(default: `<=`)*.
- `count_inverters_in_depth` *(bool, default: `true`)* — whether `inv/buf` contribute to depth.
- `max_cuts_per_node = M` *(int, default: 100–200)* — cap of stored cuts per node (memory/CPU control).
- `max_roots_per_block` *(int, optional)* — cap of candidate roots within a block (scalability knob).
- `max_grouping_degree` *(int, default: `n_Out`)* — max size of multi-root group \(R\) during exploration.

> **No manual boundaries** are required; boundaries are inferred by sequential cut.

### 2.2 Files
- **Input:** `netlist.v`
- **Outputs:**
  - `cones.jsonl` — one record per cone:
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
  - `summary.json` — counts & distributions (by #leaves/#roots/depth), per-block rollups.
  - *(Optional)* `viz/*.dot` — Graphviz render per cone (leaves=diamond gold, roots=doublecircle lightblue, internal=box lightgray).

---

## 3) Parsing & Graph Construction

1) **Verilog Parsing**
   - Structural instances, nets, constants; behavioral `always` not required.
   - Identify library cell types; detect FF/Latch via Liberty or whitelist (`DFF*`, `SDFF*`, `DLAT*`, etc.).

2) **Node Model**
   - `Node { id, type, fanin[], fanout[], is_seq?, is_pi?, is_po? }`
   - Each cell’s **output pin** corresponds to one **graph node**; build edges driver→sinks by nets.

3) **Sequential Cut (Automatic Boundary)**
   - Treat **FF.Q / Latch.Q** as **sources** (Pseudo-PI), **FF.D / Latch.D** as **sinks** (Pseudo-PO). **Do not traverse through** sequential elements.
   - **PI** acts as a source; **PO** acts as a sink.
   - Remove edges that would cross FF/Latch (i.e., break D→Q paths).
   - Async set/reset: excluded from data-path by default (configurable).

4) **Block Partitioning**
   - On the **sequentially-cut combinational graph**, compute **undirected connected components** over combinational nodes. Each component is a **block**.
   - All subsequent computations run **per block** (no cross-block cones).

5) **Leveling & Reverse Graph**
   - Within each block, topologically level from **block sources** (nodes without predecessors inside the block). `level(src)=0`; `level(v)=1+max(level(fanin))` (or 0 for inv/buf if excluded).
   - Build **reverse graph** \(R\) for fast TFI traversal.

---

## 4) Single-Output Cones via k-Feasible Cuts (`leaf_scope = any_in_block`)

- For each **root candidate** `r` (any node in the block), enumerate **cuts** of size ≤ `K = n_In`. Leaves may be **any internal nodes**.
- **Cut State**
  - `leaves` — a set of nodes, `|leaves| ≤ K`.
  - `depth` — longest path (in the induced cone) from any leaf to `r`.
  - **Dominance pruning:** if `C1.leaves ⊆ C2.leaves` and `C1.depth ≤ C2.depth`, drop `C2`.
- **DP Merge**
  - Scan nodes in topological order:
    - If `v` has no predecessors **inside the block**, `Cuts[v] = { {leaves={v}, depth=0} }`.
    - Otherwise, **merge** cuts from its fanins (Cartesian product), prune by `|leaves| ≤ K`, `depth ≤ n_Depth` (if `<=`), apply dominance and cap by `M`.
- **Emit Cone (Single Root)**
  - If `|leaves| cmp_in n_In` and `depth cmp_depth n_Depth`, build `Sub` as the union of all paths **inside the block** from `leaves` to `r`.
  - Single-root cones are inherently connected (all paths converge to `r`).
  - Output `{roots={r}, leaves, depth, ...}` for dedup.

> **Early Pruning:** compute tentative depth during merges; if exceeding `n_Depth` (with `<=`), discard. Prefer cuts with fewer leaves / smaller depth to limit combinatorics.

---

## 5) Multi-Output Cone Extension (|R| ≤ `n_Out`)

To control combinatorial blowup, use **support sharing**:

1) **Root Candidates & Sharing Filter**
   - In each block, pick root candidates (all nodes by default, optionally rate-limit by fanout/level/PO distance).
   - For each root `r`, compute an approximate **support bitset** (e.g., nodes or sources in `TFI_B(r)`). Keep **pairs** `(r1, r2)` only if their supports **intersect**. Expand to larger **groups** up to `|R| ≤ n_Out` while maintaining non-empty intersection heuristics.

2) **Evaluate Group R**
   - `Sub = ⋃_{r∈R} TFI_B(r)` (inside the block).
   - `L = frontier(Sub)` — nodes in `Sub` with no predecessors in `Sub` (**may be internal**).
   - `depth = max_{l∈L, r∈R} dist(l→r)` (inside `Sub`).
   - **Connectivity:** undirected(Sub) must have **1** connected component.
   - **Filter:** `|L| cmp_in n_In`, `|R| cmp_out n_Out`, `depth cmp_depth n_Depth`.
   - If pass, emit cone `{roots=R, leaves=L, depth, ...}`.

> Although leaves can be arbitrary internal nodes, within `Sub` they must be **the earliest nodes** (no predecessors in `Sub`).

---

## 6) Deduplication & Signature
**Why:** The same cone may be discovered via different enumeration paths or root groupings.

**Equivalence:** Two cones are identical if they have the **same node set** (of `Sub`) and the **same root set**.

**Signature:**
- Sort `Nodes(Sub)` → `H_nodes = hash(list(Nodes))`.
- Sort `Roots(R)` → `H_roots = hash(list(R))`.
- `signature = H_nodes ⊕ rotate(H_roots, k)` (128-bit recommended). Optionally include sorted edge list hash.
- On emission, compute signature, keep first occurrence; on rare collisions, fall back to exact set comparison.

---

## 7) Correctness & Testing
- **Closure:** For any cone, every path from any internal node in `Sub` to any `r ∈ R` stays **within `Sub`**; any `l ∈ L` has no predecessor in `Sub`.
- **Depth Validation:** Randomly sample cones and recompute DAG **Longest Path** to cross-check DP depth.
- **Indivisibility:** undirected(Sub) has exactly one connected component.
- **Boundary:** All nodes of `Sub` belong to the **same block**.
- **Regression:** Large designs (≥ 1e5 nodes) show stable time/memory curves as `M` varies.

---

## 8) Complexity & Performance
- Main costs: **cut enumeration** and **multi-root grouping**.
- **Controls:**
  - `M` — per-node cut cap (100–200 is a good start).
  - Root candidate pruning — by fanout, level, distance to PO/FF.D, etc.
  - Support-based prefilter — only group roots with intersecting supports.
- **Early pruning:** Apply `K`, `depth`, and comparator checks during merges.
- **Data structures:** compressed **bitsets** for supports and `Sub`; small sorted vectors for leaves; object pools for cuts.
- **Parallelism:** across blocks; within a block, across roots (single-output) and across root groups (multi-output).

---

## 9) Boundary Policies & Edge Cases
- **Latch:** treat as sequential barrier by default (`Q`=source, `D`=sink); no time-borrow modeling. (Optional override to mark some latches as transparent, at engineering risk.)
- **Constants:** valid sources; may appear as leaves depending on `Sub` frontier.
- **Multi-driver/tri-state:** normalize (insert arbitration) or reject netlists that contain them.
- **Combinational loops:** detect and break to form a DAG; report the breakpoints.
- **Buses:** model per-bit; multi-output cones may span multiple bits within a block.
- **Async set/reset:** excluded by default; optionally include as sources if desired.

---

## 10) Artifacts & Visualization
- `cones.jsonl`, `summary.json` as defined in §2.
- Optional Graphviz emission per cone:
  - leaves: `shape=diamond, fill=gold`
  - internal: `shape=box, fill=lightgray`
  - roots: `shape=doublecircle, fill=lightblue`
  - annotate `depth/#leaves/#roots`; draw only `Sub`.

---

## 11) Pseudocode (High-Level)

```text
BUILD_GRAPH(netlist):
  parse verilog -> Cells, Nets
  tag sequential cells (FF/Latch) via liberty/whitelist
  build node graph: driver(net) -> sinks(net)

SEQ_CUT_AND_BLOCKS():
  // Auto boundary
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

## 12) Default Parameters
- `cmp_in = cmp_out = cmp_depth = "<="`
- `count_inverters_in_depth = true`
- `max_cuts_per_node = 150` (tune by design size)
- `max_grouping_degree = n_Out`
- `roots_pool`: all nodes in the block by default; optionally filter by fanout/level/proximity to PO/FF.D.

---

## 13) One-Page Takeaways
- **Boundary:** automatic via sequential cut → per-block exploration; never cross FF/Latch/PI/PO.
- **General I/O:** roots & leaves may be **internal nodes**; leaves are the **frontier** within the cone subgraph.
- **Algorithm:** single-root via **k-feasible cuts** (`K=n_In`); multi-root via **support-sharing** grouping then check `|L|/|R|/depth/connectivity`.
- **Indivisible:** cone’s induced subgraph must be **one** undirected component.
- **Dedup:** signature over `(node-set, root-set)`; keep first occurrence.

---

## 14) CLI Usage (Quick Start)

### Example
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

### Key Flags
- `--netlist <path>`: structural Verilog input.
- `--n_in <int>` / `--n_out <int>` / `--n_depth <int>`: cone constraints.
- `--cmp_in|--cmp_out|--cmp_depth {<=,==}`: comparators (default: `<=`).
- `--count_inverters_in_depth {true,false}`: include `inv/buf` in depth (default: `true`).
- `--max_cuts_per_node <int>`: cap of stored cuts per node (default: 150).
- `--max_grouping_degree <int>`: max number of roots in a multi-root group (default: `n_Out`).
- `--emit-dot`: emit Graphviz `.dot` for top-K cones; combine with `--dot-topk`.
- `--out-dir <dir>`: output directory for JSON and visualization files.

### Outputs
- `results/cones.jsonl` — one cone per line (see §2.2).
- `results/summary.json` — global and per-block stats.
- `results/viz/*.dot` — optional cone visualizations.

### Exit Codes (suggested)
- `0`: success
- `1`: parsing/build-graph error
- `2`: combinational loop detected and not resolved
- `3`: output write error
