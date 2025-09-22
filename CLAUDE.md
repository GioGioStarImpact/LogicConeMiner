# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the **LogicConeMiner** project - a tool for discovering logic cones in gate-level netlists. The project implements algorithms to find combinational logic cones within Verilog netlists, automatically cutting at sequential boundaries (flip-flops and latches).

## Core Concepts

### Logic Cone Mining
- **Input**: Gate-level structural Verilog netlists with standard logic gates, flip-flops, latches, and I/O
- **Automatic Boundary**: Sequential cut at FF/Latch/PI/PO boundaries to create combinational blocks
- **General I/O**: Both roots (outputs) and leaves (inputs) can be any internal nodes within a block
- **Constraints**: Configurable limits on number of inputs (`n_In`), outputs (`n_Out`), and depth (`n_Depth`)
- **Indivisibility**: Each cone must form a single connected component

### Key Algorithms
1. **Single-Output Cones**: Uses k-feasible cuts with dynamic programming
2. **Multi-Output Cones**: Support-sharing grouping with connectivity validation
3. **Sequential Cut**: Automatic boundary detection by cutting at sequential elements

## Implementation Architecture

The specification indicates this should be implemented as a command-line tool with the following key components:

### Core Modules Expected
- **Verilog Parser**: Structural netlist parsing with FF/Latch detection
- **Graph Builder**: Node graph construction with driver→sink edges
- **Sequential Cutter**: Automatic boundary detection and block partitioning
- **Cut Enumerator**: k-feasible cuts algorithm for single-root cones
- **Multi-Root Grouper**: Support-sharing based root grouping
- **Cone Validator**: Connectivity and constraint checking
- **Deduplication**: Signature-based cone deduplication

### Data Structures
- **Node**: `{id, type, fanin[], fanout[], is_seq?, is_pi?, is_po?}`
- **Block**: Connected component of combinational nodes after sequential cut
- **Cut**: `{leaves, depth}` with dominance pruning
- **Cone**: `{roots, leaves, depth, num_nodes, num_edges, signature}`

## CLI Interface

The tool should be implemented as `cone_finder` with the following key parameters:

### Required Parameters
- `--netlist <path>`: Input Verilog file
- `--n_in <int>`: Maximum number of cone inputs (leaves)
- `--n_out <int>`: Maximum number of cone outputs (roots)
- `--n_depth <int>`: Maximum cone depth

### Key Optional Parameters
- `--cmp_in|--cmp_out|--cmp_depth {<=,==}`: Comparison operators (default: `<=`)
- `--count_inverters_in_depth {true,false}`: Include inverters in depth calculation (default: `true`)
- `--max_cuts_per_node <int>`: Memory control for cut enumeration (default: 150)
- `--max_grouping_degree <int>`: Max roots in multi-root groups (default: `n_Out`)
- `--emit-dot`: Generate Graphviz visualizations
- `--out-dir <dir>`: Output directory

## Output Files

### Required Outputs
- `cones.jsonl`: One JSON record per discovered cone with fields: `cone_id`, `block_id`, `roots`, `leaves`, `depth`, `num_nodes`, `num_edges`, `connected`, `signature`
- `summary.json`: Statistics and distributions by block and constraint values

### Optional Outputs
- `viz/*.dot`: Graphviz files for cone visualization
  - Leaves: diamond shape, gold fill
  - Roots: double circle, light blue fill
  - Internal nodes: box shape, light gray fill

## Performance Considerations

- **Memory Control**: Use `max_cuts_per_node` to limit cut storage per node
- **Root Pruning**: Filter root candidates by fanout, level, or distance to outputs
- **Support Prefiltering**: Only group roots with intersecting support sets
- **Parallelization**: Process blocks independently, parallelize within blocks

## Testing Strategy

- **Closure Validation**: Verify all paths from internal nodes to roots stay within cone
- **Depth Validation**: Cross-check dynamic programming depth with longest path algorithms
- **Connectivity Validation**: Ensure cone subgraphs form single connected components
- **Boundary Validation**: Verify all cone nodes belong to same block
- **Regression Testing**: Test performance on large designs (≥100k nodes)

## Edge Cases

- **Combinational Loops**: Detect and break to form DAG
- **Multi-driver Nets**: Normalize or reject
- **Constants**: Valid as sources/leaves depending on frontier
- **Latches**: Treat as sequential barriers by default
- **Async Set/Reset**: Excluded from data path by default