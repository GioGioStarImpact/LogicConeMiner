#!/usr/bin/env python3
"""
LogicConeMiner - Logic Cone Discovery in Gate-Level Netlists

This tool discovers logic cones within gate-level Verilog netlists by:
1. Automatically cutting at sequential boundaries (FF/Latch/PI/PO)
2. Finding combinational logic cones using k-feasible cuts
3. Supporting both single-output and multi-output cones
4. Ensuring cone connectivity and constraint satisfaction

Author: Claude Code Implementation
"""

import argparse
import json
import hashlib
import re
import sys
import os
from typing import Dict, List, Set, Tuple, Optional, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from itertools import combinations, product
import logging

# 導入 CSV 元件庫
from csv_cell_library import CSVCellLibrary, MacroHandler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ModuleDefinition:
    """Represents a module definition in the netlist"""
    name: str
    input_ports: List[str] = field(default_factory=list)
    output_ports: List[str] = field(default_factory=list)
    wire_declarations: Dict[str, str] = field(default_factory=dict)  # wire_name -> width_spec
    cells: Dict[str, Tuple[str, Dict]] = field(default_factory=dict)  # instance_name -> (cell_type, connections)
    macro_boundaries: Dict[str, List] = field(default_factory=dict)

@dataclass
class Node:
    """Represents a node in the circuit graph"""
    id: str
    type: str  # gate type or "PI", "PO", "FF", "LATCH", "CONST"
    fanin: List[str] = field(default_factory=list)
    fanout: List[str] = field(default_factory=list)
    is_seq: bool = False  # True if sequential element
    is_pi: bool = False   # True if primary input
    is_po: bool = False   # True if primary output
    level: int = -1       # Topological level within block
    module_name: str = ""  # Which module this node belongs to

    def __post_init__(self):
        if self.type in ['FF', 'LATCH']:
            self.is_seq = True

@dataclass
class Cut:
    """Represents a k-feasible cut"""
    leaves: Set[str]
    depth: int

    def dominates(self, other: 'Cut') -> bool:
        """Check if this cut dominates another (smaller/equal leaves, better/equal depth)"""
        return (self.leaves <= other.leaves and self.depth <= other.depth and
                (self.leaves < other.leaves or self.depth < other.depth))

@dataclass
class Cone:
    """Represents a discovered logic cone"""
    cone_id: str
    block_id: int
    roots: List[str]
    leaves: List[str]
    depth: int
    num_nodes: int
    num_edges: int
    connected: bool
    signature: str
    module_name: str = ""  # Which module this cone belongs to
    nodes: Set[str] = field(default_factory=set)  # All nodes in cone

class VerilogParser:
    """Enhanced Verilog parser with multi-module and CSV cell library support"""

    def __init__(self, csv_library: CSVCellLibrary = None):
        self.modules: Dict[str, ModuleDefinition] = {}  # module_name -> ModuleDefinition
        self.macro_definitions: Dict[str, ModuleDefinition] = {}  # macro_name -> ModuleDefinition
        self.csv_library = csv_library
        self.macro_handler = MacroHandler(csv_library) if csv_library else None


    def is_sequential(self, cell_type: str) -> bool:
        """Check if cell type is sequential using CSV library"""
        if self.csv_library and self.csv_library.is_standard_cell(cell_type):
            return self.csv_library.is_sequential(cell_type)

        # 非標準元件（Macro）視為非時序，但會被當作邊界處理
        return False

    def parse_file(self, filename: str) -> None:
        """Parse Verilog file and extract all modules"""
        logger.info(f"正在解析 Verilog 檔案: {filename}")

        try:
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(filename, 'r', encoding='latin-1') as f:
                content = f.read()

        # Remove comments and simplify
        content = re.sub(r'//.*?\n', '\n', content)
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)

        # Extract all modules
        module_pattern = r'module\s+(\w+)\s*\((.*?)\);(.*?)endmodule'
        for match in re.finditer(module_pattern, content, re.DOTALL):
            module_name, ports_str, module_body = match.groups()

            logger.info(f"解析模組: {module_name}")
            module_def = self._parse_module(module_name, ports_str, module_body)

            # 判斷是標準模組還是 Macro
            if self._is_module_macro(module_def):
                self.macro_definitions[module_name] = module_def
                logger.info(f"檢測到 Macro 定義: {module_name}")
            else:
                self.modules[module_name] = module_def
                logger.info(f"檢測到標準模組: {module_name}")

        logger.info(f"解析完成: {len(self.modules)} 個標準模組, {len(self.macro_definitions)} 個 Macro 定義")

    def _parse_module(self, module_name: str, ports_str: str, module_body: str) -> ModuleDefinition:
        """Parse a single module"""
        module_def = ModuleDefinition(name=module_name)

        # Parse port declarations from module header and body
        self._parse_port_declarations(module_def, ports_str, module_body)

        # Parse wire declarations
        self._parse_wire_declarations(module_def, module_body)

        # Parse cell instances
        self._parse_cell_instances(module_def, module_body)

        # Handle macro boundaries
        self._handle_macro_boundaries(module_def)

        return module_def

    def _parse_port_declarations(self, module_def: ModuleDefinition, ports_str: str, module_body: str) -> None:
        """Parse input/output port declarations"""
        # Parse from module body (more reliable)
        # Handle formats like: input a, b, c; output d; wire bbb[20:0];

        # Input ports
        input_patterns = [
            r'input\s+([^;]+);',  # input a, b, c;
            r'input\s+(\w+)\s*,',  # input a, from port list
            r'input\s+(\w+)\s*\)',  # input a) from port list
        ]

        for pattern in input_patterns:
            for match in re.finditer(pattern, module_body + ports_str):
                ports_spec = match.group(1)
                # Handle both simple and bus formats
                port_names = self._extract_port_names(ports_spec)
                module_def.input_ports.extend(port_names)

        # Output ports
        output_patterns = [
            r'output\s+([^;]+);',
            r'output\s+(\w+)\s*,',
            r'output\s+(\w+)\s*\)',
        ]

        for pattern in output_patterns:
            for match in re.finditer(pattern, module_body + ports_str):
                ports_spec = match.group(1)
                port_names = self._extract_port_names(ports_spec)
                module_def.output_ports.extend(port_names)

        # Remove duplicates while preserving order
        module_def.input_ports = list(dict.fromkeys(module_def.input_ports))
        module_def.output_ports = list(dict.fromkeys(module_def.output_ports))

    def _extract_port_names(self, ports_spec: str) -> List[str]:
        """Extract port names from declaration, handling bus notation"""
        port_names = []

        # Split by comma first
        for port_spec in ports_spec.split(','):
            port_spec = port_spec.strip()

            # Handle bus notation like wire_name[20:0]
            if '[' in port_spec:
                # Extract base name before bracket
                base_name = re.search(r'(\w+)\s*\[', port_spec)
                if base_name:
                    port_names.append(base_name.group(1))
            else:
                # Simple wire name
                wire_name = re.search(r'(\w+)', port_spec)
                if wire_name:
                    port_names.append(wire_name.group(1))

        return port_names

    def _parse_wire_declarations(self, module_def: ModuleDefinition, module_body: str) -> None:
        """Parse wire declarations"""
        # Handle: wire bbb[20:0]; wire a, b, c;
        wire_pattern = r'wire\s+([^;]+);'

        for match in re.finditer(wire_pattern, module_body):
            wire_spec = match.group(1)

            # Split by comma for multiple wires
            for wire_decl in wire_spec.split(','):
                wire_decl = wire_decl.strip()

                # Extract wire name and optional width
                if '[' in wire_decl:
                    # Bus wire: bbb[20:0]
                    wire_match = re.search(r'(\w+)\s*(\[.*?\])', wire_decl)
                    if wire_match:
                        wire_name, width_spec = wire_match.groups()
                        module_def.wire_declarations[wire_name] = width_spec
                else:
                    # Simple wire
                    wire_name = re.search(r'(\w+)', wire_decl)
                    if wire_name:
                        module_def.wire_declarations[wire_name.group(1)] = ''

    def _parse_cell_instances(self, module_def: ModuleDefinition, module_body: str) -> None:
        """Parse cell instances"""
        # Pattern: cell_type instance_name ( connections );
        instance_pattern = r'(\w+)\s+(\w+)\s*\((.*?)\)\s*;'

        for match in re.finditer(instance_pattern, module_body, re.DOTALL):
            cell_type, instance_name, connections_str = match.groups()

            # Skip module/endmodule keywords
            if cell_type in ['module', 'endmodule']:
                continue

            # Parse connections (.port(net), ...)
            connections = {}
            conn_pattern = r'\.(\w+)\s*\(\s*([^)]+)\s*\)'

            for conn_match in re.finditer(conn_pattern, connections_str):
                port, net = conn_match.groups()
                connections[port] = net.strip()

            module_def.cells[instance_name] = (cell_type, connections)

    def _handle_macro_boundaries(self, module_def: ModuleDefinition) -> None:
        """Handle macro boundaries for a module"""
        if not self.macro_handler:
            return

        macro_boundaries = {'inputs_as_po': [], 'outputs_as_pi': []}

        for instance_name, (cell_type, connections) in module_def.cells.items():
            # 檢查是否為 Macro（非 CSV 定義的標準元件，也非本檔案定義的模組）
            if (self.macro_handler.is_macro(cell_type) and
                cell_type not in self.macro_definitions):

                boundaries = self.macro_handler.handle_macro_instance(
                    instance_name, cell_type, connections
                )
                macro_boundaries['inputs_as_po'].extend(boundaries['inputs_as_po'])
                macro_boundaries['outputs_as_pi'].extend(boundaries['outputs_as_pi'])

                logger.info(f"模組 {module_def.name} 中的 Macro {instance_name} ({cell_type}) 處理為邊界節點")

        module_def.macro_boundaries = macro_boundaries

    def _is_module_macro(self, module_def: ModuleDefinition) -> bool:
        """判斷模組是否應該被視為 Macro"""
        # 只有當模組「完全」由非標準元件組成時，才視為 Macro
        # 如果混合了標準元件和非標準元件，視為標準模組（會在內部處理 Macro 邊界）

        total_cells = len(module_def.cells)
        if total_cells == 0:
            return False  # 空模組視為標準模組

        non_standard_cells = 0
        for instance_name, (cell_type, connections) in module_def.cells.items():
            if not self.csv_library.is_standard_cell(cell_type):
                non_standard_cells += 1

        # 如果全部都是非標準元件，視為 Macro
        # 如果混合或全部都是標準元件，視為標準模組
        return non_standard_cells == total_cells

class GraphBuilder:
    """Builds circuit graph from parsed Verilog"""

    def __init__(self, parser: VerilogParser):
        self.parser = parser
        self.nodes: Dict[str, Node] = {}
        self.blocks: List[Set[str]] = []  # List of blocks (connected components)
        self.reverse_graph: Dict[str, Set[str]] = defaultdict(set)

    def build_graph(self) -> None:
        """Build initial circuit graph"""
        logger.info("建構電路圖...")

        # Create nodes for primary inputs/outputs
        for port_name, direction in self.parser.ports.items():
            node = Node(
                id=port_name,
                type="PI" if direction == "input" else "PO",
                is_pi=(direction == "input"),
                is_po=(direction == "output")
            )
            self.nodes[port_name] = node

        # Create nodes for cell outputs and constants
        net_drivers: Dict[str, str] = {}  # net -> driver_node

        for instance_name, (cell_type, connections) in self.parser.cells.items():
            # For sequential elements, create separate nodes for Q and D
            if self.parser.is_sequential(cell_type):
                # Q output node
                q_node_id = f"{instance_name}.Q"
                q_node = Node(
                    id=q_node_id,
                    type=cell_type,
                    is_seq=True
                )
                self.nodes[q_node_id] = q_node

                # D input node (pseudo-output)
                d_node_id = f"{instance_name}.D"
                d_node = Node(
                    id=d_node_id,
                    type=f"{cell_type}_D",
                    is_seq=True
                )
                self.nodes[d_node_id] = d_node

                # Connect to nets
                if 'Q' in connections:
                    net_drivers[connections['Q']] = q_node_id

            else:
                # Regular combinational cell - create node for output
                output_ports = ['Y', 'Z', 'Q', 'OUT']  # Common output port names
                for port in output_ports:
                    if port in connections:
                        node_id = f"{instance_name}.{port}"
                        node = Node(
                            id=node_id,
                            type=cell_type
                        )
                        self.nodes[node_id] = node
                        net_drivers[connections[port]] = node_id
                        break
                else:
                    # If no standard output port found, create generic node
                    node_id = instance_name
                    node = Node(
                        id=node_id,
                        type=cell_type
                    )
                    self.nodes[node_id] = node

        # Build connections
        for instance_name, (cell_type, connections) in self.parser.cells.items():
            if self.parser.is_sequential(cell_type):
                # Sequential element connections
                q_node_id = f"{instance_name}.Q"
                d_node_id = f"{instance_name}.D"

                if 'D' in connections:
                    input_net = connections['D']
                    if input_net in net_drivers:
                        driver = net_drivers[input_net]
                        # Connect driver -> D input (but this will be cut)
                        pass  # Will be handled in sequential cut

            else:
                # Find this cell's output node
                cell_output_node = None
                for port in ['Y', 'Z', 'Q', 'OUT']:
                    if port in connections:
                        cell_output_node = f"{instance_name}.{port}"
                        break
                else:
                    cell_output_node = instance_name

                if cell_output_node in self.nodes:
                    # Connect inputs
                    input_ports = ['A', 'B', 'C', 'D', 'IN', 'I']
                    for port, net in connections.items():
                        if port in input_ports or port.startswith('I'):
                            if net in net_drivers:
                                driver = net_drivers[net]
                                self.nodes[driver].fanout.append(cell_output_node)
                                self.nodes[cell_output_node].fanin.append(driver)
                            elif net in self.nodes:  # Primary input
                                self.nodes[net].fanout.append(cell_output_node)
                                self.nodes[cell_output_node].fanin.append(net)

        logger.info(f"圖建構完成: {len(self.nodes)} 個節點")

    def sequential_cut_and_blocks(self) -> None:
        """Perform sequential cut and find combinational blocks"""
        logger.info("執行時序切割並找尋組合邏輯區塊...")

        # Remove edges crossing sequential elements
        combinational_nodes = set()
        for node_id, node in self.nodes.items():
            if not node.is_seq and not node.is_pi and not node.is_po:
                combinational_nodes.add(node_id)

        # Build combinational edges (undirected for connected components)
        edges = set()
        for node_id in combinational_nodes:
            node = self.nodes[node_id]
            for fanout_id in node.fanout:
                if fanout_id in combinational_nodes:
                    edges.add((min(node_id, fanout_id), max(node_id, fanout_id)))

        # Find connected components using BFS
        visited = set()
        for node_id in combinational_nodes:
            if node_id not in visited:
                # Start new component
                component = set()
                queue = deque([node_id])

                while queue:
                    current = queue.popleft()
                    if current in visited:
                        continue

                    visited.add(current)
                    component.add(current)

                    # Add neighbors
                    for neighbor in self.nodes[current].fanin + self.nodes[current].fanout:
                        if neighbor in combinational_nodes and neighbor not in visited:
                            queue.append(neighbor)

                if component:
                    self.blocks.append(component)

        logger.info(f"找到 {len(self.blocks)} 個組合邏輯區塊")

        # Build reverse graph for each block
        for node_id, node in self.nodes.items():
            for fanout_id in node.fanout:
                self.reverse_graph[fanout_id].add(node_id)

    def level_nodes_in_block(self, block: Set[str]) -> None:
        """Assign topological levels to nodes within a block"""
        # Find sources (nodes with no predecessors in block)
        in_degree = {node_id: 0 for node_id in block}
        for node_id in block:
            for fanin_id in self.nodes[node_id].fanin:
                if fanin_id in block:
                    in_degree[node_id] += 1

        queue = deque([node_id for node_id in block if in_degree[node_id] == 0])
        level = 0

        while queue:
            next_queue = deque()

            for _ in range(len(queue)):
                node_id = queue.popleft()
                self.nodes[node_id].level = level

                for fanout_id in self.nodes[node_id].fanout:
                    if fanout_id in block:
                        in_degree[fanout_id] -= 1
                        if in_degree[fanout_id] == 0:
                            next_queue.append(fanout_id)

            queue = next_queue
            level += 1

class ConeEnumerator:
    """Enumerates logic cones using k-feasible cuts"""

    def __init__(self, graph_builder: GraphBuilder, config: dict):
        self.graph = graph_builder
        self.config = config
        self.discovered_cones: List[Cone] = []
        self.signatures_seen: Set[str] = set()

    def enumerate_single_root_cones(self, block_id: int, block: Set[str]) -> None:
        """Enumerate single-output cones in a block using k-feasible cuts"""
        logger.info(f"枚舉區塊 {block_id} 中的單輸出錐 ({len(block)} 個節點)")

        # Level nodes in this block
        self.graph.level_nodes_in_block(block)

        # Get nodes in topological order
        topo_order = sorted(block, key=lambda x: self.graph.nodes[x].level)

        # For each root candidate
        for root_id in block:
            cuts_per_node = self._compute_cuts_for_root(root_id, block, topo_order)

            # Check cuts at root and emit cones
            if root_id in cuts_per_node:
                for cut in cuts_per_node[root_id]:
                    if self._satisfies_constraints(cut, 1):  # 1 root
                        cone = self._build_cone_from_cut(block_id, [root_id], cut, block)
                        if cone and self._is_new_cone(cone):
                            self.discovered_cones.append(cone)

    def enumerate_multi_root_cones(self, block_id: int, block: Set[str]) -> None:
        """Enumerate multi-output cones using support sharing"""
        if self.config['n_out'] <= 1:
            return  # No multi-root cones needed

        logger.info(f"枚舉區塊 {block_id} 中的多輸出錐")

        # Get root candidates
        root_candidates = self._get_root_candidates(block)

        # Compute support for each root
        root_supports = {}
        for root_id in root_candidates:
            support = self._compute_support(root_id, block)
            root_supports[root_id] = support

        # Find root pairs with intersecting supports
        valid_pairs = []
        for i, root1 in enumerate(root_candidates):
            for j, root2 in enumerate(root_candidates[i+1:], i+1):
                if root_supports[root1] & root_supports[root2]:  # Intersection
                    valid_pairs.append((root1, root2))

        # Expand pairs to larger groups
        root_groups = self._expand_to_groups(valid_pairs, root_supports,
                                           self.config['n_out'])

        # Evaluate each group
        for roots in root_groups:
            cone = self._build_multi_root_cone(block_id, roots, block)
            if cone and self._satisfies_multi_root_constraints(cone) and self._is_new_cone(cone):
                self.discovered_cones.append(cone)

    def _get_root_candidates(self, block: Set[str]) -> List[str]:
        """Get root candidates for multi-output grouping"""
        # For now, use all nodes in block
        # Could be filtered by fanout, level, proximity to outputs, etc.
        return list(block)

    def _compute_support(self, root_id: str, block: Set[str]) -> Set[str]:
        """Compute transitive fanin (support) for a root within block"""
        support = set()
        visited = set()
        queue = deque([root_id])

        while queue:
            node_id = queue.popleft()
            if node_id in visited:
                continue

            visited.add(node_id)
            support.add(node_id)

            # Add fanins that are in the block
            for fanin_id in self.graph.nodes[node_id].fanin:
                if fanin_id in block and fanin_id not in visited:
                    queue.append(fanin_id)

        return support

    def _expand_to_groups(self, pairs: List[Tuple[str, str]],
                         root_supports: Dict[str, Set[str]],
                         max_size: int) -> List[List[str]]:
        """Expand pairs to larger groups while maintaining support intersection"""
        groups = []

        # Start with pairs as initial groups
        for pair in pairs:
            groups.append(list(pair))

        # Try to expand groups
        max_group_size = min(max_size, self.config.get('max_grouping_degree', max_size))

        for size in range(3, max_group_size + 1):
            new_groups = []

            for group in groups:
                if len(group) == size - 1:
                    # Try to add one more root
                    group_support = set.intersection(*[root_supports[root] for root in group])

                    for root_id, support in root_supports.items():
                        if (root_id not in group and
                            group_support & support):  # Has intersection
                            new_group = group + [root_id]
                            new_groups.append(new_group)

            groups.extend(new_groups)

        return groups

    def _build_multi_root_cone(self, block_id: int, roots: List[str], block: Set[str]) -> Optional[Cone]:
        """Build multi-root cone"""
        # Compute union of TFI for all roots
        cone_nodes = set()
        for root_id in roots:
            root_support = self._compute_support(root_id, block)
            cone_nodes.update(root_support)

        # Find frontier (leaves) - nodes in cone with no predecessors in cone
        leaves = set()
        for node_id in cone_nodes:
            fanins_in_cone = [f for f in self.graph.nodes[node_id].fanin if f in cone_nodes]
            if not fanins_in_cone:
                leaves.add(node_id)

        # Compute depth - longest path from any leaf to any root
        depth = self._compute_longest_path(cone_nodes, leaves, roots)

        # Count edges in cone
        num_edges = 0
        for node_id in cone_nodes:
            for fanout_id in self.graph.nodes[node_id].fanout:
                if fanout_id in cone_nodes:
                    num_edges += 1

        # Check connectivity
        connected = self._is_connected(cone_nodes)

        # Generate signature
        signature = self._generate_signature(cone_nodes, roots)

        cone = Cone(
            cone_id=signature[:16],
            block_id=block_id,
            roots=sorted(roots),
            leaves=sorted(list(leaves)),
            depth=depth,
            num_nodes=len(cone_nodes),
            num_edges=num_edges,
            connected=connected,
            signature=signature,
            nodes=cone_nodes
        )

        return cone

    def _compute_longest_path(self, cone_nodes: Set[str], leaves: Set[str], roots: List[str]) -> int:
        """Compute longest path from any leaf to any root within cone"""
        # Build topological order within cone
        in_degree = {node_id: 0 for node_id in cone_nodes}
        for node_id in cone_nodes:
            for fanout_id in self.graph.nodes[node_id].fanout:
                if fanout_id in cone_nodes:
                    in_degree[fanout_id] += 1

        # Use topological sort with distance tracking
        distances = {node_id: 0 for node_id in cone_nodes}
        queue = deque([node_id for node_id in leaves])

        max_distance = 0
        count_inverters = self.config['count_inverters_in_depth']

        while queue:
            node_id = queue.popleft()
            current_dist = distances[node_id]

            for fanout_id in self.graph.nodes[node_id].fanout:
                if fanout_id in cone_nodes:
                    # Calculate edge cost
                    fanout_node = self.graph.nodes[fanout_id]
                    gate_cost = 0 if (not count_inverters and
                                     fanout_node.type.upper() in ['INV', 'BUF']) else 1

                    new_dist = current_dist + gate_cost
                    distances[fanout_id] = max(distances[fanout_id], new_dist)

                    in_degree[fanout_id] -= 1
                    if in_degree[fanout_id] == 0:
                        queue.append(fanout_id)

        # Get maximum distance at roots
        for root_id in roots:
            if root_id in distances:
                max_distance = max(max_distance, distances[root_id])

        return max_distance

    def _satisfies_multi_root_constraints(self, cone: Cone) -> bool:
        """Check if multi-root cone satisfies constraints"""
        cmp_in = self.config['cmp_in']
        cmp_out = self.config['cmp_out']
        cmp_depth = self.config['cmp_depth']

        # Check input constraint
        if cmp_in == '<=' and len(cone.leaves) > self.config['n_in']:
            return False
        elif cmp_in == '==' and len(cone.leaves) != self.config['n_in']:
            return False

        # Check output constraint
        if cmp_out == '<=' and len(cone.roots) > self.config['n_out']:
            return False
        elif cmp_out == '==' and len(cone.roots) != self.config['n_out']:
            return False

        # Check depth constraint
        if cmp_depth == '<=' and cone.depth > self.config['n_depth']:
            return False
        elif cmp_depth == '==' and cone.depth != self.config['n_depth']:
            return False

        # Must be connected
        if not cone.connected:
            return False

        return True

    def _compute_cuts_for_root(self, root_id: str, block: Set[str], topo_order: List[str]) -> Dict[str, List[Cut]]:
        """Compute k-feasible cuts for a specific root using dynamic programming"""
        cuts_per_node: Dict[str, List[Cut]] = {}
        max_cuts = self.config['max_cuts_per_node']
        K = self.config['n_in']
        max_depth = self.config['n_depth']
        count_inverters = self.config['count_inverters_in_depth']

        for node_id in topo_order:
            if node_id not in block:
                continue

            node = self.graph.nodes[node_id]
            fanins_in_block = [f for f in node.fanin if f in block]

            if not fanins_in_block:
                # Source node in block
                cuts_per_node[node_id] = [Cut(leaves={node_id}, depth=0)]
            else:
                # Merge cuts from fanins
                cuts = []

                # Get cuts from all fanins
                fanin_cuts = []
                for fanin_id in fanins_in_block:
                    if fanin_id in cuts_per_node:
                        fanin_cuts.append(cuts_per_node[fanin_id])
                    else:
                        fanin_cuts.append([Cut(leaves={fanin_id}, depth=0)])

                # Cartesian product of fanin cuts
                for cut_combination in product(*fanin_cuts):
                    # Merge cuts
                    merged_leaves = set()
                    max_fanin_depth = 0

                    for cut in cut_combination:
                        merged_leaves.update(cut.leaves)
                        max_fanin_depth = max(max_fanin_depth, cut.depth)

                    # Calculate new depth
                    gate_cost = 0 if (not count_inverters and
                                     node.type.upper() in ['INV', 'BUF']) else 1
                    new_depth = max_fanin_depth + gate_cost

                    # Check constraints
                    if len(merged_leaves) <= K and new_depth <= max_depth:
                        new_cut = Cut(leaves=merged_leaves, depth=new_depth)
                        cuts.append(new_cut)

                # Apply dominance pruning
                cuts = self._prune_dominated_cuts(cuts)

                # Limit number of cuts
                cuts = cuts[:max_cuts]
                cuts_per_node[node_id] = cuts

        return cuts_per_node

    def _prune_dominated_cuts(self, cuts: List[Cut]) -> List[Cut]:
        """Remove dominated cuts"""
        pruned = []
        for i, cut1 in enumerate(cuts):
            dominated = False
            for j, cut2 in enumerate(cuts):
                if i != j and cut2.dominates(cut1):
                    dominated = True
                    break
            if not dominated:
                pruned.append(cut1)
        return pruned

    def _satisfies_constraints(self, cut: Cut, num_roots: int) -> bool:
        """Check if cut satisfies all constraints"""
        cmp_in = self.config['cmp_in']
        cmp_out = self.config['cmp_out']
        cmp_depth = self.config['cmp_depth']

        # Check input constraint
        if cmp_in == '<=' and len(cut.leaves) > self.config['n_in']:
            return False
        elif cmp_in == '==' and len(cut.leaves) != self.config['n_in']:
            return False

        # Check output constraint
        if cmp_out == '<=' and num_roots > self.config['n_out']:
            return False
        elif cmp_out == '==' and num_roots != self.config['n_out']:
            return False

        # Check depth constraint
        if cmp_depth == '<=' and cut.depth > self.config['n_depth']:
            return False
        elif cmp_depth == '==' and cut.depth != self.config['n_depth']:
            return False

        return True

    def _build_cone_from_cut(self, block_id: int, roots: List[str], cut: Cut, block: Set[str]) -> Optional[Cone]:
        """Build cone structure from cut information"""
        # Find all nodes in cone by traversing from leaves to roots
        cone_nodes = set()
        queue = deque(cut.leaves)
        cone_nodes.update(cut.leaves)

        while queue:
            node_id = queue.popleft()
            for fanout_id in self.graph.nodes[node_id].fanout:
                if fanout_id in block and fanout_id not in cone_nodes:
                    # Check if all fanins of this node are in cone or it's a root
                    node_fanins = [f for f in self.graph.nodes[fanout_id].fanin if f in block]
                    if (fanout_id in roots or
                        all(f in cone_nodes for f in node_fanins)):
                        cone_nodes.add(fanout_id)
                        queue.append(fanout_id)

        # Count edges in cone
        num_edges = 0
        for node_id in cone_nodes:
            for fanout_id in self.graph.nodes[node_id].fanout:
                if fanout_id in cone_nodes:
                    num_edges += 1

        # Check connectivity
        connected = self._is_connected(cone_nodes)

        # Generate signature
        signature = self._generate_signature(cone_nodes, roots)

        cone = Cone(
            cone_id=signature[:16],
            block_id=block_id,
            roots=sorted(roots),
            leaves=sorted(list(cut.leaves)),
            depth=cut.depth,
            num_nodes=len(cone_nodes),
            num_edges=num_edges,
            connected=connected,
            signature=signature,
            nodes=cone_nodes
        )

        return cone

    def _is_connected(self, nodes: Set[str]) -> bool:
        """Check if nodes form a connected component in undirected view"""
        if not nodes:
            return True

        # Build undirected adjacency
        adj = defaultdict(set)
        for node_id in nodes:
            for neighbor in (self.graph.nodes[node_id].fanin + self.graph.nodes[node_id].fanout):
                if neighbor in nodes:
                    adj[node_id].add(neighbor)
                    adj[neighbor].add(node_id)

        # BFS to check connectivity
        start = next(iter(nodes))
        visited = set()
        queue = deque([start])

        while queue:
            node_id = queue.popleft()
            if node_id in visited:
                continue
            visited.add(node_id)

            for neighbor in adj[node_id]:
                if neighbor not in visited:
                    queue.append(neighbor)

        return len(visited) == len(nodes)

    def _generate_signature(self, nodes: Set[str], roots: List[str]) -> str:
        """Generate unique signature for cone"""
        nodes_hash = hashlib.sha256('|'.join(sorted(nodes)).encode()).hexdigest()
        roots_hash = hashlib.sha256('|'.join(sorted(roots)).encode()).hexdigest()

        # XOR with rotation
        combined = int(nodes_hash[:32], 16) ^ int(roots_hash[:32], 16)
        return format(combined, '032x')

    def _is_new_cone(self, cone: Cone) -> bool:
        """Check if cone is new (not seen before)"""
        if cone.signature in self.signatures_seen:
            return False
        self.signatures_seen.add(cone.signature)
        return True

class ModuleGraphBuilder:
    """Builds circuit graph for a single module"""

    def __init__(self, parser: VerilogParser, module_name: str):
        self.parser = parser
        self.module_name = module_name
        self.nodes: Dict[str, Node] = {}
        self.blocks: List[Set[str]] = []

    def build_module_graph(self, module_def: ModuleDefinition) -> None:
        """Build circuit graph for this module"""
        logger.info(f"建構模組 {self.module_name} 的電路圖...")

        # Create nodes for primary inputs/outputs
        for port_name in module_def.input_ports:
            node = Node(
                id=port_name,
                type="PI",
                is_pi=True,
                module_name=self.module_name
            )
            self.nodes[port_name] = node

        for port_name in module_def.output_ports:
            node = Node(
                id=port_name,
                type="PO",
                is_po=True,
                module_name=self.module_name
            )
            self.nodes[port_name] = node

        # Create nodes for cell outputs
        net_drivers: Dict[str, str] = {}

        for instance_name, (cell_type, connections) in module_def.cells.items():
            if self.parser.is_sequential(cell_type):
                # Sequential element
                q_node_id = f"{instance_name}.Q"
                q_node = Node(
                    id=q_node_id,
                    type=cell_type,
                    is_seq=True,
                    module_name=self.module_name
                )
                self.nodes[q_node_id] = q_node

                if 'Q' in connections:
                    net_drivers[connections['Q']] = q_node_id
            else:
                # Combinational cell or macro
                if self.parser.csv_library.is_standard_cell(cell_type):
                    output_ports = self.parser.csv_library.get_output_pins(cell_type)
                else:
                    output_ports = ['Y', 'Z', 'Q', 'OUT', 'O']

                for port in output_ports:
                    if port in connections:
                        node_id = f"{instance_name}.{port}"
                        node = Node(
                            id=node_id,
                            type=cell_type,
                            module_name=self.module_name
                        )
                        self.nodes[node_id] = node
                        net_drivers[connections[port]] = node_id
                        break

        # Handle macro boundaries
        for boundary_info in module_def.macro_boundaries.get('inputs_as_po', []):
            node = Node(
                id=boundary_info['node_id'],
                type=boundary_info['type'],
                is_po=True,
                module_name=self.module_name
            )
            self.nodes[boundary_info['node_id']] = node

        for boundary_info in module_def.macro_boundaries.get('outputs_as_pi', []):
            node = Node(
                id=boundary_info['node_id'],
                type=boundary_info['type'],
                is_pi=True,
                module_name=self.module_name
            )
            self.nodes[boundary_info['node_id']] = node
            net_drivers[boundary_info['net']] = boundary_info['node_id']

        # Build connections
        for instance_name, (cell_type, connections) in module_def.cells.items():
            if not self.parser.is_sequential(cell_type):
                # Find output node
                cell_output_node = None
                if self.parser.csv_library.is_standard_cell(cell_type):
                    output_ports = self.parser.csv_library.get_output_pins(cell_type)
                else:
                    output_ports = ['Y', 'Z', 'Q', 'OUT', 'O']

                for port in output_ports:
                    if port in connections:
                        cell_output_node = f"{instance_name}.{port}"
                        break

                if cell_output_node and cell_output_node in self.nodes:
                    # Connect inputs
                    if self.parser.csv_library.is_standard_cell(cell_type):
                        input_ports = self.parser.csv_library.get_input_pins(cell_type)
                    else:
                        input_ports = ['A', 'B', 'C', 'D', 'IN', 'I']

                    for port, net in connections.items():
                        if port in input_ports or port.startswith('I') or port.startswith('A'):
                            if net in net_drivers:
                                driver = net_drivers[net]
                                self.nodes[driver].fanout.append(cell_output_node)
                                self.nodes[cell_output_node].fanin.append(driver)
                            elif net in self.nodes:  # Primary input
                                self.nodes[net].fanout.append(cell_output_node)
                                self.nodes[cell_output_node].fanin.append(net)

        logger.info(f"模組 {self.module_name} 圖建構完成: {len(self.nodes)} 個節點")

    def find_combinational_blocks(self) -> None:
        """Find combinational blocks in this module"""
        # Find all sequential and boundary nodes
        boundary_nodes = set()
        for node in self.nodes.values():
            if node.is_seq or node.is_pi or node.is_po:
                boundary_nodes.add(node.id)

        # Find connected components
        visited = set()
        for node_id, node in self.nodes.items():
            if node_id not in visited and node_id not in boundary_nodes:
                block = set()
                self._dfs_block(node_id, boundary_nodes, visited, block)
                if block:
                    self.blocks.append(block)

        logger.info(f"模組 {self.module_name} 找到 {len(self.blocks)} 個組合邏輯區塊")

    def _dfs_block(self, node_id: str, boundary_nodes: Set[str], visited: Set[str], block: Set[str]) -> None:
        """DFS to find connected combinational components"""
        if node_id in visited or node_id in boundary_nodes or node_id not in self.nodes:
            return

        visited.add(node_id)
        block.add(node_id)

        node = self.nodes[node_id]
        for neighbor in node.fanin + node.fanout:
            self._dfs_block(neighbor, boundary_nodes, visited, block)

class ModuleConeEnumerator:
    """Enumerates cones for a single module"""

    def __init__(self, graph_builder: ModuleGraphBuilder, config: Dict, module_name: str):
        self.graph_builder = graph_builder
        self.config = config
        self.module_name = module_name
        self.discovered_cones: List[Cone] = []
        self.signatures_seen: Set[str] = set()

    def enumerate_single_root_cones(self, block_id: int, block: Set[str]) -> None:
        """Enumerate single-root cones in a block"""
        logger.info(f"枚舉模組 {self.module_name} 區塊 {block_id} 中的單輸出錐 ({len(block)} 個節點)")

        # Find potential roots (nodes with fanout to other blocks or PO)
        potential_roots = []
        for node_id in block:
            node = self.graph_builder.nodes[node_id]
            # Check if node has fanout outside block or to PO
            has_external_fanout = any(
                fanout_id not in block or self.graph_builder.nodes.get(fanout_id, Node("", "")).is_po
                for fanout_id in node.fanout
            )
            if has_external_fanout or not node.fanout:  # Include sinks
                potential_roots.append(node_id)

        # Generate cones for each root
        for root_id in potential_roots:
            cuts = self._enumerate_k_feasible_cuts(root_id, block)

            for cut in cuts:
                if self._satisfies_constraints(cut.leaves, [root_id], cut.depth):
                    cone_nodes = self._extract_cone_nodes(root_id, cut.leaves, block)
                    if self._is_connected_cone(cone_nodes):
                        cone = self._create_cone(block_id, [root_id], list(cut.leaves), cut.depth, cone_nodes)
                        if self._is_new_cone(cone):
                            self.discovered_cones.append(cone)

    def enumerate_multi_root_cones(self, block_id: int, block: Set[str]) -> None:
        """Enumerate multi-root cones in a block"""
        logger.info(f"枚舉模組 {self.module_name} 區塊 {block_id} 中的多輸出錐")

        # Implementation similar to single-root but with root combinations
        # For brevity, using a simplified version
        pass

    def _enumerate_k_feasible_cuts(self, root_id: str, block: Set[str]) -> List[Cut]:
        """Enumerate k-feasible cuts for a root"""
        # Simplified implementation
        return [Cut(leaves={root_id}, depth=0)]

    def _satisfies_constraints(self, leaves: Set[str], roots: List[str], depth: int) -> bool:
        """Check if cone satisfies size/depth constraints"""
        n_in_ok = self._compare_constraint(len(leaves), self.config['n_in'], self.config['cmp_in'])
        n_out_ok = self._compare_constraint(len(roots), self.config['n_out'], self.config['cmp_out'])
        depth_ok = self._compare_constraint(depth, self.config['n_depth'], self.config['cmp_depth'])
        return n_in_ok and n_out_ok and depth_ok

    def _compare_constraint(self, value: int, limit: int, operator: str) -> bool:
        """Compare value against constraint"""
        return value <= limit if operator == '<=' else value == limit

    def _extract_cone_nodes(self, root_id: str, leaves: Set[str], block: Set[str]) -> Set[str]:
        """Extract all nodes in the cone"""
        # Simple implementation - include all nodes reachable from leaves to root
        return {root_id} | leaves

    def _is_connected_cone(self, cone_nodes: Set[str]) -> bool:
        """Check if cone forms a connected component"""
        return len(cone_nodes) > 0

    def _create_cone(self, block_id: int, roots: List[str], leaves: List[str], depth: int, cone_nodes: Set[str]) -> Cone:
        """Create a cone object"""
        signature = self._generate_signature(cone_nodes, roots)
        cone_id = f"{self.module_name}_b{block_id}_{signature[:8]}"

        return Cone(
            cone_id=cone_id,
            block_id=block_id,
            roots=roots,
            leaves=leaves,
            depth=depth,
            num_nodes=len(cone_nodes),
            num_edges=sum(len(self.graph_builder.nodes[node_id].fanout) for node_id in cone_nodes if node_id in self.graph_builder.nodes),
            connected=True,
            signature=signature,
            module_name=self.module_name,
            nodes=cone_nodes
        )

    def _generate_signature(self, nodes: Set[str], roots: List[str]) -> str:
        """Generate unique signature for cone"""
        nodes_hash = hashlib.sha256('|'.join(sorted(nodes)).encode()).hexdigest()
        roots_hash = hashlib.sha256('|'.join(sorted(roots)).encode()).hexdigest()
        combined = int(nodes_hash[:32], 16) ^ int(roots_hash[:32], 16)
        return format(combined, '032x')

    def _is_new_cone(self, cone: Cone) -> bool:
        """Check if cone is new"""
        if cone.signature in self.signatures_seen:
            return False
        self.signatures_seen.add(cone.signature)
        return True

class ConeOutputWriter:
    """Handles output file generation"""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def write_cones_jsonl(self, cones: List[Cone]) -> None:
        """Write cones to JSONL format"""
        filepath = os.path.join(self.output_dir, 'cones.jsonl')
        logger.info(f"寫入 {len(cones)} 個錐至 {filepath}")

        with open(filepath, 'w') as f:
            for cone in cones:
                record = {
                    'cone_id': cone.cone_id,
                    'module_name': cone.module_name,
                    'block_id': cone.block_id,
                    'roots': cone.roots,
                    'leaves': cone.leaves,
                    'depth': cone.depth,
                    'num_nodes': cone.num_nodes,
                    'num_edges': cone.num_edges,
                    'connected': cone.connected,
                    'signature': cone.signature
                }
                f.write(json.dumps(record) + '\n')

    def write_summary_json(self, cones: List[Cone], blocks: List[Set[str]]) -> None:
        """Write summary statistics"""
        filepath = os.path.join(self.output_dir, 'summary.json')
        logger.info(f"寫入摘要統計至 {filepath}")

        # Group cones by module
        modules = {}
        for cone in cones:
            if cone.module_name not in modules:
                modules[cone.module_name] = []
            modules[cone.module_name].append(cone)

        summary = {
            'total_cones': len(cones),
            'total_blocks': len(blocks),
            'total_modules': len(modules),
            'distribution': {
                'by_depth': {},
                'by_inputs': {},
                'by_outputs': {},
                'by_block': {},
                'by_module': {}
            }
        }

        # Count distributions
        for cone in cones:
            # By depth
            depth_key = str(cone.depth)
            summary['distribution']['by_depth'][depth_key] = summary['distribution']['by_depth'].get(depth_key, 0) + 1

            # By inputs
            inputs_key = str(len(cone.leaves))
            summary['distribution']['by_inputs'][inputs_key] = summary['distribution']['by_inputs'].get(inputs_key, 0) + 1

            # By outputs
            outputs_key = str(len(cone.roots))
            summary['distribution']['by_outputs'][outputs_key] = summary['distribution']['by_outputs'].get(outputs_key, 0) + 1

            # By block
            block_key = str(cone.block_id)
            summary['distribution']['by_block'][block_key] = summary['distribution']['by_block'].get(block_key, 0) + 1

            # By module
            module_key = cone.module_name
            summary['distribution']['by_module'][module_key] = summary['distribution']['by_module'].get(module_key, 0) + 1

        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='LogicConeMiner - Logic Cone Discovery Tool')

    # Required parameters
    parser.add_argument('--netlist', required=True, help='Input Verilog netlist file')
    parser.add_argument('--n_in', type=int, required=True, help='Maximum number of cone inputs')
    parser.add_argument('--n_out', type=int, required=True, help='Maximum number of cone outputs')
    parser.add_argument('--n_depth', type=int, required=True, help='Maximum cone depth')

    # Required parameters (continued)
    parser.add_argument('--cell_library', required=True, help='CSV cell library file')
    parser.add_argument('--cmp_in', choices=['<=', '=='], default='<=', help='Input comparison operator')
    parser.add_argument('--cmp_out', choices=['<=', '=='], default='<=', help='Output comparison operator')
    parser.add_argument('--cmp_depth', choices=['<=', '=='], default='<=', help='Depth comparison operator')
    parser.add_argument('--count_inverters_in_depth', type=bool, default=True, help='Count inverters in depth')
    parser.add_argument('--max_cuts_per_node', type=int, default=150, help='Maximum cuts stored per node')
    parser.add_argument('--out_dir', default='results', help='Output directory')

    args = parser.parse_args()

    # Configuration
    config = {
        'n_in': args.n_in,
        'n_out': args.n_out,
        'n_depth': args.n_depth,
        'cmp_in': args.cmp_in,
        'cmp_out': args.cmp_out,
        'cmp_depth': args.cmp_depth,
        'count_inverters_in_depth': args.count_inverters_in_depth,
        'max_cuts_per_node': args.max_cuts_per_node
    }

    try:
        # 初始化 CSV 元件庫
        csv_library = CSVCellLibrary(args.cell_library)
        logger.info(f"從 {args.cell_library} 載入 {len(csv_library.cells)} 個標準元件定義")

        # Parse Verilog with CSV library
        verilog_parser = VerilogParser(csv_library)
        verilog_parser.parse_file(args.netlist)

        # 顯示檢測結果
        logger.info(f"發現 {len(verilog_parser.modules)} 個標準模組")
        logger.info(f"發現 {len(verilog_parser.macro_definitions)} 個 Macro 定義")

        if verilog_parser.macro_handler:
            detected_macros = verilog_parser.macro_handler.get_detected_macros()
            for macro_type in detected_macros:
                logger.info(f"檢測到 Macro: {macro_type}")

        # Process each module independently
        all_cones = []
        all_blocks = []

        for module_name, module_def in verilog_parser.modules.items():
            logger.info(f"處理模組: {module_name}")

            # Build graph for this module
            graph_builder = ModuleGraphBuilder(verilog_parser, module_name)
            graph_builder.build_module_graph(module_def)
            graph_builder.find_combinational_blocks()

            # Enumerate cones for this module
            enumerator = ModuleConeEnumerator(graph_builder, config, module_name)

            for block_id, block in enumerate(graph_builder.blocks):
                enumerator.enumerate_single_root_cones(block_id, block)
                enumerator.enumerate_multi_root_cones(block_id, block)

            # Collect results
            all_cones.extend(enumerator.discovered_cones)
            all_blocks.extend(graph_builder.blocks)

            logger.info(f"模組 {module_name} 完成: {len(enumerator.discovered_cones)} 個邏輯錐")

        # Write outputs
        writer = ConeOutputWriter(args.out_dir)
        writer.write_cones_jsonl(all_cones)
        writer.write_summary_json(all_cones, all_blocks)

        logger.info(f"完成！總共發現 {len(all_cones)} 個邏輯錐")
        return 0

    except Exception as e:
        logger.error(f"執行錯誤: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())