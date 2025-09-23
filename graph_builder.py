#!/usr/bin/env python3
"""
LogicConeMiner - Graph Builder

Handles circuit graph construction, boundary processing and combinational logic block analysis
"""

import logging
from typing import Dict, List, Set

from data_structures import ModuleDefinition, Node
from verilog_parser import VerilogParser

logger = logging.getLogger(__name__)

class ModuleGraphBuilder:
    """Builds circuit graph for a single module"""

    def __init__(self, parser: VerilogParser, module_name: str):
        self.parser = parser
        self.module_name = module_name
        self.nodes: Dict[str, Node] = {}
        self.blocks: List[Set[str]] = []

    def build_module_graph(self, module_def: ModuleDefinition) -> None:
        """Build circuit graph for this module"""
        logger.info(f"Building circuit graph for module {self.module_name}...")

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
                # Sequential element - treat as boundary
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

            elif self.parser.is_module_instantiation(cell_type):
                # Module instantiation - treat as sequential boundary
                self._handle_module_instantiation(instance_name, cell_type, connections,
                                                module_def, net_drivers)

            else:
                # Standard combinational cell
                if self.parser.csv_library and self.parser.csv_library.is_standard_cell(cell_type):
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


        # Build connections
        for instance_name, (cell_type, connections) in module_def.cells.items():
            if self.parser.is_sequential(cell_type):
                # Sequential elements - do not build combinational connections (boundary)
                continue

            elif self.parser.is_module_instantiation(cell_type):
                # Module instantiation - build connections to boundary nodes
                self._build_module_connections(instance_name, cell_type, connections, net_drivers)

            else:
                # Standard combinational cells
                cell_output_node = None
                if self.parser.csv_library and self.parser.csv_library.is_standard_cell(cell_type):
                    output_ports = self.parser.csv_library.get_output_pins(cell_type)
                else:
                    output_ports = ['Y', 'Z', 'Q', 'OUT', 'O']

                for port in output_ports:
                    if port in connections:
                        cell_output_node = f"{instance_name}.{port}"
                        break

                if cell_output_node and cell_output_node in self.nodes:
                    # Connect inputs
                    if self.parser.csv_library and self.parser.csv_library.is_standard_cell(cell_type):
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

        logger.info(f"Module {self.module_name} graph construction complete: {len(self.nodes)} nodes")

    def _handle_module_instantiation(self, instance_name: str, cell_type: str,
                                   connections: Dict[str, str],
                                   module_def: ModuleDefinition,
                                   net_drivers: Dict[str, str]) -> None:
        """Handle module instantiation boundaries"""
        if self.parser.is_black_box_module(cell_type):
            # Black box module - infer port directions
            port_directions = self.parser.infer_black_box_port_direction(
                instance_name, cell_type, connections, module_def)

            for port, net in connections.items():
                direction = port_directions.get(port, 'input')  # Default to input

                if direction == 'output':
                    # Module output - treat as pseudo PI (for subsequent logic)
                    node_id = f"{instance_name}.{port}_OUT"
                    node = Node(
                        id=node_id,
                        type=f"MODULE_OUTPUT_{cell_type}",
                        is_pi=True,  # Input source for combinational logic
                        module_name=self.module_name
                    )
                    self.nodes[node_id] = node
                    net_drivers[net] = node_id
                else:
                    # Module input - treat as pseudo PO (previous logic endpoint)
                    node_id = f"{instance_name}.{port}_IN"
                    node = Node(
                        id=node_id,
                        type=f"MODULE_INPUT_{cell_type}",
                        is_po=True,  # Output endpoint for combinational logic
                        module_name=self.module_name
                    )
                    self.nodes[node_id] = node

        else:
            # Defined module - handle according to module definition
            if cell_type in self.parser.modules:
                target_module = self.parser.modules[cell_type]
            elif cell_type in self.parser.macro_definitions:
                target_module = self.parser.macro_definitions[cell_type]
            else:
                # Should not happen, but for safety
                logger.warning(f"Module {cell_type} definition not found")
                return

            for port, net in connections.items():
                if port in target_module.input_ports:
                    # Connect to module input - treat as pseudo PO
                    node_id = f"{instance_name}.{port}_IN"
                    node = Node(
                        id=node_id,
                        type=f"MODULE_INPUT_{cell_type}",
                        is_po=True,
                        module_name=self.module_name
                    )
                    self.nodes[node_id] = node
                elif port in target_module.output_ports:
                    # Module output - treat as pseudo PI
                    node_id = f"{instance_name}.{port}_OUT"
                    node = Node(
                        id=node_id,
                        type=f"MODULE_OUTPUT_{cell_type}",
                        is_pi=True,
                        module_name=self.module_name
                    )
                    self.nodes[node_id] = node
                    net_drivers[net] = node_id

        logger.info(f"Processing module instantiation: {instance_name} ({cell_type}) - {'Black Box' if self.parser.is_black_box_module(cell_type) else 'Defined Module'}")

    def _build_module_connections(self, instance_name: str, _cell_type: str,
                                connections: Dict[str, str],
                                net_drivers: Dict[str, str]) -> None:
        """Build connections to module boundary nodes"""
        for port, net in connections.items():
            # Build appropriate connections based on port direction
            input_node_id = f"{instance_name}.{port}_IN"
            output_node_id = f"{instance_name}.{port}_OUT"

            # Check if it's an input port (pseudo PO)
            if input_node_id in self.nodes:
                # Build connection driving to module input
                if net in net_drivers:
                    driver = net_drivers[net]
                    self.nodes[driver].fanout.append(input_node_id)
                    self.nodes[input_node_id].fanin.append(driver)
                elif net in self.nodes:  # Primary input
                    self.nodes[net].fanout.append(input_node_id)
                    self.nodes[input_node_id].fanin.append(net)

            # Check if it's an output port (pseudo PI)
            if output_node_id in self.nodes:
                # Module output already registered in net_drivers, used by other logic
                pass

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

        logger.info(f"Module {self.module_name} found {len(self.blocks)} combinational logic blocks")

    def _dfs_block(self, node_id: str, boundary_nodes: Set[str], visited: Set[str], block: Set[str]) -> None:
        """DFS to find connected combinational components"""
        if node_id in visited or node_id in boundary_nodes or node_id not in self.nodes:
            return

        visited.add(node_id)
        block.add(node_id)

        node = self.nodes[node_id]
        for neighbor in node.fanin + node.fanout:
            self._dfs_block(neighbor, boundary_nodes, visited, block)