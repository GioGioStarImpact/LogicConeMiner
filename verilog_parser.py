#!/usr/bin/env python3
"""
LogicConeMiner - Verilog Parser

Handles Verilog file parsing, module detection and Black Box processing
"""

import re
import logging
from typing import Dict, List, Set

from data_structures import ModuleDefinition
from csv_cell_library import CSVCellLibrary

logger = logging.getLogger(__name__)

class VerilogParser:
    """Enhanced Verilog parser with multi-module and CSV cell library support"""

    # Verilog identifier pattern: supports both regular (word characters + $) and escaped identifiers
    # Escaped identifiers: backslash followed by non-backslash chars until mandatory space
    IDENTIFIER_PATTERN = r'(?:\\[^\\]+\s|[\w$]+)'

    def __init__(self, csv_library: CSVCellLibrary = None):
        self.modules: Dict[str, ModuleDefinition] = {}  # module_name -> ModuleDefinition
        self.macro_definitions: Dict[str, ModuleDefinition] = {}  # macro_name -> ModuleDefinition
        self.csv_library = csv_library
        self.black_box_modules: Set[str] = set()  # Undefined modules (black box)

    def _normalize_identifier(self, identifier: str) -> str:
        """Normalize escaped identifier by removing leading backslash and trailing whitespace"""
        if identifier.startswith('\\'):
            # Remove leading \ and trailing whitespace
            return identifier[1:].rstrip()
        return identifier

    def is_sequential(self, cell_type: str) -> bool:
        """Check if cell type is sequential using CSV library"""
        if self.csv_library and self.csv_library.is_standard_cell(cell_type):
            return self.csv_library.is_sequential(cell_type)

        # Non-standard components (Macros) are treated as non-sequential, but handled as boundaries
        return False

    def is_module_instantiation(self, cell_type: str) -> bool:
        """Check if cell_type is a module (defined or black box)"""
        return (cell_type in self.modules or
                cell_type in self.macro_definitions or
                cell_type in self.black_box_modules)

    def is_black_box_module(self, cell_type: str) -> bool:
        """Check if cell_type is a black box module"""
        return cell_type in self.black_box_modules

    def detect_black_box_modules(self) -> None:
        """Detect black box modules and show alerts"""
        all_instantiated_modules = set()

        # Collect all instantiated modules
        for module_def in self.modules.values():
            for _, (cell_type, _) in module_def.cells.items():
                if not self.csv_library or not self.csv_library.is_standard_cell(cell_type):
                    all_instantiated_modules.add(cell_type)

        # Check which modules are not defined
        for module_name in all_instantiated_modules:
            if (module_name not in self.modules and
                module_name not in self.macro_definitions):
                self.black_box_modules.add(module_name)
                logger.warning(f"⚠️  ALERT: Module '{module_name}' is undefined, will be treated as Black Box")

    def infer_black_box_port_direction(self, instance_name: str, cell_type: str,
                                     connections: Dict[str, str],
                                     module_def: ModuleDefinition) -> Dict[str, str]:
        """Infer port directions for black box modules"""
        port_directions = {}  # port_name -> 'input' or 'output'

        # Analyze connections, infer direction based on net driving conditions
        for port, net in connections.items():
            # Check if this net has other drivers
            has_other_driver = False
            is_driven_by_pi = False

            for other_instance, (other_cell_type, other_connections) in module_def.cells.items():
                if other_instance == instance_name:
                    continue

                # Check if other instance outputs drive this net
                if self.csv_library and self.csv_library.is_standard_cell(other_cell_type):
                    output_ports = self.csv_library.get_output_pins(other_cell_type)
                    for out_port, out_net in other_connections.items():
                        if out_port in output_ports and out_net == net:
                            has_other_driver = True
                            break

            # Check if driven by primary input
            if net in module_def.input_ports:
                is_driven_by_pi = True

            # Infer port direction
            if has_other_driver or is_driven_by_pi:
                port_directions[port] = 'input'  # Driven by others, so it's input
            else:
                port_directions[port] = 'output'  # No other drivers, inferred as output

        logger.info(f"Black Box '{cell_type}' port direction inference: {port_directions}")
        return port_directions

    def parse_file(self, filename: str) -> None:
        """Parse Verilog file and extract all modules"""
        logger.info(f"Parsing Verilog file: {filename}")

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
        module_pattern = rf'module\s+({self.IDENTIFIER_PATTERN})\s*\((.*?)\);(.*?)endmodule'
        for match in re.finditer(module_pattern, content, re.DOTALL):
            module_name, ports_str, module_body = match.groups()
            module_name = self._normalize_identifier(module_name)

            logger.info(f"Parsing module: {module_name}")
            module_def = self._parse_module(module_name, ports_str, module_body)

            # Determine if it's a standard module or Macro
            if self._is_module_macro(module_def):
                self.macro_definitions[module_name] = module_def
                logger.info(f"Detected Macro definition: {module_name}")
            else:
                self.modules[module_name] = module_def
                logger.info(f"Detected standard module: {module_name}")

        logger.info(f"Parsing complete: {len(self.modules)} standard modules, {len(self.macro_definitions)} Macro definitions")

        # Detect black box modules
        self.detect_black_box_modules()
        if self.black_box_modules:
            logger.info(f"Detected {len(self.black_box_modules)} Black Box modules: {list(self.black_box_modules)}")

    def _parse_module(self, module_name: str, ports_str: str, module_body: str) -> ModuleDefinition:
        """Parse a single module"""
        module_def = ModuleDefinition(name=module_name)

        # Parse port declarations from module header and body
        self._parse_port_declarations(module_def, ports_str, module_body)

        # Parse cell instances
        self._parse_cell_instances(module_def, module_body)

        return module_def

    def _parse_port_declarations(self, module_def: ModuleDefinition, ports_str: str, module_body: str) -> None:
        """Parse input/output port declarations"""

        # Parse from port list first (module declaration)
        self._parse_ports_from_port_list(module_def, ports_str)

        # Parse from module body (internal declarations)
        self._parse_ports_from_module_body(module_def, module_body)

        # Remove duplicates while preserving order
        module_def.input_ports = list(dict.fromkeys(module_def.input_ports))
        module_def.output_ports = list(dict.fromkeys(module_def.output_ports))

    def _parse_ports_from_port_list(self, module_def: ModuleDefinition, ports_str: str) -> None:
        """Parse ports from module declaration port list"""
        # Handle: module name(input clk, input [7:0] data, output result);

        # Input ports from port list
        input_pattern = rf'input\s+(?:\[[^\]]+\]\s*)?({self.IDENTIFIER_PATTERN})'
        for match in re.finditer(input_pattern, ports_str):
            port_name = self._normalize_identifier(match.group(1))
            module_def.input_ports.append(port_name)

        # Output ports from port list
        output_pattern = rf'output\s+(?:\[[^\]]+\]\s*)?({self.IDENTIFIER_PATTERN})'
        for match in re.finditer(output_pattern, ports_str):
            port_name = self._normalize_identifier(match.group(1))
            module_def.output_ports.append(port_name)

    def _parse_ports_from_module_body(self, module_def: ModuleDefinition, module_body: str) -> None:
        """Parse ports from module body declarations"""
        # Handle: input [7:0] data; output result; input \escaped name ;

        # Input ports from module body - support both with and without width
        input_pattern = rf'input\s+(?:\[[^\]]+\]\s*)?({self.IDENTIFIER_PATTERN})\s*;'
        for match in re.finditer(input_pattern, module_body):
            port_name = self._normalize_identifier(match.group(1))
            module_def.input_ports.append(port_name)

        # Output ports from module body - support both with and without width
        output_pattern = rf'output\s+(?:\[[^\]]+\]\s*)?({self.IDENTIFIER_PATTERN})\s*;'
        for match in re.finditer(output_pattern, module_body):
            port_name = self._normalize_identifier(match.group(1))
            module_def.output_ports.append(port_name)

    def _extract_port_names(self, ports_spec: str) -> List[str]:
        """Extract port names from declaration, handling bus notation"""
        port_names = []

        # Split by comma first
        for port_spec in ports_spec.split(','):
            port_spec = port_spec.strip()

            # Handle bus notation: [width] port_name or port_name[width] or simple port_name
            if '[' in port_spec:
                if port_spec.startswith('['):
                    # Format: [7:0] data_bus
                    bus_name = re.search(rf'\]\s*({self.IDENTIFIER_PATTERN})', port_spec)
                    if bus_name:
                        port_names.append(self._normalize_identifier(bus_name.group(1)))
                else:
                    # Format: data_bus[7:0] (old style)
                    base_name = re.search(rf'({self.IDENTIFIER_PATTERN})\s*\[', port_spec)
                    if base_name:
                        port_names.append(self._normalize_identifier(base_name.group(1)))
            else:
                # Simple port name
                wire_name = re.search(rf'({self.IDENTIFIER_PATTERN})', port_spec)
                if wire_name:
                    port_names.append(self._normalize_identifier(wire_name.group(1)))

        return port_names


    def _parse_cell_instances(self, module_def: ModuleDefinition, module_body: str) -> None:
        """Parse cell instances"""
        # Pattern: cell_type instance_name ( connections );
        # Note: for escaped identifiers, spaces are already included, so use \s* instead of \s+
        instance_pattern = rf'({self.IDENTIFIER_PATTERN})\s*({self.IDENTIFIER_PATTERN})\s*\((.*?)\)\s*;'

        for match in re.finditer(instance_pattern, module_body, re.DOTALL):
            cell_type, instance_name, connections_str = match.groups()
            cell_type = self._normalize_identifier(cell_type)
            instance_name = self._normalize_identifier(instance_name)

            # Skip module/endmodule keywords
            if cell_type in ['module', 'endmodule']:
                continue

            # Parse connections (.port(net), ...)
            connections = {}
            conn_pattern = rf'\.({self.IDENTIFIER_PATTERN})\s*\(\s*([^)]+)\s*\)'

            for conn_match in re.finditer(conn_pattern, connections_str):
                port, net = conn_match.groups()
                port = self._normalize_identifier(port)
                connections[port] = net.strip()

            module_def.cells[instance_name] = (cell_type, connections)


    def _is_module_macro(self, module_def: ModuleDefinition) -> bool:
        """Determine if module should be treated as Macro"""
        # Only treat as Macro when module is 'completely' composed of non-standard components
        # If mixed standard and non-standard components, treat as standard module (handles Macro boundaries internally)

        total_cells = len(module_def.cells)
        if total_cells == 0:
            return False  # Empty module treated as standard module

        non_standard_cells = 0
        for _, (cell_type, _) in module_def.cells.items():
            if not self.csv_library or not self.csv_library.is_standard_cell(cell_type):
                non_standard_cells += 1

        # If all are non-standard components, treat as Macro
        # If mixed or all standard components, treat as standard module
        return non_standard_cells == total_cells