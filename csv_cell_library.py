#!/usr/bin/env python3
"""
CSV Standard Cell Library Parser
Uses CSV files to define standard cell information, avoiding direct use of confidential Liberty files
"""

import csv
import os
from typing import Dict, List, Set, Optional
from dataclasses import dataclass

@dataclass
class CellDefinition:
    """Standard cell definition"""
    name:          str
    cell_type:     str  # "combinational" or "sequential"
    input_pins:    List[str]
    output_pins:   List[str]
    is_sequential: bool
    clock_pin:     Optional[str] = None
    data_pin:      Optional[str] = None

class CSVCellLibrary:
    """CSV standard cell library parser"""

    def __init__(self, csv_file: str):
        self.cells: Dict[str, CellDefinition] = {}

        if not csv_file:
            raise ValueError("CSV cell library file must be provided")

        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV cell library file not found: {csv_file}")

        self.load_from_csv(csv_file)

    def load_from_csv(self, csv_file: str) -> None:
        """Load cell definitions from CSV file"""
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)

                for row in reader:
                    cell_name = row['cell_name'].strip()
                    cell_type = row['cell_type'].strip()

                    # Parse input pins (comma-separated)
                    input_pins = [pin.strip() for pin in row['input_pins'].split(',') if pin.strip()]

                    # Parse output pins (comma-separated)
                    output_pins = [pin.strip() for pin in row['output_pins'].split(',') if pin.strip()]

                    # Parse sequential flag
                    is_sequential = row['is_sequential'].strip().lower() == 'true'

                    # Parse clock and data pins
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

            print(f"Loaded {len(self.cells)} standard cell definitions from {csv_file}")

        except Exception as e:
            raise RuntimeError(f"Failed to load CSV file: {e}")


    def is_standard_cell(self, cell_name: str) -> bool:
        """Check if cell is a standard cell"""
        return cell_name in self.cells

    def get_cell_definition(self, cell_name: str) -> Optional[CellDefinition]:
        """Get cell definition"""
        return self.cells.get(cell_name, None)

    def is_sequential(self, cell_name: str) -> bool:
        """Check if cell is sequential"""
        cell_def = self.get_cell_definition(cell_name)
        return cell_def.is_sequential if cell_def else False

    def get_input_pins(self, cell_name: str) -> List[str]:
        """Get input pin list"""
        cell_def = self.get_cell_definition(cell_name)
        return cell_def.input_pins if cell_def else []

    def get_output_pins(self, cell_name: str) -> List[str]:
        """Get output pin list"""
        cell_def = self.get_cell_definition(cell_name)
        return cell_def.output_pins if cell_def else []


    def export_template_csv(self, output_file: str) -> None:
        """Export CSV template file"""
        header = ['cell_name', 'cell_type', 'input_pins', 'output_pins',
                 'is_sequential', 'clock_pin', 'data_pin']

        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)

            # Write some examples
            examples = [
                ['AND2', 'combinational', 'A,B', 'Y', 'false', '', ''],
                ['DFF', 'sequential', 'D,CLK', 'Q', 'true', 'CLK', 'D'],
                ['MUX2', 'combinational', 'A,B,S', 'Y', 'false', '', ''],
            ]

            for example in examples:
                writer.writerow(example)

        print(f"CSV template exported to {output_file}")

