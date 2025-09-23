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
import sys
import logging

# Import CSV cell library
from csv_cell_library import CSVCellLibrary

# Import modularized LogicConeMiner components
from verilog_parser import VerilogParser
from graph_builder import ModuleGraphBuilder
from cone_enumerator import ModuleConeEnumerator
from output_writer import ConeOutputWriter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='LogicConeMiner - Logic Cone Discovery Tool')

    # Required parameters
    parser.add_argument('--netlist', required=True, help='Input Verilog netlist file')
    parser.add_argument('--n_in', type=int, required=True, help='Maximum number of cone inputs')
    parser.add_argument('--n_out', type=int, required=True, help='Maximum number of cone outputs')
    parser.add_argument('--n_depth', type=int, required=True, help='Maximum cone depth')
    parser.add_argument('--cell_library', required=True, help='CSV cell library file')

    # Optional parameters
    parser.add_argument('--cmp_in', choices=['<=', '=='], default='<=', help='Input comparison operator')
    parser.add_argument('--cmp_out', choices=['<=', '=='], default='<=', help='Output comparison operator')
    parser.add_argument('--cmp_depth', choices=['<=', '=='], default='<=', help='Depth comparison operator')
    parser.add_argument('--count_inverters_in_depth', action='store_true', default=True,
                       help='Count inverters in depth calculation (default: True)')
    parser.add_argument('--no_count_inverters_in_depth', action='store_false', dest='count_inverters_in_depth',
                       help='Do not count inverters in depth calculation')
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
        # Initialize CSV cell library
        csv_library = CSVCellLibrary(args.cell_library)
        logger.info(f"Loaded {len(csv_library.cells)} standard cell definitions from {args.cell_library}")

        # Parse Verilog with CSV library
        verilog_parser = VerilogParser(csv_library)
        verilog_parser.parse_file(args.netlist)

        # Display detection results
        logger.info(f"Found {len(verilog_parser.modules)} standard modules")
        logger.info(f"Found {len(verilog_parser.macro_definitions)} macro definitions")


        # Process each module independently
        all_cones = []
        all_blocks = []

        for module_name, module_def in verilog_parser.modules.items():
            logger.info(f"Processing module: {module_name}")

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

            logger.info(f"Module {module_name} completed: {len(enumerator.discovered_cones)} logic cones")

        # Write outputs
        writer = ConeOutputWriter(args.out_dir)
        writer.write_cones_jsonl(all_cones)
        writer.write_summary_json(all_cones, all_blocks)

        logger.info(f"Complete! Found {len(all_cones)} logic cones in total")
        return 0

    except Exception as e:
        logger.error(f"Execution error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())