#!/usr/bin/env python3
"""
LogicConeMiner - Output Writer

Handles result output, file writing and statistics generation
"""

import json
import os
import logging
from typing import List, Set

from data_structures import Cone

logger = logging.getLogger(__name__)

class ConeOutputWriter:
    """Handles output file generation"""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def write_cones_jsonl(self, cones: List[Cone]) -> None:
        """Write cones to JSONL format"""
        filepath = os.path.join(self.output_dir, 'cones.jsonl')
        logger.info(f"Writing {len(cones)} cones to {filepath}")

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
        logger.info(f"Writing summary statistics to {filepath}")

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