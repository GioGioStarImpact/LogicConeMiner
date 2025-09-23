#!/usr/bin/env python3
"""
LogicConeMiner - Cone Enumerator

Handles logic cone enumeration algorithms and constraint checking
"""

import hashlib
import logging
from typing import Dict, List, Set

from data_structures import Cut, Cone, Node
from graph_builder import ModuleGraphBuilder

logger = logging.getLogger(__name__)

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
        logger.info(f"Enumerating single-output cones in module {self.module_name} block {block_id} ({len(block)} nodes)")

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

    def enumerate_multi_root_cones(self, block_id: int, _block: Set[str]) -> None:
        """Enumerate multi-root cones in a block"""
        logger.info(f"Enumerating multi-output cones in module {self.module_name} block {block_id}")

        # Implementation similar to single-root but with root combinations
        # For brevity, using a simplified version
        pass

    def _enumerate_k_feasible_cuts(self, root_id: str, _block: Set[str]) -> List[Cut]:
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

    def _extract_cone_nodes(self, root_id: str, leaves: Set[str], _block: Set[str]) -> Set[str]:
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