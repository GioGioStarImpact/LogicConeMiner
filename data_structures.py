#!/usr/bin/env python3
"""
LogicConeMiner - Data Structures

Contains all basic data structures for the logic cone mining system
"""

from typing import Dict, List, Set, Tuple
from dataclasses import dataclass, field

@dataclass
class ModuleDefinition:
    """Represents a module definition in the netlist"""
    name: str
    input_ports: List[str] = field(default_factory=list)
    output_ports: List[str] = field(default_factory=list)
    cells: Dict[str, Tuple[str, Dict]] = field(default_factory=dict)  # instance_name -> (cell_type, connections)

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