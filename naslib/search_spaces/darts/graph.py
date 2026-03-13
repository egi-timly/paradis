import os
import pickle
import numpy as np
import random
import logging
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F

from naslib.search_spaces.core import primitives as ops
from naslib.search_spaces.core.graph import Graph, EdgeData
from naslib.search_spaces.core.query_metrics import Metric
from naslib.search_spaces.darts.conversions import (
    convert_op_indices_to_naslib,
    convert_naslib_to_op_indices,
    is_valid_arch
)

from .primitives import DARTSConcat, FactorizedReduce, AuxiliaryHeadCIFAR

logger = logging.getLogger(__name__)

NUM_CELLS = 2
NUM_EDGES = 14
NUM_INPUTS_PER_NODE = 2
NUM_OPS = 8

CELL_MULT = 4

OP_NAMES = ["Identity", "Zero", "SepConv3x3", "SepConv5x5", "DilConv3x3", "DilConv5x5", "MaxPool1x1", "AvgPool1x1"]


class DartsSearchSpace(Graph):
    """
    Implementation of the DARTS search space.
    """

    OPTIMIZER_SCOPE = [
        "first_layer",
        "second_layer",
        "stage_1",
        "reduction_1",
        "rr_stage_2",
        "post_rr_stage_2",
        "stage_2",
        "reduction_2",
        "rr_stage_3",
        "post_rr_stage_3",
        "stage_3",
    ]

    OPTIMIZER_SCOPE_REGULARS = [
        "first_layer",
        "second_layer",
        "stage_1",
        "post_rr_stage_2",
        "stage_2",
        "post_rr_stage_3",
        "stage_3",
    ]

    OPTIMIZER_SCOPE_REDUCTIONS = [
        "rr_stage_2",
        "rr_stage_3",
    ]

    QUERYABLE = True

    def __init__(self, n_classes=10, in_channels=3, init_channels=16, auxiliary=False):
        super().__init__()
        self.num_classes = n_classes
        self.op_indices = None

        self.max_epoch = 199
        self.in_channels = in_channels
        self.space_name = "darts"
        self.labeled_archs = None
        self.instantiate_model = True
        self.sample_without_replacement = False

        #
        # Makrograph definition
        #
        self.name = "makrograph"

        # Cell is on the edges
        # 1-2:               Preprocessing
        # 2-3, ..., 6-7:     cells stage 1
        # 7-8:               residual block stride 2
        # 8-9, ..., 12-13:   cells stage 2
        # 13-14:             residual block stride 2
        # 14-15, ..., 18-19: cells stage 3
        # 19-20:             post-processing

        total_num_nodes_main = 23
        self.add_nodes_from(range(1, total_num_nodes_main))
        self.add_edges_from([(i, i + 1) for i in range(1, total_num_nodes_main - 1)])
        self.add_edges_from([(i, i + 2) for i in range(1, total_num_nodes_main - 2)])

        # Auxiliary endpoint
        self.add_node(23)

        # Network endpoint
        self.add_node(24)

        # Last edge
        self.add_edge(22, 24)
        # Auxiliary branch
        self.add_edge(15, 23)

        self.channels = {
                    "first_layer": init_channels,
                    "second_layer": init_channels,
                    "stage_1": init_channels, 
                    "reduction_1": init_channels,
                    "rr_stage_2": init_channels*2,
                    "post_rr_stage_2": init_channels*2,
                    "stage_2": init_channels*2, 
                    "reduction_2": init_channels*2,
                    "rr_stage_3": init_channels*4,
                    "post_rr_stage_3": init_channels*4,
                    "stage_3": init_channels*4
                    }

        #regular cell template
        cell = self.make_regular_cell()

        #reduction cell templates
        reduction_cell_1 = self.make_reduction_cell(scope_reduction='reduction_1')
        reduction_cell_2 = self.make_reduction_cell(scope_reduction='reduction_2')

        #
        # operations at the edges
        #

        # preprocessing
        self.edges[1, 2].set("op", ops.Stem(C_in=self.in_channels,
                                            C_out=self.channels['first_layer']))

        # stage 1
        self.edges[2, 3].set("op", ops.Identity())
        self.nodes[3]["subgraph"] = cell.copy().set_scope('first_layer', recursively=False).set_input([2, 2])
        self.edges[2, 4].set("op", ops.Identity())
        self.edges[3, 4].set("op", ops.Identity())
        self.nodes[4]["subgraph"] = cell.copy().set_scope('second_layer', recursively=False).set_input([2, 3])
        for i in range(5, 8):
            self.edges[i - 1, i].set("op", ops.Identity())
            self.edges[i - 2, i].set("op", ops.Identity())
            self.nodes[i]["subgraph"] = cell.copy().set_scope('stage_1', recursively=False).set_input([i - 2, i - 1])

        # stage 2
        self.edges[6, 8].set("op", ops.Identity())
        self.edges[7, 8].set("op", ops.Identity())
        self.nodes[8]["subgraph"] = reduction_cell_1.copy().set_scope('rr_stage_2', recursively=False).set_input([6, 7])
        self.edges[7, 9].set("op", ops.Identity())
        self.edges[8, 9].set("op", ops.Identity())
        self.nodes[9]["subgraph"] = cell.copy().set_scope('post_rr_stage_2', recursively=False).set_input([7, 8])
        for i in range(10, 15):
            self.edges[i - 1, i].set("op", ops.Identity())
            self.edges[i - 2, i].set("op", ops.Identity())
            self.nodes[i]["subgraph"] = cell.copy().set_scope('stage_2', recursively=False).set_input([i - 2, i - 1])

        # stage 3
        self.edges[13, 15].set("op", ops.Identity())
        self.edges[14, 15].set("op", ops.Identity())
        self.nodes[15]["subgraph"] = reduction_cell_2.copy().set_scope('rr_stage_3', recursively=False).set_input([13, 14])
        self.edges[14, 16].set("op", ops.Identity())
        self.edges[15, 16].set("op", ops.Identity())
        self.nodes[16]["subgraph"] = cell.copy().set_scope('post_rr_stage_3', recursively=False).set_input([14, 15])
        for i in range(17, 22):
            self.edges[i - 1, i].set("op", ops.Identity())
            self.edges[i - 2, i].set("op", ops.Identity())
            self.nodes[i]["subgraph"] = cell.copy().set_scope('stage_3', recursively=False).set_input([i - 2, i - 1])

        # post-processing
        self.edges[22, 24].set(
            "op",
            ops.Sequential(
                nn.BatchNorm2d(self.channels['stage_3']*CELL_MULT),
                nn.ReLU(inplace=False),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(self.channels['stage_3']*CELL_MULT, self.num_classes),
            ),
        )

        # auxiliary head
        if auxiliary:
            self.edges[15, 23].set(
                "op",
                AuxiliaryHeadCIFAR(
                    C=self.channels['rr_stage_3']*CELL_MULT,
                    num_classes=self.num_classes
                )
            )

        self._set_cell_ops()

    def make_regular_cell(self):

        #
        # Regular cell definition
        #
        cell = Graph()
        cell.name = "cell"  # Use the same name for all cells with shared attributes

        # Input nodes
        cell.add_node(1)
        cell.add_node(2)

        # Preprocess nodes
        cell.add_node(3)
        cell.add_node(4)

        # Intermediate nodes
        cell.add_node(5)
        cell.add_node(6)
        cell.add_node(7)
        cell.add_node(8)

        # Concat and output node
        cell.add_node(9)

        # Preprocess edges
        cell.add_edge(1, 3)
        cell.add_edge(2, 4)

        # Input edges
        cell.add_edge(3, 5)
        cell.add_edge(4, 5)

        cell.add_edge(3, 6)
        cell.add_edge(4, 6)

        cell.add_edge(3, 7)
        cell.add_edge(4, 7)

        cell.add_edge(3, 8)
        cell.add_edge(4, 8)

        # Intermediate edges
        cell.add_edge(5, 6)

        cell.add_edge(5, 7)
        cell.add_edge(6, 7)

        cell.add_edge(5, 8)
        cell.add_edge(6, 8)
        cell.add_edge(7, 8)

        # Concat edges
        cell.add_edge(5, 9)
        cell.add_edge(6, 9)
        cell.add_edge(7, 9)
        cell.add_edge(8, 9)

        # Fix preprocessing edges (add operations later)
        for edge in ((1, 3), (2, 4)):
            cell.edges[edge].set("fixed", True, shared=True)
            cell.edges[edge].set("preprocessing_{:}".format(edge[0]), True, shared=True)

        # Set concat edges to identity (no operation choice on these)
        for edge in ((5, 9), (6, 9), (7, 9), (8, 9)):
            cell.edges[edge].set("op", ops.Identity())
            cell.edges[edge].set("fixed", True, shared=True)

        # Add combination operation to concat node
        cell.nodes[9]["comb_op"] = DARTSConcat()

        return(cell)

    def make_reduction_cell(self, scope_reduction):

        #
        # Reduction cell definition
        #
        cell = Graph()
        cell.name = "reduction"  # Use the same name for all cells with shared attributes

        # Input nodes
        cell.add_node(1)
        cell.add_node(2)

        # Preprocess nodes
        cell.add_node(3)
        cell.add_node(4)

        #Intermediate nodes
        cell.add_node(5)
        cell.add_node(6)
        cell.add_node(7)
        cell.add_node(8)

        # Concat node
        cell.add_node(9)

        # Preprocess edges
        cell.add_edge(1, 3)
        cell.add_edge(2, 4)

        # Input edges (with reduction)
        cell.add_edge(3, 5)
        cell.add_edge(4, 5)

        cell.add_edge(3, 6)
        cell.add_edge(4, 6)

        cell.add_edge(3, 7)
        cell.add_edge(4, 7)

        cell.add_edge(3, 8)
        cell.add_edge(4, 8)

        # Intermediate edges
        cell.add_edge(5, 6)

        cell.add_edge(5, 7)
        cell.add_edge(6, 7)

        cell.add_edge(5, 8)
        cell.add_edge(6, 8)
        cell.add_edge(7, 8)

        # Concat edges
        cell.add_edge(5, 9)
        cell.add_edge(6, 9)
        cell.add_edge(7, 9)
        cell.add_edge(8, 9)

        # Fix preprocessing edges (add operations later)
        for edge in ((1, 3), (2, 4)):
            cell.edges[edge].set("fixed", True, shared=True)
            cell.edges[edge].set("preprocessing_{:}".format(edge[0]), True, shared=True)

        # Set concat edges to identity (no operation choice on these)
        for edge in ((5, 9), (6, 9), (7, 9), (8, 9)):
            cell.edges[edge].set("op", ops.Identity())
            cell.edges[edge].set("fixed", True, shared=True)

        # Add reduction subgraph to reduction edges
        for edge in ((3, 5), (4, 5), (3, 6), (4, 6), (3, 7), (4, 7), (3, 8), (4, 8)):
            cell.edges[edge].set("op", self.make_1in_connection(name='red_connection').set_scope(scope_reduction))
            cell.edges[edge].set("fixed", True, shared=True)

        # Add combination operation to concat node (WIP)
        cell.nodes[9]["comb_op"] = DARTSConcat()

        return(cell)
        
    def make_1in_connection(self, name='connection'):

        #
        # 1 input connection cell
        #
        connection = Graph()
        connection.name = name

        # Input node
        connection.add_node(1)

        # Output node
        connection.add_node(2)

        # Edge
        connection.add_edge(1, 2)

        return(connection)

    def _set_cell_ops(self):
        for i, scope in enumerate(self.OPTIMIZER_SCOPE):

            # set choice ops
            if scope in ["first_layer", "second_layer", "stage_1", "rr_stage_2", "post_rr_stage_2", "stage_2", "rr_stage_3", "post_rr_stage_3", "stage_3"]:
                c = self.channels[scope]
                self.update_edges(
                    update_func=lambda edge: DartsSearchSpace._set_ops(edge, C=c),
                    scope=scope,
                    private_edge_data=True,
                )
            elif scope in ["reduction_1", "reduction_2"]:
                c_in = self.channels[scope]
                c_out = c_in * 2
                self.update_edges(
                    update_func=lambda edge: DartsSearchSpace._set_ops_reduction(edge, C_in=c_in, C_out=c_out),
                    scope=scope,
                    private_edge_data=True,
                )

            #set preprocessing
            if scope in ["stage_1", "stage_2", "stage_3"]:
                c = self.channels[scope]
                self.update_edges(
                    update_func=lambda edge: DartsSearchSpace._set_preprocess_regular(edge, C_in=c, C_out=c),
                    scope=scope,
                    private_edge_data=True,
                )
            elif scope in ["rr_stage_2"]:
                c = self.channels["reduction_1"]
                self.update_edges(
                    update_func=lambda edge: DartsSearchSpace._set_preprocess_regular(edge, C_in=c, C_out=c),
                    scope=scope,
                    private_edge_data=True,
                )
            elif scope in ["rr_stage_3"]:
                c = self.channels["reduction_2"]
                self.update_edges(
                    update_func=lambda edge: DartsSearchSpace._set_preprocess_regular(edge, C_in=c, C_out=c),
                    scope=scope,
                    private_edge_data=True,
                )
            elif scope in ["post_rr_stage_2"]:
                c = self.channels[scope]
                c_prev = self.channels["reduction_1"]
                self.update_edges(
                    update_func=lambda edge: DartsSearchSpace._set_preprocess_post_rr(edge, C_in=c, C_in_in=c_prev, C_out=c),
                    scope=scope,
                    private_edge_data=True,
                )
            elif scope in ["post_rr_stage_3"]:
                c = self.channels[scope]
                c_prev = self.channels["reduction_2"]
                self.update_edges(
                    update_func=lambda edge: DartsSearchSpace._set_preprocess_post_rr(edge, C_in=c, C_in_in=c_prev, C_out=c),
                    scope=scope,
                    private_edge_data=True,
                )
            elif scope in ["first_layer"]:
                c = self.channels[scope]
                self.update_edges(
                    update_func=lambda edge: DartsSearchSpace._set_preprocess_first_layer(edge, C_in=c, C_out=c),
                    scope=scope,
                    private_edge_data=True,
                )
            elif scope in ["second_layer"]:
                c = self.channels[scope]
                self.update_edges(
                    update_func=lambda edge: DartsSearchSpace._set_preprocess_second_layer(edge, C_in=c, C_out=c),
                    scope=scope,
                    private_edge_data=True,
                )

    @staticmethod
    def _set_ops(edge, C):
        if not edge.data.has("fixed") or not edge.data["fixed"]:
            edge.data.set(
                "op",
                [
                    ops.Identity(),
                    ops.Zero(stride=1),
                    ops.SepConv(C, C, kernel_size=3, stride=1, padding=1, affine=False, track_running_stats=False),
                    ops.SepConv(C, C, kernel_size=5, stride=1, padding=2, affine=False, track_running_stats=False),
                    ops.DilConv(C, C, kernel_size=3, stride=1, padding=2, dilation=2, affine=False, track_running_stats=False),
                    ops.DilConv(C, C, kernel_size=5, stride=1, padding=4, dilation=2, affine=False, track_running_stats=False),
                    ops.MaxPool1x1(C_in=C, C_out=C, kernel_size=3, stride=1, affine=False),
                    ops.AvgPool1x1(C_in=C, C_out=C, kernel_size=3, stride=1, affine=False),
                ],
            )

    @staticmethod
    def _set_ops_reduction(edge, C_in, C_out):
        if not edge.data.has("fixed") or not edge.data["fixed"]:
            edge.data.set(
                "op",
                [
                    None,
                    ops.Zero(stride=2, C_in=C_in, C_out=C_out),
                    ops.SepConv(C_in, C_out, kernel_size=3, stride=2, padding=1, affine=False, track_running_stats=False),
                    ops.SepConv(C_in, C_out, kernel_size=5, stride=2, padding=2, affine=False, track_running_stats=False),
                    ops.DilConv(C_in, C_out, kernel_size=3, stride=2, padding=2, dilation=2, affine=False, track_running_stats=False),
                    ops.DilConv(C_in, C_out, kernel_size=5, stride=2, padding=4, dilation=2, affine=False, track_running_stats=False),
                    ops.MaxPool1x1(C_in=C_in, C_out=C_out, kernel_size=3, stride=2, affine=False),
                    ops.AvgPool1x1(C_in=C_in, C_out=C_out, kernel_size=3, stride=2, affine=False),
                ],
            )

    @staticmethod
    def _set_preprocess_regular(edge, C_in, C_out):
        if (edge.data.has("preprocessing_1") and edge.data["preprocessing_1"]) or (edge.data.has("preprocessing_2") and edge.data["preprocessing_2"]):
            edge.data.set(
                "op",
                ops.ReLUConvBN(C_in*CELL_MULT, C_out, kernel_size=1, stride=1, padding=0)
            )

    @staticmethod
    def _set_preprocess_post_rr(edge, C_in, C_in_in, C_out):
        if edge.data.has("preprocessing_1") and edge.data["preprocessing_1"]:
            edge.data.set(
                "op",
                FactorizedReduce(C_in_in*CELL_MULT, C_out)
            )
        elif edge.data.has("preprocessing_2") and edge.data["preprocessing_2"]:
            edge.data.set(
                "op",
                ops.ReLUConvBN(C_in*CELL_MULT, C_out, kernel_size=1, stride=1, padding=0)
            )

    @staticmethod
    def _set_preprocess_first_layer(edge, C_in, C_out):
        if (edge.data.has("preprocessing_1") and edge.data["preprocessing_1"]) or (edge.data.has("preprocessing_2") and edge.data["preprocessing_2"]):
            edge.data.set(
                "op",
                ops.ReLUConvBN(C_in, C_out, kernel_size=1, stride=1, padding=0)
            )

    @staticmethod
    def _set_preprocess_second_layer(edge, C_in, C_out):
        if edge.data.has("preprocessing_1") and edge.data["preprocessing_1"]:
            edge.data.set(
                "op",
                ops.ReLUConvBN(C_in, C_out, kernel_size=1, stride=1, padding=0)
            )
        elif edge.data.has("preprocessing_2") and edge.data["preprocessing_2"]:
            edge.data.set(
                "op",
                ops.ReLUConvBN(C_in*CELL_MULT, C_out, kernel_size=1, stride=1, padding=0)
            )

    def get_op_indices(self):
        if self.op_indices is None:
            self.op_indices = convert_naslib_to_op_indices(self)
        return self.op_indices

    def get_hash(self):
        return tuple(self.get_op_indices())

    def get_arch_iterator(self, dataset_api=None):
        return itertools.product(range(NUM_OPS), repeat=NUM_CELLS*NUM_EDGES)

    def set_op_indices(self, op_indices):
        # This will update the edges in the naslib object to op_indices
        self.op_indices = op_indices

        if self.instantiate_model == True:
            convert_op_indices_to_naslib(op_indices, self)

    def set_spec(self, op_indices, dataset_api=None):
        self.set_op_indices(op_indices)

    def set_training(self, recursively=True):
        #Sets training mode to graph and all subgraphs
        self.training = True
        if recursively:
            for g in self._get_child_graphs(single_instances=False):
                g.training = True

    def set_testing(self, recursively=True):
        #Sets testing mode to graph and all subgraphs
        self.training = False
        if recursively:
            for g in self._get_child_graphs(single_instances=False):
                g.training = False
        
    def set_drop_prob(self, drop_prob, recursively=True):
        #Sets drop prob of all searchable subgraphs to given value
        if recursively:
            for g in self._get_child_graphs(single_instances=False):
                if g.scope in self.OPTIMIZER_SCOPE:
                    g.drop_prob = drop_prob

    def clone_no_aux(self):
        #Clones the graph while removing the auxiliary head

        new_self = self.clone()
        new_self.edges[15, 23].set(
                "op",
                ops.Identity()
            )
        return(new_self)

    def sample_random_labeled_architecture(self):
        assert self.labeled_archs is not None, "Labeled archs not provided to sample from"

        op_indices = random.choice(self.labeled_archs)

        if self.sample_without_replacement == True:
            self.labeled_archs.pop(self.labeled_archs.index(op_indices))

        self.set_spec(op_indices)

    def sample_random_architecture(self, dataset_api=None, load_labeled=False):
        """
        This will sample a random architecture and update the edges in the
        naslib object accordingly.
        """

        edge_groups = [
            [0, 1],
            [2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12, 13]
        ]

        if load_labeled == True:
            return self.sample_random_labeled_architecture()

        while True:
            op_indices = np.ones((NUM_CELLS*NUM_EDGES,), dtype=int).tolist()
            choices_idx_regular = []
            choices_idx_reduction = []
            for group in edge_groups:
                choice = np.random.choice(group, size=NUM_INPUTS_PER_NODE, replace=False).tolist()
                choices_idx_regular += choice
            for group in edge_groups:
                choice = np.random.choice(group, size=NUM_INPUTS_PER_NODE, replace=False).tolist()
                choices_idx_reduction += choice

            non_zero_ops = list(range(NUM_OPS))[:1] + list(range(NUM_OPS))[2:]
            for idx in choices_idx_regular:
                op_indices[idx] = np.random.choice(non_zero_ops)
            for idx in choices_idx_reduction:
                op_indices[NUM_EDGES+idx] = np.random.choice(non_zero_ops)

            if not is_valid_arch(op_indices):
                continue

            self.set_op_indices(op_indices)
            break
        self.compact = self.get_op_indices()

    def mutate(self, parent, mode=None, dataset_api=None):
        """
        This will mutate one op from the parent op indices, and then
        update the naslib object and op_indices
        """

        connect_possible_moves = {
            0 : [], 1 : [],
            2 : [3, 4], 3 : [2, 4], 4 : [2, 3],
            5 : [6, 7, 8], 6 : [5, 7, 8], 7 : [5, 6, 8], 8 : [5, 6, 7],
            9 : [10, 11, 12, 13], 10 : [9, 11, 12, 13], 11 : [9, 10, 12, 13], 12 : [9, 10, 11, 13], 13 : [9, 10, 11, 12],
        }

        non_zero_ops = list(range(NUM_OPS))[:1] + list(range(NUM_OPS))[2:]

        def inner_mutate(parent, mode='op'):
            parent_op_indices = parent.get_op_indices()
            op_indices = list(parent_op_indices)

            if mode=='op':
                edge = np.random.choice(len(parent_op_indices))
                while parent_op_indices[edge] == 1:
                    edge = np.random.choice(len(parent_op_indices))
                available = [o for o in non_zero_ops if o != parent_op_indices[edge]]
                op_index = np.random.choice(available)
                op_indices[edge] = op_index

            elif mode=='connect':
                edge = np.random.choice(len(parent_op_indices))
                available = connect_possible_moves[edge%NUM_EDGES]
                while parent_op_indices[edge] == 1 or len(available)==0:
                    edge = np.random.choice(len(parent_op_indices))
                    available = connect_possible_moves[edge%NUM_EDGES]
                edge_target = np.random.choice(available)
                if edge>=NUM_EDGES:
                    edge_target += NUM_EDGES
                while parent_op_indices[edge_target] != 1:
                    edge_target = np.random.choice(available)
                    if edge >= NUM_EDGES:
                        edge_target += NUM_EDGES
                op_indices[edge_target] = op_indices[edge]
                op_indices[edge] = 1
                
            return(op_indices)

        if mode==None:
            mode = random.choice(['op', 'connect'])
        while True:
            op_indices = inner_mutate(parent, mode=mode)

            if not is_valid_arch(op_indices):
                continue
        
            self.set_op_indices(op_indices)
            break
        self.compact = self.get_op_indices()

    def crossover(self, parent0, parent1):
        """
        Set this graph's op_indices as a crossover of parent1 and parent2 indices
        """

        def inner_crossover(parent0, parent1):
            parent0_op_indices = list(parent0.get_op_indices())
            parent1_op_indices = list(parent1.get_op_indices())
            n_edges = len(parent0_op_indices)

            base_parent = np.random.choice(1)
            if base_parent==0:
                base_indices = parent0_op_indices
                crossover_indices = parent1_op_indices
            else:
                base_indices = parent1_op_indices
                crossover_indices = parent0_op_indices
            n_changes = n_edges//2
            for n in range(n_changes):
                edge = np.random.choice(n_edges)
                base_indices[edge] = crossover_indices[edge]

            return(base_indices)

        while True:
            op_indices = inner_crossover(parent0, parent1)

            if not is_valid_arch(op_indices):
                continue
        
            self.set_op_indices(op_indices)
            break
        self.compact = self.get_op_indices()

    def get_type(self):
        return "darts"

    def get_loss_fn(self):
        return F.cross_entropy

    def forward_before_global_avg_pool(self, x):
        outputs = []
        def hook_fn(module, inputs, output_t):
            # print(f'Input tensor shape: {inputs[0].shape}')
            # print(f'Output tensor shape: {output_t.shape}')
            outputs.append(inputs[0])

        for m in self.modules():
            if isinstance(m, torch.nn.AdaptiveAvgPool2d):
                m.register_forward_hook(hook_fn)

        self.forward(x, None)

        assert len(outputs) == 1
        return outputs[0]
