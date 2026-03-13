import torch

from naslib.search_spaces.core.primitives import AbstractPrimitive

OP_NAMES = ["Identity", "Zero", "SepConv3x3", "SepConv5x5", "DilConv3x3", "DilConv5x5", "MaxPool1x1", "AvgPool1x1"]

EDGE_LIST_REGULAR = ((3, 5), (4, 5), (3, 6), (4, 6), (5, 6), (3, 7), (4, 7), (5, 7), (6, 7), (3, 8), (4, 8), (5, 8), (6, 8), (7, 8))
EDGE_LIST_REDUCTION_RED = ((3, 5), (4, 5), (3, 6), (4, 6), (3, 7), (4, 7), (3, 8), (4, 8))
EDGE_LIST_REDUCTION_REG = ((5, 6), (5, 7), (6, 7), (5, 8), (6, 8), (7, 8))

def convert_naslib_to_op_indices(naslib_object):

    cells = naslib_object._get_child_graphs(single_instances=True)
    #Find a regular cell and a reduction cell by scope
    found_regular = False
    found_reduction = False
    i = 0
    while i < len(cells) and (not found_regular or not found_reduction):
        cell = cells[i]
        if not found_regular and cell.name == 'cell':
            regular_cell = cell
            found_regular = True
        if not found_reduction and cell.name == 'reduction':
            reduction_cell = cell
            found_reduction = True
        i += 1
    assert found_regular and found_reduction, 'Did not find regular and reduction cells in naslib object'

    ops_regular = []
    ops_reduction = []
    for i, j in EDGE_LIST_REGULAR:
        ops_regular.append(regular_cell.edges[i, j]["op"].get_op_name)

    for i, j in EDGE_LIST_REGULAR:
        if (i, j) in EDGE_LIST_REDUCTION_RED:
            ops_reduction.append(reduction_cell.edges[i, j]["op"].edges[1, 2]["op"].get_op_name)
        elif (i, j) in EDGE_LIST_REDUCTION_REG:
            ops_reduction.append(reduction_cell.edges[i, j]["op"].get_op_name)

    return [OP_NAMES.index(name) for name in ops_regular] + [OP_NAMES.index(name) for name in ops_reduction]

def convert_op_indices_to_naslib(op_indices, naslib_object):

    assert len(op_indices) == 28, 'Wrong genome size'

    op_indices_regular = op_indices[:14]
    op_indices_reduction = op_indices[14:]

    # create dictionaries of edges to ops
    edge_op_dict_regular = {}
    edge_op_dict_reduction = {}
    for i, (index_reg, index_red) in enumerate(zip(op_indices_regular, op_indices_reduction)):
        edge_op_dict_regular[EDGE_LIST_REGULAR[i]] = OP_NAMES[index_reg]
        edge_op_dict_reduction[EDGE_LIST_REGULAR[i]] = OP_NAMES[index_red]

    def op_index_factory(cell_type):

        if cell_type == 'regular':
            def add_op_index(edge):
                # function that adds the op index from the dictionary to each edge
                if (edge.head, edge.tail) in edge_op_dict_regular:
                    for i, op in enumerate(edge.data.op):
                        if op is not None and op.get_op_name == edge_op_dict_regular[(edge.head, edge.tail)]:
                            index = i
                            break
                    edge.data.set("op_index", index, shared=True)

        elif cell_type == 'reduction':
            def add_op_index(edge):
                # function that adds the op index from the dictionary to each edge
                if (edge.head, edge.tail) in edge_op_dict_reduction:
                    if (edge.head, edge.tail) in EDGE_LIST_REDUCTION_RED:
                        for i, op in enumerate(edge.data.op.edges[1,2].op):
                            if op is not None and op.get_op_name == edge_op_dict_reduction[(edge.head, edge.tail)]:
                                index = i
                                break
                        edge.data.op.edges[1, 2].set("op_index", index, shared=True)
                    elif (edge.head, edge.tail) in EDGE_LIST_REDUCTION_REG:
                        for i, op in enumerate(edge.data.op):
                            if op is not None and op.get_op_name == edge_op_dict_reduction[(edge.head, edge.tail)]:
                                index = i
                                break
                        edge.data.set("op_index", index, shared=True)

        return(add_op_index)

    def update_ops_factory(cell_type):

        if cell_type=='regular':
            def update_ops(edge):
                # function that replaces the primitive ops at the edges with the one in op_index

                if (edge.head, edge.tail) in edge_op_dict_regular:

                    if isinstance(edge.data.op, list):
                        primitives = edge.data.op
                    else:
                        primitives = edge.data.primitives

                    chosen_op = primitives[edge.data.op_index]
                    primitives[edge.data.op_index] = update_batchnorms(chosen_op)

                    edge.data.set("op", primitives[edge.data.op_index])
                    edge.data.set("primitives", primitives)  # store for later use

        elif cell_type=='reduction':
            def update_ops(edge):
                # function that replaces the primitive ops at the edges with the one in op_index

                if (edge.head, edge.tail) in edge_op_dict_reduction:
                    if (edge.head, edge.tail) in EDGE_LIST_REDUCTION_RED:
                        if isinstance(edge.data.op.edges[1,2].op, list):
                            primitives = edge.data.op.edges[1,2].op
                        else:
                            primitives = edge.data.op.edges[1,2].primitives

                        chosen_op = primitives[edge.data.op.edges[1,2].op_index]
                        primitives[edge.data.op.edges[1,2].op_index] = update_batchnorms(chosen_op)

                        edge.data.op.edges[1,2].set("op", primitives[edge.data.op.edges[1,2].op_index])
                        edge.data.op.edges[1,2].set("primitives", primitives)

                    elif (edge.head, edge.tail) in EDGE_LIST_REDUCTION_REG:
                        if isinstance(edge.data.op, list):
                            primitives = edge.data.op
                        else:
                            primitives = edge.data.primitives

                        chosen_op = primitives[edge.data.op_index]
                        primitives[edge.data.op_index] = update_batchnorms(chosen_op)

                        edge.data.set("op", primitives[edge.data.op_index])
                        edge.data.set("primitives", primitives)

        return(update_ops)

    def update_batchnorms(op: AbstractPrimitive) -> AbstractPrimitive:
        """ Makes batchnorms in the op affine, if they exist """
        init_params = op.init_params
        has_batchnorm = False

        for module in op.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                has_batchnorm = True
                break

        if not has_batchnorm:
            return op

        if 'affine' in init_params:
            init_params['affine'] = True
        if 'track_running_stats' in init_params:
            init_params['track_running_stats'] = True

        new_op = type(op)(**init_params)
        return new_op
    
    #Update edges for the regular cells
    naslib_object.update_edges(
        op_index_factory(cell_type='regular'), scope=naslib_object.OPTIMIZER_SCOPE_REGULARS, private_edge_data=True
    )
    naslib_object.update_edges(
        update_ops_factory(cell_type='regular'), scope=naslib_object.OPTIMIZER_SCOPE_REGULARS, private_edge_data=True
    )

    #Update edges for the reduction cells
    naslib_object.update_edges(
        op_index_factory(cell_type='reduction'), scope=naslib_object.OPTIMIZER_SCOPE_REDUCTIONS, private_edge_data=True
    )
    naslib_object.update_edges(
        update_ops_factory(cell_type='reduction'), scope=naslib_object.OPTIMIZER_SCOPE_REDUCTIONS, private_edge_data=True
    )

def is_valid_arch(op_indices):

    def is_identity_reduction_edges(op_indices):

        return(op_indices[0] == 0 or op_indices[1] == 0 or \
                op_indices[2] == 0 or op_indices[3] == 0 or \
                op_indices[5] == 0 or op_indices[6] == 0 or \
                op_indices[9] == 0 or op_indices[10] == 0)
    
    def check_input_limit(op_indices):

        def is_zero_op(o):

            return(o == 1)
        
        n_inputs_node5 = sum([not(is_zero_op(o)) for o in [op_indices[0], op_indices[1]]])
        n_inputs_node6 = sum([not(is_zero_op(o)) for o in [op_indices[2], op_indices[3], op_indices[4]]])
        n_inputs_node7 = sum([not(is_zero_op(o)) for o in [op_indices[5], op_indices[6], op_indices[7], op_indices[8]]])
        n_inputs_node8 = sum([not(is_zero_op(o)) for o in [op_indices[9], op_indices[10], op_indices[11], op_indices[12], op_indices[13]]])

        return(n_inputs_node5 == 2 and n_inputs_node6 == 2 and n_inputs_node7 == 2 and n_inputs_node8 == 2)

    def is_zero_node5(op_indices):

        return(op_indices[0] == 1 and op_indices[1] == 1)
    
    def is_zero_node6(op_indices):

        return(op_indices[2] == 1 and op_indices[3] == 1 and \
                (op_indices[4] == 1 or is_zero_node5(op_indices)))
    
    def is_zero_node7(op_indices):

        return(op_indices[5] == 1 and op_indices[6] == 1 and \
                (op_indices[7] == 1 or is_zero_node5(op_indices)) and \
                (op_indices[8] == 1 or is_zero_node6(op_indices)))
    
    def is_zero_node8(op_indices):
        return(op_indices[9] == 1 and op_indices[10] == 1 and \
                (op_indices[11] == 1 or is_zero_node5(op_indices)) and \
                (op_indices[12] == 1 or is_zero_node6(op_indices)) and \
                (op_indices[13] == 1 or is_zero_node7(op_indices)))
        
    op_indices_regular = op_indices[:14]
    op_indices_reduction = op_indices[14:]

    is_valid_regular = not (is_zero_node5(op_indices_regular) and is_zero_node6(op_indices_regular) and \
                            is_zero_node7(op_indices_regular) and is_zero_node8(op_indices_regular)) and \
                            check_input_limit(op_indices_regular)
    is_valid_reduction = not ((is_zero_node5(op_indices_reduction) and is_zero_node6(op_indices_reduction) and \
                            is_zero_node7(op_indices_reduction) and is_zero_node8(op_indices_reduction)) or \
                            is_identity_reduction_edges(op_indices_reduction)) and \
                            check_input_limit(op_indices_reduction)
    
    return(is_valid_regular and is_valid_reduction)
