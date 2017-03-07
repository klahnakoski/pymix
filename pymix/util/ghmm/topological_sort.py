from pymix.util.ghmm.wrapper import ARRAY_CALLOC, ighmm_dmatrix_stat_alloc

def topological_sort(c_model):
    v = local_store_topo(c_model)
    edge_classes = dfs(c_model)
    indegrees = [0] * c_model.N

    for i in range(c_model.N):
        indegrees[i] = c_model.N
    for i in range(c_model.N):    # don't consider back edges in top sort
        for j in range(c_model.N):
            if edge_classes[i][j] == GRAY:
                indegrees[j] -= 1

    v.head = 0
    v.tail = 0
    for i in range(c_model.N):
        if indegrees[i] == 0:
            v.queue[v.tail] = i
            v.tail += 1
    dels_cnt = 0
    while v.head != v.tail:
        nodeu = v.queue[v.head]        # dequeue
        v.head += 1
        if c_model.silent[nodeu]:
            v.topo_order[dels_cnt] = nodeu        # append it to the list
            dels_cnt += 1

        for i in range(c_model.N):
            if edge_classes[nodeu][i] != GRAY:
                indegrees[i] -= 1
                if i != nodeu and indegrees[i] == 0:
                    v.queue[v.tail] = i  # enqueue
                    v.tail += 1

    return v.topo_order[0:dels_cnt]

GRAY = 0
BLACK = 1
WHITE = 2
NONE = -1


class local_store_topo():
    def __init__(self, mo):
        self.topo_order = ARRAY_CALLOC(mo.N)
        self.topo_order_length = 0
        self.queue = ARRAY_CALLOC(mo.N)
        self.head = 0
        self.tail = 0


def visit(c_model, nodev, parents, colors, edge_classes):
    """
    Implementation of DFSVisit with recursion (WS)
    """
    colors[nodev] = GRAY
    for i in range(c_model.N):
        if edge_classes[nodev][i] == NONE:       # First exploration
            edge_classes[nodev][i] = colors[i]
        if colors[i] == WHITE:
            parents[i] = nodev
            visit(c_model, i, parents, colors, edge_classes)
    colors[nodev] = BLACK        # finished



def dfs(c_model):
    """
     Return classification of edges in the model graph
       WHITE EDGE => edges in the DFS search tree
       GRAY EDGE  => edges that form cycles which must be removed
                     before running topological sort
    """
    initials = 0
    colors = ARRAY_CALLOC(c_model.N)
    parents = ARRAY_CALLOC(c_model.N)
    edge_classes = ighmm_dmatrix_stat_alloc(c_model.N, c_model.N)

    for i in range(c_model.N):
        if c_model.s[i].pi == 1.0:
            initials = i             # assuming only one initial state
        colors[i] = WHITE
        parents[i] = -1
        for j in range(c_model.N):
            edge_classes[i][j] = NONE

    visit(c_model, initials, parents, colors, edge_classes)
    for i in range(c_model.N):    # handle subtrees
        if colors[i] == WHITE:
            visit(c_model, i, parents, colors, edge_classes)

    return edge_classes
