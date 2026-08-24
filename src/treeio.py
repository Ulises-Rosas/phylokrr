import numpy as np
from collections import deque

def fas_to_dic(file):
    
    file_content = open(file, 'r').readlines()
    seqs_list   = []
    
    for i in file_content:
        line = i.strip()
        if line: # just if file starts empty
            seqs_list.append(line) 
    
    keys = [] 
    values = []    
    i = 0
    while(">" in seqs_list[i]):
        keys.append(seqs_list[i])
        i += 1 
        JustOneValue = []

        while((">" in seqs_list[i]) == False):
            JustOneValue.append(seqs_list[i]) 
            i += 1

            if(i == len(seqs_list)):
                i -= 1
                break

        values.append("".join(JustOneValue).upper().replace(" ", ""))
        
    return dict(zip(keys, values))

# region: extended newick parser
class myNode:

    def __init__(self, name = None, 
                 left = None, 
                 right = None, 
                 ancestor = None,
                 index = None, 
                 hybrid = False,
                 node_label = '',
                 branch_length = 0.0,
                 support = 0.0,
                 gamma = 1.0, # probability
                 ):
        
        self.name = name
        self.left = left
        self.right = right
        self.ancestor = ancestor
        self.index = index
        self.branch_length = branch_length
        self.label = node_label
        self.hybrid = hybrid
        self.gamma = gamma
        self.support = support


        # it is true when node is hybrid and 
        # incoming hybrid nodes  are connected 
        # from their left to this node.
        # From this node sucessors are present.
        self.isChild = False # child from hybrid nodes

    def __repr__(self):
        base_str = f"Node(i = {self.index}"

        if self.name:
            base_str += f", n = {self.name}"

        if self.left:
            base_str += f", l = {self.left.index}"

        if  self.right and not isinstance( self.right, list):
                base_str += f", r = {self.right.index}"

        return base_str + ")"

def is_leaf(p):
    """
    Check if the node is a tree leaf node
    """
    if p.left is None and p.right is None:
        return True
    else:
        return False

def is_HybLeaf(p):
    """
    Check if the node is a hybrid leaf node
    """
    if p.hybrid and not p.isChild:
        return True
    else:
        return False


def make_node_info(p):
    """
    make node info.

    Ref: RichNewick.pdf (pp. 9-10)

    parameters:
    ----------
    p : myNode
        node to set info

    returns:
    -------
    ni : str
        node info string
    """
    la = f"{p.label}" if p.label else ''
    le = f"{p.branch_length}" if p.branch_length else ''
    su = f"{p.support}" if p.support else ''
    # no new info if 1.0
    pr = f"{p.gamma}" if p.gamma != 1.0 else ''

    ni = ''
    # combinations of 3
    if le and su and pr:
        ni += f":{le}:{su}:{pr}"

    # combinations of 2
    elif (not le) and su and pr:
        ni += f"::{su}:{pr}"

    elif le and (not su) and pr:
        ni += f":{le}::{pr}"

    elif le and su and (not pr):
        ni += f":{le}:{su}"
    
    # combinations of 1
    elif le and (not su) and (not pr):
        ni += f":{le}"

    elif (not le) and su and (not pr):
        ni += f"::{su}"

    elif (not le) and (not su) and pr:
        ni += f":::{pr}"

    else: # combination of 0
        pass

    if not is_leaf(p):
        ni = la + ni

    return ni


def writeT(p):

    if not p:
        return
    
    if is_leaf(p) or is_HybLeaf(p):
        # extract leaf node info
        node_info = make_node_info(p)
        return f"{p.name}{node_info}"
    
    else:
        ln = writeT(p.left)
        rn = writeT(p.right)
   
        node = f"{ln},{rn}" if rn else ln
        # extract node info only
        # when we reach the end 
        # of the stack
        node_info = make_node_info(p)
            
        return f"({node}){node_info}"

def parseTree(all_nodes, root = -1):

    w = all_nodes[root]

    if isinstance(w.right, list):
        n1 = w.left
        n2 = w.right[0]
        n3 = w.right[1]
        
        return f"({writeT(n1)},{writeT(n2)},{writeT(n3)});"
    
    else:
        n1 = w.left
        n2 = w.right

        return f"({writeT(n1)},{writeT(n2)});"

def parseBinTree(all_nodes, root = -1):

    w = all_nodes[root]
    n1 = w.left
    n2 = w.right

    return f"({writeT(n1)},{writeT(n2)});"

def renaming_tips(all_keys, tree):
    for n,k in enumerate(all_keys):
        # n,k
        spps = k.replace(">", "")
        tree = tree.replace(f"'{n}'", spps)

    return tree

def parse_and_rename(all_nodes, all_keys):
    nwk_str = parseTree(all_nodes)
    for n,k in enumerate(all_keys):
        nwk_str = nwk_str.replace(f"'t{n}'", k.replace(">", ""))
    return nwk_str

def get_int_nodes(n, T):
    int_nodes = [0]*(n - 2)
    k = 0
    for i in range(2*n - 2):

        if not T[i].name:
            int_nodes[k] = T[i].index
            k += 1

    return int_nodes

def get_edges(nodes_indx, n, T, get_root = False):
    
    E = [[0,0]]*(n - 3)
    k = 0; root = 0
    for u in nodes_indx:

        if isinstance(T[u].right, list):
            root = u
            continue
        
        E[k] = [u, T[u].ancestor.index]
        k += 1

    if get_root:
        return E, root
    
    else:
        return E
    
def tokenize(tree):
    """
    split into tokens based on the characters '(', ')', ',', ':', ';'
    Huelsenbeck, programming for biologists, pp. 124-125

    parameters:
    -----------
    tree : str
        newick tree string

    returns:
    --------
    tokens : list
        list of tokens
    """
    # tree = mytree
    tokens = []
    ns_size = len(tree)
    i = 0
    while i < ns_size:
        c = tree[i]
        if c in '(),:;':
            tokens.append(c)
            i += 1

        else:
            j = i
            tempStr = ''
            while c not in '(),:;':
                tempStr += c
                j += 1
                c = tree[j]

            i = j
            tokens.append(tempStr)

    return tokens
    
def set_float(s, k):
    try:
        return float(s)
    
    except ValueError:
        raise ValueError(f"Error: We expect a number at non-label node info. Check {s} at character {k}")
        
def next_nodeInfo(s):
    """
    This essentially checks if the next token
    is not a comma or a semicolon. If it neither
    of these, then it is a node info.
    """
    return True if (s != ',') or (s != ';') else False

def is_hybrid(s):
    """
    check if the string is a hybrid
    """
    if s.startswith('#'):
        return True
    else:
        return False

def set_nodeInfo(p, infoType, tk, k, hyb_nodes):
    """
    set node info based on the type of info
    
    0 : label
    1 : branch length
    2 : support
    3 : probability
    
    ref: RichNewick.pdf

    parameters:
    -----------
    p : myNode
        node to set
    infoType : int
        type of info to set
    tk : str
        token to set
    k : int
        character position in the string
    hyb_nodes : dict
        dictionary of hybrid nodes
        where the key is the id and the value
        is a list of nodes
    """

    if infoType == 0: # label
        # impossible to happen, but just in case
        assert not is_leaf(p), f"Error: We expect a label for a non-leaf node. Check {tk} at character {k}"
    
        p.label = tk
        p.hybrid = is_hybrid(tk)

        if p.hybrid:
            # update the hybrid nodes
            update_hyb_nodes(p, hyb_nodes, tk)

    elif infoType == 1:
        p.branch_length = set_float(tk, k)

    elif infoType == 2:
        p.support = set_float(tk, k)

    else:
        p.gamma = set_float(tk, k)


def has_children(p):
    if p.left is None or p.right is None:
        return False
    else:
        return True

def has_child(p):
    if p.left or p.right:
        return True
    else:
        return False    

def solve_polytomy(p, n, nodes):
    """
    solve polytomy by adding a new node y on the
    edge (p, gp) where gp is the grandparent of p.
    Draw:
          gp
          | (edge can be left or (any) right)
          y
         / \
        p   n3
       / \
      n1  n2 

    Distance between p and y equal to 0
    and y and grandparent equal to the previous
    distance between p and grandparent. 

    parameters:
    -----------
    p : myNode
        parent node 
    
    n : myNode
        new node

    nodes : list
        list of nodes in the tree

    returns:
    --------
    None        
    """

    # if p is not the root, 
    # then it has an ancestor
    gp = p.ancestor 
    
    # keep track of the side of p
    p_left = False
    if gp.left == p:
        p_left = True
    
    # make a new node y
    # make p and n children of y
    if p_left:
        # y -> gp
        y = myNode(left=p, right=n, ancestor=gp, 
                   branch_length=p.branch_length)
        # y <- gp, replace p with y
        gp.left = y
    else:
        # y -> gp
        y = myNode(left=n, right=p, ancestor=gp, 
                   branch_length=p.branch_length)
        # y <- gp, replace p with y
        # root case special case
        if isinstance(gp.right, list):
            if gp.right[0] == p:
                gp.right[0] = y
            else:
                gp.right[1] = y
        else:
            gp.right = y

    # add y to the list of nodes
    nodes += [y]

    # if it is polytomy, then most likely this is already 
    # 0, but we set it to 0 anyway
    p.branch_length = 0.
    p.ancestor = y
    n.ancestor = y


def set_left_or_right(p, n, root, nodes):
    """
    set the left or right child of a node

    if p is the root, then the right child
    is a list of the current right child and the new node n

    parameters:
    -----------
    p : myNode
        parent node

    n : myNode
        new node

    root : myNode
        root node
    
    nodes : list
        list of nodes in the tree. This is used to
        to add an intermediate node in case of a polytomy
        see `solve_polytomy`

    returns:
    --------
    None

    """
    # set left child if it is None
    # as first option
    if p.left is None:
        p.left = n
        return
    
    # set right child if it is None
    # as second option
    if p.right is None:
        p.right = n
        return
    
    # if left and right children are not None
    # then the right child is a list of the current
    # right child and the new node n if p is the root
    if p == root:
        p.right = [p.right, n]
    else:
        # otherwise, we have a polytomy.
        # if p is not the root, then it has an ancestor
        # and it is a polytomy. We solve the polytomy
        solve_polytomy(p, n, nodes)


def build_up_nodes(tokens):

    hyb_nodes = {}
    # ensures O(1) appending 
    nodes = deque()
    root = None
    p = None

    readingNodeInfo = False
    # Info Type: 0 for label, 
    # 1 for edge, 2 for support, 
    # 3 for probability
    infoType = 0 

    # k tracks the position in the string
    k = 0
    for i, tk in enumerate(tokens):

        k += len(tk)

        if tk == "(":
            # internal node
            n = myNode()
            nodes.append(n)
            if p is None:
                root = n
            
            else:
                # connect the node to the parent
                n.ancestor = p
                # set new node the left or right child
                set_left_or_right(p, n, root, nodes)
            
            # let the current node 
            # be the parent
            p = n

        elif tk == "," or tk == ")":
            # move down a node
            p = p.ancestor

            if tk == ")":
                # check if the node 
                # is not empty
                if not has_child(p):
                    raise ValueError(f"Error: We expect at least a child per node. Check character {k}")
                
                # check if there info for the 
                # internal node coming
                if next_nodeInfo(tokens[i+1]):
                    # start reading internal node info.
                    readingNodeInfo = True
                    # we iterate all ':' tokens
                    # to find the type of info
                    # we start from 0
                    infoType = 0 
            
            # it means we reached ',' and we stop
            # reading node info
            else:
                readingNodeInfo = False
                infoType = 0

        elif tk == ";":
            # end of tree
            if p != root:
                raise ValueError("Error: We expect to finish at the root node")

        else:
            if readingNodeInfo:
                if tk == ":":
                    # this indicates that there 
                    # is info on the next token.

                    # Notice that if p is leaf node, 
                    # the next token is ":" and then
                    # it will start reading from length
                    # as infoType will be 1:
                    infoType += 1
                    # then we go to the next token
                    continue

                # reading node info
                if 0 <= infoType <= 3:
                    set_nodeInfo(p, infoType, tk, k, hyb_nodes)

            else:
                # leaf node
                leaf_name = tk.strip("''")
                n = myNode(name=leaf_name, hybrid=is_hybrid(leaf_name))
                
                # update set of nodes
                nodes.append(n)
                # update set of hybrid nodes
                if n.hybrid:
                    update_hyb_nodes(n, hyb_nodes, leaf_name)

                # connect the leaf node to the parent
                n.ancestor = p
                # set new node to left or right
                set_left_or_right(p, n, root, nodes)

                # let the current node 
                # be the parent
                p = n

                # check if there is leaf node info coming
                if next_nodeInfo(tokens[i+1]):
                    readingNodeInfo = True
                    # Iterations will decide 
                    # the type of info
                    infoType = 0

    if hyb_nodes:
        # process the hybrid nodes
        connect_hyb_nodes(hyb_nodes)

    return list(nodes), hyb_nodes, root

def connect_hyb_nodes(hyb_nodes):
    """
    connect the hybrid nodes to one child,
    the one that is not a leaf node (i.e., it
    has no successors). Child node will have
    multiple ancestors. The first ancestor will be
    tree base node. The rest of the ancestors
    and the rest of ancestors will be hybrid leaf nodes.

    The left of hybrid leaf nodes will
    be connected to the child node.

    parameters:
    -----------
    hyb_nodes : dict
        dictionary of hybrid nodes
        where the key is the id and the value
        is a list of nodes
        e.g. {#H1: [n1, n2, n3]}
        where n1, n2, n3 are hybrid nodes,
        with one of them being a child node
        and the rest are leaf nodes.

    returns:
    -------
        None
    """
    for k, nodes in hyb_nodes.items():    
        # recieving all the leaf nodes
        # this node has successors
        out_node = None # child node
        # injecting the leaf nodes
        in_nodes = deque()
        
        for n in nodes:
            if is_leaf(n):
                in_nodes.append(n)
            else:
                out_node = n

        assert len(in_nodes) == len(nodes) - 1, f"Error: We found {len(in_nodes)} leaf nodes representing {k} when it should be {len(nodes) - 1} (i.e. number of {k} nodes - 1)"
        assert out_node, f"Error: We found no {k} nodes with successors"

        out_node.ancestor = [out_node.ancestor] + list(in_nodes)
        out_node.isChild = True
        
        # leaf nodes are empty
        # and they have no successors
        for n in in_nodes:
            n.left = out_node


def update_hyb_nodes(p, hyb_nodes, id):
    """
    update the hybrid nodes

    parameters:
    -----------
    p : myNode
        node to set

    hyb_nodes : dict
        dictionary of hybrid nodes
        where the key is the id and the value
        is a list of nodes

    id : str
        id of the hybrid node e.g. #H1, #H2, etc. 
        if p is a internal node then
        the id is the label of the node
        if p is a leaf node then 
        the id is the name of the node

    returns:
    -------
        None
    """
    if id not in hyb_nodes:
        hyb_nodes[id] = deque([p])

    else:
        hyb_nodes[id].append(p) # O(1)

def parseNewickTree(mstr):
    # mstr = mytree
    tokens = tokenize(mstr)
    # print(tokens)

    nodes, _, root = build_up_nodes(tokens)
    # print(nodes[0].right)
    for i, n in enumerate(nodes):
        n.index = i

        if n.ancestor is None:
            root = n

    return nodes, root

def parseNewickNet(mstr):
    # mstr = mytree
    tokens = tokenize(mstr)
    # print(tokens)

    nodes, hyb_nodes, root = build_up_nodes(tokens)
    # print(nodes[0].right)
    for i, n in enumerate(nodes):
        n.index = i

        if n.ancestor is None:
            root = n

    return nodes, hyb_nodes, root
# endregion: extended newick parser

# region: tree manipulation
def flip_nodes(P, u_idx = 1, v_idx = 1):
    """
    P is the path to root.

    Flip nodes such that:
              n_anc
                | anc
                n
   left/right /  \  left/right
                 n_child
    
    Turns into:
              n_anc
                | left/right
                n
   left/right /  \  anc
                 n_child

    Since P is a path to root, all n_anc belong to P
    n_child are picked from P
    """
    assert u_idx > 0, "flips needs previous node"
    for i in range(u_idx, v_idx):
        n = P[i]

        n_anc  = n.ancestor
        n_child = P[i-1]

        n.ancestor = n_child

        if n.left == n_child:
            n.left = n_anc
        else:
            n.right = n_anc

def get_RootPath(P, tmp_n, root):
    """
    nodes from tmp_n ancestor to root.
    """
    while tmp_n != root:
        P.append(tmp_n.ancestor)
        tmp_n = tmp_n.ancestor

def reroot(n_t: myNode, root: myNode, all_nodes: list):
    """
    Re-root above n_t.
    root is the current root.

    It returns a new root.
    """

    r_new = myNode(index=root.index)
    P = [r_new]
    get_RootPath(P, n_t, root)

    n_t_anc = P[1]
    if n_t_anc == root:
        # print('Trivial re-rooting')
        return 
    
    # add to the new root
    # left and right defined 
    # on the target node and its
    # adjacent node
    r_new.right = n_t_anc
    r_new.left  = n_t

    # let target node new ancestor
    # be the r_new
    n_t.ancestor = r_new

    # let adjacent node point to the 
    # r_new
    if n_t_anc.left == n_t:
        n_t_anc.left = r_new
    else:
        n_t_anc.right = r_new

    # flip nodes from ancestor of n_t
    # to right before the root
    flip_nodes(P, 1, len(P) - 1)

    # root child 1 on path
    r_c1 = P[-2] 
    # root child 2
    r_c2 = root.left if root.left != r_c1 else root.right

    # make r_c2 (=> its substree) ancestor be
    # r_c1
    r_c2.ancestor = r_c1

    # r_c1 must point to the root
    # after flipping
    if r_c1.left == root:
        r_c1.left = r_c2
    else:
        r_c1.right = r_c2

    all_nodes[r_new.index] = r_new
    return r_new
# endregion: tree manipulation

# region: vcv matrix
def get_nodes(n):
    # 3*(2n-2) = O(n)
    out = []
    if isinstance(n, list):
        for ni in n:
            out.append(ni)
    else:
        if n is not None:
            out.append(n)

    return out

def get_adj(nodes):
    adj = {}
    for n in nodes:
        adj[n] = []
        adj[n] += get_nodes(n.left)
        adj[n] += get_nodes(n.right)
        adj[n] += get_nodes(n.ancestor)

    return adj

def dfs_distances(adj, start, C):
    # O(n) since it's a tree
    S = [(start, -1, 0)]   # (node, parent, distance)
    dist = {}

    while S:
        node, parent, d = S.pop()
        dist[node] = d

        for u in adj[node]:
            if u != parent:
                S.append((u, node, d + C[(node, u)]))

    return dist

def get_weight(v, u):
    # if isinstance(v.ancestor, list):
    if u.left == v or u.right == v:
        return v.branch_length
    else:        
        return u.branch_length

def get_C(nodes, adj):
    """
    get edge weights
    O(n) since it's a tree
    """
    C = {} # O(n)
    for v in nodes:
        for u in adj[v]:
            w = get_weight(v, u)
            C[(v, u)] = w
            C[(u, v)] = w

    return C

def get_APSP(leaves, adj, C):
    n = len(leaves)
    D = np.zeros((n, n))

    # O(n^2)
    for i in range(n - 1): # O(n)
        u = leaves[i]
        tmp_dist = dfs_distances(adj, u, C) # O(n)
        for j in range(i + 1, n):
            v = leaves[j]
            D[i, j] = tmp_dist[v]
            D[j, i] = tmp_dist[v]

    return D

def get_BM(D, root, leaves, adj, C):
    n = len(leaves)
    dist_root = dfs_distances(adj, root, C)
    D_r = np.zeros(n)
    # O(n)
    for i in range(n):
        D_r[i] = dist_root[leaves[i]]

    # O(n^2)
    BM_VCV = np.zeros((n, n))
    for i in range(n):
        BM_VCV[i, :] = (D_r[i] + D_r - D[i, :])/2

    return BM_VCV

def get_cov_mat(nodes, root, process = "BM",
                 sigma2 = 1, alpha = 1):
    # target nodes: leaves 
    leaves = [n for n in nodes if is_leaf(n)]
    names = [n.name for n in leaves]

    adj = get_adj(nodes) # O(n)
    C = get_C(nodes, adj) # O(n)

    # O(n^2), it transverses the tree n times, and
    # each traversal is O(n)
    D = get_APSP(leaves, adj, C)
    # print(D)
    if process == "BM":
        # makes for each row i it makes n calculations from D.
        # since there n rows, the time complexity is O(n^2)
        BM_D = get_BM(D, root, leaves, adj, C) # O(n^2)
        return sigma2*BM_D, names
    
    elif process == "OU":
        c1 = 2*alpha
        return (sigma2/c1)*np.exp(-alpha*D), names
    
    else:
        raise ValueError("process must be either 'BM' or 'OU'")

def get_vcv(mytree, process = "BM", sigma2 = 1, alpha = 0.5):
    """
    get the variance-covariance matrix from
    a newick tree string.

    O(n^2) time complexity, where n is the number of tips in the tree.

    parameters:
    -----------
    mytree : str
        newick tree string

    process : str
        "BM" for Brownian motion
        "OU" for Ornstein-Uhlenbeck

    sigma2 : float
        evolutionary rate

    alpha : float
        selection parameter, only used for OU process

    returns:
    --------
    vcv : list
        variance-covariance matrix without
        being multiplied by the evolutionary rate

    name_list : list
        ordered list of tip names
        used to build the vcv matrix
    """

    nodes, root = parseNewickTree(mytree)
    assert not isinstance(root.right, list), "Unrooted tree not supported"

    vcv, names = get_cov_mat(nodes, root, process = process, sigma2 = sigma2, alpha = alpha)
    # return the variance-covariance matrix
    return vcv, names

# endregion: vcv matrix

# # testings of code
# mytree1 = "((t2:3,t3:2):1,t4,t1:4);"
# mytree2 = "(t4,(t2:3,t3:2):1,t1:4);"
# mytree3 = "(t4,t1:4,(t2:3,t3:2):1);"
# mytree4 = "(t4,t1:4,(t2:3,t3:2,t5:1):1);"
# for i in [mytree1, mytree2, mytree3, mytree4]:
#     tmp_nodes, tmp_root = parseNewickTree(i)
#     print(parseTree(tmp_nodes, root = tmp_root.index))

# mytree = "(t4,t1:4,(t2:3):1);"
# nodes, root = parseNewickTree(mytree)
# print(root.right)
# print(parseTree(nodes, root = root.index))

# mytree = "((t4,t1:4),(t2:3):1);"
# vcv, names = get_vcv(mytree)

# import numpy as np
# np.array(vcv)