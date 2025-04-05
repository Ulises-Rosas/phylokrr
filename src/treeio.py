
class myNode:

    def __init__(self, name = None, 
                 left = None, 
                 right = None, 
                 ancestor = None,
                 index = None, 
                 node_label = None,
                 branch_length = 0.):
        self.name = name
        self.left = left
        self.right = right
        self.ancestor = ancestor
        self.index = index
        self.branch_length = branch_length
        self.label = node_label


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


def writeT(p):

    if not p:
        return

    bl = f":{p.branch_length}"

    if p.left is None and p.right is None:
        return f"{p.name}{bl}"
    
    else:
        ln = writeT(p.left)
        rn = writeT(p.right)
        nlabel = '' if not p.label else p.label
        node = f"{ln},{rn}" if rn else ln
            
        return f"({node}){nlabel}{bl}"

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

def check_if_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
    

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

    nodes = []
    root = None
    p = None
    readingBranchLength = False
    readingLabel = False
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
                n.ancestor = p
                set_left_or_right(p, n, root, nodes)

            p = n

        elif tk == "," or tk == ")":
            # move down a node
            p = p.ancestor

            if tk == ")" and not has_child(p):
                raise ValueError(f"Error: We expect at least a child per node. Check character {k}")
            
            # check if the next token is a number
            next_number = check_if_number(tokens[i+1])
            
            if tk == ")" and next_number:
                readingLabel = True
            
        elif tk == ":":
            readingBranchLength = True

        elif tk == ";":
            # end of tree
            if p != root:
                raise ValueError("Error: We expect to finish at the root node")

        else:
            if readingBranchLength:
                p.branch_length = float(tk)
                readingBranchLength = False

            elif readingLabel:
                p.label = float(tk)
                readingLabel = False
                
            else:
                # leaf node
                n = myNode(name = tk.strip("''"))
                nodes.append(n)
                n.ancestor = p
                set_left_or_right(p, n, root, nodes)

                p = n

    return nodes, root

def parseNewickTree(mstr):
    # mstr = mytree
    tokens = tokenize(mstr)
    # print(tokens)

    nodes, root = build_up_nodes(tokens)
    # print(nodes[0].right)
    for i, n in enumerate(nodes):
        n.index = i

        if n.ancestor is None:
            root = n

    return nodes, root

def dfs_ur(n, path, paths):
    """
    DFS for unrooted trees
    """
    if n:
        path += [n.index]

        if n.name:
            paths.append(list(path)) 

        
        dfs_ur(n.left, path, paths)

        if n.ancestor:
            dfs_ur(n.right, path, paths)
        else:
            dfs_ur(n.right[0], path, paths)
            dfs_ur(n.right[1], path, paths)

        path.pop()

def dfs_r(n, path, paths):
    """
    DFS for rooted trees
    """
    if n:
        path += [n.index]

        if n.name:
            paths.append(list(path)) 

        dfs_r(n.left, path, paths)
        dfs_r(n.right, path, paths)

        path.pop()

def get_vcv_paths(paths, nodes):
    """
    get the variance-covariance matrix from
    a set of paths from the root to the tips

    parameters:
    -----------
    paths : list
        list of paths from the root to the tips

    nodes : list
        list of nodes in the tree

    returns:
    --------

    vcv : list
        variance-covariance matrix

    name_list : list
        ordered list of tip names
        used to build the vcv matrix
    """

    n = len(paths)
    vcv = [[0.]*n for i in range(n)]
    # O(n^3)
    for i in range(n):
        for j in range(i+1, n):

            path_i = paths[i]
            path_j = paths[j]
            min_len = min(len(path_i), len(path_j))
            tmp_corr = 0.

            for k in range(1, min_len):            
                if path_i[k] == path_j[k]:
                    # print(i,j,k)
                    tmp_corr += nodes[path_i[k]].branch_length

                else:
                    break

            vcv[i][j] = tmp_corr
            vcv[j][i] = tmp_corr

    name_list = ['']*n
    # O(n^2)
    for i in range(n):
        path_i = paths[i]
        corr = 0.
        for k in range(1, len(path_i)):
            corr += nodes[path_i[k]].branch_length

        vcv[i][i] = corr
        name_list[i] = nodes[path_i[-1]].name

    return vcv, name_list

def get_vcv(mytree):
    """
    get the variance-covariance matrix from
    a newick tree string

    parameters:
    -----------
    mytree : str
        newick tree string

    returns:
    --------
    vcv : list
        variance-covariance matrix

    name_list : list
        ordered list of tip names
        used to build the vcv matrix
    """

    nodes, root = parseNewickTree(mytree)
    assert not isinstance(root.right, list), "Unrooted tree not supported"
    # paths from root to tips
    path = []
    paths = []
    dfs_r(root, path, paths)
    # pairwise comparisons of paths to
    # get the variance-covariance matrix
    vcv, names = get_vcv_paths(paths, nodes)
    # return the variance-covariance matrix
    return vcv, names

# testings of code
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

# vcv, names = get_vcv(mytree)

# import numpy as np
# np.array(vcv)
