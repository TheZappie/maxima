import networkx as nx
import numpy as np
from skimage import morphology


def prune(G, node):
    """
    Transform a canonical max tree to a max tree.

    returns np.array denoting to which each node is grouped
    """
    reference = np.arange(len(G.nodes))

    def _prune(G, node):
        value = G.nodes[node]['value']
        preds = [p for p in G.predecessors(node)]
        for p in preds:
            if G.nodes[p]['value'] == value:
                G.remove_node(p)
                reference[p] = node
            else:
                _prune(G, p)

    _prune(G, node)
    return reference


def find_nearby_maxima(image, seeds, connectivity=1):
    """Find indices of the maximum in 2d-array that can be reached from seed points

    seeds: list of tuples containing the indices of every seed
    When identical values are present in the image (plateaus), a random peak is chosen
    """
    image[np.isnan(image)] = np.nanmin(image)
    parent, traverser = morphology.max_tree(image, connectivity)
    image_rav = image.ravel()
    tree = nx.DiGraph()
    tree.add_nodes_from(traverser)
    P_rav = parent.ravel()
    for node in tree.nodes():
        tree.nodes[node]['value'] = image_rav[node]
    tree.add_edges_from([(n, P_rav[n]) for n in traverser[1:]])
    links = prune(tree, traverser[0])
    leaves = set(x for x in tree.nodes() if tree.in_degree(x) == 0)
    seeds_reveled = np.ravel_multi_index(np.array(seeds).T, image.shape)

    def find(nxtree, seed):
        if seed in leaves:
            # return original seed point
            return np.unravel_index(seed, image.shape)
        ancestors = nx.ancestors(nxtree, links[seed])
        local_maxima = list(leaves & ancestors)
        if not local_maxima:
            raise ValueError("Should not happen")
        local_maximum = local_maxima[np.argmax(image_rav[local_maxima])]
        return np.unravel_index(local_maximum, image.shape)
    # TODO overload input to allow for single seed
    # try:
    #     iter(raveled_indices)
    # except TypeError:
    #     return find(tree, raveled_indices)
    return [find(tree, s) for s in seeds_reveled]


# TODO find other solution. Base on while loop to avoid recursion
current_node = seed
while True:
    break