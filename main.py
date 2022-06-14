import networkx as nx
import numpy as np
from skimage import morphology


def find_nearby_maxima(image, seeds, connectivity=1):
    """Find indices of the maximum in 2d-array that can be reached from seed points

    seeds: list of tuples containing the indices of every seed
    Unexpected result may happen when identical values are present in the image (plateaus)
    """
    parent, traverser = morphology.max_tree(image, connectivity)
    image_rav = image.ravel()
    tree = nx.DiGraph()
    tree.add_nodes_from(traverser)
    P_rav = parent.ravel()
    tree.add_edges_from([(n, P_rav[n]) for n in traverser[1:]])
    leaves = set(x for x in tree.nodes() if tree.in_degree(x) == 0)
    raveled_indices = np.ravel_multi_index(np.array(seeds).T, image.shape)

    def find(nxtree, seed):
        if seed in leaves:
            # return original seed point
            return np.unravel_index(seed, image.shape)
        ancestors = nx.ancestors(nxtree, seed)
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
    return [find(tree, s) for s in raveled_indices]
