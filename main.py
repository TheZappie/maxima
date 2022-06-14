import networkx as nx
import numpy as np
from skimage import morphology


def find_nearby_maxima(image, seeds, connectivity=1):
    """Find indices of the maximum in 2d-array that can be reached from seeds points
    seeds: list of tuples containing the indices of every seed
    Unexpected result may happen when identical values are present in the image (plateaus)
    """
    P, S = morphology.max_tree(image, connectivity)
    image_rav = image.ravel()
    tree = nx.DiGraph()
    tree.add_nodes_from(S)
    P_rav = P.ravel()
    tree.add_edges_from([(n, P_rav[n]) for n in S[1:]])
    leaves = set(x for x in tree.nodes() if tree.in_degree(x) == 0)
    raveled_indices = np.ravel_multi_index(np.array(seeds).T, image.shape)

    def find(tree, seed):
        ancestors = nx.ancestors(tree, seed)
        local_maxima = list(leaves & ancestors)
        if not local_maxima:
            return np.unravel_index(seed, image.shape)
        local_maximum = local_maxima[np.argmax(image_rav[local_maxima])]
        return np.unravel_index(local_maximum, image.shape)

    # try:
    #     iter(raveled_indices)
    # except TypeError:
    #     return find(tree, raveled_indices)
    return [find(tree, s) for s in raveled_indices]


if __name__ == '__main__':
    image = np.array([[40, 40, 39, 39, 38],
                      [40, 41, 39, 39, 39],
                      [30, 30, 30, 32, 32],
                      [33, 33, 30, 32, 35],
                      [30, 30, 30, 33, 36]], dtype=np.uint8)
    seeds = ((0, 4), (2, 4), (3, 4), (4, 4))
    find_nearby_maxima(image, seeds)
