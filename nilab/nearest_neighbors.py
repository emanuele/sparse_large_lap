import numpy as np
from scipy.spatial import cKDTree
from functools import reduce


def flip_streamlines_flattened(streamlines_flattened):
    """Flip the direction of the streamlines each repesented as a vector
    of the sequence of coordinates (flattening).
    """
    a, b = streamlines_flattened.shape
    streamlines_flattened_flip = streamlines_flattened.reshape(a, -1, 3)[:, ::-1, :]
    return np.resize(streamlines_flattened_flip, (a, b))
    return streamlines_flattened_flip


def kdt_query_with_flip(kdt, x, x_flip, k, verbose=False):
    """Perform two KDTree queries (x, x_flip) and merge their results
    returning k neighbors, as the closest one either to x or x_flip,
    removing duplicate neighbors.
    """
    if verbose: print("Querying kdtree")
    ds, idxs = kdt.query(x, k=k, n_jobs=-1)

    if verbose: print("Querying kdtree with flipped queries")
    ds_flip, idxs_flip = kdt.query(x_flip, k=k, n_jobs=-1)

    if verbose: print("Merging results")
    if k == 1:
        ds_all = np.hstack([ds[:, None], ds_flip[:, None]])
        idxs_all = np.hstack([idxs[:, None], idxs_flip[:, None]])
    else:
        ds_all = np.hstack([ds, ds_flip])
        idxs_all = np.hstack([idxs, idxs_flip])

    ds_all_argsort = ds_all.argsort(1)
    distances = np.zeros((len(x), k))
    neighbors = np.zeros((len(x), k), dtype=np.int)
    for i in range(len(ds_all_argsort)):
        idxs_all_sorted_ds = idxs_all[i][ds_all_argsort[i]]
        ds_all_sorted = ds_all[i][ds_all_argsort[i]]
        size = len(idxs_all_sorted_ds)
        # remove duplicates in idxs_all_sorted_ds:
        # (see https://stackoverflow.com/questions/12926898/numpy-unique-without-sort )
        nodup = sorted(np.unique(idxs_all_sorted_ds, return_index=True)[1])
        idxs_all_sorted_ds = idxs_all_sorted_ds[nodup]
        ds_all_sorted = ds_all_sorted[nodup]
        # if len(idxs_all_sorted_ds) < size:
        #     print(size - len(idxs_all_sorted_ds))

        # and keep the first k
        neighbors[i, :] = idxs_all_sorted_ds[:k]
        distances[i, :] = ds_all_sorted[:k]

    # squeeze() is necessaey to match the API of kdtree.query():
    return distances.squeeze(), neighbors.squeeze()


def streamlines_neighbors(streamlines, tractogram, k):
    """Compute the k-nearest neighbors of a set of streamlines within a
    tractogram. Returns distances of neighbors and IDs of neighbors
    within the tractogram. Assumes ALL streamlines have the same
    number of points.
    """
    x = streamlines.reshape(len(streamlines), -1)
    x_flip = streamlines[:, ::-1, :].reshape(len(streamlines), -1)
    kdt = cKDTree(tractogram.reshape(len(tractogram), -1))  # Memory intensive!
    return kdt_query_with_flip(kdt, x, x_flip, k)


def get_streamlines_id(streamlines, tractogram, tolerance=1.0e-10):
    """Returns the IDs of the streamlines in the tractogram as the closest
    ones within a certain (Euclidean) tolerance. If at least one is
    above the tolerance an Exception is raised.

    Streamlines and tractogram are assumed to have the same number of
    points.

    """
    distances, neighbors = streamlines_neighbors(streamlines, tractogram, k=1)
    assert((distances <= tolerance).all())
    return neighbors


def compute_superset(streamlines, tractogram, k,
                     exclude_streamlines=True, tolerance=1.0e10):
    """Returns the k-neareast neighbors of the streamlines within the
    tractogram avoiding duplicates. If exclude_streamlines is True,
    the IDs of the streamlines are excluded from the result too if
    they are within a certain distance (tolerance).

    Notice that if tolerance is very large, then this last step is
    equivalent to remove the nearest neighbor streamlines, which is
    safer than removing only the matching streamlines (which
    corresponds to a very low tolerance).
    """
    distances, neighbors = streamlines_neighbors(streamlines, tractogram, k=k)
    # merging all neighbors:
    # superset = reduce(np.union1d, neighbors)  # SLOWER
    superset = np.array(list(set(neighbors.flatten().tolist())))  # 10x FASTER
    if exclude_streamlines:
        if k == 1:
            to_be_excluded = neighbors[distances <= tolerance]
        else:
            to_be_excluded = neighbors[:, 0][distances[:, 0] <= tolerance]

        superset = np.setdiff1d(superset, to_be_excluded)
        if len(to_be_excluded) < len(streamlines):
            print("WARNING: AT LEAST SOME streamlines do not belong to the tractogram!!!")

    return superset, distances.max()


if __name__ == '__main__':
    np.random.seed(0)

    n = 100000
    d = 48
    n_queries = 1000
    k = 100

    print("Generating vector data")
    X = np.random.uniform(size=(n, d))
    print("X.shape: %s" % (X.shape,))
    print("Generaring %s queries, subset of X" % n_queries)
    idx_query = np.random.permutation(n)[:n_queries]
    x = X[idx_query, :]
    print("Generating flipped queries")
    # x_flip = x[:, ::-1]
    x_flip = flip_streamlines_flattened(x)

    print("Flipping half entries of X")
    idx = np.random.permutation(n)[:int(n/2)]
    # X[idx, :] = X[idx, ::-1]
    X[idx] = flip_streamlines_flattened(X[idx])

    print("Building kdtree")
    kdt = cKDTree(X)

    distances, neighbors = kdt_query_with_flip(kdt, x, x_flip, k=k)
    # check that all first neighbors match idx_query:
    assert((idx_query == neighbors[:, 0]).all())
    assert((distances[:, 0] == 0).all())

    print('')
    print('Generating random tractogram and streamlines as its subset')
    n_points = 16
    tractogram = np.random.normal(size=(n, n_points, 3))
    n_streamlines = 1000
    subset_ids = np.random.choice(len(tractogram), n_streamlines)
    streamlines = tractogram[subset_ids]
    print('Flipping half of the streamlines')
    streamlines[:int(n_streamlines/2)] = streamlines[:int(n_streamlines/2), ::-1, :]
    k = 10
    print('Computing %s neighbors of each streamline in the subset wrt the tractogram' % k)
    distances, neighbors = streamlines_neighbors(streamlines, tractogram, k=k)
    print('Checking that 1-NN have distance 0')
    assert((distances[:, 0] == 0.0).all())
    print('OK')

    print('')
    print('Verifying that streamlines IDs are correctly computed')
    streamlines_ids = get_streamlines_id(streamlines, tractogram)
    assert((streamlines_ids == subset_ids).all())
    print('OK')

    print('')
    n_streamlines = 100
    print('Generating other %s streamlines' % n_streamlines)
    streamlines = np.random.normal(size=(n_streamlines, n_points, 3))
    k = 1
    print('Computing their closest (k=1) neighbor of each streamline in the tractogram')
    distances, neighbors = streamlines_neighbors(streamlines, tractogram, k=k)
    print(distances)
    print(neighbors)

    print('')
    k = 5
    print('Computing the superset of the streamlines (k=%s)' % k)
    superset, max_distance = compute_superset(streamlines, tractogram, k=k)
    print(superset)
    print('The superset is composed of %s streamlines' % len(superset))
    print("The max distance is %s" % max_distance)
