import numpy as np
import pandas as pd

from sklearn.neighbors import NearestNeighbors
from annoy import AnnoyIndex
import itertools
import networkx as nx
import hnswlib
from sklearn.preprocessing import normalize

def nn_approx(ds1, ds2, names1, names2, knn=50):
    """
    Approximate nearest neighbor search using HNSW (hnswlib).

    Parameters
    ----------
    ds1 : np.ndarray
        Query dataset (N1, D).
    ds2 : np.ndarray
        Reference dataset (N2, D).
    names1 : list of str
        Identifiers for rows in `ds1`.
    names2 : list of str
        Identifiers for rows in `ds2`.
    knn : int, default=50
        Number of nearest neighbors to find.

    Returns
    -------
    Set[Tuple[str, str]]
        Set of matched (query_name, reference_name) pairs.
    """
    dim = ds2.shape[1]
    num_elements = ds2.shape[0]
    p = hnswlib.Index(space='l2', dim=dim)
    p.init_index(max_elements=num_elements, ef_construction=100, M = 16)
    p.set_ef(10)
    p.add_items(ds2)
    ind,  distances = p.knn_query(ds1, k=knn)
    match = set()
    for a, b in zip(range(ds1.shape[0]), ind):
        for b_i in b:
            match.add((names1[a], names2[b_i]))
    return match


def nn(ds1, ds2, names1, names2, knn=50, metric_p=2):
    """
    Exact nearest neighbor search using scikit-learn.

    Parameters
    ----------
    ds1 : np.ndarray
        Query dataset.
    ds2 : np.ndarray
        Reference dataset.
    names1 : list of str
        Identifiers for `ds1`.
    names2 : list of str
        Identifiers for `ds2`.
    knn : int
        Number of nearest neighbors.
    metric_p : int
        Minkowski distance parameter (e.g., 2 for Euclidean).

    Returns
    -------
    Set[Tuple[str, str]]
        Set of matched nearest neighbor pairs.
    """
    # Find nearest neighbors of first dataset.
    nn_ = NearestNeighbors(knn, p=metric_p)
    nn_.fit(ds2)
    ind = nn_.kneighbors(ds1, return_distance=False)

    match = set()
    for a, b in zip(range(ds1.shape[0]), ind):
        for b_i in b:
            match.add((names1[a], names2[b_i]))

    return match

def nn_annoy(ds1, ds2, names1, names2, norm=True, knn = 20, metric='euclidean', n_trees = 10, save_on_disk = False):
    """
    Approximate nearest neighbor search using Annoy index.

    Parameters
    ----------
    ds1 : np.ndarray
        Query dataset.
    ds2 : np.ndarray
        Reference dataset.
    names1 : list of str
        Identifiers for `ds1`.
    names2 : list of str
        Identifiers for `ds2`.
    norm : bool, default=True
        Whether to L2 normalize datasets before indexing.
    knn : int
        Number of nearest neighbors to retrieve.
    metric : str, default='euclidean'
        Distance metric ('euclidean', 'manhattan', etc.).
    n_trees : int, default=10
        Number of trees to build in Annoy index.
    save_on_disk : bool, default=False
        Whether to write index to disk.

    Returns
    -------
    Set[Tuple[str, str]]
        Set of nearest neighbor pairs.
    """
    if norm:
        ds1 = normalize(ds1)
        ds2 = normalize(ds2) 

    """ Assumes that Y is zero-indexed. """
    # Build index.
    a = AnnoyIndex(ds2.shape[1], metric=metric)
    if(save_on_disk):
        a.on_disk_build('annoy.index')
    for i in range(ds2.shape[0]):
        a.add_item(i, ds2[i, :])
    a.build(n_trees)

    # Search index.
    ind = []
    for i in range(ds1.shape[0]):
        ind.append(a.get_nns_by_vector(ds1[i, :], knn, search_k=-1))
    ind = np.array(ind)

    # Match.
    match = set()
    for a, b in zip(range(ds1.shape[0]), ind):
        for b_i in b:
            match.add((names1[a], names2[b_i]))

    return match


def mnn(ds1, ds2, names1, names2, knn1=20, knn2=20,
        approx = True, metric='euclidean', way='hnsw', norm=False):
    """
    Compute Mutual Nearest Neighbors (MNN) between two datasets.

    Parameters
    ----------
    ds1 : np.ndarray
        First dataset (queries).
    ds2 : np.ndarray
        Second dataset (references).
    names1 : list of str
        Identifiers for `ds1`.
    names2 : list of str
        Identifiers for `ds2`.
    knn1 : int
        Number of neighbors for `ds1` → `ds2`.
    knn2 : int
        Number of neighbors for `ds2` → `ds1`.
    approx : bool, default=True
        Whether to use approximate neighbor search.
    metric : str, default='euclidean'
        Distance metric to use.
    way : str, default='hnsw'
        Method for approximate search: 'hnsw' or 'annoy'.
    norm : bool, default=False
        Whether to normalize data before Annoy search.

    Returns
    -------
    Set[Tuple[str, str]]
        Set of mutual nearest neighbor pairs.
    """
    
    if approx: 
        if way=='hnsw':
            # Find nearest neighbors in first direction.
            # output KNN point for each point in ds1.  match1 is a set(): (points in names1, points in names2), the size of the set is ds1.shape[0]*knn
            match1 = nn_approx(ds1, ds2, names1, names2, knn=knn1)
            # Find nearest neighbors in second direction.
            match2 = nn_approx(ds2, ds1, names2, names1, knn=knn2)
        else:
            match1 = nn_annoy(ds1, ds2, names1, names2, norm=norm, knn=knn1, metric=metric)
            match2 = nn_annoy(ds2, ds1, names2, names1, norm=norm, knn=knn2, metric=metric)
    else:
        match1 = nn(ds1, ds2, names1, names2, knn=knn1)
        match2 = nn(ds2, ds1, names2, names1, knn=knn2)
    # Compute mutual nearest neighbors.
    mutual = match1 & set([ (b, a) for a, b in match2 ])

    return mutual
