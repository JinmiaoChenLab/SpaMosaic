import numpy as np
import pandas as pd

from sklearn.neighbors import NearestNeighbors
from annoy import AnnoyIndex
import itertools
import networkx as nx
import hnswlib
from sklearn.preprocessing import normalize

def nn_approx(ds1, ds2, names1, names2, knn=50):
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


def mnn(ds1, ds2, names1, names2, knn = 20, approx = True, metric='euclidean', way='hnsw', norm=False):
    if approx: 
        if way=='hnsw':
            # Find nearest neighbors in first direction.
            # output KNN point for each point in ds1.  match1 is a set(): (points in names1, points in names2), the size of the set is ds1.shape[0]*knn
            match1 = nn_approx(ds1, ds2, names1, names2, knn=knn)
            # Find nearest neighbors in second direction.
            match2 = nn_approx(ds2, ds1, names2, names1, knn=knn)
        else:
            match1 = nn_annoy(ds1, ds2, names1, names2, norm=norm, knn=knn, metric=metric)
            match2 = nn_annoy(ds2, ds1, names2, names1, norm=norm, knn=knn, metric=metric)
    else:
        match1 = nn(ds1, ds2, names1, names2, knn=knn)
        match2 = nn(ds2, ds1, names2, names1, knn=knn)
    # Compute mutual nearest neighbors.
    mutual = match1 & set([ (b, a) for a, b in match2 ])

    return mutual
