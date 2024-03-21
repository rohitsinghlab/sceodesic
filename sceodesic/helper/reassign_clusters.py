# code for reassigning small clusters 

import numpy as np 
from collections import Counter
from sklearn.cluster import KMeans


def reassign_clusters(data, kmeans, min_size, rct=0):

    print("flag 270: reassigning clusters")

    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    counter = Counter(labels)
    small_clusters = [key for key, val in counter.items() if val < min_size]

    print("flag 270.5: number of small clusters:", len(small_clusters))

    # we shouldn't ever have more than one recursion 
    assert rct < 2
    
    if len(small_clusters) == 0:
        return kmeans

    print("flag 271: clusters to remove:", small_clusters)
    print("flag 271: number of clusters to remove:", len(small_clusters))

    good_clusters = np.where(~np.isin(np.arange(centers.shape[0]), small_clusters))[0]
    good_centers = centers[good_clusters, :]
    ngood = len(good_clusters)
    # we only need to update the assignment, 
    # centers shouldn't change much if we simply add a few points

    print("flag 272: refitting kmeans using", ngood, "new clusters")

    new_kmeans = KMeans(ngood, init=good_centers, max_iter=1)
    new_kmeans.fit(data)
    return reassign_clusters(data, new_kmeans, min_size, rct+1)
