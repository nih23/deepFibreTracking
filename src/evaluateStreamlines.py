from dipy.segment.metric import SumPointwiseEuclideanMetric
from dipy.segment import quickbundles


def pairwiseDistance(streamlines_a, streamlines_b, nb_points = 20):
    
    sl_est_2 = set_number_of_points(Streamlines(streamlines_a), nb_points = nb_points)
    sl_val_2 = set_number_of_points(Streamlines(streamlines_b), nb_points = nb_points)
    dist = quickbundles.bundles_distances_mdf(sl_est_2, sl_val_2)
    
    dist_idx = np.argmin(dist, axis = 1)
    dist_val = np.min(dist, axis = 1)
    
    return dist_val, dist_idx
    