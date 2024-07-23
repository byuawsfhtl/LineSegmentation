import tensorflow as tf
from sklearn.cluster import DBSCAN
from scipy.ndimage.filters import median_filter
import numpy as np


@tf.function
def model_inference(model, imgs):
    return model(imgs, training=False)

def cluster(image, eps=5, min_points=50):
    """
    Cluster the points on the image using the DBSCAN clustering algorithm. Perform some form of skeletonization.

    :param image: The predicted baseline image
    :param min_points: The minimum number of line pixels (after skeletonization) that must be included for the cluster
                       to be considered.
    :return: The baselines as a list of lists of points
    """
    # Perform clustering according to the DBSCAN algorithm
    points = tf.where(image).numpy()  # Find the coordinates that are non-zero
    if len(points) == 0:
        return []  # If we didn't predict any baselines, return an empty baseline cluster array
    clustered_points = DBSCAN(eps=eps, min_samples=15).fit(points)

    # Create a list of lists to hold the clusters based on the labeling
    unique_labels = np.unique(clustered_points.labels_)
    if -1 in unique_labels:
        num_labels = len(unique_labels) - 1
    else:
        num_labels = len(unique_labels)

    clusters = [[] for _ in range(num_labels)]

    # Place points corresponding to a given label into their own list
    for label, point in zip(clustered_points.labels_, points):
        if label != -1:
            clusters[label].append(point.tolist())

    # Sort the clusters from left to right
    for c in clusters:
        c.sort(key=lambda p: p[1])

    # Perform non-maximum suppression so we only have one point per column
    nms_clusters = []
    for c in clusters:  # For each cluster
        c_cluster = []
        current = -1
        for point in c:  # For each point in a cluster
            if point[1] > current:
                c_cluster.append(point)
                current = point[1]
        nms_clusters.append(c_cluster)

    # Filter out minimum points
    nms_clusters = list(filter(lambda cl: len(cl) > min_points, nms_clusters))

    for nms_cluster in nms_clusters:
        first_x_point = nms_cluster[0][1] - 5
        if first_x_point < 0:
            first_x_point = 0
        first_point = (nms_cluster[0][0], first_x_point)

        last_x_point = nms_cluster[-1][1] + 5
        if last_x_point >= image.shape[1]:
            last_x_point = image.shape[1] - 1
        last_point = (nms_cluster[-1][0], last_x_point)

        nms_cluster.insert(0, first_point)
        nms_cluster.append(last_point)

    return nms_clusters

def sharpen_image(image_prediction, thresh=.1, filter_sizes=(3, 3)):
    """
    Sharpen an image by using a serious of median filters.

    :param image_prediction: The image prediction
    :param thresh: Threshold at start before filtering for binarization
    :param filter_sizes: Sizes of the median filters to be used
    :return: The sharpened image
    """
    clean_seam_image = np.where(image_prediction > thresh, 1, 0)

    # Perform filtering
    for kernel_size in filter_sizes:
        clean_seam_image = median_filter(clean_seam_image, size=kernel_size)

    return clean_seam_image
