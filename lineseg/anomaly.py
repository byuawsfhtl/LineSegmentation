import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from skimage.draw import polygon
from util import cluster, sharpen_image


class AnomalyDetector():
    def __init__(self, threshold):
        self.density_score = 0
        self.threshold = threshold
        self.image_score = 0

    def load_image(self, image_path):
        # Load the image from the file path
        image = tf.io.read_file(image_path)

        # Decode the image as a tensor
        image = tf.image.decode_image(image, channels=3)

        # normalize pixel values
        image /= 255
        return image

    def flag_anomalies(self, baseline_img_path, min_points, plot_hull=False):
        
        baseline_prediction = self.load_image(baseline_img_path)

        baseline_img = tf.squeeze(baseline_prediction[:, :, 1])
        baseline_img = tf.cast(baseline_img, tf.float32)
        baseline_img = sharpen_image(baseline_img)

        # cluster baselines with DBSCAN
        baselines = cluster(baseline_img, 5, min_points=min_points)

        if len(baselines) == 0:
            print("No baseline clusters found.")
            return

        num_clusters = len(baselines)

        # Prepare the plot
        plt.figure(figsize=(10, 10))
        plt.imshow(np.array(baseline_img), cmap='gray')

        quality_score = 0
        
        # Ensure that each cluster is a consistent structure
        for cluster_points in baselines:
            cluster_points = np.array(cluster_points)
            if len(cluster_points) == 0:
                continue

            # Perform convex hull calculation and anomaly scoring
            hull = ConvexHull(cluster_points)
            hull_vertices = cluster_points[hull.vertices]

            # shift vertices down a bit
            hull_vertices += 3

            if plot_hull:
                # Plot the convex hull
                plt.plot(hull_vertices[:, 1], hull_vertices[:, 0], 'r-', linewidth=1)
            
            # take the x and y vals from hull vertices to get all pixel coordinates inside convex hull, then set those pixel values to True for mask
            mask = np.zeros(baseline_img.shape, np.bool_)
            rr, cc = polygon(hull_vertices[:, 0], hull_vertices[:, 1], baseline_img.shape)
            mask[rr, cc] = True

            # calculate average pixel value with convex hull mask over original image
            average_pixel_value = np.mean(np.array(baseline_img)[mask])
            print(f'Cluster with average pixel value: {average_pixel_value}')

            quality_score += average_pixel_value

        # take average quality score to decide whether to tag as anomaly or not
        quality_score /= num_clusters
        if quality_score < self.threshold:
            print(f'Anomaly detected for image: {os.path.basename(baseline_img_path)}')
            print(f"Total quality score was: {(quality_score * 100):.2f}%")
        else:
            print(f"No anomaly detected. Here is the overall quality score: {(quality_score * 100):.2f}%")
    
        if plot_hull:
            # Show the plot and block execution until the plot window is closed
            plt.title('Convex Hulls over Baseline Image')
            plt.show()


anamoly_detector = AnomalyDetector(0.8)
anamoly_detector.flag_anomalies(r'C:\Users\Mark Clement\FHTL\LineSegmentation\data\example\baselines\004005955_00781.jpg', 50, True)
