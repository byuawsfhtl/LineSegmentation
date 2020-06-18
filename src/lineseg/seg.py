import os

import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from shapely.geometry import Polygon
from scipy.ndimage.filters import median_filter


def segment_from_predictions(original_image, baseline_prediction, seam_prediction, filename, step_size=1,
                             save_images=True, plot_images=False, save_path='./data/out/snippets'):
    original_image = tf.squeeze(original_image).numpy()
    baseline_image = tf.squeeze(tf.argmax(baseline_prediction, axis=3)).numpy()
    seam_image = tf.squeeze(seam_prediction[:, :, :, 1])

    sharpened_seam_image = sharpen_image(seam_image)

    baselines = cluster(baseline_image)

    baselines = sort_lines(baselines, original_image.shape)

    # Search the cleaned-up seam image for upper/lower seams
    # Create a polygon around the line based on the seam data
    polygons = []
    for baseline in baselines:
        seam_top = []
        seam_top_founds = []
        seam_bottom = []
        seam_bottom_founds = []
        for index, point in enumerate(baseline):
            if index % step_size == 0:
                seam_top_point, seam_top_found = search_up(point, sharpened_seam_image)
                seam_bottom_point, seam_bottom_found = search_down(point, sharpened_seam_image)

                seam_top.append(seam_top_point)
                seam_bottom.append(seam_bottom_point)

                seam_top_founds.append(seam_top_found)
                seam_bottom_founds.append(seam_bottom_found)

        seam_top_clean = clean_seam(seam_top, seam_top_founds)
        seam_bottom_clean = clean_seam(seam_bottom, seam_bottom_founds)

        if len(seam_top_clean) != 0 and len(seam_bottom_clean) != 0:
            polygons.append(np.concatenate((seam_top_clean, seam_bottom_clean[::-1])))

    # Iterate over all baselines/polygons - segment, dewarp, and crop
    for index, (poly, baseline) in enumerate(zip(polygons, baselines)):
        segment, segment_baseline = segment_from_polygon(Polygon(poly), Image.fromarray(original_image), baseline)
        dewarped_segment = dewarp(segment, segment_baseline)
        final_segment = final_crop(dewarped_segment)

        snippet_name = filename + '_' + str(index) + '.jpg'

        if save_images:
            save_image(final_segment, save_path, snippet_name)
        if plot_images:
            plot_image(final_segment, snippet_name)


def sharpen_image(seam_image, thresh_start=.1, thresh_end=.9, filter_sizes=(4, 4, 3)):
    clean_seam_image = np.where(seam_image > thresh_start, 1, 0)

    # Perform filtering
    for kernel_size in filter_sizes:
        clean_seam_image = median_filter(clean_seam_image, size=kernel_size)

    clean_seam_image = np.where(clean_seam_image > thresh_end, 1, 0)

    return clean_seam_image


def cluster(image, min_points=10):
    # Perform clustering according to the DBSCAN algorithm
    points = tf.where(image).numpy()  # Find the coordinates that are non-zero
    clustered_points = DBSCAN(eps=2.5, min_samples=2).fit(points)

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

    return nms_clusters


def search_up(point, image, max_height=20, min_height=4):
    y, x = point
    y_start = y

    while image[y][x] == 0 or y_start - y < min_height:
        y -= 1
        if y < 0:  # Bounds check
            break
        if y_start - y > max_height:
            return [x, y], False  # Return False if no seam was found

    return [x, y], True


def search_down(point, image, max_height=12, min_height=2):
    y_max = image.shape[0] - 1
    y, x = point
    y_start = y

    while image[y][x] == 0 or y - y_start < min_height:
        y += 1
        if y > y_max:  # Bounds check
            break
        if y - y_start > max_height:
            return [x, y], False  # Return False if no seam was found

    return [x, y], True


def clean_seam(seam, founds):
    new_seam = []

    prev_none_x = -1
    for point, seam_found in zip(seam, founds):
        if seam_found:
            if prev_none_x != -1:
                new_seam.append([prev_none_x, point[1]])

            new_seam.append(point)
            prev_none_x = -1
        else:
            if prev_none_x == -1:
                prev_none_x = point[0]

    # If we weren't able to clean up the seam, return the old
    if len(new_seam) == 0:
        return seam
    else:
        # If the last point was none, add the last point to the new seam with the y-value
        # mimicking the previous y_value
        if prev_none_x != -1:
            new_seam.append([seam[-1][0], new_seam[-1][1]])

        return new_seam


def save_image(img, path, name):
    if not os.path.exists(path):
        os.makedirs(path)

    img = Image.fromarray(img)
    img.save(os.path.join(path, name))


def plot_image(img, title=None, figsize=(20, 20)):
    plt.figure(figsize=figsize)
    if title is not None:
        plt.title(title)
    plt.imshow(img, cmap='gray')
    plt.show()


def segment_from_polygon(polygon: Polygon, original_image, baseline, cushion=8):
    poly_coords = polygon.exterior.coords[:]
    bounds = polygon.bounds

    blank_img = Image.new("L", original_image.size, 255)
    mask = Image.new("1", original_image.size, 0)
    poly_draw = ImageDraw.Draw(mask)
    poly_draw.polygon(poly_coords, fill=255)

    y_max = original_image.size[1] - 1  # The size dim in pillow is backwards compared to numpy

    # Add a cushion to boundaries, so we don't cut off text when dewarping
    y_start = int(bounds[1]) - cushion
    if y_start < 0:
        y_start = 0
    y_end = int(bounds[3]) + cushion
    if y_end > y_max:
        y_end = y_max

    # We're only dewarping y_coordinates, so we don't worry about it here.
    x_start = int(bounds[0])
    x_end = int(bounds[2])

    new_img = Image.composite(original_image, blank_img, mask)
    new_baseline = [(point[0] - y_start, point[1] - x_start) for point in baseline]

    new_img_cropped = np.array(new_img)[y_start:y_end, x_start:x_end]
    new_baseline_cropped = list(filter(lambda p: 0 <= int(p[1]) < new_img_cropped.shape[1], new_baseline))

    return new_img_cropped, new_baseline_cropped


def dewarp(img, baseline):
    img_copy = img.copy()

    baseline_y = [point[0] for point in baseline]
    baseline_median = np.median(baseline_y)

    for point in baseline:
        # The x-coordinate represents a column in the image
        column = int(point[1])

        # Calculate the shift based on the difference between the y-coordinate and the median
        shift = int(baseline_median - point[0])

        # Shift the column up or down depending on the difference calculated
        shift_column(img_copy, column, shift)

    return img_copy


def shift_column(im, column: int, shift: int):
    im[:, column] = np.roll(im[:, column], shift, axis=0)

    # When shifting, fill the ends with white pixels. Don't roll the numbers.
    if shift > 0:
        im[:, column][:shift] = 255
    if shift < 0:
        im[:, column][shift:] = 255


def final_crop(im):
    # Mask of non-black pixels (assuming image has a single channel).
    mask = im < 255

    # Coordinates of non-black pixels.
    coords = np.argwhere(mask)

    # Bounding box of non-black pixels.
    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0) + 1  # slices are exclusive at the top

    # Get the contents of the bounding box.
    cropped = im[x0:x1, y0:y1]

    return cropped


def sort_lines(lines, img_shape, num_columns=2, kernel_size=10):
    sorted_lines_list = []

    height, width = img_shape

    col_step = width // num_columns

    for col in range(0, width, col_step):
        x_start = col
        x_end = col + col_step

        for row in range(0, height, kernel_size):
            y_start = row
            y_end = row + kernel_size

            for line in lines:
                y, x = line[0]
                if y_start <= y < y_end and x_start <= x < x_end:
                    sorted_lines_list.append(line)

    assert len(sorted_lines_list) == len(lines)

    return sorted_lines_list
