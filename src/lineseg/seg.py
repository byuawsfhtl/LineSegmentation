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
    """
    Segment the baseline and seam predictions and write the segments to the specified path.

    :param original_image: The original image to be segmented
    :param baseline_prediction: The predicted baselines (ARU-Net output)
    :param seam_prediction: The predicted seams (ARU-Net output)
    :param filename: The name of the file that is being segmented
    :param step_size: How many columns along the baseline to look at when searching the seam image to find
                      the bounding polygon
    :param save_images: Whether or not to save the images that are segmented
    :param plot_images: Whether or not to plot the images that are segmented
    :param save_path: The path to save the images to
    :return: None
    """
    original_image = tf.squeeze(original_image).numpy()
    # baseline_image = tf.squeeze(tf.argmax(baseline_prediction, axis=3)).numpy()
    baseline_image = tf.squeeze(baseline_prediction[:, :, :, 1])
    seam_image = tf.squeeze(seam_prediction[:, :, :, 1])

    sharpened_baseline_image = sharpen_image(baseline_image)
    sharpened_seam_image = sharpen_image(seam_image)

    baselines = cluster(sharpened_baseline_image)
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


def sharpen_image(image_prediction, thresh_start=.1, thresh_end=.9, filter_sizes=(4, 4, 3)):
    """
    Sharpen an image by using a serious of median filters.

    :param image_prediction: The image prediction
    :param thresh_start: Threshold at start before filtering for binarization
    :param thresh_end: Threshold at end after filtering for binarization
    :param filter_sizes: Sizes of the median filters to be used
    :return: The sharpened image
    """
    clean_seam_image = np.where(image_prediction > thresh_start, 1, 0)

    # Perform filtering
    for kernel_size in filter_sizes:
        clean_seam_image = median_filter(clean_seam_image, size=kernel_size)

    clean_seam_image = np.where(clean_seam_image > thresh_end, 1, 0)

    return clean_seam_image


def cluster(image, min_points=100):
    """
    Cluster the points on the image using the DBSCAN clustering algorithm. Perform some form of skeletonization.

    :param image: The predicted baseline image
    :param min_points: The minimum number of line pixels (after skeletonization) that must be included for the cluster
                       to be considered.
    :return: The baselines as a list of lists of points
    """
    # Perform clustering according to the DBSCAN algorithm
    points = tf.where(image).numpy()  # Find the coordinates that are non-zero
    clustered_points = DBSCAN(eps=10, min_samples=5).fit(points)

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


def search_up(point, image, max_height=60, min_height=8):
    """
    Search for a seam point above the given baseline point.

    :param point: The baseline point to be searched from
    :param image: The image to be searched
    :param max_height: The max number of pixels to be searched until the max point is returned
    :param min_height: The min number of pixels to be searched before a seam point can be considered found
    :return: The found seam point
    """
    y, x = point
    y_start = y

    while y > 0 and (image[y][x] == 0 or y_start - y < min_height):
        y -= 1
        if y_start - y > max_height:
            return [x, y], False  # Return False if no seam was found
    seam_begin = y

    while y > 0 and y_start - y <= max_height * 2 and (image[y][x] == 1 or y_start - y < min_height):
        y -= 1
    seam_end = y

    final_y = np.floor((seam_begin + seam_end) / 2)

    return [x, final_y], True


def search_down(point, image, max_height=30, min_height=6):
    """
    Search for a seam point below the given baseline point.

    :param point: The baseline point to be searched from
    :param image: The image to be searched
    :param max_height: The max number of pixels to be searched until the max point is returned
    :param min_height: The min number of pixels to be searched before a seam point can be considered found
    :return: The found seam point
    """
    y_max = image.shape[0] - 1
    y, x = point
    y_start = y

    while y < y_max and (image[y][x] == 0 or y - y_start < min_height):
        y += 1
        if y - y_start > max_height:
            return [x, y], False  # Return False if no seam was found
    seam_begin = y

    while y < y_max and y - y_start <= max_height * 2 and (image[y][x] == 1 or y - y_start < min_height):
        y += 1
    seam_end = y

    final_y = np.ceil((seam_begin + seam_end) / 2)

    return [x, final_y], True


def clean_seam(seam, founds):
    """
    Clean the extracted seam by removing outliers

    :param seam: The seam as list of lists
    :param founds: A list of whether or not the search algorithm found a seam or if the current point is the max value
    :return: The cleaned seam
    """
    new_seam = []

    # Iterate over the seams and replace outliers (where a seam point *was not* found) with
    # a nearby seam point that *was* found
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
    """
    Save the image in the specified path and name
    :param img: Image to be saved as numpy array
    :param path: Path to directory to be saved
    :param name: Image name
    :return: None
    """
    if not os.path.exists(path):
        os.makedirs(path)

    img = Image.fromarray(img)
    img.save(os.path.join(path, name))


def plot_image(img, title=None, figsize=(20, 20)):
    """
    Plot the image. Requires user input to continue program execution.

    :param img: Image to plot
    :param title: Title of the plot
    :param figsize: Size of the plot as tuple
    :return: None
    """
    plt.figure(figsize=figsize)
    if title is not None:
        plt.title(title)
    plt.imshow(img, cmap='gray')
    plt.show()


def segment_from_polygon(polygon: Polygon, original_image, baseline, cushion=0):
    """
    Given a Shapely Polygon, segment the image and return the new image segment
    with its new corresponding baseline.

    :param polygon: The bounding polygon around the text-line to be extracted
    :param original_image: The original image that contains the bounding polygon
    :param baseline: The baseline that corresponds to the given text-line
    :param cushion: How much whitespace we should add above and below to account for dewarping
    :return: The segmented image, new baseline corresponding to segmented image
    """
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
    """
    Dewarp the image according to the baseline.

    :param img: Image to be warped
    :param baseline: The baseline corresponding to the text-line as list of points
    :return:
    """
    # Make a copy so we can modify this image without affecting the original image
    img_copy = img.copy()

    # Find the median y point on the baseline
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
    """
    This function will shift a given column in an image up or down.
    The image will be shifted in-place.
    Used for dewarping an image in the dewarp function.

    Pixels shifted out of the image will not be wrapped to bottom or top of the image

    :param im: Image whose column will be shifted
    :param column: Column to shift
    :param shift: The number of pixels to be shifted up or down
    :return: None
    """
    im[:, column] = np.roll(im[:, column], shift, axis=0)

    # When shifting, fill the ends with white pixels. Don't roll the numbers.
    if shift > 0:
        im[:, column][:shift] = 255
    if shift < 0:
        im[:, column][shift:] = 255


def final_crop(im):
    """
    After text-line extraction and dewarping, there is often a great deal of white space around the image.
    This function will crop the white space out and return only the image bounded by the text line.
    :param im: The image to be cropped
    :return: The cropped image
    """
    # Mask of non-black pixels (assuming image has a single channel).
    mask = im < 255

    # Coordinates of non-black pixels.
    coords = np.argwhere(mask)

    if len(coords) == 0:
        return im  # Return the original image if no black coordinates are found

    # Bounding box of non-black pixels.
    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0) + 1  # slices are exclusive at the top

    # Get the contents of the bounding box.
    cropped = im[x0:x1, y0:y1]

    return cropped


def sort_lines(lines, img_shape, num_columns=1, kernel_size=10):
    """
    This function will sort baselines from top-down. It also has the capability to sort from top-down
    one column at a time. This can be particularly useful if baselines need to be outputted in a
    a specific order

    :param lines: The lines to be sorted in list of lists format
    :param img_shape: tuple giving the image shape (height, width)
    :param num_columns: The number of columns used when sorting from top-down
    :param kernel_size: The kernel size used when scanning for baselines from top-down
    :return: The sorted lines
    """
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
