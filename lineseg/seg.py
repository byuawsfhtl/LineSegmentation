import os

import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.ndimage.filters import median_filter


def segment_from_predictions(original_img, baseline_prediction, filename, save_path, save_line_images=True,
                             plot_images=False, max_above=25, max_below=15, include_coords_in_path=False,
                             save_raw=False, raw_path=None):
    """
    Produce line-level segmentations based on the baseline prediction and write the segments to the specified path.

    :param original_img: The original image given as input to the segmentation model
    :param baseline_prediction: The output of the segmentation model
    :param filename: The name of the file that is being segmented (will be used when creating snippet names)
    :param save_path: The path which text line snippets will be saved
    :param save_line_images: Boolean indicating whether or not to save text-line images
    :param plot_images: Boolean indicating whether or not to plot text-line images
    :param max_above: The maximum threshold in pixels the search algorithm will look above a baseline for the next line
                      to create the bounding polygon.
    :param max_below: The maximum threshold in pixels the search algorithm will look below a baseline for the next line
                      to create the bounding polygon.
    :param include_coords_in_path: Boolean indicating whether or not the coordinate information for the line will be
                                   included in the name of each line snippet.
    :param save_raw: Boolean indicating whether or not to save the raw baseline prediction output
    :param raw_path: If save_raw is True, the path to save the raw basline prediction output images
    :return: None
    """
    # Image pre-processing and manipulations to get the images in the right format
    original_img = tf.squeeze(original_img).numpy()
    original_img = original_img * 255
    original_img = original_img.astype(np.uint8)

    baseline_image = tf.squeeze(baseline_prediction[:, :, 1])
    baseline_image = sharpen_image(baseline_image)

    if save_raw:
        if raw_path is None:
            raise Exception("Raw path must be set to save raw output")

        # Convert to jpeg and write file
        encoded = tf.image.encode_jpeg(tf.expand_dims(tf.cast(baseline_image * 255, tf.uint8), 2))
        tf.io.write_file(os.path.join(raw_path, filename + '.jpg'), encoded)

    # Cluster the baselines given in baseline image using DBSCAN
    baselines = cluster(baseline_image)

    # Create a new cleaner baseline image using the clustered and filtered baselines
    new_baseline_image = create_new_image_from_baselines(baselines, (baseline_prediction.shape[1],
                                                                     baseline_prediction.shape[0]))

    # Sort the clustered lines top to bottom on each half of the page
    columns = sort_lines(baselines, baseline_prediction.shape[0:2])

    # Iterate over each column
    for col_index, baselines in enumerate(columns):

        # Iterate over all baselines and segment the text lines
        for baseline_index in range(len(baselines)):
            baseline = baselines[baseline_index]

            # Lists to hold upper and lower poly-lines
            upper_polyline, lower_polyline = [], []
            upper_polyline_found, lower_polyline_found = [], []

            # Iterate over each pixel in a baseline and create bounding polygon
            for point in baseline:
                # Search above the baseline
                above_point, above_found = search_up(point, new_baseline_image, max_height=int(max_above / .7))
                if not above_found:
                    above_space = max_above
                else:
                    above_space = int((point[0] - above_point[1]) * 0.7)
                upper_point_y = point[0] - above_space
                if upper_point_y < 0:
                    upper_point_y = 0
                upper_point_x = point[1]

                # Search below the baseline
                below_point, below_found = search_down(point, new_baseline_image, max_height=int(max_below / .4))
                if not below_found:
                    below_space = max_below
                else:
                    below_space = int((below_point[1] - point[0]) * 0.4)
                lower_point_y = point[0] + below_space
                if lower_point_y >= baseline_prediction.shape[0]:
                    lower_point_y = baseline_prediction.shape[0] - 1
                lower_point_x = point[1]

                # Append points to poly-line lists
                upper_polyline_found.append(above_found)
                lower_polyline_found.append(below_found)
                upper_polyline.append((upper_point_x, upper_point_y))
                lower_polyline.append((lower_point_x, lower_point_y))

            # Clean the poly-lines depending on whether or not the poly-line was found
            upper_polyline = clean_seam(upper_polyline, upper_polyline_found)
            lower_polyline = clean_seam(lower_polyline, lower_polyline_found)

            # Merge the two poly-lines into a single polygon
            polygon = np.concatenate((upper_polyline, lower_polyline[::-1], np.expand_dims(upper_polyline[0], 0)))

            polygon = map_points_to_original_img(polygon, original_img.shape, baseline_prediction.shape)

            x_coords = [poly[0] for poly in polygon]
            y_coords = [poly[1] for poly in polygon]

            left_y = np.min(y_coords)
            left_x = np.min(x_coords)
            right_y = np.max(y_coords)
            right_x = np.max(x_coords)

            # Segment the text line from the original image based on the given polygon
            segment, segment_baseline = segment_from_polygon(polygon, Image.fromarray(original_img), baseline)
            dewarped_segment = dewarp(segment, segment_baseline)
            final_segment = final_crop(dewarped_segment)

            if include_coords_in_path:
                snippet_name = filename + '_' + str(col_index) + '_' + str(baseline_index).zfill(3) + '_'\
                               + str(left_y).zfill(4) + '_' + str(left_x).zfill(4) + '_' + str(right_y).zfill(4) + '_'\
                               + str(right_x).zfill(4) + '.jpg'
            else:
                snippet_name = filename + '_' + str(col_index) + '_' + str(baseline_index).zfill(3) + '.jpg'

            if save_line_images:
                save_image(final_segment, save_path, snippet_name)
            if plot_images:
                plot_image(final_segment, snippet_name)


def create_new_image_from_baselines(baselines, img_size):
    """
    Create new clean image from baselines

    :param baselines: A list of points that will be drawn onto a new binary image
    :param img_size: The size of the new image to be created
    :return: The new image as a numpy array
    """
    img = Image.new('1', img_size, 0)
    draw = ImageDraw.Draw(img)

    for baseline in baselines:
        baseline = [(point[1], point[0]) for point in baseline]
        draw.line(baseline, fill=1, width=1)

    return np.array(img)


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


def cluster(image, min_points=50):
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
    clustered_points = DBSCAN(eps=5, min_samples=15).fit(points)

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


def search_up(point, image, max_height=100, min_height=1):
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


def search_down(point, image, max_height=50, min_height=1):
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


def plot_image(img, title=None, figsize=(8, 8)):
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


def segment_from_polygon(polygon, original_image, baseline, cushion=0):
    """
    Given a polygon (as numpy array), segment the image and return the new image segment
    with its new corresponding baseline.

    :param polygon: The bounding polygon around the text-line to be extracted
    :param original_image: The original image that contains the bounding polygon
    :param baseline: The baseline that corresponds to the given text-line
    :param cushion: How much whitespace we should add above and below to account for dewarping
    :return: The segmented image, new baseline corresponding to segmented image
    """
    x_values = [x for x, y in polygon]
    y_values = [y for x, y in polygon]

    y_start = np.min(y_values)
    y_end = np.max(y_values)
    x_start = np.min(x_values)
    x_end = np.max(x_values)

    blank_img = Image.new("L", original_image.size, 255)
    mask = Image.new("1", original_image.size, 0)
    poly_draw = ImageDraw.Draw(mask)
    poly_draw.polygon([(xcoord, ycoord) for xcoord, ycoord in polygon], fill=255)

    y_max = original_image.size[1] - 1  # The size dim in pillow is backwards compared to numpy

    # Add a cushion to boundaries, so we don't cut off text when dewarping
    y_start -= cushion
    if y_start < 0:
        y_start = 0
    y_end += cushion
    if y_end > y_max:
        y_end = y_max

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
    median = np.median(baseline_y)

    for point in baseline:
        # The x-coordinate represents a column in the image
        column = int(point[1])

        # Calculate the shift based on the difference between the y-coordinate and the median
        shift = int(median - point[0])

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


def sort_lines(lines, img_shape, num_columns=2, kernel_size=10):
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
    height, width = img_shape

    col_step = width // num_columns

    columns_list = []
    for col in range(0, width, col_step):
        x_start = col
        x_end = col + col_step

        column_lines = []

        for row in range(0, height, kernel_size):
            y_start = row
            y_end = row + kernel_size

            for line in lines:
                y, x = line[0]
                if y_start <= y < y_end and x_start <= x < x_end:
                    column_lines.append(line)

        columns_list.append(column_lines)

    return columns_list


def map_points_to_original_img(polygon, start_size, end_size):
    """
    This function takes two points on the resized image and will return what the points are in the original image

    :param polygon: The polygon that is getting mapped to the original image
    :param start_size: The original size dimensions of the image before the resizing
    :param end_size: The size dimensions of the image after the resizing
    """
    original_ratio = start_size[1] / start_size[0]
    scales = [start_size[1] / end_size[1], start_size[0] / end_size[0]]

    # In this case there is padding along the X Axis
    if scales[0] < scales[1]:
        padding = end_size[1] - (original_ratio * end_size[0])
        scale = start_size[0] / end_size[0]
        padding = padding / 2
        for poly_index in range(len(polygon)):
            point = polygon[poly_index]
            polygon[poly_index][0] = int(round((point[0] - padding) * scale))
            polygon[poly_index][1] = int(round(point[1] * scale))

    # In this case there is padding along the Y axis
    elif scales[0] > scales[1]:
        padding = end_size[0] - ((1 / original_ratio) * end_size[1])
        scale = start_size[1] / end_size[1]
        padding = padding / 2
        for poly_index in range(len(polygon)):
            point = polygon[poly_index]
            polygon[poly_index][0] = int(round(point[0] * scale))
            polygon[poly_index][1] = int(round((point[1] - padding) * scale))

    # There is no padding the image was evenly scaled
    elif scales[0] == scales[1]:
        scale = scales[0]
        for poly_index in range(len(polygon)):
            point = polygon[poly_index]
            polygon[poly_index][0] = int(round(point[0] * scale))
            polygon[poly_index][1] = int(round(point[1] * scale))

    return polygon
