import tensorflow as tf
import PIL
import numpy
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def display(img, x_left, y_left, x_right, y_right) :
    img  = PIL.Image.fromarray(img.numpy().astype('uint8'), mode='RGB')
    fig, ax = plt.subplots()
    ax.imshow(img)
    rect = patches.Rectangle((x_left, y_left), x_right - x_left, y_right - y_left, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    plt.show()

def map_points_to_original_img(x_point, y_point, start_size, end_size) :
    original_ratio = start_size[1] / start_size[0]
    resized_ratio = end_size[1] /end_size[0]
    scales = []
    scales.append(start_size[1] / end_size[1])
    scales.append(start_size[0] / end_size[0])

    print('Image Start Size: {}'.format(start_size))
    print('Image End Size: {}'.format(end_size))
    print('Original Ratio: {}'.format(original_ratio))
    print('Resized Ratio: {}'.format(resized_ratio))

    #In this case there is padding along the X Axis
    if scales[0] < scales[1] :
        print('Padding in along the X axis')
        scales.clear()
        padding = end_size[0] * (resized_ratio - original_ratio)
        scales.append(start_size[1] / ( end_size[1] - padding)  )
        scales.append(start_size[0] / end_size[0]  )
        padding = padding // 2
        scale_x = scales[0]
        scale_y = scales[1]
        print('Scales: {}'.format(scales))
        print('Padding {}'.format(padding))
        return int(( x_point - padding ) * scale_x), int(y_point * scale_y)

    #In this case there is padding along the Y axis
    elif scales[0] > scales[1] :
        scales.clear()
        padding = end_size[1] * ( (1 / resized_ratio) - (1 / original_ratio) )
        scales.append(start_size[1] / end_size[1])
        scales.append(start_size[0] / ( end_size[0] - padding) )
        padding = padding // 2
        scale_x = scales[0]
        scale_y = scales[1]
        print('Scales: {}'.format(scales))
        print('Padding {}'.format(padding))
        return int(x_point * scale_x), int(( y_point - padding ) * scale_y)

    #There is no padding the image was evenly scaled
    elif scales[0] == scales[1] :
        scale_x = scales[0]
        scale_y = scales[1]
        return int(x_point * scale_x), int(y_point * scale_y)

path = 'download.jfif'
img = PIL.Image.open(path)
start_img = tf.convert_to_tensor(numpy.asarray(img))
original_size = start_img.get_shape()

#Fix for non even divisions
end_img = tf.image.resize_with_pad(start_img, 500, 1000)
resized_size = end_img.get_shape()

x_left, y_left, x_right, y_right = 600, 50, 725, 380
display(end_img, x_left, y_left, x_right, y_right)
x_left, y_left = map_points_to_original_img(x_left, y_left, original_size, resized_size)
x_right, y_right = map_points_to_original_img(x_right, y_right, original_size, resized_size)
print('X Left: {} Y Left: {} X Right: {} Y Right: {}'.format(x_left, y_left, x_right, y_right))
display(start_img, x_left, y_left, x_right, y_right)
