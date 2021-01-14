import sys
import os
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
from xml.etree import ElementTree as ET

SCHEMA = '{http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15}'

def convert_to_tuple_list(string):
    return [tuple(int(i) for i in x.split(',')) for x in string.split(' ')]


def create_gt(img_name):
    img = Image.open(img_name)
    gt_img = Image.new('1', img.size, 0)

    output_path = os.path.join('gt', img_name)

    draw = ImageDraw.Draw(gt_img)

    root = ET.parse(os.path.join('page', img_name.split('.')[0] + '.xml'))

    page = root.find(SCHEMA + 'Page')

    text_regions = page.findall(SCHEMA + 'TextRegion')
    for text_region in text_regions:

        text_lines = text_region.findall(SCHEMA + 'TextLine')
        for text_line in text_lines:
            baseline = text_line.find(SCHEMA + 'Baseline')
            points = baseline.get('points')
            tuple_list = convert_to_tuple_list(points)
            draw.line(tuple_list, fill=1, width=12)
    
    # Create the ground-truth directory if it hasn't been created already.
    if not os.path.exists('gt'):
        os.makedirs('gt')

    # Save the image to the output directory
    gt_img.save(output_path)


def main(args):
    if len(args) > 0:
        directory = args[0]
    else:
        directory = './'

    files = os.listdir(directory)
    for f in files:
        if f.endswith('.jpg') or f.endswith('.tif') or f.endswith('.jpeg'):
            print('Processing', f)
            create_gt(f)


if __name__ == "__main__":
    main(sys.argv[1:])
