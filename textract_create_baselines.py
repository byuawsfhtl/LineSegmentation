import json
import os
from PIL import Image, ImageDraw
import argparse

# Function to process the JSON file and filter out lines
def filter_lines(json_data):
    lines = []
    for block in json_data['Blocks']:
        if block['BlockType'] == 'LINE':
            polygon = block['Geometry']['Polygon']
            lines.append({'polygon': polygon})
    return lines

# Function to create the ground truth image with white baselines on black background
def draw_gt(image_size, lines, output_path, offset):
    gt_img = Image.new('1', image_size, 0)  # Create a new binary image with black background
    draw = ImageDraw.Draw(gt_img)

    for line in lines:
        polygon = [(point['X'] * image_size[0], point['Y'] * image_size[1]) for point in line['polygon']]
        midpoints = [( (polygon[i][0] + polygon[-(i + 1)][0]) / 2,
                       (polygon[i][1] + polygon[-(i + 1)][1]) / 2 + offset) for i in range(len(polygon) // 2)]
        midpoints = sorted(midpoints, key=lambda x: x[0])
        draw.line(midpoints, fill=1, width=8)  # Draw white lines on the binary image

    gt_img.save(output_path)

# Function to create the overlay image with red baselines on the original image
def draw_overlay(img, lines, output_path, offset):
    overlay_img = img.copy().convert("RGBA")
    draw = ImageDraw.Draw(overlay_img)

    for line in lines:
        polygon = [(point['X'] * img.width, point['Y'] * img.height) for point in line['polygon']]
        midpoints = [( (polygon[i][0] + polygon[-(i + 1)][0]) / 2,
                       (polygon[i][1] + polygon[-(i + 1)][1]) / 2 + offset) for i in range(len(polygon) // 2)]
        midpoints = sorted(midpoints, key=lambda x: x[0])
        draw.line(midpoints, fill=(255, 0, 0, 255), width=8)  # Draw red lines with full opacity

    # Convert the image to RGB mode before saving as JPEG
    overlay_img = overlay_img.convert("RGB")
    overlay_img.save(output_path)

def process_directory(input_dir, json_dir, output_dir, overlay_output_dir=None, offset=20):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if overlay_output_dir and not os.path.exists(overlay_output_dir):
        os.makedirs(overlay_output_dir)

    for image_name in os.listdir(input_dir):
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_dir, image_name)
            json_path = os.path.join(json_dir, os.path.splitext(image_name)[0], "analyzeDocResponse.json")
            gt_output_path = os.path.join(output_dir, image_name)

            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    json_data = json.load(f)

                lines = filter_lines(json_data)
                img = Image.open(image_path).convert("RGBA")
                
                # Create the ground truth image
                draw_gt(img.size, lines, gt_output_path, offset)

                # Create the overlay image if the directory is specified
                if overlay_output_dir:
                    overlay_output_path = os.path.join(overlay_output_dir, image_name)
                    draw_overlay(img, lines, overlay_output_path, offset)
            else:
                print(f"JSON file for {image_name} not found at {json_path}")

def main():
    parser = argparse.ArgumentParser(description="Create binary images with baselines based on JSON data.")
    parser.add_argument("input_dir", help="Path to the directory containing input image files")
    parser.add_argument("json_dir", help="Path to the directory containing JSON files")
    parser.add_argument("output_dir", help="Path to the directory where processed image files will be saved")
    parser.add_argument("--overlay_output_dir", help="Directory to save overlay images")
    parser.add_argument("--offset", type=int, default=20, help="Offset to apply to the Y-coordinate of the baseline")

    args = parser.parse_args()

    process_directory(args.input_dir, args.json_dir, args.output_dir, args.overlay_output_dir, args.offset)

if __name__ == "__main__":
    main()
