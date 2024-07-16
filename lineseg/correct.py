import cv2
import numpy as np
import argparse

def process_image(image_path, output_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to load image {image_path}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Apply erosion and then dilation to separate text lines
    erosion_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    erosion_kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 3))

    eroded = cv2.erode(gray, erosion_kernel, iterations=1)
    dilated = cv2.dilate(eroded, dilation_kernel, iterations=1)
    final = cv2.erode(dilated, erosion_kernel2, iterations=1)

    cv2.imwrite(output_path, final)
    print(f"Processed image saved as {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Process an image to separate text lines.")
    parser.add_argument("input_image", help="Path to the input image file")
    parser.add_argument("output_image", help="Path to save the processed image file")

    args = parser.parse_args()
    process_image(args.input_image, args.output_image)

if __name__ == "__main__":
    main()
