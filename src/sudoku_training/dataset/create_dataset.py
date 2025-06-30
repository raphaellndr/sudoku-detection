from pathlib import Path

import cv2
import numpy as np
from detector import get_biggest_contour, preprocess_image, reorder, split_into_boxes

IMG_WIDTH, IMG_HEIGHT = 450, 450


def process_sudoku_dataset(input_folder, output_folder, img_extensions=(".png", ".jpg", ".jpeg")):
    """
    Process all sudoku images in a folder and extract individual digit boxes.

    Args:
        input_folder (str): Path to folder containing sudoku images
        output_folder (str): Path to folder where digit boxes will be saved
        img_extensions (tuple): Supported image extensions
    """

    # Create output directory if it doesn't exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # Get all image files from input folder
    input_path = Path(input_folder)
    image_files = []
    for ext in img_extensions:
        image_files.extend(input_path.glob(f"*{ext}"))
        image_files.extend(input_path.glob(f"*{ext.upper()}"))

    print(f"Found {len(image_files)} images to process")

    successful_processed = 0
    failed_processed = 0

    for idx, image_path in enumerate(image_files):
        if idx < 130:
            pass
        elif idx > 275:
            break
        else:
            try:
                print(f"Processing image {idx + 1}/{len(image_files)}: {image_path.name}")

                # Read and resize image
                image = cv2.imread(str(image_path))
                if image is None:
                    print(f"  Error: Could not read image {image_path}")
                    failed_processed += 1
                    continue

                image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))

                # Preprocess image
                thresh = preprocess_image(image)

                # Find contours
                contours, hierarchy = cv2.findContours(
                    thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                # Get biggest contour (should be the sudoku grid)
                biggest_contour, max_area = get_biggest_contour(contours)

                if biggest_contour.size == 0:
                    print(f"  Error: No suitable contour found in {image_path}")
                    failed_processed += 1
                    continue

                # Reorder points and apply perspective transform
                biggest_contour = reorder(biggest_contour)
                points_1 = np.float32(biggest_contour)
                points_2 = np.float32(
                    [
                        [0, 0],
                        [IMG_WIDTH, 0],
                        [0, IMG_HEIGHT],
                        [IMG_WIDTH, IMG_HEIGHT],
                    ]
                )

                matrix = cv2.getPerspectiveTransform(points_1, points_2)
                warped = cv2.warpPerspective(image, matrix, (IMG_WIDTH, IMG_HEIGHT))

                # Convert to grayscale and split into boxes
                gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
                boxes = split_into_boxes(gray)

                # Save individual boxes
                image_name = image_path.stem  # filename without extension
                save_digit_boxes(boxes, output_folder, image_name)

                successful_processed += 1
                print(f"  Successfully processed: {image_name}")

            except Exception as e:
                print(f"  Error processing {image_path}: {e!s}")
                failed_processed += 1
                continue

    print("\nProcessing complete!")
    print(f"Successfully processed: {successful_processed} images")
    print(f"Failed to process: {failed_processed} images")
    print(f"Total digit boxes extracted: {successful_processed * 81}")


def save_digit_boxes(boxes, output_folder, image_name):
    """
    Save individual digit boxes to the output folder.

    Args:
        boxes (list): List of 81 digit boxes (9x9 grid)
        output_folder (str): Path to output folder
        image_name (str): Name of the original image (used as prefix)
    """

    for i, box in enumerate(boxes):
        # Calculate row and column (0-8)
        row = i // 9
        col = i % 9

        # Create filename: imagename_row_col.png
        filename = f"{image_name}_r{row}_c{col}.png"
        filepath = Path(output_folder) / filename

        # Save the box
        cv2.imwrite(str(filepath), box)


def create_organized_dataset(input_folder, output_folder, img_extensions=(".png", ".jpg", ".jpeg")):
    """
    Create an organized dataset with subfolders for empty cells and digits 1-9.
    This version creates folders for manual sorting later.

    Args:
        input_folder (str): Path to folder containing sudoku images
        output_folder (str): Path to folder where organized dataset will be created
    """

    # Create main output directory
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create subfolders for each digit class (0=empty, 1-9=digits)
    class_folders = ["empty", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    for class_name in class_folders:
        (output_path / class_name).mkdir(exist_ok=True)

    # Process all images and save to 'unsorted' folder initially
    unsorted_folder = output_path / "unsorted"
    unsorted_folder.mkdir(exist_ok=True)

    process_sudoku_dataset(input_folder, str(unsorted_folder), img_extensions)

    print(f"\nDigit boxes saved to {unsorted_folder}")


if __name__ == "__main__":
    input_folder = (
        "/Users/raphaellndr/Documents/programmation/python/sudoku-detection/sudoku_dataset"
    )
    output_folder = (
        "/Users/raphaellndr/Documents/programmation/python/sudoku-detection/digits_datasetv2"
    )
    create_organized_dataset(input_folder, output_folder)
