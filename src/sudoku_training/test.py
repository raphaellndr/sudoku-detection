import cv2
import numpy as np
from detector import detect_digits, get_biggest_contour, preprocess_image, reorder, split_into_boxes

IMG_WIDTH, IMG_HEIGHT = 450, 450

image = cv2.imread(
    "/Users/raphaellndr/Documents/programmation/python/sudoku-detection/data/hard_sudoku.png"
)
image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
cv2.imshow("Original Image", image)
cv2.waitKey(0)

thresh = preprocess_image(image)
cv2.imshow("Thresholded Image", thresh)
cv2.waitKey(0)

img_contours = image.copy()
img_big_contours = image.copy()
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 3)
cv2.imshow("Contours", img_contours)
cv2.waitKey(0)

biggest_contour, max_area = get_biggest_contour(contours)
if biggest_contour.size != 0:
    biggest_contour = reorder(biggest_contour)
    cv2.drawContours(img_big_contours, biggest_contour, -1, (0, 0, 255), 10)
    cv2.imshow("Biggest Contour", img_big_contours)
    cv2.waitKey(0)
else:
    print("No suitable contour found.")

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
cv2.imshow("Warped Image", warped)
cv2.waitKey(0)

gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
boxes = split_into_boxes(gray)

digits = detect_digits(boxes)
print(digits)
