import argparse
import cv2
import numpy as np
import os


# Function we'll use to display contour area

def get_contour_areas(contours):
    # returns the areas of all contours as list
    all_areas = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        all_areas.append(area)
    return all_areas


def set_filename_parts(inpath, seg):
    # Extract file name
    fname = os.path.basename(os.path.normpath(inpath))
    # Split name and extension
    fn, fe = os.path.splitext(fname)
    return fn + "_seg_{}".format(seg) + fe


def x_coord_contour(contours):
    # Returns the X coordinate for the contour centroid
    if cv2.contourArea(contours) > 10:
        M = cv2.moments(contours)
        return (int(M['m10']/M['m00']))


def label_contour_center(image, c):
    # Places the red circle on the center of contours
    M = cv2.moments(c)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    # Draw the contour number on the image
    cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)
    return image


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--infile",
                    help="Input image file to work on it.")
args = parser.parse_args()

print("Reading from file {}".format(args.infile))

# Original image
image = cv2.imread(args.infile)

# Create a copy of our original image
orginal_image = image.copy()

# Grayscale our image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Find Canny edges
edged = cv2.Canny(gray, 50, 200)

# Find contours and print how many were found
_, contours, hierarchy = cv2.findContours(
    edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
print("Number of contours found = ", len(contours))

# Computer Center of Mass or centroids and draw them on our image
# for (i, c) in enumerate(contours):
#     orig = label_contour_center(image, c)

# cv2.imshow("4 - Contour Centers ", image)
# cv2.waitKey(0)

# Sort contours large to small
sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
# sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]

# Iterate over our contours and draw one at a time
# for c in sorted_contours:
#     cv2.drawContours(orginal_image, [c], -1, (255, 0, 0), 3)
#     cv2.waitKey(0)
#     cv2.imshow('Contours by area', orginal_image)

# cv2.waitKey(0)

# Sort by left to right using our x_cord_contour function
contours_left_to_right = sorted(contours, key=x_coord_contour, reverse=False)

# Labeling Contours left to right
for (i, c) in enumerate(contours_left_to_right):
    cv2.drawContours(orginal_image, [c], -1, (0, 0, 255), 3)
    M = cv2.moments(c)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    cv2.putText(orginal_image, str(i+1), (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('6 - Left to Right Contour', orginal_image)
    cv2.waitKey(0)
    (x, y, w, h) = cv2.boundingRect(c)

    # Let's now crop each contour and save these images
    cropped_contour = orginal_image[y:y + h, x:x + w]

    image_name = set_filename_parts(args.infile, str(i+1))
    print(image_name)
    cv2.imwrite(image_name, cropped_contour)

cv2.destroyAllWindows()
