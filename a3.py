import cv2
import numpy as np
import random
import math

img = cv2.imread("Rainier1.png")
img2 = cv2.imread("Rainier2.png")


def compute_harris(my_img):
    img_grey = np.float32(cv2.cvtColor(my_img, cv2.COLOR_BGR2GRAY))
    # Calculate the derivatives
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])

    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])

    cv2.GaussianBlur(img_grey, (3, 3), 0, 0, cv2.BORDER_DEFAULT)

    derivative_x = cv2.filter2D(img_grey, -1, sobel_x)
    derivative_y = cv2.filter2D(img_grey, -1, sobel_y)

    derivative_xy = cv2.multiply(derivative_x, derivative_y)
    derivative_xx = cv2.multiply(derivative_x, derivative_x)
    derivative_yy = cv2.multiply(derivative_y, derivative_y)

    # Apply Gaussian to the derivatives
    smoothed_yy = cv2.GaussianBlur(derivative_yy, (3, 3), cv2.BORDER_DEFAULT)
    smoothed_xx = cv2.GaussianBlur(derivative_xx, (3, 3), cv2.BORDER_DEFAULT)
    smoothed_xy = cv2.GaussianBlur(derivative_xy, (3, 3), cv2.BORDER_DEFAULT)

    # Calculate the trace and determinant
    determinant = cv2.subtract(cv2.multiply(smoothed_xx, smoothed_yy), cv2.multiply(smoothed_xy, smoothed_xy))
    trace = cv2.add(smoothed_xx, smoothed_yy)

    # Compute Corner Response
    corner_response = cv2.divide(determinant, cv2.add(trace, 0.00000001))

    # Threshold the Corner Response
    threshold = corner_response > 0.3 * corner_response.max()

    keypoints = np.where(threshold)
    keypoints = list(zip(keypoints[0], keypoints[1]))

    # Create a list of Keypoint Objects
    img_keypoints = []
    for kp in keypoints:
        (row, col) = kp
        img_keypoints.append(cv2.KeyPoint(col, row, 8))

    return img_keypoints


def project(x1, y1, H):
    x2 = (H[0][0] * x1 + H[0][1] * y1 + H[0][2]) / (H[2][0] * x1 + H[2][1] * y1 + H[2][2] + 0.0000001)
    y2 = (H[1][0] * x1 + H[1][1] * y1 + H[1][2]) / (H[2][0] * x1 + H[2][1] * y1 + H[2][2] + 0.0000001)

    res = (x2, y2)
    return res


# Counts the number of inliers
def compute_inlier_count(homography, matches, inlier_threshold):
    return len(get_inliers(homography, matches, inlier_threshold))


# Returns a list of the inliers
def get_inliers(homography, matches, inlier_threshold):
    inliers = []
    for match in matches:
        (x1, y1) = match[0]
        (srcx, srcy) = match[1]
        try:
            projected_point = project(x1, y1, homography)
            ssd = math.sqrt((srcy - projected_point[1]) ** 2 + (srcx - projected_point[0]) ** 2)

            if ssd <= inlier_threshold:
                inliers.append(match)
        except TypeError:
            print("bad")

    return inliers


def ransac(matches, num_iterations, inlier_threshold):
    max_inliers = 0
    best_homography = None

    # Convert list of DMatches to points
    obj_list = [kp1[m.queryIdx].pt for m in matches]
    scene_list = [kp2[m.trainIdx].pt for m in matches]
    merged_list = tuple(zip(obj_list, scene_list))

    # Main body of RANSAC
    for i in range(num_iterations):
        random_indexes = []

        # Generate 4 random indexes
        while len(random_indexes) != 4:
            temp = random.randint(0, len(obj_list) - 1)
            random_indexes.append(temp)

        # Take the 4 matches that are at the randomly generated indices
        obj_matches = []
        scene_matches = []
        for index in random_indexes:
            obj_matches.append(obj_list[index])
            scene_matches.append(scene_list[index])

        homography = cv2.findHomography(np.array(obj_matches), np.array(scene_matches), 0)[0]

        # Compute the number of inliers and modify max_inliers and best homography
        num_inliers = compute_inlier_count(homography, merged_list, inlier_threshold)

        if num_inliers > max_inliers:
            max_inliers = num_inliers
            best_homography = homography

    all_inliers = get_inliers(best_homography, merged_list, inlier_threshold)
    obj_inliers = [m[0] for m in all_inliers]
    scene_inliers = [m[1] for m in all_inliers]

    # Calculate the homography using only inliers
    refined_homography = cv2.findHomography(np.array(obj_inliers), np.array(scene_inliers), 0)[0]
    inverse_homography = cv2.findHomography(np.array(scene_inliers), np.array(obj_inliers), 0)[0]

    new_inliers = get_inliers(refined_homography, merged_list, inlier_threshold)

    # Display the results
    img1_points = []
    img2_points = []
    d_matches = []

    for count, inlier in enumerate(new_inliers):
        img1_points.append(cv2.KeyPoint(inlier[0][0], inlier[0][1], 16))
        img2_points.append(cv2.KeyPoint(inlier[1][0], inlier[1][1], 16))
        d_matches.append(cv2.DMatch(count, count, 0))

    # Display Ransac results and write to the a file
    result = cv2.drawMatches(img, img1_points, img2, img2_points, d_matches, None)
    cv2.imshow("Ransac Matches", result)
    cv2.imwrite("3.png", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return refined_homography, inverse_homography


def stitch(image1, image2, homography, homography_inv):
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]

    # Project the corners of img2 to img1
    pts = [[0, 0], [0, h2 - 1], [w2 - 1, h2 - 1], [w2 - 1, 0]]
    projected_pts = []
    for w, h in pts:
        projected_pts.append(project(w, h, homography_inv))

    # convert negatives and make everything int
    projected_list = []
    for pt_x, pt_y in projected_pts:
        pt_x = int(pt_x) if pt_x > 0 else int(w1 - pt_x)
        pt_y = int(pt_y) if pt_y > 0 else int(h1 - pt_y)
        projected_list.append((pt_x, pt_y))

    # Set the frame dimensions by taking the max between image 1 and the projected points
    stitched_height = max(h1, max(projected_list, key=lambda item: item[1])[1])
    stitched_width = max(w1, max(projected_list, key=lambda item: item[0])[0])
    stitched = np.zeros((stitched_height, stitched_width, 3), dtype=np.uint8)

    # Calculate offset by finding the lowest negative value
    offset_y = min(projected_pts, key=lambda item: item[1])[1]
    offset_y = int(offset_y * -1) if offset_y < 0 else 0

    offset_x = min(projected_pts, key=lambda item: item[0])[0]
    offset_x = int(offset_x * -1) if offset_x < 0 else 0

    # Put image 1 on the new frame
    stitched[offset_y: offset_y + h1, offset_x: offset_x + w1] = image1
    cv2.imwrite("Stitched.png", stitched)
    projected_image2 = np.zeros((stitched_height, stitched_width, 3), dtype=np.uint8)
    # Stitching
    # Loop through the new frame and project every pixel to image2. Shift the image by the offset where necessary
    for row in range(stitched_height):
        for col in range(stitched_width):
            ppt_list = project(col - offset_x, row - offset_y, homography)

            # If the point lies within the boundaries of image2, calculate the value and blend it
            if 0 < ppt_list[1] < h2 and 0 < ppt_list[0] < w2:
                point_value = cv2.getRectSubPix(image2, (1, 1), ppt_list)
                projected_image2[row, col] = point_value
                alpha = 1.0 if np.all(stitched[row, col] == 0) else 0.7
                alpha = 0 if np.all(point_value == 0) else alpha

                stitched[row, col] = stitched[row, col] * (1.0 - alpha) + point_value * alpha

    cv2.imwrite("Projected.png", projected_image2)
    return stitched


# Compute interest points using Harris Detector
kp1 = compute_harris(img)
kp2 = compute_harris(img2)

# Create SIFT
sift = cv2.xfeatures2d.SIFT_create()

# Compute SIFT Descriptors
kp1, desc1 = sift.compute(img, kp1)
kp2, desc2 = sift.compute(img2, kp2)

# Match descriptors.
bf = cv2.BFMatcher()
match_list = bf.match(desc1, desc2)
match_list = sorted(match_list, key=lambda x: x.distance)
# Draw Matches
img_matches = cv2.drawMatches(img, kp1, img2, kp2, match_list, None, flags=2)
cv2.imshow("Original Matches", img_matches)
cv2.waitKey(0)

# Perform Ransac on the matches
(hom, hom_inv) = ransac(match_list, 1000, 1)

print("Finished RANSAC")
stiched_image = stitch(img, img2, hom, hom_inv)

cv2.imshow("Stitched Image", stiched_image)
cv2.waitKey(0)
