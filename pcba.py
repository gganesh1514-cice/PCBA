import cv2
import numpy as np

def align_images(image1, image2):
    """
    Aligns two images using ORB feature detection and homography.

    Args:
        image1 (np.ndarray): The reference image (golden sample).
        image2 (np.ndarray): The image to be aligned (test sample).

    Returns:
        np.ndarray: The aligned test image.
        tuple: A tuple containing the homography matrix and good matches,
               or None if alignment fails.
    """
    # Convert images to grayscale for feature detection
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Initialize the ORB detector (Oriented FAST and Rotated BRIEF)
    orb = cv2.ORB_create()

    # Find keypoints and descriptors in both images
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    # Use a Brute-Force Matcher with Hamming distance
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match the descriptors
    matches = bf.match(des1, des2)

    # Sort matches by their distance (best matches first)
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract the keypoints from the best matches
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Find the homography matrix
    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

    # Warp the test image to align with the golden image
    aligned_image = cv2.warpPerspective(image2, M, (image1.shape[1], image1.shape[0]))

    return aligned_image, (M, matches)

def find_missing_components(golden_image, test_image):
    """
    Compares two aligned images to find differences.

    Args:
        golden_image (np.ndarray): The aligned reference image.
        test_image (np.ndarray): The aligned test image.

    Returns:
        np.ndarray: An image with the missing components highlighted.
    """
    # Convert images to grayscale
    gray_golden = cv2.cvtColor(golden_image, cv2.COLOR_BGR2GRAY)
    gray_test = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

    # Calculate the absolute difference between the two images
    # A difference value of 0 means the pixels are identical
    diff = cv2.absdiff(gray_golden, gray_test)

    # Threshold the difference image to create a binary mask
    # A low threshold highlights small differences, while a high one
    # focuses on significant differences (like missing components)
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    # Dilate the thresholded image to connect close components
    # This helps in finding larger, more complete contours
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=2)

    # Find contours in the dilated image
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a copy of the golden image to draw on
    output_image = golden_image.copy()

    # Draw bounding boxes around the detected differences
    for contour in contours:
        # Filter out small contours that might be noise
        if cv2.contourArea(contour) > 100:
            (x, y, w, h) = cv2.boundingRect(contour)
            # Draw a green rectangle with a thickness of 2
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return output_image

def main():
    """
    Main function to run the PCBA comparison program.
    """
    # Load the golden sample image and the test image
    # Note: Using 'r' before the string to handle backslashes in Windows file paths.
    golden_pcba = cv2.imread(r'C:\Users\ADMIN\Desktop\pcba\golden.jpg')
    test_pcba = cv2.imread(r'C:\Users\ADMIN\Desktop\pcba\test.jpg')

    if golden_pcba is None or test_pcba is None:
        print("Error: Could not load one or both images. "
              "Please ensure the file paths are correct.")
        return

    # 1. Align the test image to the golden image
    print("Aligning images...")
    aligned_test_pcba, _ = align_images(golden_pcba, test_pcba)
    

    if aligned_test_pcba is None:
        print("Error: Image alignment failed. "
              "Please check if the images have enough distinct features for matching.")
        return

    print("Alignment complete.")

    # 2. Find and highlight missing components
    print("Finding missing components...")
    result_image = find_missing_components(golden_pcba, aligned_test_pcba)

    # 3. Display the results
    cv2.imshow("Golden PCBA (Reference)", golden_pcba)
    cv2.imshow("Test PCBA (Aligned)", aligned_test_pcba)
    cv2.imshow("Result - Missing Components Highlighted", result_image)

    # Wait for a key press and close all windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
