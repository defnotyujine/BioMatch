import cv2

# Function to load images and handle errors
def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load {image_path}. Please check the file path and integrity.")
    else:
        print(f"{image_path} loaded successfully.")
    return image

# Function to calculate percentage similarity
def calculate_similarity(matches, keypoints1, keypoints2):
    # Number of good matches to keypoints ratio
    good_matches = len(matches)
    total_keypoints = (len(keypoints1) + len(keypoints2)) / 2  # Average number of keypoints in both images
    similarity_percentage = (good_matches / total_keypoints) * 100  # Calculate the percentage similarity
    return similarity_percentage

# Main function to analyze the fingerprints
def analyze_fingerprints():
    # Ask for the image file paths
    image1_path = input("Enter the name or path of the first fingerprint image: ")
    image2_path = input("Enter the name or path of the second fingerprint image: ")

    # Load images
    image1 = load_image(image1_path)
    image2 = load_image(image2_path)

    # If both images are loaded, proceed with processing
    if image1 is not None and image2 is not None:
        # Convert images to grayscale
        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        # Initialize ORB detector (you can use other detectors as well)
        orb = cv2.ORB_create()

        # Detect keypoints and descriptors
        keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
        keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

        # Use BFMatcher to find matches between the two images
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(descriptors1, descriptors2)

        # Sort matches based on distance (best matches first)
        matches = sorted(matches, key=lambda x: x.distance)

        # Print the number of keypoints and matches
        print(f"Keypoints in image 1: {len(keypoints1)}")
        print(f"Keypoints in image 2: {len(keypoints2)}")
        print(f"Number of matches found: {len(matches)}")

        # Calculate and print the similarity percentage
        similarity_percentage = calculate_similarity(matches, keypoints1, keypoints2)
        print(f"Percentage similarity: {similarity_percentage:.2f}%")

        # Optionally, draw matches on the images and display them
        result_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imshow("Matches", result_image)

        # Wait for the user to press a key to close the window and terminate
        cv2.waitKey(0)  # Wait indefinitely until a key is pressed
        cv2.destroyAllWindows()  # Close the window after keypress

# Run the analysis
analyze_fingerprints()
