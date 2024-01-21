import cv2 as cv
import numpy as np
import argparse

# Function to overlay the shirt image on the person image at specified points
def overlay_shirt(person_image, shirt_image, points):
    for point in points:
        if point is not None:
            x, y = point
            h, w, _ = shirt_image.shape

            # Ensure the indices are integers
            x, y = int(x), int(y)

            # Draw a point on the person image at the specified coordinates
            cv.circle(person_image, (x, y), 5, (0, 255, 0), -1)

            # Extract the region of interest (ROI) from the person image
            roi = person_image[y:y+h, x:x+w]

            # Get the alpha channel from the shirt image
            alpha_channel = shirt_image[:, :, 3] / 255.0

            # Resize the shirt image to match the ROI size
            resized_shirt = cv.resize(shirt_image[:, :, :3], (w, h))

            # Blend the shirt image with the person image
            for c in range(0, 3):
                person_image[y:y+h, x:x+w, c] = (1 - alpha_channel) * person_image[y:y+h, x:x+w, c] + alpha_channel * resized_shirt[:, :, c]

    return person_image

if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description='Overlay shirt on a person image at specified points')
    parser.add_argument('--person_image', required=True, help='Path to the person image')
    parser.add_argument('--shirt_image', required=True, help='Path to the shirt image')
    parser.add_argument('--points', required=True, help='Comma-separated list of points, e.g., "3,4,2,0,5,6,7"')

    args = parser.parse_args()

    # Load person image and shirt image
    person_image = cv.imread(args.person_image)
    shirt_image = cv.imread(args.shirt_image, cv.IMREAD_UNCHANGED)

    # Convert points argument to a list of tuples
    points = [tuple(map(int, point.split(','))) for point in args.points.split(',')]

    # Overlay shirt on the person image and draw points
    result_image = overlay_shirt(person_image.copy(), shirt_image, points)

    # Display the person image with points
    cv.imshow('Person Image with Points', result_image)
    cv.waitKey(0)
    cv.destroyAllWindows()
