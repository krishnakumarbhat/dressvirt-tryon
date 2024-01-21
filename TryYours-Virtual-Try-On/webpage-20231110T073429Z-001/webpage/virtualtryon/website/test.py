import cv2
import numpy as np

# Load the person and shirt images
person_image = cv2.imread('person.png')
shirt_image = cv2.imread('2.png')




if person_image is None or shirt_image is None:
    print("Error: Unable to load one or both of the images.")
else:
    # Resize the shirt image to fit the person
    shirt_height, shirt_width, _ = shirt_image.shape
    # Ensure that the shirt image and person image have the same dimensions
    shirt_image = cv2.resize(shirt_image, (person_image.shape[1], person_image.shape[0]))

    # Create a mask for the shirt
    gray_shirt = cv2.cvtColor(shirt_image, cv2.COLOR_BGR2GRAY)
    ret, shirt_mask = cv2.threshold(gray_shirt, 200, 255, cv2.THRESH_BINARY)

    # Invert the mask
    shirt_mask_inv = cv2.bitwise_not(shirt_mask)

    # Extract the person without the shirt
    person_without_shirt = cv2.bitwise_and(person_image, person_image, mask=shirt_mask_inv)

    # Extract the shirt
    shirt = cv2.bitwise_and(shirt_image, shirt_image, mask=shirt_mask)

    # Add the shirt to the person
    result = cv2.add(person_without_shirt, shirt)

# Display the result
    cv2.imshow('Person with Shirt', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
