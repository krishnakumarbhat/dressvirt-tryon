from pathlib import Path
import cv2
from yolov5 import YOLOv5

# Load the pre-trained YOLOv5 model
model = YOLOv5()

# Load the image
image = cv2.imread('image.jpg')

# Detect objects in the image
results = model(image)

# Filter for ambulance detections based on class ID (e.g., 0 for ambulance)
ambulances = results.pandas().query('name == "ambulance"')

# Draw bounding boxes around the detected ambulances
for i in range(len(ambulances)):
    x = int(ambulances.loc[i, 'xmin'])
    y = int(ambulances.loc[i, 'ymin'])
    w = int(ambulances.loc[i, 'xmax'] - x)
    h = int(ambulances.loc[i, 'ymax'] - y)
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Display the image with bounding boxes
cv2.imshow('Image with detected ambulances', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
