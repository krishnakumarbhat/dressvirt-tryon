import torch
from torchvision import models, transforms
from PIL import Image, ImageDraw

# Load the pre-trained human pose estimation model
pose_model = models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
pose_model.eval()

# Load the image
image_path = "person.png"
image = Image.open(image_path)

# Define a transform to preprocess the image
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Preprocess the image
input_image = transform(image).unsqueeze(0)

# Perform human pose estimation
with torch.no_grad():
    prediction = pose_model(input_image)

# Extract keypoint coordinates from the prediction
keypoints = prediction[0]['keypoints'][0]

# Draw keypoints on the image
draw = ImageDraw.Draw(image)
for keypoint in keypoints:
    x, y, _ = keypoint.numpy().astype(int)
    draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill='red')

# Save or display the image with keypoints
image.save("out.jpg")
image.show()
