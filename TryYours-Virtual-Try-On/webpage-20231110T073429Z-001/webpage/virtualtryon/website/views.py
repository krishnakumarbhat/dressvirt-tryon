views.py
from flask import Blueprint, render_template, request, flash, redirect, url_for
from flask_login import login_required, current_user
import os
import cv2 as cv
import numpy as np
import argparse
from PIL import Image
from flask import current_app 


views = Blueprint('views', __name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@views.route('/', methods=['GET', 'POST'])
@login_required
def home():
    person_image_path = None
    shirt_image_path = None
    result_image_path = None
    output_filename = None 

    if request.method == 'POST':
        person_image = request.files['person_image']
        shirt_image = request.files['shirt_image']

        if person_image and allowed_file(person_image.filename) and shirt_image and allowed_file(shirt_image.filename):
            person_image_path = os.path.join(UPLOAD_FOLDER, person_image.filename)
            shirt_image_path = os.path.join(UPLOAD_FOLDER, shirt_image.filename)

            person_image.save(person_image_path)
            shirt_image.save(shirt_image_path)

            person_image = cv.imread(person_image_path)
            shirt_image = cv.imread(shirt_image_path, cv.IMREAD_UNCHANGED)  # Load shirt image with an alpha channel


            parser = argparse.ArgumentParser()
            parser.add_argument('--thr', default=0.2, type=float, help='Threshold value for pose parts heat map')
            parser.add_argument('--width', default=368, type=int, help='Resize input to a specific width.')
            parser.add_argument('--height', default=368, type=int, help='Resize input to a specific height.')

            args = parser.parse_args()

            BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                        "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                        "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
                        "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

            POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
                        ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
                        ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
                        ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
                        ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]

            inWidth = args.width
            inHeight = args.height
            model_path = os.path.join(current_app.root_path, "graph_opt.pb")
            # model_path = os.path.abspath("graph_opt.pb")
            try:
                with open(model_path, 'rb') as file:
                    file_content = file.read()
                print("File opened successfully.")
            except Exception as e:
                print(f"Error opening the file: {e}")
            net = cv.dnn.readNetFromTensorflow(model_path)
            
            # Perform pose estimation on the "person.png" image
            net.setInput(cv.dnn.blobFromImage(person_image, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
            out = net.forward()
            out = out[:, :19, :, :]  # MobileNet output [1, 57, -1, -1], we only need the first 19 elements

            assert(len(BODY_PARTS) == out.shape[1])

            points = []
            for i in range(len(BODY_PARTS)):
                # Slice heatmap of corresponding body part.
                heatMap = out[0, i, :, :]

                # Find the maximum confidence point.
                _, conf, _, point = cv.minMaxLoc(heatMap)
                x = (person_image.shape[1] * point[0]) / out.shape[3]
                y = (person_image.shape[0] * point[1]) / out.shape[2]
                # Add a point if its confidence is higher than the threshold.
                points.append((int(x), int(y)) if conf > args.thr else None)

            # Assuming you have found the points "RShoulder" (point 2) and "LShoulder" (point 5)
            shoulder2 = points[2]
            shoulder5 = points[5]

            # Calculate the distance between the shoulders
            distance = np.linalg.norm(np.array(shoulder2) - np.array(shoulder5))

            # Resize the "shirt.png" image to match the distance
            shirt_height, shirt_width, _ = shirt_image.shape
            shirt_scale = (distance / shirt_width) *1.5
            shirt_image_resized = cv.resize(shirt_image, (int(shirt_scale * shirt_width), int(shirt_scale * shirt_height)))

            # Overlay the shirt image on the person image, shifting left and above
            person_image_with_shirt = person_image.copy()

            # Calculate the position to place the shirt on the person, shifting left and above
            x_offset = shoulder2[0] - int(shirt_image_resized.shape[1] * 0.15)  # Shift left by 20% of the shirt's width
            y_offset = shoulder2[1] - int(shirt_image_resized.shape[0] * 0.1)  # Shift above by 10% of the shirt's height

            # Extract the alpha channel from the shirt image (transparency)
            alpha_channel = shirt_image_resized[:, :, 3] / 255.0

            # Blend the shirt onto the person using the alpha channel
            for c in range(0, 3):
                person_image_with_shirt[y_offset:y_offset + shirt_image_resized.shape[0], x_offset:x_offset + shirt_image_resized.shape[1], c] = \
                    person_image_with_shirt[y_offset:y_offset + shirt_image_resized.shape[0], x_offset:x_offset + shirt_image_resized.shape[1], c] * \
                    (1 - alpha_channel) + shirt_image_resized[:, :, c] * alpha_channel
                

                # Generate a dynamic filename using f-string
            output_filename = f'person_with_shirt_{person_image[0:-3]}_{shirt_image[0:-3]}.png'

                # Save the resulting image with the dynamic filename
            cv.imwrite(output_filename, person_image_with_shirt)

            # Save the resulting image with a different filename
            output_filename = 'output_image.JPEG'  # Specify a valid filename here
            pil_image = Image.fromarray(cv.cvtColor(person_image_with_shirt, cv.COLOR_BGR2RGB))
            pil_image.save(output_filename, 'PNG')


            # Display the resulting image
            cv.imshow('Person with Shirt', person_image_with_shirt)
            cv.waitKey(0)
            cv.destroyAllWindows()

            flash("Images successfully uploaded and saved and here is the output.", 'success')
            return redirect(url_for('views.home'))

    return render_template("home.html", user=current_user, person_image_path=person_image_path, shirt_image_path=shirt_image_path,output_image_path=output_filename)