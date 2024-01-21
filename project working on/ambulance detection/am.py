import torch
from IPython.display import Image, clear_output

!python detect.py --weights yolov5s.pt --img 640 --conf 0.25 --source images.jpg

Image(filename='runs/detect/exp/ambulance.jpg', width=600)
