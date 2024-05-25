"""
### validate.py

### This module is used to validate our custom model on the goat images.

"""
from ultralytics import YOLO

# Load a model
model = YOLO('./runs/detect/train5/weights/best.pt')  # load our custom model

# Validate the model
metrics = model.val()  # no arguments needed, dataset and settings remembered
metrics.box.map    # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps   # a list contains map50-95 of each category
