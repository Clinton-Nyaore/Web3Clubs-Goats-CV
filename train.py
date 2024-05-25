"""
### train.py

### This module is used to train the detection model on the goat images.

"""

from ultralytics import YOLO

model = YOLO()

# Train the custom model
model.train(data='data.yaml', hyp='hyp.yaml', epochs=3)