import cv2
import numpy as np
import os

# Paths to the YOLO model files
CONFIG_PATH = "yolov3.cfg"
WEIGHTS_PATH = "yolov3.weights"
NAMES_PATH = "coco.names"

# Load YOLO model
net = cv2.dnn.readNet(WEIGHTS_PATH, CONFIG_PATH)

# Get the output layer names (correct method for latest OpenCV versions)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load COCO class labels
with open(NAMES_PATH, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Load an image for object detection
image_path = 'images/fruits1.jpg'  # Path to the input image
image = cv2.imread(image_path)

# Get image dimensions
height, width, channels = image.shape

# Prepare the image for YOLO
blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

# Lists to hold detected object info
class_ids = []
confidences = []
boxes = []

# Confidence threshold and NMS threshold (adjustable)
CONFIDENCE_THRESHOLD = 0.3  # Lowered to capture more objects
NMS_THRESHOLD = 0.4  # You can adjust this value based on your needs

# Process each detection
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > CONFIDENCE_THRESHOLD:  # Only consider detections with confidence > threshold
            # Get the bounding box coordinates
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Get top-left corner coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Apply Non-Maximum Suppression to remove redundant boxes
indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=CONFIDENCE_THRESHOLD, nms_threshold=NMS_THRESHOLD)

# Draw bounding boxes and labels on the image
if len(indices) > 0:
    for i in indices.flatten():
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = str(round(confidences[i], 2))
        
        # Draw bounding box and label on the image
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, label + " " + confidence, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display the output image
cv2.imshow("Object Detection Output", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Optionally, save the output image
cv2.imwrite("output_image.jpg", image)