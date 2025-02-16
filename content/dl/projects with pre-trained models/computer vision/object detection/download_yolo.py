import urllib.request

CONFIG_PATH = "yolov3.cfg"
WEIGHTS_PATH = "yolov3.weights"
NAMES_PATH = "coco.names"

print("Downloading YOLO model files...")
url_base = "https://huggingface.co/homohapiens/darknet-yolov4/resolve/main/"

# Download YOLOv3 configuration, weights, and COCO names
urllib.request.urlretrieve(url_base + "yolov3.cfg", CONFIG_PATH)
urllib.request.urlretrieve(url_base + "yolov3.weights", WEIGHTS_PATH)
urllib.request.urlretrieve("https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names", NAMES_PATH)

print("YOLO model files downloaded.")