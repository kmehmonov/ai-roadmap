import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
import os

# Define the model file path
MODEL_PATH = "resnet50.pth"

# Check if the model file exists locally; if not, download and save it
if not os.path.exists(MODEL_PATH):
    print("Downloading pre-trained ResNet50 model...")
    model = resnet50(pretrained=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model downloaded and saved as '{MODEL_PATH}'")
else:
    print(f"Loading model from '{MODEL_PATH}'...")
    model = resnet50(pretrained=False)
    model.load_state_dict(torch.load(MODEL_PATH))
print("Model loaded successfully.")

# Set the model to evaluation mode
model.eval()

# Define the image preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),   # Resize to 224x224 (ResNet input size)
    transforms.ToTensor(),           # Convert image to PyTorch tensor
    transforms.Normalize(            # Normalize with ImageNet's mean and std
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load ImageNet class labels
# Download the file if it doesn't exist
IMAGENET_CLASSES_PATH = "imagenet_classes.txt"
if not os.path.exists(IMAGENET_CLASSES_PATH):
    print("Downloading ImageNet class labels...")
    import urllib.request
    url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    urllib.request.urlretrieve(url, IMAGENET_CLASSES_PATH)
    print(f"ImageNet class labels downloaded and saved as '{IMAGENET_CLASSES_PATH}'")

# Load class labels
with open(IMAGENET_CLASSES_PATH, "r") as f:
    imagenet_classes = eval(f.read())

# Initialize the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

print("Press 'q' to quit.")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Unable to read frame.")
        break

    # Convert the frame to PIL image
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Preprocess the image
    input_tensor = transform(pil_image).unsqueeze(0)  # Add batch dimension

    # Perform object classification
    with torch.no_grad():
        outputs = model(input_tensor)
        _, indices = outputs.topk(3)  # Get the top 3 predictions
        predictions = [(imagenet_classes[idx], outputs[0, idx].item()) for idx in indices[0]]

    # Display predictions on the frame
    for i, (label, score) in enumerate(predictions):
        text = f"{label}: {score:.2f}"
        cv2.putText(frame, text, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Real-Time Object Classification (PyTorch)', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
