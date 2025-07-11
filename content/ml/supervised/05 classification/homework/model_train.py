import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Get current working directory
cwd = Path(__file__).parent.resolve()

# Load MNIST dataset
print("Loading MNIST dataset...")
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
y = y.astype(int)  # Convert labels to integers

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train SVM classifier
print("Training SVM classifier...")
svm = SVC(kernel='rbf', random_state=42)
svm.fit(X_train_scaled, y_train)

# Save the trained model and scaler
joblib.dump(svm, cwd / 'svm_digit_classifier.pkl')
joblib.dump(scaler, cwd / 'scaler.pkl')
print("Model and scaler saved as 'svm_digit_classifier.pkl' and 'scaler.pkl'")

# Predict on test set
y_pred = svm.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Generate and plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix for MNIST Digit Classifier')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Visualize some predictions
def plot_digits(X, y_true, y_pred, n_samples=5):
    fig, axes = plt.subplots(1, n_samples, figsize=(10, 2))
    for i, ax in enumerate(axes):
        ax.imshow(X[i].reshape(28, 28), cmap='gray')
        ax.set_title(f"True: {y_true[i]}\nPred: {y_pred[i]}")
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# Display 5 random test images with predictions
indices = np.random.choice(len(X_test), 5, replace=False)
plot_digits(X_test[indices], y_test[indices], y_pred[indices])