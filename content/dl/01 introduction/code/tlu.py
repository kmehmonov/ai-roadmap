import numpy as np


class ThresholdLogicUnit:
    def __init__(self, input_size):
        # Initialize weights and bias randomly
        self.weights = np.random.rand(input_size)
        self.bias = np.random.rand(1)

    def forward(self, x):
        # Calculate the weighted sum
        linear_output = np.dot(x, self.weights) + self.bias
        # Apply the step activation function
        return np.where(linear_output >= 0, 1, 0)

    def parameters(self):
        # Return trainable parameters
        return [self.weights, self.bias]
    
    def update_parameters(self, gradients, learning_rate):
        # Apply gradient updates to weights and bias
        self.weights -= learning_rate * gradients['weights']
        self.bias -= learning_rate * gradients['bias']

    
def train(model, x, y, learning_rate=0.01, epochs=10):
    for epoch in range(epochs):
        epoch_loss = 0
        for i in range(len(x)):
            # Forward pass
            output = model.forward(x[i])
            # Compute error
            error = y[i] - output
            # Compute gradients (manual backward pass)
            gradients = {
                'weights': -error * x[i],
                'bias': -error
            }
            # Update parameters
            model.update_parameters(gradients, learning_rate)
            # Accumulate loss
            epoch_loss += error[0]**2
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")



if __name__ == '__main__':
    # Simulated dataset: Student scores in two subjects and pass/fail labels
    # Features: [Score in Subject 1, Score in Subject 2]
    # Labels: 1 = Pass, 0 = Fail
    inputs = np.array([
        [85, 90], # Pass
        [40, 50], # Fail
        [60, 65], # Pass
        [30, 40], # Fail
        [75, 80], # Pass
        [20, 30]  # Fail
    ])
    targets = np.array([1, 0, 1, 0, 1, 0])

    # Initialize the TLU
    tlu = ThresholdLogicUnit(input_size=2)

    # Train the model
    train(tlu, inputs, targets, learning_rate=0.01, epochs=7)

    # outputs = np.array([tlu.forward(x) for x in inputs])
    outputs = tlu.forward(inputs)
    
    # Print final results
    print("\nFinal Results:")
    print("True Labels (Pass=1, Fail=0):", targets)
    print("Predicted Labels:", outputs)

    # Calculate accuracy
    accuracy = np.mean(outputs == targets) * 100
    print(f"\nAccuracy: {accuracy:.2f}%")
    
