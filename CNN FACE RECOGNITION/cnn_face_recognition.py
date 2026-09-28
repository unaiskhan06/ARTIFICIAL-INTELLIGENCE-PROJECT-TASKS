"""
Task 6 - CNN-Based Face Recognition System
==========================================
A Convolutional Neural Network (CNN) that recognizes faces.

Dataset: Olivetti Faces (built into scikit-learn)
  - 400 grayscale face images, 64x64 pixels each
  - 40 different people, 10 images per person
  - Perfect for learning CNN on a CPU-only machine

Architecture (the 6 key layers you need to understand):
  1. INPUT LAYER       - receives the 64x64 face image as a tensor
  2. CONVOLUTION LAYER - extracts features (edges, textures, facial parts)
  3. MAX POOLING LAYER - downsamples, keeps important features, reduces size
  4. FLATTEN LAYER     - converts 2D feature maps into a 1D vector
  5. DENSE LAYER       - fully-connected reasoning layer (learns combinations)
  6. OUTPUT LAYER      - final classification (which person's face is it?)

Run:
  python cnn_face_recognition.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib
matplotlib.use("Agg")  # save plots to file instead of showing
import matplotlib.pyplot as plt
import os


# =========================================================================== #
#  STEP 0: LOAD THE DATASET
# =========================================================================== #
# The Olivetti dataset has 400 face images of 40 people (10 photos each).
# Each image is 64x64 grayscale pixels (values 0-1).
# =========================================================================== #

def load_data():
    print("Loading Olivetti Faces dataset...")
    faces = fetch_olivetti_faces()
    X = faces.images      # shape: (400, 64, 64) - 400 images of 64x64
    y = faces.target      # shape: (400,)       - label 0-39 for each image

    print(f"  Dataset: {X.shape[0]} images, {len(np.unique(y))} people")
    print(f"  Image size: {X.shape[1]}x{X.shape[2]} pixels (grayscale)")

    # Split into training (80%) and testing (20%) sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Training: {len(X_train)} images | Testing: {len(X_test)} images\n")

    # Add a channel dimension: (N, 64, 64) -> (N, 1, 64, 64)
    # The "1" means 1 color channel (grayscale). RGB images would have 3.
    X_train = X_train.reshape(-1, 1, 64, 64).astype(np.float32)
    X_test = X_test.reshape(-1, 1, 64, 64).astype(np.float32)
    y_train = y_train.astype(np.int64)
    y_test = y_test.astype(np.int64)

    return X_train, X_test, y_train, y_test


def save_sample_images(X, y, path):
    """Save a grid of sample face images for visualization."""
    fig, axes = plt.subplots(4, 10, figsize=(12, 5))
    for i, ax in enumerate(axes.flat):
        ax.imshow(X[i].reshape(64, 64), cmap="gray")
        ax.set_title(f"ID:{y[i]}", fontsize=7)
        ax.axis("off")
    plt.suptitle("Sample Faces from Olivetti Dataset", fontsize=14)
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()
    print(f"  Sample images saved to: {path}\n")


# =========================================================================== #
#  THE CNN MODEL - This is the heart of the task.
#  Each layer is explained in detail below.
# =========================================================================== #

class FaceRecognitionCNN(nn.Module):
    """
    A CNN for face recognition.

    The image flows through the network like this:

    INPUT IMAGE (1x64x64)
        |
        v
    [CONV LAYER 1]  -->  learns simple features (edges, lines)
        |
        v
    [MAX POOL 1]   -->  shrinks 64x64 to 32x32
        |
        v
    [CONV LAYER 2]  -->  learns complex features (eyes, nose, mouth shapes)
        |
        v
    [MAX POOL 2]   -->  shrinks 32x32 to 16x16
        |
        v
    [FLATTEN]      -->  converts 2D maps to 1D vector (32 x 16 x 16 = 8192)
        |
        v
    [DENSE LAYER]  -->  fully-connected reasoning (128 neurons)
        |
        v
    [OUTPUT LAYER] -->  40 neurons (one per person), picks the best match
    """

    def __init__(self, num_classes=40):
        super().__init__()

        # ----------------------------------------------------------------- #
        # LAYER 1: CONVOLUTION LAYER (Conv2d)
        # ----------------------------------------------------------------- #
        # WHAT IT DOES: Slides small "filters" (kernels) across the image.
        #              Each filter detects a specific pattern (edge, curve,
        #              texture). Where the pattern matches, the output is high.
        #
        # PARAMETERS:
        #   in_channels=1    -> 1 input channel (grayscale image)
        #   out_channels=32  -> 32 different filters = 32 feature maps
        #   kernel_size=3    -> each filter is 3x3 pixels
        #   padding=1        -> pad edges with zeros so output size stays same
        #
        # After this layer: image becomes 32 feature maps of 64x64
        # ----------------------------------------------------------------- #
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32,
                               kernel_size=3, padding=1)

        # ReLU activation: sets all negative values to 0.
        # This adds non-linearity (without it, the CNN can only learn
        # linear relationships, which aren't powerful enough).
        self.relu1 = nn.ReLU()

        # ----------------------------------------------------------------- #
        # LAYER 2: MAX POOLING LAYER (MaxPool2d)
        # ----------------------------------------------------------------- #
        # WHAT IT DOES: Divides the image into 2x2 blocks and keeps only
        #              the MAXIMUM value in each block. This:
        #              - Reduces the image size by half (64x64 -> 32x32)
        #              - Keeps the strongest features
        #              - Makes the network faster and more robust
        #              - Provides translation invariance (small shifts don't matter)
        #
        # PARAMETERS:
        #   kernel_size=2  -> 2x2 pooling window
        #   stride=2       -> window moves 2 pixels at a time (no overlap)
        # ----------------------------------------------------------------- #
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # ----------------------------------------------------------------- #
        # LAYER 3: SECOND CONVOLUTION LAYER
        # ----------------------------------------------------------------- #
        # Same idea as Layer 1, but now operates on the 32 feature maps
        # from Layer 1. It learns HIGHER-LEVEL features by combining
        # the simple features from Layer 1 (e.g., edges -> eye shapes).
        #
        #   in_channels=32   -> receives 32 feature maps from conv1
        #   out_channels=32  -> outputs 32 new feature maps
        # ----------------------------------------------------------------- #
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32,
                               kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()

        # Second max pooling: shrinks 32x32 to 16x16
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # ----------------------------------------------------------------- #
        # LAYER 4: FLATTEN LAYER
        # ----------------------------------------------------------------- #
        # WHAT IT DOES: Converts the 2D feature maps into a 1D vector.
        #
        # Before flatten: 32 channels x 16 height x 16 width = 8192 values
        # After flatten:  a single row of 8192 numbers
        #
        # WHY: Dense (fully-connected) layers only accept 1D input.
        #      The flatten layer bridges 2D image features to 1D reasoning.
        # ----------------------------------------------------------------- #
        self.flatten = nn.Flatten()

        # ----------------------------------------------------------------- #
        # LAYER 5: DENSE LAYER (Fully Connected)
        # ----------------------------------------------------------------- #
        # WHAT IT DOES: Every neuron is connected to every input value.
        #              This layer learns COMBINATIONS of features -
        #              e.g., "if edge detector A fired AND eye-shape detector
        #              B fired, this might be person 5."
        #
        # PARAMETERS:
        #   in_features=32*16*16  -> 8192 input values from flatten
        #   out_features=128      -> 128 neurons (a bottleneck that forces
        #                            the network to learn compact representations)
        # ----------------------------------------------------------------- #
        self.dense1 = nn.Linear(in_features=32 * 16 * 16, out_features=128)
        self.relu3 = nn.ReLU()

        # Dropout: randomly turns off 30% of neurons during training.
        # This prevents overfitting (memorizing the training data).
        self.dropout = nn.Dropout(0.3)

        # ----------------------------------------------------------------- #
        # LAYER 6: OUTPUT LAYER
        # ----------------------------------------------------------------- #
        # WHAT IT DOES: The final layer that produces the prediction.
        #              It has one neuron per class (40 people = 40 neurons).
        #              Each neuron outputs a "score" for that person.
        #              The highest score = the predicted identity.
        #
        # During training, we use CrossEntropyLoss which internally applies
        # softmax (converts scores to probabilities that sum to 1.0).
        # ----------------------------------------------------------------- #
        self.output = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x):
        """
        Forward pass: defines how data flows through the network.
        x starts as the input image and gets transformed layer by layer.
        """
        # --- Conv Block 1: extract simple features, then shrink ---
        x = self.conv1(x)       # Convolution: 1x64x64 -> 32x64x64
        x = self.relu1(x)       # Activation: remove negatives
        x = self.pool1(x)       # Max Pool:   32x64x64 -> 32x32x32

        # --- Conv Block 2: extract complex features, then shrink ---
        x = self.conv2(x)       # Convolution: 32x32x32 -> 32x32x32
        x = self.relu2(x)       # Activation: remove negatives
        x = self.pool2(x)       # Max Pool:   32x32x32 -> 32x16x16

        # --- Flatten: 2D feature maps -> 1D vector ---
        x = self.flatten(x)     # 32x16x16 -> 8192

        # --- Dense reasoning ---
        x = self.dense1(x)      # 8192 -> 128
        x = self.relu3(x)       # Activation
        x = self.dropout(x)     # Regularization (training only)

        # --- Output: 40 class scores ---
        x = self.output(x)      # 128 -> 40 (one score per person)
        return x


# =========================================================================== #
#  STEP 2: TRAINING THE CNN
# =========================================================================== #
# Training = repeatedly showing the network images, checking its prediction,
# and adjusting the weights to reduce errors. This is done with:
#   - Loss function: measures how wrong the prediction is
#   - Optimizer: updates the weights to reduce the loss
#   - Epochs: how many times we go through the entire dataset
# =========================================================================== #

def train_model(model, X_train, y_train, X_test, y_test, epochs=15):
    # Create PyTorch DataLoaders for batch training.
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    # CrossEntropyLoss = standard loss for classification.
    # It measures the difference between predicted probabilities and true labels.
    criterion = nn.CrossEntropyLoss()

    # Adam optimizer: adjusts weights using gradients + momentum.
    # lr=0.001 = learning rate (step size for weight updates).
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Training the CNN...")
    print(f"  Epochs: {epochs} | Batch size: 32 | Optimizer: Adam | LR: 0.001\n")

    train_losses = []
    test_accs = []

    for epoch in range(epochs):
        # --- Training phase ---
        model.train()
        running_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()        # reset gradients
            outputs = model(batch_X)     # forward pass: predict
            loss = criterion(outputs, batch_y)  # calculate error
            loss.backward()              # backward pass: compute gradients
            optimizer.step()             # update weights
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)

        # --- Evaluation phase ---
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():  # no gradient computation needed for eval
            for batch_X, batch_y in test_loader:
                outputs = model(batch_X)
                _, predicted = torch.max(outputs, 1)  # pick highest score
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

        accuracy = 100 * correct / total
        test_accs.append(accuracy)

        print(f"  Epoch {epoch+1:2d}/{epochs} | "
              f"Loss: {avg_loss:.4f} | Test Accuracy: {accuracy:.1f}%")

    print(f"\nTraining complete! Final accuracy: {test_accs[-1]:.1f}%\n")
    return train_losses, test_accs


# =========================================================================== #
#  STEP 3: VISUALIZE TRAINING PROGRESS
# =========================================================================== #

def plot_training(train_losses, test_accs, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(train_losses, "b-o", markersize=4)
    ax1.set_title("Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)

    ax2.plot(test_accs, "g-o", markersize=4)
    ax2.set_title("Test Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()
    print(f"  Training plot saved to: {path}\n")


# =========================================================================== #
#  STEP 4: TEST ON A SINGLE IMAGE
# =========================================================================== #

def predict_single(model, image, true_label, path):
    """Run the CNN on a single face image and show the result."""
    model.eval()
    with torch.no_grad():
        # image shape: (1, 64, 64) -> add batch dim -> (1, 1, 64, 64)
        img_tensor = torch.from_numpy(image).unsqueeze(0)
        output = model(img_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.imshow(image.reshape(64, 64), cmap="gray")
    color = "green" if predicted.item() == true_label else "red"
    ax.set_title(f"Predicted: Person {predicted.item()}\n"
                 f"Actual: Person {true_label}\n"
                 f"Confidence: {confidence.item():.1%}",
                 fontsize=10, color=color)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()
    print(f"  Prediction: Person {predicted.item()} "
          f"(actual: Person {true_label}, confidence: {confidence.item():.1%})")
    print(f"  Result saved to: {path}\n")


# =========================================================================== #
#  STEP 5: VISUALIZE CONVOLUTIONAL FILTERS
# =========================================================================== #
# This shows what the CNN actually "sees" in the first convolution layer.
# Each filter learns to detect a specific pattern (edges, textures, etc.)
# =========================================================================== #

def visualize_filters(model, path):
    """Show the learned filters from the first convolution layer."""
    # Get the weights from conv1: shape (32, 1, 3, 3) = 32 filters of 3x3
    filters = model.conv1.weight.data.clone()

    fig, axes = plt.subplots(4, 8, figsize=(10, 5))
    for i, ax in enumerate(axes.flat):
        if i < filters.shape[0]:
            # Each filter is 3x3, normalize to 0-1 for display
            f = filters[i, 0].numpy()
            f = (f - f.min()) / (f.max() - f.min() + 1e-8)
            ax.imshow(f, cmap="gray")
        ax.axis("off")
    plt.suptitle("Learned Convolution Filters (Layer 1)\n"
                 "Each square is a 3x3 pattern the CNN detects", fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()
    print(f"  Filter visualization saved to: {path}\n")


# =========================================================================== #
#  STEP 6: VISUALIZE FEATURE MAPS
# =========================================================================== #
# Show what each convolution layer produces when processing a face.
# This shows HOW the image is transformed as it passes through the CNN.
# =========================================================================== #

def visualize_feature_maps(model, image, path):
    """Show the feature maps produced by each conv layer for one image."""
    model.eval()
    img_tensor = torch.from_numpy(image).unsqueeze(0)

    with torch.no_grad():
        # Pass through first conv block
        x1 = model.conv1(img_tensor)
        x1 = model.relu1(x1)
        x1_pool = model.pool1(x1)

        # Pass through second conv block
        x2 = model.conv2(x1_pool)
        x2 = model.relu2(x2)
        x2_pool = model.pool2(x2)

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))

    # Original image
    axes[0].imshow(image.reshape(64, 64), cmap="gray")
    axes[0].set_title("Input\n(64x64)", fontsize=9)
    axes[0].axis("off")

    # After Conv1 + Pool: show first 8 feature maps combined as a grid
    for ax, fm, title in [
        (axes[1], x1[0], "After Conv1\n(32 maps)"),
        (axes[2], x1_pool[0], "After Pool1\n(32x32x32)"),
        (axes[3], x2_pool[0], "After Conv2+Pool2\n(32x16x16)"),
    ]:
        # Show first 8 feature maps in a small grid
        grid = fm[:8].permute(1, 2, 0).numpy() if fm.shape[0] >= 3 else \
               fm[0].numpy()
        if len(grid.shape) == 3:
            grid = grid[:, :, 0]
        ax.imshow(grid, cmap="viridis")
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    plt.suptitle("How the CNN Processes a Face Image", fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()
    print(f"  Feature map visualization saved to: {path}\n")


# =========================================================================== #
#  MAIN
# =========================================================================== #

def main():
    output_dir = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(output_dir, "output_plots")
    os.makedirs(plots_dir, exist_ok=True)

    print("=" * 65)
    print("  CNN-Based Face Recognition System")
    print("=" * 65)

    # Step 0: Load data
    X_train, X_test, y_train, y_test = load_data()
    save_sample_images(X_train, y_train,
                       os.path.join(plots_dir, "01_sample_faces.png"))

    # Step 1: Create the CNN model
    model = FaceRecognitionCNN(num_classes=40)
    print("CNN Model Architecture:")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    # Step 2: Train
    train_losses, test_accs = train_model(
        model, X_train, y_train, X_test, y_test, epochs=15
    )

    # Step 3: Plot training progress
    plot_training(train_losses, test_accs,
                  os.path.join(plots_dir, "02_training_progress.png"))

    # Step 4: Test on a single image
    print("Testing on a single image...")
    predict_single(model, X_test[0], y_test[0],
                   os.path.join(plots_dir, "03_single_prediction.png"))

    # Step 5: Visualize learned filters
    print("Visualizing learned convolution filters...")
    visualize_filters(model,
                      os.path.join(plots_dir, "04_conv_filters.png"))

    # Step 6: Visualize feature maps
    print("Visualizing feature maps...")
    visualize_feature_maps(model, X_test[0],
                           os.path.join(plots_dir, "05_feature_maps.png"))

    print("=" * 65)
    print("  All outputs saved to: " + plots_dir)
    print("  Files generated:")
    print("    01_sample_faces.png      - sample faces from dataset")
    print("    02_training_progress.png  - loss & accuracy curves")
    print("    03_single_prediction.png  - one face with prediction")
    print("    04_conv_filters.png       - learned 3x3 filters")
    print("    05_feature_maps.png       - how image transforms through CNN")
    print("=" * 65)


if __name__ == "__main__":
    main()
