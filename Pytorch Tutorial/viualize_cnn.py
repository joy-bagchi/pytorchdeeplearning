import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import math

# ---------------------------
# 1. Define a simple CNN
# ---------------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # MNIST is 1-channel (grayscale), we'll create 8 filters in the first layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(16 * 7 * 7, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # [B, 8, 14, 14]
        x = self.pool(F.relu(self.conv2(x)))  # [B, 16, 7, 7]
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# ---------------------------
# 2. Utility: visualize conv filters
# ---------------------------
def visualize_filters(weight_tensor, title="Filters"):
    """
    weight_tensor: [out_channels, in_channels, kH, kW]
    For MNIST conv1: [8, 1, 3, 3]
    We'll visualize each [1, kH, kW] filter as a 2D image.
    """
    # detach and move to cpu
    w = weight_tensor.detach().cpu()

    out_channels, in_channels, kH, kW = w.shape
    assert in_channels == 1, "This visualizer assumes 1 input channel (grayscale)."

    num_filters = out_channels
    cols = min(num_filters, 8)
    rows = math.ceil(num_filters / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows))
    fig.suptitle(title)

    if rows == 1:
        axes = [axes]

    for i in range(rows * cols):
        ax = axes[i // cols][i % cols] if rows > 1 else axes[i % cols]
        ax.axis("off")

        if i < num_filters:
            # filter i, first (and only) channel
            filt = w[i, 0, :, :].numpy()
            # normalize for nicer visualization
            filt_min, filt_max = filt.min(), filt.max()
            if filt_max > filt_min:
                filt = (filt - filt_min) / (filt_max - filt_min)
            ax.imshow(filt, cmap="gray")
        else:
            ax.imshow([[0]], cmap="gray")

    plt.tight_layout()
    plt.show()

# ---------------------------
# 3. Load MNIST data
# ---------------------------
transform = transforms.Compose([
    transforms.ToTensor(),                         # [0,1]
    transforms.Normalize((0.1307,), (0.3081,)),    # standard MNIST normalization
])

train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# ---------------------------
# 4. Create model, loss, optimizer
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# ---------------------------
# 5. Visualize filters BEFORE training
# ---------------------------
print("Visualizing conv1 filters BEFORE training...")
visualize_filters(model.conv1.weight, title="Conv1 Filters - BEFORE Training")

# ---------------------------
# 6. Training loop (few epochs)
# ---------------------------
num_epochs = 3  # keep small for a quick demo

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (batch_idx + 1) % 200 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], "
                  f"Loss: {running_loss / 200:.4f}")
            running_loss = 0.0

print("Training complete.")

# ---------------------------
# 7. Visualize filters AFTER training
# ---------------------------
print("Visualizing conv1 filters AFTER training...")
visualize_filters(model.conv1.weight, title="Conv1 Filters - AFTER Training")
