import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda import device
import helper_utils

device = device("cuda" if torch.cuda.is_available() else "cpu")
print("Available devices:", torch.cuda.device_count())
print(f"Current device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
print(f"Using device: {device}")

model = nn.Sequential(
    nn.Linear(1, 1),
)

# Distances in miles for recent bike deliveries
distances = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float32)
# Corresponding delivery times in minutes
times = torch.tensor([[6.96], [12.11], [16.77], [22.21]], dtype=torch.float32)

loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
for epoch in range(100):
    outputs = model(distances)
    loss = loss_fn(outputs, times)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item()}")


with torch.no_grad():
    predicted_time = model(torch.tensor([7.0], dtype=torch.float32))[0]
    print(f"Predicted time: {predicted_time.item()}")

helper_utils.plot_results(model, distances, times)

