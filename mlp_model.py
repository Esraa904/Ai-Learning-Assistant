import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# Load the dataset
data = pd.read_excel(r"D:\Esraa\Ai Learning Assistant/student_learning_data.csv")

# Features and labels
X = data[['Score', 'Time_Spent', 'Attempts', 'Correct_Ratio', 'Motion_Score']]
y = (data['Score'] >= 60).astype(int)  # 1 for Good performance, 0 for Needs Improvement

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

# Define the MLP model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(5, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

model = MLP()

# Define optimizer and loss
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

# Training loop
epochs = 30
batch_size = 8

for epoch in range(epochs):
    model.train()
    permutation = torch.randperm(X_train.size()[0])

    epoch_loss = 0.0
    correct = 0
    total = 0

    for i in range(0, X_train.size()[0], batch_size):
        indices = permutation[i:i+batch_size]
        batch_x, batch_y = X_train[indices], y_train[indices]

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        preds = (outputs > 0.5).float()
        correct += (preds == batch_y).sum().item()
        total += batch_y.size(0)

    avg_loss = epoch_loss / (X_train.size(0) / batch_size)
    accuracy = correct / total

    print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f}")

# Evaluation
model.eval()
with torch.no_grad():
    outputs = model(X_test)
    preds = (outputs > 0.5).float()
    test_accuracy = (preds == y_test).sum().item() / y_test.size(0)

print(f"\nTest Accuracy: {test_accuracy:.4f}")
