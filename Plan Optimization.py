# Step 6: Plan Optimization Setup

import torch.optim as optim
import torch.nn as nn

model = nn.Linear(10, 2)  # مثال بسيط على موديل
# Optimizer planned to use
# Adam Optimizer to improve training efficiency and stability
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Learning Rate options for experiments
# Will try different rates later to find the best one
learning_rates = [0.01, 0.001, 0.0005, 0.0001]

# Early Stopping Plan (Manual Monitoring)
# - Monitor validation loss manually during training
# - If validation loss does not improve after 5 epochs, stop training early

# Note: Early stopping will be implemented manually in Step 8
