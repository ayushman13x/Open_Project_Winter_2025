import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from model import QST_Model

# Loading the dataset
df = pd.read_csv('data/training/dataset.csv')

# Splitting: Using the first 8000 samples for training
train_df = df.iloc[:8000]

# Converting to Tensors
X_train = torch.tensor(train_df[['x', 'y', 'z']].values, dtype=torch.float32)
y_train = torch.tensor(train_df[['r00', 'r11', 'r01_real', 'r01_imag']].values, dtype=torch.float32)

# Initializing our model and choosing the 'Adam' optimizer to handle the learning
model = QST_Model()
optimizer = optim.Adam(model.parameters(), lr=0.0005)
criterion = nn.MSELoss() # Our 'grading' tool to measure errors

print("Starting the training process...")

# Running the training loop for 200 'epochs' 
for epoch in range(200):
    # Resetting the gradients so we start fresh each round
    optimizer.zero_grad()
    
    # Passing the measurements through the model to get a prediction
    predictions = model(X_train)
    
    # Calculating the loss (the difference between prediction and reality)
    loss = criterion(predictions, y_train)
    
    # Performing 'backpropagation' to tell the neurons how to change
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1}/200], Loss: {loss.item():.6f}")

# Saving to our outputs folder as required
torch.save(model.state_dict(), 'outputs/model_weights.pt')
print("Training complete! Model weights saved to outputs/model_weights.pt")