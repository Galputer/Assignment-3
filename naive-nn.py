from math import sqrt
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from feature_engineer import feature_engineer
from parse_board import parse_board

# check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = nn.Sequential(
    nn.Linear(6, 10),
    nn.LeakyReLU(),
    nn.Dropout(0.5),
    
    nn.Linear(10, 1)
)

# move model to device
model.to(device)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

board_df = parse_board()
engineered_df = feature_engineer(board_df)

# Select the relevant features and target variable
X = engineered_df.values
y = board_df['astar'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# move data to device
X_train = torch.tensor(X_train).float()
X_test = torch.tensor(X_test).float()
y_train = torch.tensor(y_train).float()
y_test = torch.tensor(y_test).float()

X_train = X_train.to(device)
X_test = X_test.to(device)
y_train = y_train.to(device)
y_test = y_test.to(device)

model.train()
for epoch in range(1000):
    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = model(torch.tensor(X_train).float())
    loss = criterion(outputs.squeeze(), torch.tensor(y_train).float())
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    # print statistics
    running_loss = sqrt(loss.item())
    print(f"Epoch {epoch+1}, RMSE: {running_loss:.4f}")
    
# evaluate the trained neural network on the test set
with torch.no_grad():
    model.eval()
    outputs = model(X_test)
    test_loss = criterion(outputs.squeeze(), y_test)

print(f"Test RMSE: {sqrt(test_loss.item()):.4f}")