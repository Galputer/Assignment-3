from math import sqrt
import os
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
    nn.Linear(5, 10),
    nn.LeakyReLU(),
    nn.Dropout(0.5),
    
    nn.Linear(10, 1)
)

# move model to device
model.to(device)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

print(f'{"Reading board data" :=<100}')
board_df = parse_board()
print(f'{"Generating features" :=<100}')
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
X = torch.tensor(X).float()
y = torch.tensor(y).float()

X_train = X_train.to(device)
X_test = X_test.to(device)
y_train = y_train.to(device)
y_test = y_test.to(device)
X = X.to(device)
y = y.to(device)

print(f'{"Running NN test-train split" :=<100}')
model.train()
for epoch in range(500):
    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = model(X_train)
    loss = criterion(outputs.squeeze(), y_train)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    # print statistics
    # running_loss = sqrt(loss.item())
    # print(f"Epoch {epoch+1}, RMSE: {running_loss:.4f}")
    
# evaluate the trained neural network on the test set
with torch.no_grad():
    model.eval()
    outputs = model(X_test)
    test_loss = criterion(outputs.squeeze(), y_test)

print(f"Test RMSE: {sqrt(test_loss.item()):.4f}")


print(f'{"Running NN full dataset" :=<100}')
model.train()
for epoch in range(500):
    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = model(X)
    loss = criterion(outputs.squeeze(), y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    # print statistics
    # running_loss = sqrt(loss.item())
    # print(f"Epoch {epoch+1}, RMSE: {running_loss:.4f}")
    
# evaluate the trained neural network on the test set
with torch.no_grad():
    model.eval()
    outputs = model(X)
    test_loss = criterion(outputs.squeeze(), y)

print(f"Model RMSE: {sqrt(test_loss.item()):.4f}")

# Save the model as a pt file
dir_path = os.path.dirname(os.path.realpath(__file__))
filename = 'models/nn-3.pt'
if os.path.isfile(os.path.join(dir_path,filename)):
    print(f'{"Model exists!" :=<100}')
else:
    print(f'{"Saving model to disk" :=<100}')
    torch.save(model.state_dict(), os.path.join(dir_path,filename))