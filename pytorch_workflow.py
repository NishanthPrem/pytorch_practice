#%% Importing the libraries

import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

#%% Setting the known parameters

weight = 0.3
bias = 0.9

X = torch.arange(0, 1, 0.01).unsqueeze(1)
y = weight * X + bias

print(f"Number of X samples: {len(X)}")
print(f"Number of y samples: {len(y)}")
print(f"First 10 X & y samples:\nX: {X[:10]}\ny: {y[:10]}")

#%% Splitting the data

train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

len(X_train), len(y_train), len(X_test), len(y_test)

#%% Visualizing the data

def plot_predictions(train_data=X_train, 
                     train_labels=y_train, 
                     test_data=X_test, 
                     test_labels=y_test, 
                     predictions=None):
  """
  Plots training data, test data and compares predictions.
  """
  plt.figure(figsize=(10, 7))

  # Plot training data in blue
  plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
  
  # Plot test data in green
  plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

  if predictions is not None:
    # Plot the predictions in red (predictions were made on the test data)
    plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

  # Show the legend
  plt.legend(prop={"size": 14});

#%% Calling the function

plot_predictions(X_train, y_train, X_test, y_test)

#%% Creating the LinearRegression

class LinearRegressionCustom(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1), requires_grad=True)
        self.bias = nn.Parameter(torch.randn(1), requires_grad=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias

#%% Instantiating the model

model = LinearRegressionCustom().to(device)
print(model.state_dict())

#%% Setting the hyperparameters

lr = 0.01

loss_fn = nn.L1Loss()
optimizer = optim.SGD(model.parameters(), lr)

#%% Training the model
torch.manual_seed(42)
epochs = 300

X_train = X_train.to(device)
X_test = X_test.to(device)
y_train = y_train.to(device)
y_test = y_test.to(device)

for epoch in range(epochs):
    model.train()
    
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    model.eval()   
    with torch.inference_mode():

        yt_pred = model(X_test)
        test_loss = loss_fn(yt_pred, y_test )
        if epoch % 20 == 0:
    
         print(f"Epoch: {epoch} | Train loss: {loss:.3f} | Test loss: {test_loss:.3f}")
    
#%% Making the predictions

model.eval()
with torch.inference_mode():
    y_pred = model(X_test)
print(y_pred)

#%% Plotting the predictions

plot_predictions(predictions=y_pred.cpu())

#%% Saving the model

from pathlib import Path

MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "01_pytorch_model"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

print(f"Saving model to {MODEL_SAVE_PATH}")
torch.save(obj = model.state_dict(),f = MODEL_SAVE_PATH)
