#%% Importing the libraries

import matplotlib.pyplot as plt
import torch

#%% Setting up the device

RANDOM_SEED=42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%% Creating the dataset

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

n_samples = 1000
X, y = make_moons(n_samples, noise=0.1, random_state=RANDOM_SEED)
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

#%% Visualizing the data

plt.scatter(x=X[:, 0], 
            y=X[:, 1], 
            c=y, 
            cmap=plt.cm.RdYlBu);


#%% Building the Model
import torch.nn as nn

class MoonModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 1)
        
        self.activation = nn.ReLU()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        
        return self.fc3(x)

moon_model = MoonModel().to(device)

#%% Setting up the loss function and optimizer

import torch.optim as optim

loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(moon_model.parameters(), lr=0.3)

#%% Checking our data
print("Logits:")
print(moon_model(X_train.to(device)[:10]).squeeze())

print("Pred Probs:")
print(torch.sigmoid(moon_model(X_train.to(device)[:10].squeeze())))

print("Pred Labels:")
print(torch.round(torch.sigmoid(moon_model(X_train.to(device)[:10].squeeze()))))

#%% Evaluating metrics

from torchmetrics import Accuracy

acc_fn = Accuracy(task="multiclass", num_classes=2).to(device)

#%% Training the data

torch.manual_seed(RANDOM_SEED)

epochs = 1000

X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

for epoch in range(1, epochs+1):
    moon_model.train()
    y_logits = moon_model(X_train).squeeze()
    y_pred_probs = torch.sigmoid(y_logits)
    y_pred = torch.round(y_pred_probs)
    
    loss = loss_fn(y_logits, y_train)
    acc = acc_fn(y_pred, y_train.int())
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    moon_model.eval()
    with torch.inference_mode():
        test_logits = moon_model(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))
        
        test_loss = loss_fn(test_logits, test_pred)
        test_acc = acc_fn(test_pred, y_test.int())
        if epoch % 100 == 0:
         print(f"Epoch: {epoch} | Train loss: {loss:.3f} | Test loss: {test_loss:.3f}")

#%% Predicting the data

import numpy as np

# TK - this could go in the helper_functions.py and be explained there
def plot_decision_boundary(model, X, y):
  
    # Put everything to CPU (works better with NumPy + Matplotlib)
    model.to("cpu")
    X, y = X.to("cpu"), y.to("cpu")

    # Source - https://madewithml.com/courses/foundations/neural-networks/ 
    # (with modifications)
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), 
                         np.linspace(y_min, y_max, 101))

    # Make features
    X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

    # Make predictions
    model.eval()
    with torch.inference_mode():
        y_logits = model(X_to_pred_on)

    # Test for multi-class or binary and adjust logits to prediction labels
    if len(torch.unique(y)) > 2:
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1) # mutli-class
    else: 
        y_pred = torch.round(torch.sigmoid(y_logits)) # binary
    
    # Reshape preds and plot
    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

#%% Plot decision boundaries for training and test sets

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(moon_model, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(moon_model, X_test, y_test)

