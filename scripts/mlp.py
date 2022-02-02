
import numpy as np

import torch



import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from make_dataset import transform_data

import matplotlib.pyplot as plt


class Multiclass_Net(nn.Module):
    def __init__(self, n_input, n_hidden1, n_hidden2, n_hidden3, n_output):
        super().__init__()
        self.hidden1 = nn.Linear(n_input, n_hidden1)
        self.hidden2 = nn.Linear(n_hidden1, n_hidden2)
        self.hidden3 = nn.Linear(n_hidden2, n_hidden3)
        self.out = nn.Linear(n_hidden3, n_output)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = self.out(x)
        return x

net = Multiclass_Net(n_input=224*224*3, n_hidden1=100,  n_hidden2=50, n_hidden3=20, n_output=3)


def train_model(model, criterion, optimizer, trainloader, num_iter, device):
    model = model.to(device)
    model.train()  # Set the model to training mode

    cost = []

    for epoch in range(num_iter):

        running_loss = 0.0

        for i, data in enumerate(trainloader):
            # Get the inputs X and labels y for the minibatch
            inputs, labels = data[0].to(device), data[1].to(device)

            # Zero the gradients of the weights each iteration
            optimizer.zero_grad()

            # Calculate the predictions and the cost/loss
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Use autograd to calculate the gradient of the cost with respect to each weight
            loss.backward()

            # Use the optimizer to do the weights update
            optimizer.step()

            # Add the loss to running loss for the epoch
            running_loss += loss.item()

        cost.append(running_loss)
    return cost


images, dataloaders, batch_size, class_names, dataset_sizes = transform_data()
criterion1 = nn.CrossEntropyLoss()
# Define the method of updating the weights each iteration
optimizer1 = optim.SGD(net.parameters(), lr=0.01)
# Number of iterations (epochs) to train
n_iter = 135
# Set device
device1 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trainloader1 = dataloaders.get('train')
valloader1 = dataloaders.get('val')
# Train model
cost_path = train_model(net,criterion1,optimizer1,trainloader1,n_iter,device1)

# Plot the cost over training
plt.plot(cost_path)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

def test_model(model,test_loader,device):
    # Turn autograd off
    with torch.no_grad():

        # Set the model to evaluation mode
        model = model.to(device)
        model.eval()

        # Set up lists to store true and predicted values
        y_true = []
        test_preds = []

        # Calculate the predictions on the test set and add to list
        for data in test_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            # Feed inputs through model to get raw scores
            logits = net.forward(inputs)
            # Convert raw scores to probabilities (not necessary since we just care about discrete probs in this case)
            probs = F.softmax(logits,dim=1)
            # Get discrete predictions using argmax
            preds = np.argmax(probs.cpu().numpy(),axis=1)
            # Add predictions and actuals to lists
            test_preds.extend(preds)
            y_true.extend(labels)

        # Calculate the accuracy
        test_preds = np.array(test_preds)
        y_true = np.array(y_true)
        test_acc = np.sum(test_preds == y_true)/y_true.shape[0]
        
        # Recall for each class
        recall_vals = []
        for i in range(3):
            class_idx = np.argwhere(y_true==i)
            total = len(class_idx)
            correct = np.sum(test_preds[class_idx]==i)
            recall = correct / total
            recall_vals.append(recall)
    
    return test_acc,recall_vals

classes = ['English', 'Russian', 'Telugu']
acc,recall_vals1 = test_model(net,valloader1,device1)
print('Test set accuracy is {:.3f}'.format(acc))
for j in range(3):
    print('For class {}, recall is {}'.format(classes[j],recall_vals1[j]))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
