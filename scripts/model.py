import copy
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchsummary import summary


# Train the model
def __train(
    model,
    criterion,
    optimizer,
    dataloaders,
    device,
    dataset_sizes,
    num_epochs=25,
):
    """
    Model training
    :param model:
    :param criterion:
    :param optimizer:
    :param dataloaders:
    :param scheduler:
    :param device:
    :param dataset_sizes:
    :param num_epochs:
    :return:
    """
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    cost = []
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))
            if phase == "train":
                cost.append(epoch_loss)
            # deep copy the model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best val Acc: {:4f}".format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, cost


# Display a batch of predictions
def __visualize_results(model, images, class_names, dataloaders, device):
    """
    Visualize a batch of predictions
    :param model:
    :param images:
    :param class_names:
    :param dataloaders:
    :param device:
    :return:
    """
    model = model.to(device)  # Send model to GPU if available
    with torch.no_grad():
        model.eval()
        # Get a batch of validation images
        images, labels = iter(dataloaders["val"]).next()
        images, labels = images.to(device), labels.to(device)
        # Get predictions
        _, preds = torch.max(model(images), 1)
        preds = np.squeeze(preds.cpu().numpy())
        images = images.cpu().numpy()

    # Plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(15, 10))
    for idx in np.arange(len(preds)):
        ax = fig.add_subplot(2, len(preds) // 2, idx + 1, xticks=[], yticks=[])
        image = images[idx]
        image = image.transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)
        ax.imshow(image)
        ax.set_title(
            "{} ({})".format(class_names[preds[idx]], class_names[labels[idx]]),
            color=("green" if preds[idx] == labels[idx] else "red"),
        )
    return


def __test_model(model, test_loader, device):
    """
    Test the model given the net, test loader and device
    :param model:
    :param test_loader:
    :param device:
    :return:
    """
    model = model.to(device)
    # Turn autograd off
    with torch.no_grad():

        # Set the model to evaluation mode
        model.eval()

        # Set up lists to store true and predicted values
        y_true = []
        test_preds = []

        # Calculate the predictions on the test set and add to list
        for data in test_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            # Feed inputs through model to get raw scores
            logits = model.forward(inputs)
            # Convert raw scores to probabilities (not necessary since we just care about discrete probs in this case)
            probs = F.softmax(logits, dim=1)
            # Get discrete predictions using argmax
            preds = np.argmax(probs.cpu().numpy(), axis=1)
            # Add predictions and actuals to lists
            test_preds.extend(preds)
            y_true.extend(labels)

        # Calculate the accuracy
        test_preds = np.array(test_preds)
        y_true = np.array(y_true)
        test_acc = np.sum(test_preds == y_true) / y_true.shape[0]

        # Recall for each class
        recall_vals = []
        for i in range(10):
            class_idx = np.argwhere(y_true == i)
            total = len(class_idx)
            correct = np.sum(test_preds[class_idx] == i)
            recall = correct / total
            recall_vals.append(recall)

    return test_acc, recall_vals


def train_model(
    images, dataloaders, batch_size, class_names, dataset_sizes, num_epochs=10
):
    # We will used a pre-trained ResNet34 model, so our architecture has already been defined.
    # The cell below loads the ResNet34 model
    # displays a summary of the model layers and the output shape of the input after passing through each layer.
    # Instantiate pre-trained resnet

    net = torchvision.models.resnet34(pretrained=True)

    # Display a summary of the layers of the model and output shape after each layer
    summary(net, (images.shape[1:]), batch_size=batch_size, device="cpu")

    # Get the number of inputs to final Linear layer
    num_ftrs = net.fc.in_features

    # Replace final Linear layer with a new Linear with the same number of inputs but just 2 outputs,
    # since we have 3 classes
    net.fc = nn.Linear(in_features=num_ftrs, out_features=3)

    # We will use Cross Entropy as the cost/loss function and Adam for the optimizer.

    # Cross entropy loss combines softmax and nn.NLLLoss() in one single class.
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=0.0001)

    # Decay LR by a factor of 0.1 every 7 epochs (not used since Adam converges better)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        net.cuda()

    # Train the model
    net, cost_path = __train(
        net,
        criterion,
        optimizer,
        dataloaders,
        device,
        dataset_sizes,
        num_epochs,
    )

    __visualize_results(net, images, class_names, dataloaders, device)

    # Test the pre-trained model
    acc, recall_vals = __test_model(net, dataloaders["val"], device)
    print("Test set accuracy is {:.3f}".format(acc))
    for i in range(3):
        print("For class {}, recall is {}".format(class_names[i], recall_vals[i]))
    plt.plot(cost_path)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()
    print("All done!")
    return net
