import os

import matplotlib.pyplot as plt
import numpy as np

# this package make graping labeled image folders from a dir
# and splitting them into train, val, and test
import splitfolders
import torch
import torch.utils.data
from torchvision import datasets, transforms


def transform_data():
    TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
    CUDA_VERSION = torch.__version__.split("+")[-1]
    print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)

    data_dir = "../data/raw/"

    # Split with a ratio.
    # To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
    splitfolders.ratio(
        "../data/raw/data_to_transform",
        output="../data/split_data",
        seed=101,
        ratio=(0.75, 0.15, 0.1),
        group_prefix=None,
    )  # default values

    # Set up transformations for training and validation (test) data
    # For training data we will center crop to get to 224 * 224, and normalization
    # We will grayscale on 3 levels (may want to drop it to one channel, but this works for the time being
    # For test set we will do only center cropping to get to 224 * 224 and normalization

    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.CenterCrop(224),
                transforms.Grayscale(3),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                transforms.Lambda(
                    lambda x: torch.flatten(x)  # pylint: disable=unnecessary-lambda
                ),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.CenterCrop(224),
                transforms.Grayscale(3),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                transforms.Lambda(
                    lambda x: torch.flatten(x)  # pylint: disable=unnecessary-lambda
                ),
            ]
        ),
    }

    # Create Datasets for training and validation sets
    data_dir = "../data/split_data"

    train_dataset = datasets.ImageFolder(
        os.path.join(data_dir, "train"), data_transforms["train"]
    )

    val_dataset = datasets.ImageFolder(
        os.path.join(data_dir, "val"), data_transforms["val"]
    )

    # Create DataLoaders for training and validation sets
    batch_size = 4
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=len(train_dataset), shuffle=True, num_workers=2
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=len(val_dataset), shuffle=False, num_workers=2
    )

    # Set up dict for dataloaders
    dataloaders = {"train": train_loader, "val": val_loader}

    # Store size of training and validation sets
    dataset_sizes = {"train": len(train_dataset), "val": len(val_dataset)}
    # Get class names associated with labels
    class_names = train_dataset.classes

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Visualize a batch of images
    images, labels = iter(train_loader).next()
    train_dataset_array = next(iter(train_loader))[0].numpy()
    train_dataset_label = next(iter(train_loader))[1].numpy()
    val_dataset_array = next(iter(val_loader))[0].numpy()
    val_dataset_label = next(iter(val_loader))[1].numpy()
    print(train_dataset_array.shape)
    print(train_dataset_label.shape)
    print(images.shape)
    images = images.numpy()
    fig = plt.figure(figsize=(10, 6))
    for idx in np.arange(batch_size):
        ax = fig.add_subplot(2, batch_size // 2, idx + 1, xticks=[], yticks=[])
        image = images[idx]
        image = image.transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)
        ax.imshow(image)
        ax.set_title("{}".format(class_names[labels[idx]]))

    plt.show()
    print("All done!")
    return (
        train_dataset_array,
        train_dataset_label,
        val_dataset_array,
        val_dataset_label,
        images,
        dataloaders,
        batch_size,
        class_names,
        dataset_sizes,
    )
