import os

import matplotlib.pyplot as plt
import numpy as np

# this package make graping labeled image folders from a dir
# and splitting them into train, val, and test
import splitfolders
import torch
import torch.utils.data
from torchvision import datasets, transforms, utils


def __imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def transform_data():
    TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
    CUDA_VERSION = torch.__version__.split("+")[-1]
    print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)

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

    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.Grayscale(3),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.Grayscale(3),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }

    # Create Datasets for training and validation sets
    data_dir = "../data/split_data"
    batch_size = 4
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ["train", "val"]
    }
    dataloaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x], batch_size=4, shuffle=True, num_workers=4
        )
        for x in ["train", "val"]
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}
    class_names = image_datasets["train"].classes

    # Get a batch of training data
    images, classes = next(iter(dataloaders["train"]))
    print(images.shape)

    # Make a grid from batch
    out = utils.make_grid(images)

    __imshow(out, title=[class_names[x] for x in classes])

    return (images, dataloaders, batch_size, class_names, dataset_sizes)


if __name__ == "__main__":
    transform_data()
