from torchvision import models, transforms
import torch
from PIL import Image
import torch.nn as nn


def predict(image_path):
    # Model class must be defined somewhere
    resnet = models.resnet34(pretrained=True)
    num_ftrs = resnet.fc.in_features
    resnet.fc = nn.Linear(in_features=num_ftrs, out_features=3)
    resnet.load_state_dict(
        torch.load("../models/model.pt", map_location=torch.device("cpu"))
    )

    # https://pytorch.org/docs/stable/torchvision/models.html
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Grayscale(3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    img = Image.open(image_path)
    batch_t = torch.unsqueeze(transform(img), 0)

    resnet.eval()
    out = resnet(batch_t)

    classes = ["english", "telugu", "russian"]

    prob = torch.nn.functional.softmax(out, dim=1)[0] * 100
    _, indices = torch.sort(out, descending=True)
    return [(classes[idx], prob[idx].item()) for idx in indices[0][:3]]
