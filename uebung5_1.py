from image_generator import ImageGenerator
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available else "cpu")


# input: torch.Size([64, 1, 64, 64])
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=9, stride=1, padding=4),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=10, stride=1),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=10, stride=1),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=10, stride=1),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(32, 10, kernel_size=3, stride=1, padding=1),
            nn.Softmax(),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


def performance_metrics(output, expected, centers):
    output_max = torch.amax(output * centers, dim=(2, 3), keepdim=True)
    expected_max = torch.amax(expected, dim=(2, 3), keepdim=True)
    evaluated_max = expected_max - output_max
    true_positives = torch.sum((0.5 > evaluated_max) * (evaluated_max >= 0))
    false_negatives = torch.sum(evaluated_max >= 0.5)
    false_positives = torch.sum(evaluated_max > -0.5)
    return true_positives, false_negatives, false_positives


# Hyperparameter
num_epochs = 10
batch_size = 128
learning_rate = 0.005

model = ConvNet().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    loss_total = 0
    tp = 0
    fn = 0
    fp = 0
    train_images = ImageGenerator(True, 64)
    for images, expected, centers in train_images:
        # Forward
        output = model(images)
        loss = criterion(output, expected)

        # Backwards
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_total += loss

        # calculate metrics
        results = performance_metrics(output, expected, centers)
        tp += results[0]
        fn += results[1]
        fp += results[2]

    print(f"epoch {epoch}:")
    print(f"loss = {loss_total / batch_size}")
    print(f"precision = {tp / (tp + fp)}")
    print(f"recall = {tp / (tp + fn)}\n")

with torch.no_grad():
    tp = 0
    fn = 0
    fp = 0
    test_images = ImageGenerator(False, 64)

    for images, expected, centers in test_images:
        # Forward
        output = model(images)

        # calculate metrics
        results = performance_metrics(output, expected, centers)
        tp += results[0]
        fn += results[1]
        fp += results[2]
    print("test results:")
    print(f"precision = {tp / (tp + fp)}")
    print(f"recall = {tp / (tp + fn)}\n")

