import torch
import torch.nn as nn
from torch import optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available else "cpu")

# Hyperparameter
num_epochs = 10
batch_size = 128
learning_rate = 0.005

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
trainset_load = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
testset_load = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

trainset = torch.utils.data.DataLoader(
    trainset_load, batch_size=batch_size, shuffle=True, num_workers=2
)
testset = torch.utils.data.DataLoader(
    testset_load, batch_size=batch_size, shuffle=False, num_workers=2
)


def plot_grid(grid):
    np_grid = grid.numpy().transpose((1, 2, 0))

    # display the grid using imshow
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(np_grid)
    ax.axis("off")
    plt.show()


# input: torch.Size([64, 1, 28, 28])
class Convolutional_Net(nn.Module):
    def __init__(self):
        super(Convolutional_Net, self).__init__()
        # layer 1
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1
        )
        # W_out = (28 - 3 + 2*1)/1 + 1= 28
        # 28 x 28 x 16
        self.batch1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # W_out = (28-2)/2 + 1 = 14
        # 14 x 14 x 16

        # layer 2
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=4, stride=1, padding=2
        )
        # W_out = (14 - 4 + 2*2)/1 = 14
        # 14 x 14 x 32
        self.batch2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # W_out = (14 - 2)/2 + 1 = 7
        # 7 x 7 x 32

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=32 * 7 * 7, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        # Conv 1
        x = self.conv1(x)
        x = self.batch1(x)
        x = self.relu(x)

        # Max Pool 1
        x = self.pool1(x)

        # Conv 2
        x = self.conv2(x)
        x = self.batch2(x)
        x = self.relu(x)

        # Max Pool 2
        x = self.pool2(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        return x


model = Convolutional_Net().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


for epoch in range(num_epochs):
    loss_total = 0
    correct_total = 0
    for i, (images, labels) in enumerate(trainset):
        images = images.to(device)
        labels = labels.to(device)

        # Forward
        output = model(images)
        loss = criterion(output, labels)

        # Backwards
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_total += loss

        # calculate false classified
        _, predicted = torch.max(output, 1)
        correct_total += (predicted == labels).sum().item()
    print(
        f"Training - Epoch: {epoch}, loss: {loss_total}, accuracy: {100*correct_total/60_000:.4f}%"
    )

with torch.no_grad():
    correct_total = 0
    false_tests_total = [
        torch.empty((0, 1, 28, 28)).to(device),
        torch.empty((0)).to(device),
    ]

    for batch in testset:
        images, labels = [x.to(device) for x in batch]
        output = model(images)

        _, predicted = torch.max(output, 1)
        correct_total += (predicted == labels).sum().item()
        false_tests = [x[predicted != labels] for x in [images, labels]]
        false_tests_total = [
            torch.cat([false_tests_total[i], false_tests[i]], dim=0) for i in [0, 1]
        ]

    print(
        f"Test set - loss: {loss_total:.4f}, accuracy: {100*correct_total/10_000:.4f}%, falsely classified = {10_000 - correct_total}"
    )
    grids = []
    for i in range(10):
        images, labels = false_tests_total
        grids.append(
            torchvision.utils.make_grid(
                images[labels == i][:10],
                nrow=10,
                padding=2,
                normalize=True,
                scale_each=True,
            ).cpu()
        )

    for grid in grids:
        plot_grid(grid)

# Verglichen mit dem Ergebnis aus 3.2 schneidet das CNN nochmal um einiges besser ab.
