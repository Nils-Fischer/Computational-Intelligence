import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms
from torchvision import datasets
import torch.optim as optim
from PIL import Image

transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=10_000, shuffle=False
)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Linear(784, 128)
        self.layer2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor):
        x = x.view(-1, 784)  # turn into 2d tensor
        x = self.relu(self.layer1(x))
        x = self.layer2(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = Net().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)

# input: torch.Size([64, 1, 28, 28])
# labels: torch.Size([64])
for epoch in range(10):
    train_loss = 0
    for data in train_loader:
        inputs, labels = data

        # forward
        outputs = net(inputs.to(device))
        loss = criterion(outputs, labels.to(device))

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    print(f"Epoch: {epoch} loss: {train_loss/len(train_loader)}")


for data in test_loader:  # only one iteration
    inputs, labels = data
    inputs = inputs.to(device)
    labels = labels.to(device)
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    print(f"test loss: {loss.item()}")

    # evaluate the vector output to a number
    _, index = torch.max(outputs, dim=1)
    labels = torch.stack((labels, index), dim=1)
    # filter for wrong output
    mask = labels[:, 0] != labels[:, 1]
    inputs = inputs[mask]
    labels = labels[mask]
    print(f"Wrong Outputs: {len(labels)}, in percentage: {len(labels)/10_000}%")

    masks = [labels[:, 0] == x for x in range(10)]
    for i, mask in enumerate(masks):
        tensor = inputs[mask][0, 0].cpu()
        array = np.uint8(255 * (tensor - tensor.min()) / (tensor.max() - tensor.min()))
        img = Image.fromarray(array)
        print(f"Supposed to be {i}, classified as {labels[mask][0,1]}")
        # uncomment to display image:
        # img.show()
