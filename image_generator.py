import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


def reduce(tensor):
    tensor = F.interpolate(tensor.unsqueeze(0).unsqueeze(0), size=(37, 37), mode="bicubic").squeeze()
    return torch.where(tensor > 0.9, torch.tensor(1.), torch.tensor(0.))


def apply_padding(tensors, padding, buffer):
    tensors[0] = F.pad(tensors[0], (
        padding[0][0] + buffer, 36 + buffer - padding[0][0], padding[0][1] + buffer, 36 + buffer - padding[0][1]))
    tensors[1] = F.pad(tensors[1], (
        padding[1][0] + buffer, 36 + buffer - padding[1][0], padding[1][1] + buffer, 36 + buffer - padding[1][1]))
    tensors[2] = F.pad(tensors[2], (
        padding[2][0] + buffer, 36 + buffer - padding[2][0], padding[2][1] + buffer, 36 + buffer - padding[2][1]))
    return tensors


class ImageGenerator:

    def __init__(self, train: bool, batch_size: int):
        self.device = torch.device("cuda" if torch.cuda.is_available else "cpu")
        transform = transforms.ToTensor()
        dataset = torchvision.datasets.MNIST(
            root="./data", train=train, download=True, transform=transform
        )
        g0 = (img.squeeze() for img, label in dataset if label == 0)
        g1 = (img.squeeze() for img, label in dataset if label == 1)
        g2 = (img.squeeze() for img, label in dataset if label == 2)
        g3 = (img.squeeze() for img, label in dataset if label == 3)
        g4 = (img.squeeze() for img, label in dataset if label == 4)
        g5 = (img.squeeze() for img, label in dataset if label == 5)
        g6 = (img.squeeze() for img, label in dataset if label == 6)
        g7 = (img.squeeze() for img, label in dataset if label == 7)
        g8 = (img.squeeze() for img, label in dataset if label == 8)
        g9 = (img.squeeze() for img, label in dataset if label == 9)
        # list comprehension und loops hat aus irgendeinem grund nicht geklappt mit generators
        self.sorted_dataset = [g0, g1, g2, g3, g4, g5, g6, g7, g8, g9]
        self.batch_size = batch_size
        self.size = 1000 + 9000 * train
        self.counter = 0

    def __iter__(self):
        return self

    def __next__(self):
        images = torch.empty(0, 64, 64)
        expected = torch.empty(0, 10, 37, 37)
        centers = torch.empty(0, 10, 37, 37)

        torch.set_printoptions(profile="full")
        for _ in range(self.batch_size):
            if self.counter == self.size:
                raise StopIteration
            # extract images
            n1, n2, n3 = random.sample(range(10), 3)
            image1 = next(self.sorted_dataset[n1])
            image2 = next(self.sorted_dataset[n2])
            image3 = next(self.sorted_dataset[n3])
            # transform images
            padding = [(random.randint(0, 36), random.randint(0, 36)) for _ in range(3)]
            padded_images = apply_padding([image1, image2, image3], padding, 0)
            # generate new image
            stacked = torch.stack(padded_images)
            image, _ = torch.max(stacked, dim=0)
            # combine all into one tensor
            images = torch.cat((images, image.unsqueeze(0)), dim=0)
            # generate expected tensors
            ones = torch.ones((4, 4))
            padded_expected = apply_padding([ones for _ in range(3)], padding, 12)
            # reduce and combine expected
            t = torch.zeros((10, 37, 37))
            t[n1] += reduce(padded_expected[0])
            t[n2] += reduce(padded_expected[1])
            t[n3] += reduce(padded_expected[2])
            expected = torch.cat((expected, t.unsqueeze(0)), dim=0)
            # generate tensors to indicate the centers
            ones = torch.ones((26, 26))
            padded_center = apply_padding([ones for _ in range(3)], padding, 1)
            center = torch.zeros((10, 37, 37))
            center[n1] += reduce(padded_center[0])
            center[n2] += reduce(padded_center[1])
            center[n3] += reduce(padded_center[2])
            centers = torch.cat((centers, center.unsqueeze(0)), dim=0)

            self.counter += 1

        return [images.unsqueeze(1).to(self.device), expected.to(self.device), centers.to(self.device)]
