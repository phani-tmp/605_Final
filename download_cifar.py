import torchvision
from torchvision import transforms

transform = transforms.ToTensor()
torchvision.datasets.CIFAR100(root="./cifar100_data", train=True, download=True, transform=transform)
