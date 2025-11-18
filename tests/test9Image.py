import torch 
from torch.utils import data
from torchvision import datasets, transforms
from PIL import Image

# transform = transforms.Compose([transforms.ToTensor()])

# # mnist = datasets.MNIST(root='data', download=True, train=True, transform=None)

# mnist = datasets.CIFAR10(root='data', download=True, train=True, transform=transform #transfrom=None)
#                          )

# img, label = mnist[0]

# print(img.size())

torch.manual_seed(20)

x = torch.randint(1, 100, (4, 5))


print(x)
print(torch.argmax(x, dim=0))