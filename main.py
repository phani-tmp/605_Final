# # import torch
# # import torchvision
# # import torch.nn as nn
# # import torch.optim as optim
# # from torchvision import transforms
# # import time
# # import argparse
# # import psutil

# # # Load MNIST data efficiently
# # def get_data_loader(batch_size=64):
# #     transform = transforms.ToTensor()
# #     train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# #     return torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2)

# # # Logistic Regression Model
# # class LogisticRegression(nn.Module):
# #     def __init__(self):
# #         super().__init__()
# #         self.linear = nn.Linear(28 * 28, 10)

# #     def forward(self, x):
# #         return self.linear(x.view(-1, 28 * 28))

# # def train(device):
# #     model = LogisticRegression().to(device)
# #     loader = get_data_loader()
# #     loss_fn = nn.CrossEntropyLoss()
# #     optimizer = optim.SGD(model.parameters(), lr=0.01)

# #     print(f"\nTraining on: {device}")
# #     start_time = time.time()
# #     for epoch in range(5):
# #         total_loss = 0
# #         for X, y in loader:
# #             X, y = X.to(device), y.to(device)
# #             optimizer.zero_grad()
# #             outputs = model(X)
# #             loss = loss_fn(outputs, y)
# #             loss.backward()
# #             optimizer.step()
# #             total_loss += loss.item()
# #         print(f"Epoch {epoch+1}, Loss: {total_loss:.2f}")
# #     end_time = time.time()

# #     print(f"Training completed in {(end_time - start_time):.2f} seconds")
# #     print(f"Memory used: {psutil.Process().memory_info().rss / 1e6:.2f} MB")

# # if __name__ == "__main__":
# #     parser = argparse.ArgumentParser()
# #     parser.add_argument("--device", default="cpu", help="Device to run on: cpu or cuda")
# #     args = parser.parse_args()

# #     device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
# #     train(device)
 
# import torch
# import torchvision
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import transforms
# import time
# import argparse
# import psutil
# import os

# # Dynamically use SLURM-allocated cores
# cpu_count = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
# torch.set_num_threads(cpu_count)

# def get_data_loader(batch_size=256):
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.5,), (0.5,))
#     ])
#     train = torchvision.datasets.CIFAR10(root='./cifar_data', train=True, download=False, transform=transform)
#     # Use max 4 workers or cpu_count, whichever is lower
#     num_workers = min(cpu_count, 4)
#     return torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers)

# class LogisticRegression(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.linear = nn.Linear(32 * 32 * 3, 10)

#     def forward(self, x):
#         return self.linear(x.view(-1, 32 * 32 * 3))

# def train(device):
#     model = LogisticRegression().to(device)
#     loader = get_data_loader()
#     loss_fn = nn.CrossEntropyLoss()
#     optimizer = optim.SGD(model.parameters(), lr=0.01)

#     print(f"\nTraining on: {device}")
#     print(f"Using {torch.get_num_threads()} CPU threads")

#     start_time = time.time()
#     for epoch in range(5):
#         total_loss = 0
#         for X, y in loader:
#             X, y = X.to(device), y.to(device)
#             optimizer.zero_grad()
#             outputs = model(X)
#             loss = loss_fn(outputs, y)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#         print(f"Epoch {epoch+1}, Loss: {total_loss:.2f}")
#     end_time = time.time()

#     print(f"Training completed in {(end_time - start_time):.2f} seconds")
#     print(f"Memory used: {psutil.Process().memory_info().rss / 1e6:.2f} MB")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--device", default="cpu", help="Device to run on: cpu or cuda")
#     args = parser.parse_args()

#     device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
#     train(device)
#=======================================================================================
# import torch
# import torchvision
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import transforms
# import time
# import argparse
# import psutil
# import os

# # Dynamically use SLURM-allocated cores
# cpu_count = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
# torch.set_num_threads(cpu_count)

# # Data loader for CIFAR-100
# def get_data_loader(batch_size=256):
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.5,), (0.5,))
#     ])
#     train = torchvision.datasets.CIFAR100(root='./cifar100_data', train=True, download=False, transform=transform)
#     num_workers = min(cpu_count, 4)
#     return torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers)

# # Logistic Regression Model for 100 classes
# class LogisticRegression(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.linear = nn.Linear(32 * 32 * 3, 100)  # Note: output changed to 100

#     def forward(self, x):
#         return self.linear(x.view(-1, 32 * 32 * 3))

# def train(device):
#     model = LogisticRegression().to(device)
#     loader = get_data_loader()
#     loss_fn = nn.CrossEntropyLoss()
#     optimizer = optim.SGD(model.parameters(), lr=0.01)

#     print(f"\nTraining on: {device}")
#     print(f"Using {torch.get_num_threads()} CPU threads")

#     start_time = time.time()
#     for epoch in range(5):
#         total_loss = 0
#         for X, y in loader:
#             X, y = X.to(device), y.to(device)
#             optimizer.zero_grad()
#             outputs = model(X)
#             loss = loss_fn(outputs, y)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#         print(f"Epoch {epoch+1}, Loss: {total_loss:.2f}")
#     end_time = time.time()

#     print(f"Training completed in {(end_time - start_time):.2f} seconds")
#     print(f"Memory used: {psutil.Process().memory_info().rss / 1e6:.2f} MB")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--device", default="cpu", help="Device to run on: cpu or cuda")
#     args = parser.parse_args()

#     device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
#     train(device)
#=========================================================================================================
# import torch
# import torchvision
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import transforms
# import time
# import argparse
# import psutil
# import os

# # Dynamically use SLURM-allocated cores
# cpu_count = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
# torch.set_num_threads(cpu_count)

# # Data loader for CIFAR-100
# def get_data_loader(batch_size=256):
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.5,), (0.5,))
#     ])
#     train = torchvision.datasets.CIFAR100(root='./cifar100_data', train=True, download=False, transform=transform)
#     num_workers = min(cpu_count, 4)
#     return torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers)

# # Models
# def get_model(model_type):
#     input_size = 32 * 32 * 3
#     output_size = 100

#     if model_type == "logistic":
#         return nn.Sequential(nn.Flatten(), nn.Linear(input_size, output_size))

#     elif model_type == "mlp":
#         return nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(input_size, 512),
#             nn.ReLU(),
#             nn.Linear(512, output_size)
#         )

#     elif model_type == "cnn":
#         return nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#             nn.Flatten(),
#             nn.Linear(32 * 16 * 16, output_size)
#         )

#     elif model_type == "deepcnn":
#         return nn.Sequential(
#             nn.Conv2d(3, 32, 3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, 3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#             nn.Flatten(),
#             nn.Linear(64 * 16 * 16, output_size)
#         )

#     elif model_type == "resblock":
#         class ResBlock(nn.Module):
#             def __init__(self):
#                 super().__init__()
#                 self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
#                 self.relu = nn.ReLU()
#                 self.conv2 = nn.Conv2d(32, 3, 3, padding=1)

#             def forward(self, x):
#                 return self.relu(x + self.conv2(self.relu(self.conv1(x))))

#         return nn.Sequential(
#             ResBlock(),
#             nn.Flatten(),
#             nn.Linear(input_size, output_size)
#         )

#     elif model_type == "batchnorm":
#         return nn.Sequential(
#             nn.Conv2d(3, 16, 3, padding=1),
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#             nn.Flatten(),
#             nn.Linear(16 * 16 * 16, output_size)
#         )

#     elif model_type == "dropout":
#         return nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(input_size, 512),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(512, output_size)
#         )

#     else:
#         raise ValueError(f"Unsupported model type: {model_type}")

# # def train(device, model_type):
# #     model = get_model(model_type).to(device)
# #     loader = get_data_loader()
# #     loss_fn = nn.CrossEntropyLoss()
# #     optimizer = optim.SGD(model.parameters(), lr=0.01)

# #     print(f"\nTraining {model_type} model on: {device}")
# #     print(f"Using {torch.get_num_threads()} CPU threads")

# #     start_time = time.time()
# #     for epoch in range(5):
# #         total_loss = 0
# #         for X, y in loader:
# #             X, y = X.to(device), y.to(device)
# #             optimizer.zero_grad()
# #             outputs = model(X)
# #             loss = loss_fn(outputs, y)
# #             loss.backward()
# #             optimizer.step()
# #             total_loss += loss.item()
# #         print(f"Epoch {epoch+1}, Loss: {total_loss:.2f}")
# #     end_time = time.time()

# #     print(f"Training completed in {(end_time - start_time):.2f} seconds")
# #     print(f"Memory used: {psutil.Process().memory_info().rss / 1e6:.2f} MB")
# def train(device, model_type):
#     if model_type in ["logreg", "mlp"]:
#         model = get_model(model_type).to(device)
#         loader = get_data_loader()
#         loss_fn = nn.CrossEntropyLoss()
#         optimizer = optim.SGD(model.parameters(), lr=0.01)

#         print(f"\nTraining on: {device}")
#         print(f"Using {torch.get_num_threads()} CPU threads")

#         start_time = time.time()
#         for epoch in range(5):
#             total_loss = 0
#             for X, y in loader:
#                 X, y = X.to(device), y.to(device)
#                 optimizer.zero_grad()
#                 outputs = model(X)
#                 loss = loss_fn(outputs, y)
#                 loss.backward()
#                 optimizer.step()
#                 total_loss += loss.item()
#             print(f"Epoch {epoch+1}, Loss: {total_loss:.2f}")
#         end_time = time.time()

#         print(f"Training completed in {(end_time - start_time):.2f} seconds")
#         print(f"Memory used: {psutil.Process().memory_info().rss / 1e6:.2f} MB")

#     else:
#         # Handle sklearn models
#         print(f"Training Scikit-learn model: {model_type}")
#         from torchvision.datasets import CIFAR100
#         transform = transforms.Compose([transforms.ToTensor()])
#         data = CIFAR100(root='./cifar100_data', train=True, download=False, transform=transform)
#         X = torch.stack([d[0] for d in data]).view(len(data), -1).numpy()
#         y = [d[1] for d in data]

#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#         model = get_model(model_type)
#         start_time = time.time()
#         model.fit(X_train, y_train)
#         end_time = time.time()
#         preds = model.predict(X_test)
#         acc = accuracy_score(y_test, preds)

#         print(f"Training completed in {(end_time - start_time):.2f} seconds")
#         print(f"Accuracy: {acc*100:.2f}%")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--device", default="cpu", help="Device to run on: cpu or cuda")
#     parser.add_argument("--model", default="logistic", help="Model type: logistic, mlp, cnn, deepcnn, resblock, batchnorm, dropout")
#     args = parser.parse_args()

#     device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
#     train(device, args.model)
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import time
import psutil
import argparse

from sklearn.linear_model import LogisticRegression as SKLogReg
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def get_model(model_type):
    model_type = model_type.lower()
    if model_type == "logreg":
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*32*3, 100)
        )
    elif model_type == "mlp":
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*32*3, 512),
            nn.ReLU(),
            nn.Linear(512, 100)
        )
    elif model_type == "svm":
        return SVC()
    elif model_type == "knn":
        return KNeighborsClassifier(n_neighbors=3)
    elif model_type == "decision_tree":
        return DecisionTreeClassifier()
    elif model_type == "random_forest":
        return RandomForestClassifier()
    elif model_type == "extra_trees":
        return ExtraTreesClassifier()
    elif model_type == "naive_bayes":
        return GaussianNB()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def get_data_loader():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.CIFAR100(root='./cifar100_data', train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    return loader


def train(device, model_type):
    if model_type in ["logreg", "mlp"]:
        model = get_model(model_type).to(device)
        loader = get_data_loader()
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        print(f"\nTraining on: {device}")
        print(f" CPU Threads: {torch.get_num_threads()} (SLURM_CPUS_PER_TASK = {torch.get_num_threads()})")

        start_time = time.time()
        for epoch in range(5):
            total_loss = 0
            for X, y in loader:
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                outputs = model(X)
                loss = loss_fn(outputs, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f" Epoch {epoch+1}, Loss: {total_loss:.2f}")
        end_time = time.time()

        print(f" Training completed in {(end_time - start_time):.2f} seconds")
        print(f" Memory used: {psutil.Process().memory_info().rss / 1e6:.2f} MB")

    else:
        print(f"Training Scikit-learn model: {model_type}")
        transform = transforms.Compose([transforms.ToTensor()])
        data = datasets.CIFAR100(root='./cifar100_data', train=True, download=True, transform=transform)
        X = torch.stack([d[0] for d in data]).view(len(data), -1).numpy()
        y = [d[1] for d in data]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = get_model(model_type)
        start_time = time.time()
        model.fit(X_train, y_train)
        end_time = time.time()
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        print(f" Training completed in {(end_time - start_time):.2f} seconds")
        print(f" Accuracy: {acc*100:.2f}%")
        print(f" Memory used: {psutil.Process().memory_info().rss / 1e6:.2f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Model to train (logreg, mlp, svm, etc.)')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use (cpu or cuda)')
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    train(device, args.model)
    print(" Job done.")
    import sys; sys.stdout.flush()

