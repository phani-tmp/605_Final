# import os
# import torch
# import torch.distributed as dist
# import torch.nn as nn
# import torch.optim as optim
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.utils.data import DataLoader, DistributedSampler
# from torchvision import datasets, transforms

# # 1. Set up distributed training
# def setup():
#     if os.name == "posix" and int(os.environ.get("RANK", 0)) == 0:
#         os.environ["GLOO_SOCKET_IFNAME"] = "en0"  # Mac: Adjust if needed

#     dist.init_process_group(backend="gloo")
#     print(f"[Rank {os.environ['RANK']}] DDP setup complete with world size {os.environ['WORLD_SIZE']}")

# def cleanup():
#     dist.destroy_process_group()

# # 2. Simple model
# class SimpleModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.linear = nn.Linear(32 * 32 * 3, 100)

#     def forward(self, x):
#         return self.linear(x.view(x.size(0), -1))

# # 3. Training loop
# def train(rank, world_size):
#     print(f"[Rank {rank}] Starting training on device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

#     transform = transforms.ToTensor()
#     dataset = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform)
#     sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
#     dataloader = DataLoader(dataset, batch_size=64, sampler=sampler)

#     model = SimpleModel()
#     ddp_model = DDP(model)

#     loss_fn = nn.CrossEntropyLoss()
#     optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)

#     for epoch in range(1, 4):
#         print(f"[Rank {rank}] Starting Epoch {epoch}")
#         sampler.set_epoch(epoch)
#         total_loss = 0
#         for X, y in dataloader:
#             optimizer.zero_grad()
#             outputs = ddp_model(X)
#             loss = loss_fn(outputs, y)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#         print(f"[Rank {rank}] Epoch {epoch} | Loss: {total_loss:.2f}")

# # 4. Entry point
# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--rank", type=int, required=True)
#     parser.add_argument("--world_size", type=int, required=True)
#     parser.add_argument("--master_addr", type=str, required=True)
#     parser.add_argument("--master_port", type=int, default=29500)
#     args = parser.parse_args()

#     os.environ["RANK"] = str(args.rank)
#     os.environ["WORLD_SIZE"] = str(args.world_size)
#     os.environ["MASTER_ADDR"] = args.master_addr
#     os.environ["MASTER_PORT"] = str(args.master_port)

#     setup()
#     train(args.rank, args.world_size)
#     cleanup()
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
import argparse
import time


class LogisticRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(32 * 32 * 3, 100)

    def forward(self, x):
        return self.linear(x.view(-1, 32 * 32 * 3))

def setup(rank, world_size, master_addr, master_port):
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    print(f"[Rank {rank}] DDP initialized with master at {master_addr}:{master_port}")

def cleanup():
    dist.destroy_process_group()

def get_data_loader(rank, world_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.CIFAR100(root="./cifar100_data", train=True, download=True, transform=transform)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(dataset, batch_size=256, sampler=sampler, num_workers=2)
    return loader, sampler

def train(rank, world_size, master_addr, master_port):
    setup(rank, world_size, master_addr, master_port)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LogisticRegression().to(device)
    ddp_model = DDP(model)

    loader, sampler = get_data_loader(rank, world_size)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)

    print(f"[Rank {rank}] Training on {device} with {torch.get_num_threads()} threads")

    start_time = time.time()
    for epoch in range(5):
        sampler.set_epoch(epoch)
        total_loss = 0
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = ddp_model(X)
            loss = loss_fn(outputs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[Rank {rank}] Epoch {epoch + 1} | Loss: {total_loss:.2f}")

    print(f"[Rank {rank}] Training complete in {(time.time() - start_time):.2f} seconds")
    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world_size", type=int, required=True)
    parser.add_argument("--master_addr", type=str, required=True)
    parser.add_argument("--master_port", type=int, default=29500)
    args = parser.parse_args()

    train(rank=args.rank, world_size=args.world_size, master_addr=args.master_addr, master_port=args.master_port)
