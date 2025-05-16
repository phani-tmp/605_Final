import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
import os

def setup(rank, world_size, master_addr, master_port):
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port

    # 🔒 Force Windows to bind the right interface
    if os.name == "nt" and rank == 1:
        os.environ["GLOO_SOCKET_IFNAME"] = "Ethernet"  # OR "Wi-Fi" depending on active interface

    # On Mac (Rank 0)
    if os.name == "posix" and rank == 0:
        os.environ["GLOO_SOCKET_IFNAME"] = "en0"  # or whatever macOS interface has IP 10.0.0.249

    print(f"[Rank {rank}] Connecting to {master_addr}:{master_port}")
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://{master_addr}:{master_port}",
        rank=rank,
        world_size=world_size
    )
    print(f"[Rank {rank}] Initialized process group!")

def cleanup():
    dist.destroy_process_group()

class SimpleModel(nn.Module):
    def _init_(self):
        super()._init_()
        self.linear = nn.Linear(32 * 32 * 3, 100)

    def forward(self, x):
        return self.linear(x.view(x.size(0), -1))

def train(rank, world_size, master_addr, master_port):
    setup(rank, world_size, master_addr, master_port)

    transform = transforms.ToTensor()
    dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=64, sampler=sampler)

    model = SimpleModel()
    ddp_model = DDP(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)

    for epoch in range(1, 4):
        total_loss = 0
        for images, labels in dataloader:
            optimizer.zero_grad()
            outputs = ddp_model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[Rank {rank}] Epoch {epoch}, Loss: {total_loss:.2f}")

    cleanup()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world_size", type=int, required=True)
    parser.add_argument("--master_addr", type=str, required=True)
    parser.add_argument("--master_port", type=str, default="29500")
    args = parser.parse_args()

    # Force correct network interface on macOS (ONLY RANK 0)
    if args.rank == 0:
        os.environ["GLOO_SOCKET_IFNAME"] = "Wi-Fi"  # change if not your active interface

    train(args.rank, args.world_size, args.master_addr, args.master_port)