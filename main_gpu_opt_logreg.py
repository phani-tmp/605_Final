#!/usr/bin/env python
"""
main_gpu_opt.py  –  GPU-tuned version
* cudnn.benchmark for algo-search
* Mixed precision (AMP + GradScaler)
* Larger default batch for higher GPU util
"""

import torch, torchvision, time, argparse, psutil
from torch import nn, optim
from torchvision import transforms

class LogisticRegression(nn.Module):
    def __init__(self):  
        super().__init__()
        self.linear = nn.Linear(32 * 32 * 3, 100)

    def forward(self, x):
        return self.linear(x.view(x.size(0), -1))

def get_loader(batch=4096, workers=6):
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    ds = torchvision.datasets.CIFAR100(root='./cifar100_data', train=True, download=False, transform=tfm)
    return torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True,
                                       num_workers=workers, pin_memory=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "No GPU found!"
    torch.backends.cudnn.benchmark = True
    scaler = torch.cuda.amp.GradScaler()

    loader = get_loader()
    model = LogisticRegression().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    start_time = time.time()
    for e in range(args.epochs):
        model.train()
        total_loss = 0
        for X, y in loader:
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                outputs = model(X)
                loss = loss_fn(outputs, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        print(f"Epoch {e+1}, Loss: {total_loss:.2f}")
    end_time = time.time()

    print(f"Training completed in {(end_time - start_time):.2f} seconds")
    print(f"Memory used: {psutil.Process().memory_info().rss / 1e6:.2f} MB")
    print(f"Peak RSS: {psutil.Process().memory_info().rss / 1e6:.1f} MB")
    print(f"Peak GPU-mem: {torch.cuda.max_memory_allocated() / 1e6:.1f} MB")

if __name__ == "__main__":
    main()