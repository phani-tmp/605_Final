#!/usr/bin/env python
"""
main_cpu_opt.py  –  CPU-tuned version
* OneDNN (MKL-DNN) on
* Explicit thread & inter-op settings
* Denormals flush for minor speed gain
* Non-blocking host-to-device transfers kept off (CPU run)
"""
import os, time, argparse, torch, torchvision, psutil
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
import torch
torch.set_num_threads(8)
torch.set_num_interop_threads(4)
from torch import nn, optim
from torchvision import transforms

class LogisticRegression(nn.Module):
    def __init__(self): 
        super().__init__()
        self.linear = nn.Linear(32*32*3,100)
    def forward(self,x): return self.linear(x.view(x.size(0),-1))

def get_loader(batch=512, workers=4):
    tfm = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,))])
    ds = torchvision.datasets.CIFAR100(root='./cifar100_data', train=True, download=False, transform=tfm)
    return torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True,
                                       num_workers=workers, pin_memory=False)

def epoch(model, loader, device, optimizer, loss_fn):
    model.train()
    total_loss = 0
    for X, y in loader:
        X, y = X.to(device, non_blocking=False), y.to(device, non_blocking=False)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(X)
        loss = loss_fn(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--threads",type=int,default=8)
    ap.add_argument("--epochs",type=int,default=1); args = ap.parse_args()

    # ---- OneDNN + thread tuning ----
    torch.backends.mkldnn.enabled = True
    torch.set_flush_denormal(True)
    os.environ["OMP_NUM_THREADS"]=str(args.threads)
    os.environ["MKL_NUM_THREADS"]=str(args.threads)

    device = torch.device("cpu")
    print(f"Optimised-CPU | threads={args.threads}")
    loader = get_loader(workers=min(args.threads,4))
    model  = LogisticRegression().to(device)
    optimizer    = optim.SGD(model.parameters(), lr=0.01)
    loss_fn  = nn.CrossEntropyLoss()

    start_time = time.time()
    for e in range(args.epochs):
        loss = epoch(model, loader, device, optimizer, loss_fn)
        print(f"Epoch {e + 1}, Loss: {loss:.2f}")
    end_time = time.time()

    print(f"Training completed in {(end_time - start_time):.2f} seconds")
    print(f"Memory used: {psutil.Process().memory_info().rss / 1e6:.2f} MB")


    print(f"Peak RSS: {psutil.Process().memory_info().rss/1e6:.1f} MB")

if __name__ == "__main__": main()