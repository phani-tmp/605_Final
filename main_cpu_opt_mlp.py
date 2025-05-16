#!/usr/bin/env python
import os, time, argparse, torch, torchvision, psutil
from torch import nn, optim
from torchvision import transforms

def get_loader(batch=512, workers=4):
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train = torchvision.datasets.CIFAR100('./cifar100_data', train=True, download=True, transform=tfm)
    return train

def train_torch(model, train, dev, threads, epochs=1):
    loader = torch.utils.data.DataLoader(train, batch_size=512, shuffle=True,
                                         num_workers=min(threads, 4), pin_memory=False)
    opt = optim.Adam(model.parameters(), lr=0.01)
    lossf = nn.CrossEntropyLoss()
    model.to(dev)
    model.train()

    total_images = 0
    start = time.time()
    for epoch in range(epochs):
        for X, y in loader:
            X, y = X.to(dev), y.to(dev)
            opt.zero_grad(); out = model(X)
            loss = lossf(out, y); loss.backward(); opt.step()
            total_images += X.size(0)
    elapsed = time.time() - start
    return total_images, elapsed

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=1)
    args = ap.parse_args()

    torch.backends.mkldnn.enabled = True
    torch.set_flush_denormal(True)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(max(1, args.threads // 2))
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)

    train = get_loader()

    print("Model: mlp | Threads:", args.threads)

    model = nn.Sequential(nn.Flatten(), nn.Linear(32*32*3, 512), nn.ReLU(), nn.Linear(512, 100))

    total_images, elapsed = train_torch(model, train, torch.device('cpu'), args.threads, args.epochs)
    throughput = total_images / elapsed

    print(f"✅ Training Time: {elapsed:.2f} seconds")
    print(f"🚀 Throughput: {throughput:,.1f} img/s")
    print("📦 Peak RSS: %.1f MB" % (psutil.Process().memory_info().rss / 1e6))

if __name__ == "__main__":
    main()
