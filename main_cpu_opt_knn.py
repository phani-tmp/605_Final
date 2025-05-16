#!/usr/bin/env python
import os, time, argparse, torch, torchvision, psutil
from torch import nn, optim
from torchvision import transforms
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score

def get_loader(batch=512, workers=4):
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train = torchvision.datasets.CIFAR100('./cifar100_data', train=True, download=True, transform=tfm)
    test = torchvision.datasets.CIFAR100('./cifar100_data', train=False, download=True, transform=tfm)
    return train, test

def train_sklearn(model, train, test):
    import numpy as np
    X_train = torch.stack([x for x, _ in train]).view(len(train), -1).numpy()
    y_train = np.array([y for _, y in train])
    X_test = torch.stack([x for x, _ in test]).view(len(test), -1).numpy()
    y_test = np.array([y for _, y in test])
    st = time.time(); model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return acc, time.time() - st

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

    train, test = get_loader()

    print("Model: knn | Threads:", args.threads)

    model = KNeighborsClassifier(n_neighbors=3)

    acc, t = train_sklearn(model, train, test)
    print(f'Accuracy: {acc*100:.2f}% | Time: {t:.2f}s')

    print("Peak RSS: %.1f MB" % (psutil.Process().memory_info().rss / 1e6))

if __name__ == "__main__": main()