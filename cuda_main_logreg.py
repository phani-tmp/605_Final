import numpy as np
import math
import time
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from numba import cuda

# -----------------------------
# CUDA Kernels
# -----------------------------

@cuda.jit
def matmul(A, B, C):
    row, col = cuda.grid(2)
    if row < C.shape[0] and col < C.shape[1]:
        tmp = 0.
        for k in range(A.shape[1]):
            tmp += A[row, k] * B[k, col]
        C[row, col] = tmp

@cuda.jit
def add_bias(Z, bias):
    row, col = cuda.grid(2)
    if row < Z.shape[0] and col < Z.shape[1]:
        Z[row, col] += bias[col]

@cuda.jit
def sigmoid(Z, A):
    row, col = cuda.grid(2)
    if row < Z.shape[0] and col < Z.shape[1]:
        A[row, col] = 1.0 / (1.0 + math.exp(-Z[row, col]))

@cuda.jit
def compute_error(preds, y, error_out):
    row = cuda.grid(1)
    if row < y.shape[0]:
        error_out[row, 0] = preds[row, 0] - y[row, 0]  # ✅ both are now 2D

@cuda.jit
def update_weights(X, error, grad_w, grad_b):
    row, col = cuda.grid(2)
    if row < grad_w.shape[0] and col < grad_w.shape[1]:
        acc = 0.
        for i in range(X.shape[0]):
            acc += X[i, row] * error[i, col]
        grad_w[row, col] = acc / X.shape[0]

    if row == 0 and col < grad_b.shape[0]:
        acc_b = 0.
        for i in range(error.shape[0]):
            acc_b += error[i, 0]
        grad_b[col] = acc_b / error.shape[0]

@cuda.jit
def apply_gradients(W, b, grad_w, grad_b, lr):
    row, col = cuda.grid(2)
    if row < W.shape[0] and col < W.shape[1]:
        W[row, col] -= lr * grad_w[row, col]
    if row == 0 and col < b.shape[0]:
        b[col] -= lr * grad_b[col]

# -----------------------------
# CUDA Logistic Regression Class
# -----------------------------

class CUDAJITLogReg:
    def __init__(self, input_dim, output_dim):
        self.W = cuda.to_device(np.random.randn(input_dim, output_dim).astype(np.float32) * 0.01)
        self.b = cuda.to_device(np.zeros((output_dim,), dtype=np.float32))

    def fit(self, X, y, epochs=10, lr=0.01):
        X_d = cuda.to_device(X.astype(np.float32))
        y_d = cuda.to_device(y.astype(np.float32).reshape(-1, 1))  # ✅ force it to 2D right here


        threadsperblock = (16, 16)
        blockspergrid = (
            math.ceil(X.shape[0] / threadsperblock[0]),
            math.ceil(self.W.shape[1] / threadsperblock[1])
        )

        for _ in range(epochs):
            Z = cuda.device_array((X.shape[0], self.W.shape[1]), dtype=np.float32)
            A = cuda.device_array_like(Z)

            matmul[blockspergrid, threadsperblock](X_d, self.W, Z)
            add_bias[blockspergrid, threadsperblock](Z, self.b)
            sigmoid[blockspergrid, threadsperblock](Z, A)

            error = cuda.device_array_like(A)
            compute_error[math.ceil(X.shape[0]/32), 32](A, y_d, error)

            grad_w = cuda.device_array_like(self.W)
            grad_b = cuda.device_array_like(self.b)

            update_weights[blockspergrid, threadsperblock](X_d, error, grad_w, grad_b)
            apply_gradients[blockspergrid, threadsperblock](self.W, self.b, grad_w, grad_b, lr)

    def predict(self, X):
        X_d = cuda.to_device(X.astype(np.float32))
        Z = cuda.device_array((X.shape[0], self.W.shape[1]), dtype=np.float32)
        A = cuda.device_array_like(Z)

        threadsperblock = (16, 16)
        blockspergrid = (
            math.ceil(X.shape[0] / threadsperblock[0]),
            math.ceil(self.W.shape[1] / threadsperblock[1])
        )

        matmul[blockspergrid, threadsperblock](X_d, self.W, Z)
        add_bias[blockspergrid, threadsperblock](Z, self.b)
        sigmoid[blockspergrid, threadsperblock](Z, A)

        return (A.copy_to_host() > 0.5).astype(np.int32)

# -----------------------------
# Main Runner
# -----------------------------

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))
    ])

    train_data = datasets.CIFAR100(root="./cifar100_data", train=True, download=True, transform=transform)
    test_data = datasets.CIFAR100(root="./cifar100_data", train=False, download=True, transform=transform)

    X_train = np.stack([x.numpy() for x, _ in train_data]).astype(np.float32)
    # y_train = np.array([1 if y == 0 else 0 for _, y in train_data])
    y_train = np.array([1 if y == 0 else 0 for _, y in train_data], dtype=np.float32).reshape(-1, 1)

    X_test = np.stack([x.numpy() for x, _ in test_data]).astype(np.float32)
    y_test = np.array([1 if y == 0 else 0 for _, y in test_data], dtype=np.float32).reshape(-1, 1)


    model = CUDAJITLogReg(X_train.shape[1], 1)

    start = time.time()
    model.fit(X_train, y_train, epochs=10, lr=0.1)
    train_time = time.time() - start

    start = time.time()
    preds = model.predict(X_test).flatten()
    test_time = time.time() - start

    accuracy = np.mean(preds == y_test) * 100

    print(f"Training Time: {train_time:.2f}s")
    print(f"Prediction Time: {test_time:.2f}s")
    print(f"Accuracy: {accuracy:.2f}%")
