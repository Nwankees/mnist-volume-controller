import torch
import cv2
import time

# Load TorchScript model
model = torch.jit.load("../cpp/mnist_cnn.pt")
model.eval()

# Load the same test image
img = cv2.imread("digit.png", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (28, 28))
img = img.astype("float32") / 255.0
x = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)  # shape (1,1,28,28)

# Warm-up
for _ in range(10):
    _ = model(x)

# Benchmark (1000 runs)
iters = 1000
t0 = time.perf_counter()
for _ in range(iters):
    _ = model(x)
t1 = time.perf_counter()

avg_ms = (t1 - t0) * 1000 / iters
print(f"Python CPU inference avg: {avg_ms:.3f} ms")

# Prediction
out = model(x)
pred = out.argmax(1).item()
print("Predicted digit:", pred)
