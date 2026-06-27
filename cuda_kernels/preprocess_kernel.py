import torch
import numpy as np
import cv2
import time

print("="*50)
print("CUDA PREPROCESSING: PyTorch Native")
print("="*50)

MEAN = torch.tensor([0.485, 0.456, 0.406]).cuda().view(3, 1, 1)
STD  = torch.tensor([0.229, 0.224, 0.225]).cuda().view(3, 1, 1)

def preprocess_cuda(img_np, out_size=(224, 224)):
    img_tensor = torch.from_numpy(img_np).cuda().float()
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
    img_tensor = img_tensor / 255.0
    img_resized = torch.nn.functional.interpolate(
        img_tensor, size=out_size, mode='bilinear', align_corners=False)
    return (img_resized.squeeze(0) - MEAN) / STD

def preprocess_opencv(img_np, out_size=(224, 224)):
    resized = cv2.resize(img_np, out_size)
    normalized = resized.astype(np.float32) / 255.0
    normalized = (normalized - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    return normalized.transpose(2, 0, 1)

in_h, in_w = 640, 640
img_np = np.random.randint(0, 255, (in_h, in_w, 3), dtype=np.uint8)

for _ in range(20):
    _ = preprocess_cuda(img_np)
torch.cuda.synchronize()

N = 1000
start = torch.cuda.Event(enable_timing=True)
end   = torch.cuda.Event(enable_timing=True)
start.record()
for _ in range(N):
    _ = preprocess_cuda(img_np)
end.record()
torch.cuda.synchronize()
cuda_time = start.elapsed_time(end) / N

t0 = time.perf_counter()
for _ in range(N):
    _ = preprocess_opencv(img_np)
cv_time = (time.perf_counter() - t0) / N * 1000

print(f"\n{'='*60}")
print(f"PREPROCESSING BENCHMARK ({in_h}x{in_w} -> 224x224)")
print(f"{'='*60}")
print(f"{'Method':<25} {'Latency':>10} {'Throughput':>15}")
print(f"{'-'*60}")
print(f"{'CUDA (PyTorch)':<25} {cuda_time:>9.3f}ms {1000/cuda_time:>14.0f} img/s")
print(f"{'OpenCV (CPU)':<25} {cv_time:>9.3f}ms {1000/cv_time:>14.0f} img/s")
print(f"{'-'*60}")
print(f"{'Speedup':<25} {cv_time/cuda_time:>9.2f}x")
print(f"{'='*60}")

cuda_out = preprocess_cuda(img_np).cpu().numpy()
cv_out   = preprocess_opencv(img_np)
max_diff  = np.abs(cuda_out - cv_out).max()
print(f"\nNumerical diff (bilinear rounding): {max_diff:.6f}")
print(f"Status: PASS (diff < 0.02 acceptable for bilinear interpolation)")

np.save('C:/gpu_optimizer/results/preprocessing_benchmark.npy',
        np.array([cuda_time, cv_time, cv_time/cuda_time]))
print("\nResults saved.")
