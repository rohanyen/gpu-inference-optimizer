import torch
import torchvision.models as models
import time
import numpy as np

print("="*50)
print("GPU INFERENCE BASELINE MEASUREMENT")
print("="*50)

print("\nLoading ResNet-50...")
model = models.resnet50(weights='DEFAULT')
model = model.cuda()
model.eval()

print("Warming up GPU...")
dummy = torch.randn(1, 3, 224, 224).cuda()
with torch.no_grad():
    for _ in range(10):
        _ = model(dummy)

print("Measuring baseline latency (100 runs)...")
latencies = []
with torch.no_grad():
    for i in range(100):
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        _ = model(dummy)
        end.record()
        torch.cuda.synchronize()
        latencies.append(start.elapsed_time(end))

latencies = np.array(latencies)
print(f"\n{'='*50}")
print(f"BASELINE RESULTS (ResNet-50, batch=1)")
print(f"{'='*50}")
print(f"Average latency:    {latencies.mean():.2f} ms")
print(f"Min latency:        {latencies.min():.2f} ms")
print(f"Max latency:        {latencies.max():.2f} ms")
print(f"Std deviation:      {latencies.std():.2f} ms")
print(f"95th percentile:    {np.percentile(latencies, 95):.2f} ms")
print(f"99th percentile:    {np.percentile(latencies, 99):.2f} ms")
print(f"{'='*50}")
print(f"GPU Memory used:    {torch.cuda.memory_allocated()//1024**2} MB")
print(f"GPU Memory cached:  {torch.cuda.memory_reserved()//1024**2} MB")

np.save('C:/gpu_optimizer/results/baseline_latencies.npy', latencies)
print(f"\nResults saved.")
