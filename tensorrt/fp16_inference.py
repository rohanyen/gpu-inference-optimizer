import torch
import torchvision.models as models
import numpy as np

print("="*50)
print("FP16 OPTIMIZATION")
print("="*50)

model = models.resnet50(weights='DEFAULT').cuda().eval()
model_fp16 = model.half()
dummy_fp16 = torch.randn(1, 3, 224, 224, dtype=torch.float16).cuda()

with torch.no_grad():
    for _ in range(20):
        _ = model_fp16(dummy_fp16)
torch.cuda.synchronize()

print("Measuring FP16 latency (100 runs)...")
latencies = []
with torch.no_grad():
    for i in range(100):
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        _ = model_fp16(dummy_fp16)
        end.record()
        torch.cuda.synchronize()
        latencies.append(start.elapsed_time(end))

latencies = np.array(latencies)
baseline_avg = 11.18
baseline_std = 3.89
baseline_p99 = 23.17

print(f"\n{'='*60}")
print(f"{'Metric':<20} {'FP32 Baseline':>15} {'FP16':>15} {'Speedup':>10}")
print(f"{'-'*60}")
print(f"{'Average (ms)':<20} {baseline_avg:>15.2f} {latencies.mean():>15.2f} {baseline_avg/latencies.mean():>9.2f}x")
print(f"{'Std Dev (ms)':<20} {baseline_std:>15.2f} {latencies.std():>15.2f} {baseline_std/latencies.std():>9.2f}x")
print(f"{'99th %ile (ms)':<20} {baseline_p99:>15.2f} {np.percentile(latencies,99):>15.2f} {baseline_p99/np.percentile(latencies,99):>9.2f}x")
print(f"{'='*60}")
print(f"GPU Memory: {torch.cuda.memory_allocated()//1024**2} MB")
np.save('C:/gpu_optimizer/results/fp16_latencies.npy', latencies)
print("Results saved.")
