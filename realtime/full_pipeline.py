import torch
import torchvision.models as models
import numpy as np
import time

print("="*50)
print("OPTIMIZED FULL PIPELINE")
print("="*50)

# Load compiled model
model = models.resnet50(weights='DEFAULT').cuda().eval()
model_compiled = torch.compile(model, mode='reduce-overhead', backend='eager')

MEAN = torch.tensor([0.485, 0.456, 0.406]).cuda().view(3, 1, 1)
STD  = torch.tensor([0.229, 0.224, 0.225]).cuda().view(3, 1, 1)

N = 200
images_np = [np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8) for _ in range(N)]

# Warmup
print("Warming up optimized pipeline...")
with torch.no_grad():
    for i in range(30):
        img = torch.from_numpy(images_np[i]).float().pin_memory().cuda(non_blocking=True)
        img = torch.nn.functional.interpolate(
            img.permute(2,0,1).unsqueeze(0)/255.0,
            size=(224,224), mode='bilinear', align_corners=False)
        img = (img - MEAN) / STD
        _ = model_compiled(img)
torch.cuda.synchronize()

# Measure optimized pipeline
print("Measuring optimized pipeline (200 runs)...")
latencies_opt = []
with torch.no_grad():
    for i in range(N):
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()

        # Pinned memory transfer
        img = torch.from_numpy(images_np[i]).float().pin_memory().cuda(non_blocking=True)
        # CUDA resize + normalize
        img = torch.nn.functional.interpolate(
            img.permute(2,0,1).unsqueeze(0)/255.0,
            size=(224,224), mode='bilinear', align_corners=False)
        img = (img - MEAN) / STD
        # Compiled inference
        _ = model_compiled(img)

        end.record()
        torch.cuda.synchronize()
        latencies_opt.append(start.elapsed_time(end))

latencies_opt = np.array(latencies_opt)

# Baseline numbers
baseline_avg = 11.18
baseline_std = 3.89
baseline_p99 = 23.17

print(f"\n{'='*65}")
print(f"FINAL COMPARISON: Baseline vs Fully Optimized Pipeline")
print(f"{'='*65}")
print(f"{'Metric':<22} {'FP32 Baseline':>15} {'Optimized':>15} {'Speedup':>10}")
print(f"{'-'*65}")
print(f"{'Average (ms)':<22} {baseline_avg:>15.2f} {latencies_opt.mean():>15.2f} {baseline_avg/latencies_opt.mean():>9.2f}x")
print(f"{'Std Dev (ms)':<22} {baseline_std:>15.2f} {latencies_opt.std():>15.2f} {baseline_std/latencies_opt.std():>9.2f}x")
print(f"{'Min (ms)':<22} {7.93:>15.2f} {latencies_opt.min():>15.2f}")
print(f"{'Max (ms)':<22} {24.16:>15.2f} {latencies_opt.max():>15.2f}")
print(f"{'99th %ile (ms)':<22} {baseline_p99:>15.2f} {np.percentile(latencies_opt,99):>15.2f} {baseline_p99/np.percentile(latencies_opt,99):>9.2f}x")
print(f"{'='*65}")

# Real-time guarantee check
p99 = np.percentile(latencies_opt, 99)
p999 = np.percentile(latencies_opt, 99.9) if N >= 100 else latencies_opt.max()
print(f"\nREAL-TIME GUARANTEE ANALYSIS:")
print(f"  99th percentile:   {p99:.2f}ms")
print(f"  Max latency:       {latencies_opt.max():.2f}ms")
print(f"  Latency budget:    50ms")
print(f"  Guarantee:         {'? PASSES' if latencies_opt.max() < 50 else '? FAILS'} (<50ms)")
print(f"  Tighter guarantee: {'? PASSES' if p99 < 20 else '? FAILS'} (99p < 20ms)")

np.save('C:/gpu_optimizer/results/optimized_latencies.npy', latencies_opt)
print("\nResults saved.")
