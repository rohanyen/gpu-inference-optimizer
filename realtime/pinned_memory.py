import torch
import torchvision.models as models
import numpy as np
import time

print("="*50)
print("PINNED MEMORY OPTIMIZATION")
print("="*50)

model = models.resnet50(weights='DEFAULT').cuda().eval()
MEAN = torch.tensor([0.485, 0.456, 0.406]).cuda().view(3, 1, 1)
STD  = torch.tensor([0.229, 0.224, 0.225]).cuda().view(3, 1, 1)

N = 200
images_np = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(N)]

# Method 1: Regular memory (pageable)
print("Method 1: Regular (pageable) memory...")
latencies_regular = []
with torch.no_grad():
    for i in range(N):
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        img = torch.from_numpy(images_np[i]).float().cuda()
        img = img.permute(2,0,1).unsqueeze(0) / 255.0
        img = (img - MEAN) / STD
        _ = model(img)
        end.record()
        torch.cuda.synchronize()
        latencies_regular.append(start.elapsed_time(end))

# Method 2: Pinned memory (page-locked)
print("Method 2: Pinned memory...")
latencies_pinned = []
with torch.no_grad():
    for i in range(N):
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        img = torch.from_numpy(images_np[i]).float()
        img = img.pin_memory().cuda(non_blocking=True)
        img = img.permute(2,0,1).unsqueeze(0) / 255.0
        img = (img - MEAN) / STD
        _ = model(img)
        end.record()
        torch.cuda.synchronize()
        latencies_pinned.append(start.elapsed_time(end))

latencies_regular = np.array(latencies_regular)
latencies_pinned  = np.array(latencies_pinned)

print(f"\n{'='*65}")
print(f"{'Metric':<22} {'Pageable':>14} {'Pinned':>14} {'Speedup':>10}")
print(f"{'-'*65}")
print(f"{'Average (ms)':<22} {latencies_regular.mean():>14.2f} {latencies_pinned.mean():>14.2f} {latencies_regular.mean()/latencies_pinned.mean():>9.2f}x")
print(f"{'Std Dev (ms)':<22} {latencies_regular.std():>14.2f} {latencies_pinned.std():>14.2f}")
print(f"{'99th %ile (ms)':<22} {np.percentile(latencies_regular,99):>14.2f} {np.percentile(latencies_pinned,99):>14.2f}")
print(f"{'='*65}")

np.save('C:/gpu_optimizer/results/pinned_latencies.npy', latencies_pinned)
print("\nResults saved.")
