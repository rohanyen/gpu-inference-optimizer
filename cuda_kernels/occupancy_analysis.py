import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np

print("="*60)
print("PHASE 2: OCCUPANCY ANALYSIS")
print("="*60)

kernel_code = """
__global__ void preprocess_v2(
    const unsigned char* input, float* output,
    int in_h, int in_w, int out_h, int out_w)
{
    __shared__ float tile_r[32][32];
    __shared__ float tile_g[32][32];
    __shared__ float tile_b[32][32];

    int out_x = blockIdx.x * 16 + threadIdx.x;
    int out_y = blockIdx.y * 16 + threadIdx.y;

    float scale_x = (float)in_w / out_w;
    float scale_y = (float)in_h / out_h;
    int in_base_x = (int)(blockIdx.x * 16 * scale_x);
    int in_base_y = (int)(blockIdx.y * 16 * scale_y);

    for(int dy = 0; dy < 2; dy++) {
        for(int dx = 0; dx < 2; dx++) {
            int ty = threadIdx.y * 2 + dy;
            int tx = threadIdx.x * 2 + dx;
            if(ty < 32 && tx < 32) {
                int gy = min(in_base_y + ty, in_h-1);
                int gx = min(in_base_x + tx, in_w-1);
                int idx = (gy * in_w + gx) * 3;
                tile_r[ty][tx] = input[idx+0];
                tile_g[ty][tx] = input[idx+1];
                tile_b[ty][tx] = input[idx+2];
            }
        }
    }
    __syncthreads();

    if(out_x >= out_w || out_y >= out_h) return;

    float in_x = out_x * scale_x - in_base_x;
    float in_y = out_y * scale_y - in_base_y;
    int x0=min((int)in_x,30), y0=min((int)in_y,30);
    int x1=x0+1, y1=y0+1;
    float dx=in_x-x0, dy2=in_y-y0;
    float mean[3]={0.485f,0.456f,0.406f};
    float std_v[3]={0.229f,0.224f,0.225f};

    float r=tile_r[y0][x0]*(1-dx)*(1-dy2)+tile_r[y0][x1]*dx*(1-dy2)
           +tile_r[y1][x0]*(1-dx)*dy2+tile_r[y1][x1]*dx*dy2;
    float g=tile_g[y0][x0]*(1-dx)*(1-dy2)+tile_g[y0][x1]*dx*(1-dy2)
           +tile_g[y1][x0]*(1-dx)*dy2+tile_g[y1][x1]*dx*dy2;
    float b=tile_b[y0][x0]*(1-dx)*(1-dy2)+tile_b[y0][x1]*dx*(1-dy2)
           +tile_b[y1][x0]*(1-dx)*dy2+tile_b[y1][x1]*dx*dy2;

    output[0*out_h*out_w+out_y*out_w+out_x]=((r/255.0f)-mean[0])/std_v[0];
    output[1*out_h*out_w+out_y*out_w+out_x]=((g/255.0f)-mean[1])/std_v[1];
    output[2*out_h*out_w+out_y*out_w+out_x]=((b/255.0f)-mean[2])/std_v[2];
}
"""

mod = SourceModule(kernel_code)
fn  = mod.get_function("preprocess_v2")

# GTX 1650 specs
device = cuda.Device(0)
props  = device.get_attributes()

max_threads    = props[cuda.device_attribute.MAX_THREADS_PER_BLOCK]
max_shared     = props[cuda.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK]
warp_size      = props[cuda.device_attribute.WARP_SIZE]
max_regs       = props[cuda.device_attribute.MAX_REGISTERS_PER_BLOCK]
sm_count       = props[cuda.device_attribute.MULTIPROCESSOR_COUNT]
max_threads_sm = props[cuda.device_attribute.MAX_THREADS_PER_MULTIPROCESSOR]

threads_per_block = 16 * 16  # 256
shared_per_block  = 3 * 32 * 32 * 4  # 3 channels * 32*32 * float32
blocks_per_sm     = min(
    max_threads_sm // threads_per_block,
    max_shared // shared_per_block
)
occupancy = (blocks_per_sm * threads_per_block) / max_threads_sm * 100

print(f"\n{'='*60}")
print(f"GTX 1650 HARDWARE SPECS")
print(f"{'='*60}")
print(f"Streaming Multiprocessors:  {sm_count}")
print(f"Max threads per SM:         {max_threads_sm}")
print(f"Max shared memory per block:{max_shared//1024} KB")
print(f"Warp size:                  {warp_size}")
print(f"Max registers per block:    {max_regs}")

print(f"\n{'='*60}")
print(f"KERNEL CONFIGURATION ANALYSIS")
print(f"{'='*60}")
print(f"Threads per block:          {threads_per_block} (16x16)")
print(f"Shared memory per block:    {shared_per_block//1024} KB")
print(f"Warps per block:            {threads_per_block//warp_size}")
print(f"Blocks per SM:              {blocks_per_sm}")
print(f"Active threads per SM:      {blocks_per_sm * threads_per_block}")
print(f"Theoretical occupancy:      {occupancy:.1f}%")

print(f"\n{'='*60}")
print(f"OCCUPANCY INTERPRETATION")
print(f"{'='*60}")
if occupancy >= 75:
    print(f"  ? EXCELLENT ({occupancy:.1f}%) - GPU well utilized")
elif occupancy >= 50:
    print(f"  ? GOOD ({occupancy:.1f}%) - Acceptable utilization")
elif occupancy >= 25:
    print(f"  ??  MODERATE ({occupancy:.1f}%) - Room for improvement")
else:
    print(f"  ? LOW ({occupancy:.1f}%) - GPU underutilized")

print(f"\nWhy 16x16 blocks?")
print(f"  - 256 threads = 8 warps per block")
print(f"  - Fits in shared memory budget")
print(f"  - Good balance of parallelism vs overhead")
print(f"  - Standard choice for 2D image kernels")

np.save('C:/gpu_optimizer/results/occupancy.npy',
        np.array([occupancy, blocks_per_sm, sm_count]))
print("\nResults saved.")
