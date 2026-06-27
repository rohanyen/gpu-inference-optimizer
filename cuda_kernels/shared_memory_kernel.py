import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import cv2
import time

print("="*60)
print("PHASE 2: SHARED MEMORY TILING KERNEL (FIXED)")
print("="*60)

kernel_v1 = """
__global__ void preprocess_v1(
    const unsigned char* input, float* output,
    int in_h, int in_w, int out_h, int out_w)
{
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (out_x >= out_w || out_y >= out_h) return;

    float scale_x = (float)in_w / out_w;
    float scale_y = (float)in_h / out_h;
    float in_x = out_x * scale_x;
    float in_y = out_y * scale_y;
    int x0=min((int)in_x, in_w-2);
    int y0=min((int)in_y, in_h-2);
    int x1=x0+1, y1=y0+1;
    float dx=in_x-x0, dy=in_y-y0;

    float mean[3]={0.485f,0.456f,0.406f};
    float std_v[3]={0.229f,0.224f,0.225f};

    for(int c=0;c<3;c++){
        float p00=input[(y0*in_w+x0)*3+c];
        float p01=input[(y0*in_w+x1)*3+c];
        float p10=input[(y1*in_w+x0)*3+c];
        float p11=input[(y1*in_w+x1)*3+c];
        float val=p00*(1-dx)*(1-dy)+p01*dx*(1-dy)
                 +p10*(1-dx)*dy+p11*dx*dy;
        output[c*out_h*out_w+out_y*out_w+out_x]=
            ((val/255.0f)-mean[c])/std_v[c];
    }
}
"""

kernel_v2 = """
#define BLOCK_W 16
#define BLOCK_H 16
#define TILE_W 32
#define TILE_H 32

__global__ void preprocess_v2(
    const unsigned char* input, float* output,
    int in_h, int in_w, int out_h, int out_w)
{
    __shared__ float tile_r[TILE_H][TILE_W];
    __shared__ float tile_g[TILE_H][TILE_W];
    __shared__ float tile_b[TILE_H][TILE_W];

    int out_x = blockIdx.x * BLOCK_W + threadIdx.x;
    int out_y = blockIdx.y * BLOCK_H + threadIdx.y;

    float scale_x = (float)in_w / out_w;
    float scale_y = (float)in_h / out_h;

    // Each thread loads 4 input pixels into shared memory
    int in_base_x = (int)(blockIdx.x * BLOCK_W * scale_x);
    int in_base_y = (int)(blockIdx.y * BLOCK_H * scale_y);

    // Load 2x2 region per thread into shared memory
    for(int dy = 0; dy < 2; dy++) {
        for(int dx = 0; dx < 2; dx++) {
            int ty = threadIdx.y * 2 + dy;
            int tx = threadIdx.x * 2 + dx;
            if(ty < TILE_H && tx < TILE_W) {
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

    // Read from shared memory instead of global
    float in_x = out_x * scale_x - in_base_x;
    float in_y = out_y * scale_y - in_base_y;

    int x0 = min((int)in_x, TILE_W-2);
    int y0 = min((int)in_y, TILE_H-2);
    int x1 = x0+1, y1 = y0+1;
    float dx = in_x-x0, dy2 = in_y-y0;

    float mean[3]={0.485f,0.456f,0.406f};
    float std_v[3]={0.229f,0.224f,0.225f};

    float channels[3][4];
    channels[0][0]=tile_r[y0][x0]; channels[0][1]=tile_r[y0][x1];
    channels[0][2]=tile_r[y1][x0]; channels[0][3]=tile_r[y1][x1];
    channels[1][0]=tile_g[y0][x0]; channels[1][1]=tile_g[y0][x1];
    channels[1][2]=tile_g[y1][x0]; channels[1][3]=tile_g[y1][x1];
    channels[2][0]=tile_b[y0][x0]; channels[2][1]=tile_b[y0][x1];
    channels[2][2]=tile_b[y1][x0]; channels[2][3]=tile_b[y1][x1];

    for(int c=0;c<3;c++){
        float val=channels[c][0]*(1-dx)*(1-dy2)
                 +channels[c][1]*dx*(1-dy2)
                 +channels[c][2]*(1-dx)*dy2
                 +channels[c][3]*dx*dy2;
        output[c*out_h*out_w+out_y*out_w+out_x]=
            ((val/255.0f)-mean[c])/std_v[c];
    }
}
"""

print("Compiling kernels...")
mod_v1 = SourceModule(kernel_v1)
fn_v1  = mod_v1.get_function("preprocess_v1")
mod_v2 = SourceModule(kernel_v2)
fn_v2  = mod_v2.get_function("preprocess_v2")
print("Compiled!")

in_h, in_w = 640, 640
out_h, out_w = 224, 224
img_np = np.random.randint(0, 255, (in_h, in_w, 3), dtype=np.uint8)
img_g  = cuda.mem_alloc(img_np.nbytes)
out_g  = cuda.mem_alloc(3 * out_h * out_w * 4)
cuda.memcpy_htod(img_g, img_np)

block = (16, 16, 1)
grid  = ((out_w+15)//16, (out_h+15)//16, 1)

# Warmup v1
print("Warming up v1...")
for _ in range(20):
    fn_v1(img_g, out_g,
          np.int32(in_h), np.int32(in_w),
          np.int32(out_h), np.int32(out_w),
          block=block, grid=grid)
cuda.Context.synchronize()

# Warmup v2
print("Warming up v2...")
for _ in range(20):
    fn_v2(img_g, out_g,
          np.int32(in_h), np.int32(in_w),
          np.int32(out_h), np.int32(out_w),
          block=block, grid=grid)
cuda.Context.synchronize()
print("Warmup done!")

N = 1000

# Benchmark v1
t0 = time.perf_counter()
for _ in range(N):
    fn_v1(img_g, out_g,
          np.int32(in_h), np.int32(in_w),
          np.int32(out_h), np.int32(out_w),
          block=block, grid=grid)
cuda.Context.synchronize()
time_v1 = (time.perf_counter()-t0)/N*1000

# Benchmark v2
t0 = time.perf_counter()
for _ in range(N):
    fn_v2(img_g, out_g,
          np.int32(in_h), np.int32(in_w),
          np.int32(out_h), np.int32(out_w),
          block=block, grid=grid)
cuda.Context.synchronize()
time_v2 = (time.perf_counter()-t0)/N*1000

# OpenCV
img_cpu = np.random.randint(0,255,(in_h,in_w,3),dtype=np.uint8)
t0 = time.perf_counter()
for _ in range(N):
    r = cv2.resize(img_cpu,(out_w,out_h))
    n = (r.astype(np.float32)/255.0-[0.485,0.456,0.406])/[0.229,0.224,0.225]
    _ = n.transpose(2,0,1)
time_cv = (time.perf_counter()-t0)/N*1000

print(f"\n{'='*65}")
print(f"PHASE 2: Shared Memory Tiling Results")
print(f"{'='*65}")
print(f"{'Method':<30} {'Latency':>10} {'vs OpenCV':>12} {'vs V1':>8}")
print(f"{'-'*65}")
print(f"{'OpenCV CPU':<30} {time_cv:>9.3f}ms {'1.00x':>12}")
print(f"{'V1 Global Memory':<30} {time_v1:>9.3f}ms {time_cv/time_v1:>11.2f}x {'1.00x':>8}")
print(f"{'V2 Shared Memory':<30} {time_v2:>9.3f}ms {time_cv/time_v2:>11.2f}x {time_v1/time_v2:>7.2f}x")
print(f"{'='*65}")

np.save('C:/gpu_optimizer/results/phase2_times.npy',
        np.array([time_cv, time_v1, time_v2]))
print("Results saved.")
