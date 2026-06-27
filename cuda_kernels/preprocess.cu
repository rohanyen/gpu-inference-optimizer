#include <cuda_runtime.h>
#include <stdio.h>

// Custom CUDA kernel: resize + normalize image
// Input: uint8 RGB image (H x W x 3)
// Output: float32 normalized tensor (3 x 224 x 224)
__global__ void preprocess_kernel(
    const unsigned char* input,
    float* output,
    int in_h, int in_w,
    int out_h, int out_w,
    float mean_r, float mean_g, float mean_b,
    float std_r,  float std_g,  float std_b)
{
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (out_x >= out_w || out_y >= out_h) return;

    // Bilinear interpolation
    float scale_x = (float)in_w / out_w;
    float scale_y = (float)in_h / out_h;

    float in_x = out_x * scale_x;
    float in_y = out_y * scale_y;

    int x0 = (int)in_x;
    int y0 = (int)in_y;
    int x1 = min(x0 + 1, in_w - 1);
    int y1 = min(y0 + 1, in_h - 1);

    float dx = in_x - x0;
    float dy = in_y - y0;

    // Read pixels (HWC format)
    for (int c = 0; c < 3; c++) {
        float p00 = input[(y0 * in_w + x0) * 3 + c];
        float p01 = input[(y0 * in_w + x1) * 3 + c];
        float p10 = input[(y1 * in_w + x0) * 3 + c];
        float p11 = input[(y1 * in_w + x1) * 3 + c];

        float val = p00 * (1-dx) * (1-dy)
                  + p01 * dx * (1-dy)
                  + p10 * (1-dx) * dy
                  + p11 * dx * dy;

        // Normalize: (val/255 - mean) / std
        val = val / 255.0f;
        float mean, std;
        if (c == 0) { mean = mean_r; std = std_r; }
        else if (c == 1) { mean = mean_g; std = std_g; }
        else { mean = mean_b; std = std_b; }
        val = (val - mean) / std;

        // Output in CHW format
        output[c * out_h * out_w + out_y * out_w + out_x] = val;
    }
}

int main() {
    printf("CUDA Preprocessing Kernel compiled successfully\n");
    printf("Kernel: bilinear resize + normalize\n");
    printf("Input:  uint8 HWC -> Output: float32 CHW\n");
    return 0;
}
