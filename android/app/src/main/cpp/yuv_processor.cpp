// Ghost Radar v2.0 - YUV_420_888 to NCHW Float32 preprocessor (native C++)
// Pipeline: YUV (NV12-like from CameraX) -> RGB BT.601 -> letterbox resize -> normalize [0,1]
// Output: Float32 NCHW tensor [1, 3, S, S] for YOLOv8 input
//
// All JNI functions use direct ByteBuffer for zero-copy (where possible).
// YUV layout from CameraX ImageProxy: 3 planes, Y + U/2 + V/2 with rowStride/pixelStride.

#include <jni.h>
#include <cmath>
#include <cstring>
#include <android/log.h>

#define LOG_TAG "GhostRadarNative"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace {

// Clamp helper
inline float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// BT.601 limited range YUV -> RGB (fast integer math, scale 0-255)
// Y in [0, 255], U/V in [16, 240]
inline void yuv2rgb(int y, int u, int v, uint8_t& r, uint8_t& g, uint8_t& b) {
    int c = y - 16;
    int d = u - 128;
    int e = v - 128;
    // ITU-R BT.601 limited range
    int rI = (298 * c + 409 * e + 128) >> 8;
    int gI = (298 * c - 100 * d - 208 * e + 128) >> 8;
    int bI = (298 * c + 516 * d + 128) >> 8;
    r = (uint8_t)clampf((float)rI, 0.0f, 255.0f);
    g = (uint8_t)clampf((float)gI, 0.0f, 255.0f);
    b = (uint8_t)clampf((float)bI, 0.0f, 255.0f);
}

} // namespace

extern "C" {

// JNI: preprocess YUV frame into NCHW Float32 tensor
// Args:
//   yPlane, uPlane, vPlane: ByteBuffer of each plane
//   yRowStride, uvRowStride, uvPixelStride: int
//   width, height: int (image dimensions in pixels)
//   inputSize: int (target model size, e.g. 640)
//   outTensor: ByteBuffer for output (Float32, capacity >= 3*inputSize*inputSize*4 bytes)
JNIEXPORT void JNICALL
Java_com_example_ghost_1radar_YuvProcessor_nativePreprocess(
    JNIEnv* env,
    jclass /* clazz */,
    jobject yPlane,
    jobject uPlane,
    jobject vPlane,
    jint yRowStride,
    jint uvRowStride,
    jint uvPixelStride,
    jint width,
    jint height,
    jint inputSize,
    jobject outTensor
) {
    // Get direct ByteBuffer pointers (zero-copy)
    const uint8_t* yBuf = (const uint8_t*)env->GetDirectBufferAddress(yPlane);
    const uint8_t* uBuf = (const uint8_t*)env->GetDirectBufferAddress(uPlane);
    const uint8_t* vBuf = (const uint8_t*)env->GetDirectBufferAddress(vPlane);
    float* outBuf = (float*)env->GetDirectBufferAddress(outTensor);

    if (!yBuf || !uBuf || !vBuf || !outBuf) {
        LOGE("Failed to get direct buffer address (y=%p u=%p v=%p out=%p)",
             yBuf, uBuf, vBuf, outBuf);
        return;
    }

    const int S = inputSize;
    const int outSize = 3 * S * S;

    // Compute letterbox scale (preserve aspect ratio, pad to square)
    const float scale = fminf((float)S / (float)width, (float)S / (float)height);
    const int newW = (int)((float)width * scale);
    const int newH = (int)((float)height * scale);
    const int padX = (S - newW) / 2;
    const int padY = (S - newH) / 2;

    // Pre-fill output with 0.5 (gray for padded area) - normalize to [0,1]
    // Using memset on the full buffer is wrong because it's float, but zeroing
    // 0.0 then overwriting 0.5 only for valid pixels is fine.
    for (int i = 0; i < outSize; i++) {
        outBuf[i] = 0.5f; // normalized gray (128/255)
    }

    // For each output pixel, map to source (sx, sy) in original image
    // NCHW layout: outBuf[c * S * S + y * S + x] = normalized RGB
    const float inv255 = 1.0f / 255.0f;

    for (int oy = 0; oy < newH; oy++) {
        const int sy = (int)floorf((float)oy / scale);
        if (sy < 0 || sy >= height) continue;
        for (int ox = 0; ox < newW; ox++) {
            const int sx = (int)floorf((float)ox / scale);
            if (sx < 0 || sx >= width) continue;
            // Y plane: full resolution
            const int yIdx = sy * yRowStride + sx;
            // U/V planes: half resolution (YUV_420), with pixel stride
            const int uvIdx = (sy / 2) * uvRowStride + (sx / 2) * uvPixelStride;
            const int yVal = (int)yBuf[yIdx];
            const int uVal = (int)uBuf[uvIdx];
            const int vVal = (int)vBuf[uvIdx];
            uint8_t r, g, b;
            yuv2rgb(yVal, uVal, vVal, r, g, b);
            // NCHW: output index
            const int dx = padX + ox;
            const int dy = padY + oy;
            const int base = dy * S + dx;
            outBuf[0 * S * S + base] = (float)r * inv255;
            outBuf[1 * S * S + base] = (float)g * inv255;
            outBuf[2 * S * S + base] = (float)b * inv255;
        }
    }
}

// JNI: letterbox params (for un-letterbox when decoding boxes)
JNIEXPORT void JNICALL
Java_com_example_ghost_1radar_YuvProcessor_nativeLetterboxParams(
    JNIEnv* env,
    jclass /* clazz */,
    jint width,
    jint height,
    jint inputSize,
    jintArray outParams  // [padX, padY, scale*1000]
) {
    const float scale = fminf((float)inputSize / (float)width, (float)inputSize / (float)height);
    const int newW = (int)((float)width * scale);
    const int newH = (int)((float)height * scale);
    const int padX = (inputSize - newW) / 2;
    const int padY = (inputSize - newH) / 2;
    jint buf[3] = { padX, padY, (int)(scale * 1000.0f) };
    env->SetIntArrayRegion(outParams, 0, 3, buf);
}

} // extern "C"
