package com.example.ghost_radar

import android.util.Log
import java.nio.ByteBuffer

/**
 * Native C++ JNI bridge for YUV preprocessing.
 * YUV_420_888 -> RGB BT.601 -> letterbox -> normalize [0,1] NCHW float32
 * See: android/app/src/main/cpp/yuv_processor.cpp
 */
class YuvProcessor {
    init {
        try {
            System.loadLibrary("ghost_radar_native")
        } catch (e: UnsatisfiedLinkError) {
            Log.e(TAG, "Failed to load native lib: ${e.message}")
        }
    }

    data class LetterboxParams(val padX: Int, val padY: Int, val scale: Float)

    /**
     * Run YUV preprocessing on a single frame.
     * @param yPlane Y plane (YUV_420_888, full resolution)
     * @param uPlane U plane (half resolution, CB in BT.601)
     * @param vPlane V plane (half resolution, CR in BT.601)
     * @param yRowStride bytes per row in Y plane
     * @param uvRowStride bytes per row in U/V planes
     * @param uvPixelStride bytes per pixel in U/V planes (usually 1 or 2)
     * @param width image width
     * @param height image height
     * @param inputSize target model size (e.g. 640)
     * @param outTensor output buffer (Float32, size 3*inputSize*inputSize)
     */
    fun preprocess(
        yPlane: ByteBuffer,
        uPlane: ByteBuffer,
        vPlane: ByteBuffer,
        yRowStride: Int,
        uvRowStride: Int,
        uvPixelStride: Int,
        width: Int,
        height: Int,
        inputSize: Int,
        outTensor: ByteBuffer
    ) {
        if (yPlane.isDirect && uPlane.isDirect && vPlane.isDirect && outTensor.isDirect) {
            nativePreprocess(
                yPlane, uPlane, vPlane,
                yRowStride, uvRowStride, uvPixelStride,
                width, height, inputSize, outTensor
            )
        } else {
            // Fallback: copy to direct buffers (slower but works)
            val yDirect = if (yPlane.isDirect) yPlane else yPlane.duplicate()
            val uDirect = if (uPlane.isDirect) uPlane else uPlane.duplicate()
            val vDirect = if (vPlane.isDirect) vPlane else vPlane.duplicate()
            val outDirect = if (outTensor.isDirect) outTensor else outTensor.duplicate()
            nativePreprocess(
                yDirect, uDirect, vDirect,
                yRowStride, uvRowStride, uvPixelStride,
                width, height, inputSize, outDirect
            )
        }
    }

    fun letterboxParams(width: Int, height: Int, inputSize: Int): LetterboxParams {
        val arr = IntArray(3)
        nativeLetterboxParams(width, height, inputSize, arr)
        return LetterboxParams(arr[0], arr[1], arr[2] / 1000.0f)
    }

    private external fun nativePreprocess(
        yPlane: ByteBuffer,
        uPlane: ByteBuffer,
        vPlane: ByteBuffer,
        yRowStride: Int,
        uvRowStride: Int,
        uvPixelStride: Int,
        width: Int,
        height: Int,
        inputSize: Int,
        outTensor: ByteBuffer
    )

    private external fun nativeLetterboxParams(
        width: Int,
        height: Int,
        inputSize: Int,
        outParams: IntArray
    )

    companion object {
        private const val TAG = "YuvProcessor"
    }
}
