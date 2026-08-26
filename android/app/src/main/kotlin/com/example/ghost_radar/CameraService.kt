package com.example.ghost_radar

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.util.Log
import android.util.Size
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import io.flutter.plugin.common.EventChannel
import io.flutter.plugin.common.MethodChannel
import io.flutter.view.TextureRegistry
import java.nio.ByteBuffer
import java.util.concurrent.Executors

/**
 * CameraX-based camera service (v2.0+).
 * Replaces the Flutter `camera` package.
 * - Owns CameraX lifecycle
 * - Provides PreviewView for rendering (via TextureEntry from Flutter)
 * - Streams YUV frames to Dart via EventChannel for inference
 */
class CameraService(
    private val context: Context,
    private val lifecycleOwner: LifecycleOwner
) {
    companion object {
        private const val TAG = "CameraService"
        const val METHOD_CHANNEL = "ghost_radar/camera"
        const val EVENT_CHANNEL_FRAMES = "ghost_radar/camera_frames"
    }

    private var cameraProvider: ProcessCameraProvider? = null
    private val yuvProcessor = YuvProcessor()
    private val analysisExecutor = Executors.newSingleThreadExecutor()

    // Sinks
    private var frameSink: EventChannel.EventSink? = null
    private var previewTextureEntry: TextureRegistry.SurfaceTextureEntry? = null
    private var previewView: PreviewView? = null

    /**
     * Bind to a Flutter TextureRegistry.SurfaceTextureEntry (for Flutter rendering).
     * The PreviewView is then attached to the texture's Surface for live preview.
     */
    fun start(
        textureEntry: TextureRegistry.SurfaceTextureEntry,
        onError: (String) -> Unit
    ) {
        previewTextureEntry = textureEntry
        val previewView = PreviewView(context).apply {
            implementationMode = PreviewView.ImplementationMode.PERFORMANCE
            scaleType = PreviewView.ScaleType.FILL_CENTER
        }
        this.previewView = previewView

        // CameraX provider
        val providerFuture = ProcessCameraProvider.getInstance(context)
        providerFuture.addListener({
            try {
                val provider = providerFuture.get()
                cameraProvider = provider

                // Bind CameraX preview output to the Flutter SurfaceTexture
                val surfaceTexture = textureEntry.surfaceTexture()
                val preview = Preview.Builder()
                    .setTargetResolution(Size(640, 480))
                    .build()
                preview.setSurfaceProvider { request ->
                    val surface = android.view.Surface(surfaceTexture)
                    request.provideSurface(surface, ContextCompat.getMainExecutor(context)) { result ->
                        surface.release()
                    }
                }

                val analysis = ImageAnalysis.Builder()
                    .setTargetResolution(Size(640, 480))
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()
                analysis.setAnalyzer(analysisExecutor, YuvAnalyzer { frame ->
                    frameSink?.success(frame)
                })

                provider.unbindAll()
                provider.bindToLifecycle(
                    lifecycleOwner,
                    CameraSelector.DEFAULT_BACK_CAMERA,
                    preview,
                    analysis
                )
                Log.i(TAG, "CameraX started")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to start CameraX: ${e.message}", e)
                onError(e.message ?: "CameraX init failed")
            }
        }, ContextCompat.getMainExecutor(context))
    }

    fun stop() {
        try {
            cameraProvider?.unbindAll()
            cameraProvider = null
            previewView = null
        } catch (e: Exception) {
            Log.e(TAG, "Error stopping camera: ${e.message}")
        }
    }

    fun setFrameSink(sink: EventChannel.EventSink?) {
        frameSink = sink
    }

    fun release() {
        stop()
        setFrameSink(null)
        analysisExecutor.shutdown()
        previewTextureEntry?.release()
        previewTextureEntry = null
    }

    /**
     * ImageAnalysis.Analyzer that converts ImageProxy (YUV_420_888) to a
     * Flutter-friendly frame payload and dispatches to the sink.
     */
    private inner class YuvAnalyzer(
        private val onFrame: (Map<String, Any>) -> Unit
    ) : ImageAnalysis.Analyzer {

        @SuppressLint("UnsafeOptInUsageError")
        override fun analyze(image: ImageProxy) {
            try {
                val planes = image.planes
                if (planes.size < 3) {
                    image.close()
                    return
                }
                val yPlane = planes[0]
                val uPlane = planes[1]
                val vPlane = planes[2]

                val width = image.width
                val height = image.height

                // Frame payload: 3 ByteBuffers + metadata
                val frame = mapOf(
                    "width" to width,
                    "height" to height,
                    "yRowStride" to yPlane.rowStride,
                    "uvRowStride" to uPlane.rowStride,
                    "uvPixelStride" to uPlane.pixelStride,
                    "y" to yPlane.buffer,
                    "u" to uPlane.buffer,
                    "v" to vPlane.buffer,
                    "rotation" to image.imageInfo.rotationDegrees
                )
                onFrame(frame)
            } catch (e: Exception) {
                Log.e(TAG, "Analyzer error: ${e.message}")
            } finally {
                image.close()
            }
        }
    }
}
