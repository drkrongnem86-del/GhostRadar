package com.example.ghost_radar

import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.EventChannel
import io.flutter.plugin.common.MethodChannel
import io.flutter.view.TextureRegistry

/**
 * Ghost Radar v2.0 - MainActivity
 *
 * Bridges Flutter with native CameraX + native C++ preprocessing.
 *
 * MethodChannel `ghost_radar/camera`:
 *   - createTexture(): Long - create SurfaceTexture, return textureId
 *   - startCamera(textureId: Long)
 *   - stopCamera()
 *   - disposeTexture(textureId: Long)
 *
 * EventChannel `ghost_radar/camera_frames`:
 *   - Streams YUV frame payload to Dart: {width, height, y, u, v, strides, rotation}
 */
class MainActivity : FlutterActivity() {
    private var cameraService: CameraService? = null
    private val textureEntries = mutableMapOf<Long, TextureRegistry.SurfaceTextureEntry>()

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)

        val messenger = flutterEngine.dartExecutor.binaryMessenger

        // MethodChannel for camera control
        MethodChannel(messenger, CameraService.METHOD_CHANNEL)
            .setMethodCallHandler { call, result ->
                when (call.method) {
                    "createTexture" -> {
                        try {
                            val entry = flutterEngine.renderer.createSurfaceTexture()
                            textureEntries[entry.id()] = entry
                            result.success(entry.id())
                        } catch (e: Exception) {
                            result.error("CREATE_TEXTURE_FAIL", e.message, null)
                        }
                    }
                    "startCamera" -> {
                        val textureId = call.argument<Number>("textureId")?.toLong()
                        if (textureId == null) {
                            result.error("BAD_ARGS", "textureId required", null)
                            return@setMethodCallHandler
                        }
                        val entry = textureEntries[textureId]
                        if (entry == null) {
                            result.error("TEXTURE_NOT_FOUND", "Texture $textureId not found", null)
                            return@setMethodCallHandler
                        }
                        if (cameraService == null) {
                            cameraService = CameraService(this, this)
                        }
                        cameraService?.start(entry) { err ->
                            result.error("CAMERA_FAIL", err, null)
                        }
                        result.success(true)
                    }
                    "stopCamera" -> {
                        cameraService?.stop()
                        result.success(true)
                    }
                    "disposeTexture" -> {
                        val textureId = call.argument<Number>("textureId")?.toLong()
                        if (textureId != null) {
                            textureEntries.remove(textureId)?.release()
                        }
                        result.success(true)
                    }
                    "preprocess" -> {
                        // Optional: Dart can call back for preprocessing if not using EventChannel
                        // Currently not used (EventChannel preferred)
                        result.notImplemented()
                    }
                    "letterboxParams" -> {
                        val w = call.argument<Int>("width") ?: 0
                        val h = call.argument<Int>("height") ?: 0
                        val s = call.argument<Int>("inputSize") ?: 640
                        val proc = YuvProcessor()
                        val p = proc.letterboxParams(w, h, s)
                        result.success(mapOf(
                            "padX" to p.padX,
                            "padY" to p.padY,
                            "scale" to p.scale.toDouble()
                        ))
                    }
                    else -> result.notImplemented()
                }
            }

        // EventChannel for YUV frame stream
        EventChannel(messenger, CameraService.EVENT_CHANNEL_FRAMES)
            .setStreamHandler(object : EventChannel.StreamHandler {
                override fun onListen(arguments: Any?, events: EventChannel.EventSink?) {
                    if (cameraService == null) {
                        cameraService = CameraService(this@MainActivity, this@MainActivity)
                    }
                    cameraService?.setFrameSink(events)
                }

                override fun onCancel(arguments: Any?) {
                    cameraService?.setFrameSink(null)
                }
            })
    }

    override fun onDestroy() {
        cameraService?.release()
        cameraService = null
        textureEntries.values.forEach { it.release() }
        textureEntries.clear()
        super.onDestroy()
    }
}
