package com.example.ghost_radar

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder
import androidx.core.app.NotificationCompat

/**
 * Ghost Radar v2.0.2 - Foreground Service
 *
 * Keeps the app alive when the screen turns off and the user is in the field
 * doing clinical review. Android 8+ requires a foreground service for
 * long-running camera + microphone capture so the OS doesn't kill the process.
 *
 * Started/stopped from MainActivity via Intent (METHOD startForeground/stopForeground).
 * Notification updates from Dart via MainActivity -> Intent (METHOD updateNotification)
 * with extras "tracks" (int) and "logs" (int) to show live count.
 *
 * Channel: "ghost_radar_scan" (importance LOW - no sound)
 * Notification id: 1001
 */
class GhostRadarService : Service() {

    companion object {
        const val CHANNEL_ID = "ghost_radar_scan"
        const val CHANNEL_NAME = "Ghost Radar scan"
        const val NOTIFICATION_ID = 1001
        const val ACTION_START = "com.example.ghost_radar.START"
        const val ACTION_STOP = "com.example.ghost_radar.STOP"
        const val ACTION_UPDATE = "com.example.ghost_radar.UPDATE"
        const val EXTRA_TRACKS = "tracks"
        const val EXTRA_LOGS = "logs"

        fun start(ctx: Context) {
            val intent = Intent(ctx, GhostRadarService::class.java).setAction(ACTION_START)
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                ctx.startForegroundService(intent)
            } else {
                ctx.startService(intent)
            }
        }

        fun stop(ctx: Context) {
            val intent = Intent(ctx, GhostRadarService::class.java).setAction(ACTION_STOP)
            ctx.startService(intent)
        }

        fun update(ctx: Context, tracks: Int, logs: Int) {
            val intent = Intent(ctx, GhostRadarService::class.java)
                .setAction(ACTION_UPDATE)
                .putExtra(EXTRA_TRACKS, tracks)
                .putExtra(EXTRA_LOGS, logs)
            ctx.startService(intent)
        }
    }

    private var trackCount: Int = 0
    private var logCount: Int = 0

    override fun onCreate() {
        super.onCreate()
        ensureChannel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_STOP -> {
                stopForeground(STOP_FOREGROUND_REMOVE)
                stopSelf()
                return START_NOT_STICKY
            }
            ACTION_UPDATE -> {
                trackCount = intent.getIntExtra(EXTRA_TRACKS, trackCount)
                logCount = intent.getIntExtra(EXTRA_LOGS, logCount)
                pushNotification()
                return START_STICKY
            }
            else -> {
                // ACTION_START (or null) -> promote to foreground
                trackCount = intent?.getIntExtra(EXTRA_TRACKS, 0) ?: 0
                logCount = intent?.getIntExtra(EXTRA_LOGS, 0) ?: 0
                promoteToForeground()
            }
        }
        return START_STICKY
    }

    override fun onBind(intent: Intent?): IBinder? = null

    private fun ensureChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val mgr = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            if (mgr.getNotificationChannel(CHANNEL_ID) == null) {
                val channel = NotificationChannel(
                    CHANNEL_ID, CHANNEL_NAME, NotificationManager.IMPORTANCE_LOW
                ).apply {
                    description = "Giữ Ghost Radar chạy nền khi screen off"
                    setSound(null, null)
                    enableVibration(false)
                    setShowBadge(false)
                }
                mgr.createNotificationChannel(channel)
            }
        }
    }

    private fun buildNotification(): Notification {
        // Tap notification -> open MainActivity
        val openIntent = Intent(this, MainActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_SINGLE_TOP or Intent.FLAG_ACTIVITY_CLEAR_TOP
        }
        val piFlags = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        } else {
            PendingIntent.FLAG_UPDATE_CURRENT
        }
        val openPi = PendingIntent.getActivity(this, 0, openIntent, piFlags)

        val title = "👻 Ghost Radar đang quét"
        val text = "0-20Hz + YOLOv8 · $trackCount tracks · $logCount logs"

        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(android.R.drawable.ic_menu_view)
            .setContentTitle(title)
            .setContentText(text)
            .setStyle(NotificationCompat.BigTextStyle().bigText(text))
            .setOngoing(true)
            .setSilent(true)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .setContentIntent(openPi)
            .setCategory(NotificationCompat.CATEGORY_SERVICE)
            .build()
    }

    private fun pushNotification() {
        val mgr = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        mgr.notify(NOTIFICATION_ID, buildNotification())
    }

    private fun promoteToForeground() {
        val notification = buildNotification()
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            // Android 14+ requires explicit foregroundServiceType per use-case
            val type = ServiceInfo.FOREGROUND_SERVICE_TYPE_CAMERA or
                ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE
            startForeground(NOTIFICATION_ID, notification, type)
        } else {
            startForeground(NOTIFICATION_ID, notification)
        }
    }
}
