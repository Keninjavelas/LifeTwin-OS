package com.lifetwin.mlp.sensors

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import com.lifetwin.mlp.db.AppEventEntity
import com.lifetwin.mlp.db.DBHelper
import android.util.Log

/**
 * Lightweight SensorFusionManager stub.
 * - Registers accelerometer (and other) listeners on start()
 * - Performs a tiny, conservative fusion to detect simple motion and emits events
 *   via `DBHelper` so the rest of the pipeline can be exercised.
 */
class SensorFusionManager(private val context: Context) : SensorEventListener {
    private var sensorManager: SensorManager? = null
    private var accelerometer: Sensor? = null

    fun start() {
        sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as? SensorManager
        accelerometer = sensorManager?.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        accelerometer?.let { sensorManager?.registerListener(this, it, SensorManager.SENSOR_DELAY_NORMAL) }
        Log.i("SensorFusionManager", "started")
    }

    fun stop() {
        sensorManager?.unregisterListener(this)
        Log.i("SensorFusionManager", "stopped")
    }

    override fun onSensorChanged(event: SensorEvent?) {
        if (event?.sensor?.type != Sensor.TYPE_ACCELEROMETER) return
        val ax = event.values[0]
        val ay = event.values[1]
        val az = event.values[2]
        val magnitude = Math.sqrt((ax * ax + ay * ay + az * az).toDouble())

        // Very simple thresholding fusion: emit 'moving' if magnitude exceeds 1.5g
        if (magnitude > 15.0) {
            val e = AppEventEntity(timestamp = System.currentTimeMillis(), type = "motion", packageName = "moving")
            DBHelper.insertEventAsync(context, e)
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
        // no-op
    }
}
