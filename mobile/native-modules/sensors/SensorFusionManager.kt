package com.lifetwin.mlp.sensors

// Stub manager that will periodically combine accelerometer, screen events, and
// other signals into higher-level activity features (e.g., moving, idle, walking).

import android.content.Context

class SensorFusionManager(private val context: Context) {

    fun start() {
        // TODO: register sensor listeners and schedule periodic fusion jobs
    }

    fun stop() {
        // TODO: unregister sensors
    }
}
