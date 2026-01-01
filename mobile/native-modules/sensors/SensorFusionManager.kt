package com.lifetwin.mlp.sensors

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.util.Log
import androidx.core.content.ContextCompat
import com.lifetwin.mlp.db.*
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import java.util.UUID
import java.util.concurrent.ConcurrentLinkedQueue
import kotlin.math.sqrt

private const val TAG = "SensorFusionManager"
private const val BATCH_SIZE = 50
private const val BATCH_INTERVAL_MS = 30000L // 30 seconds
private const val ACTIVITY_WINDOW_MS = 10000L // 10 seconds for activity detection

/**
 * Battery-optimized sensor fusion manager for activity detection
 * - Batches sensor readings to minimize battery drain
 * - Uses intelligent sampling rates based on detected activity
 * - Implements activity classification using accelerometer data
 * - Provides contextual features for ML model training
 */
class SensorFusionManager(private val context: Context) : SensorEventListener, com.lifetwin.mlp.db.SensorFusionManager {
    
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    
    private var sensorManager: SensorManager? = null
    private var accelerometer: Sensor? = null
    private var gyroscope: Sensor? = null
    private var magnetometer: Sensor? = null
    
    @Volatile
    private var isCollecting = false
    
    @Volatile
    private var currentSamplingRate = SensorManager.SENSOR_DELAY_NORMAL
    
    // Sensor data batching
    private val sensorDataBatch = ConcurrentLinkedQueue<SensorReading>()
    private var lastBatchTime = System.currentTimeMillis()
    
    // Activity detection state
    private var currentActivity: ActivityContext? = null
    private val activityWindow = ConcurrentLinkedQueue<AccelerometerReading>()
    private var lastActivityUpdate = System.currentTimeMillis()

    // Implementation of SensorFusionManager interface
    
    override suspend fun startCollection() {
        if (!isSensorPermissionGranted()) {
            Log.w(TAG, "Sensor permissions not granted")
            return
        }
        
        sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as? SensorManager
        accelerometer = sensorManager?.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        gyroscope = sensorManager?.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
        magnetometer = sensorManager?.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD)
        
        // Register listeners with battery-optimized sampling rate
        accelerometer?.let { sensor ->
            sensorManager?.registerListener(this, sensor, currentSamplingRate)
        }
        
        gyroscope?.let { sensor ->
            sensorManager?.registerListener(this, sensor, SensorManager.SENSOR_DELAY_UI)
        }
        
        magnetometer?.let { sensor ->
            sensorManager?.registerListener(this, sensor, SensorManager.SENSOR_DELAY_UI)
        }
        
        isCollecting = true
        Log.i(TAG, "Sensor collection started")
        
        scope.launch {
            logAuditEvent(AuditEventType.COLLECTOR_ENABLED, "Sensor fusion started")
        }
    }

    override suspend fun stopCollection() {
        sensorManager?.unregisterListener(this)
        isCollecting = false
        
        // Flush any remaining batched data
        flushSensorBatch()
        
        Log.i(TAG, "Sensor collection stopped")
        
        scope.launch {
            logAuditEvent(AuditEventType.COLLECTOR_DISABLED, "Sensor fusion stopped")
        }
    }

    override fun isCollectionActive(): Boolean = isCollecting

    override fun getCollectorType(): CollectorType = CollectorType.SENSORS

    override suspend fun getCollectedDataCount(): Int {
        return try {
            val database = AppDatabase.getInstance(context)
            val endTime = System.currentTimeMillis()
            val startTime = endTime - (24 * 60 * 60 * 1000L) // Last 24 hours
            
            val contexts = database.activityContextDao().getContextsByTimeRange(startTime, endTime)
            contexts.size
        } catch (e: Exception) {
            Log.e(TAG, "Failed to get collected data count", e)
            0
        }
    }

    override suspend fun getCurrentActivity(): ActivityContext? {
        return currentActivity
    }

    override suspend fun getActivityHistory(timeRange: TimeRange): List<ActivityContext> {
        return try {
            val database = AppDatabase.getInstance(context)
            val entities = database.activityContextDao().getContextsByTimeRange(
                timeRange.startTime,
                timeRange.endTime
            )
            
            entities.map { entity ->
                ActivityContext(
                    id = entity.id,
                    activityType = ActivityType.valueOf(entity.activityType),
                    confidence = entity.confidence,
                    timestamp = entity.timestamp,
                    duration = entity.duration
                )
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to get activity history", e)
            emptyList()
        }
    }

    override fun isSensorPermissionGranted(): Boolean {
        // Check for body sensors permission (required for some activity recognition)
        val bodySensorsPermission = ContextCompat.checkSelfPermission(
            context,
            Manifest.permission.BODY_SENSORS
        ) == PackageManager.PERMISSION_GRANTED
        
        // Check if required sensors are available
        val sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as? SensorManager
        val hasAccelerometer = sensorManager?.getDefaultSensor(Sensor.TYPE_ACCELEROMETER) != null
        val hasGyroscope = sensorManager?.getDefaultSensor(Sensor.TYPE_GYROSCOPE) != null
        val hasMagnetometer = sensorManager?.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD) != null
        
        // At minimum, we need accelerometer for basic activity detection
        return hasAccelerometer
    }

    override suspend fun requestSensorPermission(): Boolean {
        // Basic sensors (accelerometer, gyroscope, magnetometer) don't require runtime permissions
        // Body sensors permission is optional and would be handled by the calling activity
        Log.i(TAG, "Basic sensor permissions are automatically granted. Body sensors permission is optional.")
        
        // Check if sensors are available on the device
        val available = isSensorPermissionGranted()
        if (!available) {
            Log.w(TAG, "Required sensors are not available on this device")
            scope.launch {
                logAuditEvent(AuditEventType.PERMISSION_REVOKED, "Required sensors not available")
            }
        }
        
        return available
    }

    /**
     * Handles sensor permission changes gracefully by implementing fallback behavior
     */
    suspend fun handlePermissionChange(hasPermission: Boolean) {
        if (!hasPermission && isCollecting) {
            Log.w(TAG, "Sensor permissions lost, switching to reduced functionality mode")
            
            // Stop active sensor collection
            stopCollection()
            
            // Enable fallback mode with reduced functionality
            enableFallbackMode()
            
            scope.launch {
                logAuditEvent(AuditEventType.PERMISSION_REVOKED, "Sensor permissions lost, fallback mode enabled")
            }
        } else if (hasPermission && !isCollecting) {
            Log.i(TAG, "Sensor permissions restored, resuming normal operation")
            
            // Disable fallback mode
            disableFallbackMode()
            
            // Resume sensor collection
            startCollection()
            
            scope.launch {
                logAuditEvent(AuditEventType.PERMISSION_GRANTED, "Sensor permissions restored")
            }
        }
    }

    /**
     * Enables reduced functionality mode when sensors are unavailable
     */
    private suspend fun enableFallbackMode() {
        Log.i(TAG, "Enabling sensor fallback mode")
        
        // Store fallback activity context indicating limited functionality
        val fallbackActivity = ActivityContext(
            activityType = ActivityType.UNKNOWN,
            confidence = 0.1f,
            timestamp = System.currentTimeMillis(),
            duration = 0L
        )
        
        currentActivity = fallbackActivity
        
        // Log the fallback mode activation
        val database = AppDatabase.getInstance(context)
        val rawEvent = RawEventEntity(
            id = java.util.UUID.randomUUID().toString(),
            timestamp = System.currentTimeMillis(),
            eventType = "sensor_fallback",
            packageName = null,
            duration = null,
            metadata = DBHelper.encryptMetadata(
                com.google.gson.Gson().toJson(
                    mapOf(
                        "reason" to "sensor_unavailable",
                        "fallbackMode" to true,
                        "reducedFunctionality" to true
                    )
                )
            )
        )
        
        database.rawEventDao().insert(rawEvent)
    }

    /**
     * Disables fallback mode when sensors become available again
     */
    private suspend fun disableFallbackMode() {
        Log.i(TAG, "Disabling sensor fallback mode")
        
        // Clear fallback activity
        currentActivity = null
        
        // Log the fallback mode deactivation
        val database = AppDatabase.getInstance(context)
        val rawEvent = RawEventEntity(
            id = java.util.UUID.randomUUID().toString(),
            timestamp = System.currentTimeMillis(),
            eventType = "sensor_restored",
            packageName = null,
            duration = null,
            metadata = DBHelper.encryptMetadata(
                com.google.gson.Gson().toJson(
                    mapOf(
                        "reason" to "sensor_available",
                        "fallbackMode" to false,
                        "fullFunctionality" to true
                    )
                )
            )
        )
        
        database.rawEventDao().insert(rawEvent)
    }

    /**
     * Provides reduced functionality activity detection using alternative methods
     */
    suspend fun getFallbackActivityContext(): ActivityContext? {
        if (isSensorPermissionGranted()) {
            return null // No fallback needed
        }
        
        // Use screen events and usage patterns as fallback for basic activity inference
        try {
            val database = AppDatabase.getInstance(context)
            val currentTime = System.currentTimeMillis()
            val lookbackTime = currentTime - (5 * 60 * 1000L) // 5 minutes
            
            // Check recent screen activity
            val recentScreenSessions = database.screenSessionDao().getSessionsByTimeRange(lookbackTime, currentTime)
            val recentUsageEvents = database.usageEventDao().getEventsByTimeRange(lookbackTime, currentTime)
            
            val activityType = when {
                recentScreenSessions.isNotEmpty() && recentUsageEvents.isNotEmpty() -> {
                    // Active screen usage suggests user is stationary and engaged
                    ActivityType.STATIONARY
                }
                recentScreenSessions.isEmpty() && recentUsageEvents.isEmpty() -> {
                    // No recent activity suggests device is idle
                    ActivityType.UNKNOWN
                }
                else -> {
                    // Some activity but limited data
                    ActivityType.UNKNOWN
                }
            }
            
            return ActivityContext(
                activityType = activityType,
                confidence = 0.3f, // Low confidence for fallback detection
                timestamp = currentTime,
                duration = 0L
            )
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to get fallback activity context", e)
            return ActivityContext(
                activityType = ActivityType.UNKNOWN,
                confidence = 0.1f,
                timestamp = System.currentTimeMillis(),
                duration = 0L
            )
        }
    }

    // SensorEventListener implementation

    override fun onSensorChanged(event: SensorEvent?) {
        if (!isCollecting || event == null) return
        
        scope.launch {
            try {
                processSensorEvent(event)
            } catch (e: Exception) {
                Log.w(TAG, "Failed to process sensor event", e)
            }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
        Log.d(TAG, "Sensor accuracy changed: ${sensor?.name}, accuracy: $accuracy")
    }

    // Private methods for sensor processing

    private suspend fun processSensorEvent(event: SensorEvent) {
        val currentTime = System.currentTimeMillis()
        
        when (event.sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> {
                processAccelerometerData(event, currentTime)
            }
            Sensor.TYPE_GYROSCOPE -> {
                processGyroscopeData(event, currentTime)
            }
            Sensor.TYPE_MAGNETIC_FIELD -> {
                processMagnetometerData(event, currentTime)
            }
        }
        
        // Check if we need to flush the batch
        if (shouldFlushBatch(currentTime)) {
            flushSensorBatch()
        }
        
        // Check if we need to update activity detection
        if (shouldUpdateActivity(currentTime)) {
            updateActivityDetection(currentTime)
        }
    }

    private suspend fun processAccelerometerData(event: SensorEvent, timestamp: Long) {
        val ax = event.values[0]
        val ay = event.values[1]
        val az = event.values[2]
        val magnitude = sqrt(ax * ax + ay * ay + az * az)
        
        // Add to sensor batch
        val reading = SensorReading(
            sensorType = "accelerometer",
            timestamp = timestamp,
            values = floatArrayOf(ax, ay, az),
            magnitude = magnitude
        )
        sensorDataBatch.offer(reading)
        
        // Add to activity detection window
        val accelReading = AccelerometerReading(timestamp, ax, ay, az, magnitude)
        activityWindow.offer(accelReading)
        
        // Keep activity window size manageable
        while (activityWindow.size > 100) {
            activityWindow.poll()
        }
        
        // Legacy compatibility
        if (magnitude > 15.0) {
            DBHelper.insertEventAsync(
                context,
                AppEventEntity(
                    timestamp = timestamp,
                    type = "motion",
                    packageName = "moving"
                )
            )
        }
    }

    private suspend fun processGyroscopeData(event: SensorEvent, timestamp: Long) {
        val gx = event.values[0]
        val gy = event.values[1]
        val gz = event.values[2]
        val magnitude = sqrt(gx * gx + gy * gy + gz * gz)
        
        val reading = SensorReading(
            sensorType = "gyroscope",
            timestamp = timestamp,
            values = floatArrayOf(gx, gy, gz),
            magnitude = magnitude
        )
        sensorDataBatch.offer(reading)
    }

    private suspend fun processMagnetometerData(event: SensorEvent, timestamp: Long) {
        val mx = event.values[0]
        val my = event.values[1]
        val mz = event.values[2]
        val magnitude = sqrt(mx * mx + my * my + mz * mz)
        
        val reading = SensorReading(
            sensorType = "magnetometer",
            timestamp = timestamp,
            values = floatArrayOf(mx, my, mz),
            magnitude = magnitude
        )
        sensorDataBatch.offer(reading)
    }

    private fun shouldFlushBatch(currentTime: Long): Boolean {
        return sensorDataBatch.size >= BATCH_SIZE || 
               (currentTime - lastBatchTime) >= BATCH_INTERVAL_MS
    }

    private fun shouldUpdateActivity(currentTime: Long): Boolean {
        return (currentTime - lastActivityUpdate) >= ACTIVITY_WINDOW_MS
    }

    private suspend fun flushSensorBatch() {
        if (sensorDataBatch.isEmpty()) return
        
        try {
            val batchData = mutableListOf<SensorReading>()
            while (sensorDataBatch.isNotEmpty() && batchData.size < BATCH_SIZE) {
                sensorDataBatch.poll()?.let { batchData.add(it) }
            }
            
            if (batchData.isNotEmpty()) {
                storeSensorBatch(batchData)
                lastBatchTime = System.currentTimeMillis()
                
                Log.d(TAG, "Flushed sensor batch: ${batchData.size} readings")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to flush sensor batch", e)
        }
    }

    private suspend fun storeSensorBatch(batch: List<SensorReading>) {
        try {
            // Create aggregated sensor data for storage
            val batchStartTime = batch.minOf { it.timestamp }
            val batchEndTime = batch.maxOf { it.timestamp }
            
            // Group by sensor type and calculate statistics
            val sensorStats = batch.groupBy { it.sensorType }.mapValues { (_, readings) ->
                mapOf(
                    "count" to readings.size,
                    "avgMagnitude" to readings.map { it.magnitude }.average(),
                    "maxMagnitude" to readings.maxOf { it.magnitude },
                    "minMagnitude" to readings.minOf { it.magnitude }
                )
            }
            
            // Store as raw event with encrypted metadata
            val database = AppDatabase.getInstance(context)
            val rawEvent = RawEventEntity(
                id = UUID.randomUUID().toString(),
                timestamp = batchEndTime,
                eventType = "sensor_batch",
                packageName = null,
                duration = batchEndTime - batchStartTime,
                metadata = DBHelper.encryptMetadata(
                    com.google.gson.Gson().toJson(
                        mapOf(
                            "batchSize" to batch.size,
                            "sensorStats" to sensorStats,
                            "samplingRate" to currentSamplingRate
                        )
                    )
                )
            )
            
            database.rawEventDao().insert(rawEvent)
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to store sensor batch", e)
        }
    }

    private suspend fun updateActivityDetection(currentTime: Long) {
        try {
            val detectedActivity = classifyActivity()
            
            if (detectedActivity != null) {
                // Check if activity has changed
                val activityChanged = currentActivity?.activityType != detectedActivity.activityType ||
                                   (currentTime - lastActivityUpdate) > (5 * 60 * 1000L) // 5 minutes
                
                if (activityChanged) {
                    // End previous activity if exists
                    currentActivity?.let { prevActivity ->
                        val endedActivity = prevActivity.copy(
                            duration = currentTime - prevActivity.timestamp
                        )
                        storeActivityContext(endedActivity)
                    }
                    
                    // Start new activity
                    currentActivity = detectedActivity.copy(timestamp = currentTime)
                    lastActivityUpdate = currentTime
                    
                    Log.d(TAG, "Activity changed to: ${detectedActivity.activityType} (confidence: ${detectedActivity.confidence})")
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to update activity detection", e)
        }
    }

    private fun classifyActivity(): ActivityContext? {
        if (activityWindow.size < 10) return null
        
        val recentReadings = activityWindow.takeLast(50)
        
        // Calculate activity features
        val magnitudes = recentReadings.map { it.magnitude }
        val avgMagnitude = magnitudes.average()
        val stdMagnitude = calculateStandardDeviation(magnitudes)
        val maxMagnitude = magnitudes.maxOrNull() ?: 0.0
        
        // Simple activity classification based on accelerometer patterns
        val (activityType, confidence) = when {
            avgMagnitude < 9.5 && stdMagnitude < 1.0 -> {
                Pair(ActivityType.STATIONARY, 0.8f)
            }
            avgMagnitude in 9.5..11.0 && stdMagnitude < 2.0 -> {
                Pair(ActivityType.WALKING, 0.7f)
            }
            avgMagnitude > 11.0 && stdMagnitude > 2.0 -> {
                if (maxMagnitude > 15.0) {
                    Pair(ActivityType.RUNNING, 0.6f)
                } else {
                    Pair(ActivityType.WALKING, 0.6f)
                }
            }
            stdMagnitude > 3.0 -> {
                Pair(ActivityType.IN_VEHICLE, 0.5f)
            }
            else -> {
                Pair(ActivityType.UNKNOWN, 0.3f)
            }
        }
        
        return ActivityContext(
            activityType = activityType,
            confidence = confidence,
            timestamp = System.currentTimeMillis(),
            duration = 0L
        )
    }

    private fun calculateStandardDeviation(values: List<Double>): Double {
        if (values.isEmpty()) return 0.0
        
        val mean = values.average()
        val variance = values.map { (it - mean) * (it - mean) }.average()
        return sqrt(variance)
    }

    private suspend fun storeActivityContext(activityContext: ActivityContext) {
        try {
            val database = AppDatabase.getInstance(context)
            
            // Store in activity context table
            val entity = ActivityContextEntity(
                id = activityContext.id,
                activityType = activityContext.activityType.name,
                confidence = activityContext.confidence,
                timestamp = activityContext.timestamp,
                duration = activityContext.duration,
                sensorData = null // Don't store raw sensor data for privacy
            )
            
            database.activityContextDao().insert(entity)
            
            Log.d(TAG, "Stored activity context: ${activityContext.activityType} for ${activityContext.duration}ms")
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to store activity context", e)
        }
    }

    private suspend fun logAuditEvent(eventType: AuditEventType, details: String) {
        try {
            val database = AppDatabase.getInstance(context)
            val auditEntry = AuditLogEntity(
                timestamp = System.currentTimeMillis(),
                eventType = eventType.name,
                details = """{"component":"SensorFusionManager","details":"$details"}""",
                userId = null
            )
            
            database.auditLogDao().insert(auditEntry)
        } catch (e: Exception) {
            Log.w(TAG, "Failed to log audit event", e)
        }
    }

    // Data classes for sensor processing
    
    private data class SensorReading(
        val sensorType: String,
        val timestamp: Long,
        val values: FloatArray,
        val magnitude: Double
    )
    
    private data class AccelerometerReading(
        val timestamp: Long,
        val x: Float,
        val y: Float,
        val z: Float,
        val magnitude: Double
    )

    companion object {
        /**
         * Adjusts sampling rate based on detected activity for battery optimization
         */
        fun getOptimalSamplingRate(activityType: ActivityType): Int {
            return when (activityType) {
                ActivityType.STATIONARY -> SensorManager.SENSOR_DELAY_NORMAL
                ActivityType.WALKING -> SensorManager.SENSOR_DELAY_UI
                ActivityType.RUNNING -> SensorManager.SENSOR_DELAY_GAME
                ActivityType.IN_VEHICLE -> SensorManager.SENSOR_DELAY_NORMAL
                else -> SensorManager.SENSOR_DELAY_NORMAL
            }
        }
    }
}
