package com.lifetwin.mlp.performance

import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import android.os.BatteryManager
import android.util.Log
import androidx.work.*
import com.lifetwin.mlp.db.*
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import java.util.concurrent.TimeUnit

private const val TAG = "BatteryOptimizer"

/**
 * Battery and resource optimization system
 * - Adds WorkManager constraints for charging and Wi-Fi
 * - Implements memory pressure handling and data structure optimization
 * - Adds intelligent scheduling for intensive processing
 * - Provides adaptive resource management based on system conditions
 */
class BatteryOptimizer(private val context: Context) {
    
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    
    // Battery and resource state
    private val _batteryState = MutableStateFlow(BatteryState())
    val batteryState: StateFlow<BatteryState> = _batteryState.asStateFlow()
    
    private val _resourceState = MutableStateFlow(ResourceState())
    val resourceState: StateFlow<ResourceState> = _resourceState.asStateFlow()
    
    // Optimization settings
    private val _optimizationSettings = MutableStateFlow(OptimizationSettings())
    val optimizationSettings: StateFlow<OptimizationSettings> = _optimizationSettings.asStateFlow()
    
    // Work manager for scheduling
    private val workManager = WorkManager.getInstance(context)
    
    init {
        startResourceMonitoring()
    }

    /**
     * Starts continuous resource monitoring
     */
    private fun startResourceMonitoring() {
        scope.launch {
            while (isActive) {
                try {
                    updateBatteryState()
                    updateResourceState()
                    updateOptimizationSettings()
                    delay(30000) // Update every 30 seconds
                } catch (e: Exception) {
                    Log.e(TAG, "Error in resource monitoring", e)
                    delay(60000) // Wait longer on error
                }
            }
        }
    }

    /**
     * Updates current battery state
     */
    private suspend fun updateBatteryState() {
        val batteryIntent = context.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        
        val level = batteryIntent?.getIntExtra(BatteryManager.EXTRA_LEVEL, -1) ?: -1
        val scale = batteryIntent?.getIntExtra(BatteryManager.EXTRA_SCALE, -1) ?: -1
        val status = batteryIntent?.getIntExtra(BatteryManager.EXTRA_STATUS, -1) ?: -1
        val plugged = batteryIntent?.getIntExtra(BatteryManager.EXTRA_PLUGGED, -1) ?: -1
        val temperature = batteryIntent?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, -1) ?: -1
        
        val batteryLevel = if (level >= 0 && scale > 0) (level * 100 / scale) else 100
        val isCharging = status == BatteryManager.BATTERY_STATUS_CHARGING
        val isPlugged = plugged != 0
        val batteryTemperature = temperature / 10.0 // Convert from tenths of degrees
        
        val newState = BatteryState(
            level = batteryLevel,
            isCharging = isCharging,
            isPlugged = isPlugged,
            temperature = batteryTemperature,
            timestamp = System.currentTimeMillis()
        )
        
        _batteryState.value = newState
        
        // Log significant battery changes
        val oldState = _batteryState.value
        if (kotlin.math.abs(oldState.level - batteryLevel) >= 5 || oldState.isCharging != isCharging) {
            Log.i(TAG, "Battery state changed: level=$batteryLevel%, charging=$isCharging, plugged=$isPlugged")
        }
    }

    /**
     * Updates current resource state
     */
    private suspend fun updateResourceState() {
        val runtime = Runtime.getRuntime()
        val totalMemory = runtime.totalMemory()
        val freeMemory = runtime.freeMemory()
        val maxMemory = runtime.maxMemory()
        val usedMemory = totalMemory - freeMemory
        
        val memoryUsagePercent = (usedMemory.toDouble() / maxMemory.toDouble()) * 100
        val isWifiConnected = isWifiConnected()
        val isOnMeteredConnection = isOnMeteredConnection()
        
        val newState = ResourceState(
            memoryUsagePercent = memoryUsagePercent,
            totalMemoryMB = (totalMemory / (1024 * 1024)).toDouble(),
            usedMemoryMB = (usedMemory / (1024 * 1024)).toDouble(),
            freeMemoryMB = (freeMemory / (1024 * 1024)).toDouble(),
            maxMemoryMB = (maxMemory / (1024 * 1024)).toDouble(),
            isWifiConnected = isWifiConnected,
            isOnMeteredConnection = isOnMeteredConnection,
            timestamp = System.currentTimeMillis()
        )
        
        _resourceState.value = newState
    }

    /**
     * Updates optimization settings based on current conditions
     */
    private suspend fun updateOptimizationSettings() {
        val battery = _batteryState.value
        val resources = _resourceState.value
        
        val newSettings = OptimizationSettings(
            enableBatteryOptimization = battery.level < 30,
            enableMemoryOptimization = resources.memoryUsagePercent > 80,
            enableNetworkOptimization = resources.isOnMeteredConnection,
            preferWifiForUploads = !resources.isWifiConnected && resources.isOnMeteredConnection,
            requireChargingForIntensiveWork = battery.level < 20 && !battery.isCharging,
            reducedCollectionFrequency = battery.level < 15,
            batchSizeMultiplier = calculateBatchSizeMultiplier(battery, resources),
            collectionFrequencyMultiplier = calculateCollectionFrequencyMultiplier(battery, resources),
            timestamp = System.currentTimeMillis()
        )
        
        _optimizationSettings.value = newSettings
    }

    /**
     * Schedules intensive work with appropriate constraints
     */
    fun scheduleIntensiveWork(
        workName: String,
        workClass: Class<out ListenableWorker>,
        inputData: Data = Data.EMPTY,
        requireCharging: Boolean = true,
        requireWifi: Boolean = false,
        requireBatteryNotLow: Boolean = true
    ): Operation {
        
        val constraints = Constraints.Builder().apply {
            if (requireCharging) {
                setRequiresCharging(true)
            }
            if (requireWifi) {
                setRequiredNetworkType(NetworkType.UNMETERED)
            } else {
                setRequiredNetworkType(NetworkType.CONNECTED)
            }
            if (requireBatteryNotLow) {
                setRequiresBatteryNotLow(true)
            }
            setRequiresStorageNotLow(true)
        }.build()
        
        val workRequest = OneTimeWorkRequestBuilder(workClass)
            .setConstraints(constraints)
            .setInputData(inputData)
            .setBackoffCriteria(
                BackoffPolicy.EXPONENTIAL,
                WorkRequest.MIN_BACKOFF_MILLIS,
                TimeUnit.MILLISECONDS
            )
            .build()
        
        Log.i(TAG, "Scheduling intensive work: $workName with constraints")
        return workManager.enqueueUniqueWork(
            workName,
            ExistingWorkPolicy.REPLACE,
            workRequest
        )
    }

    /**
     * Schedules periodic work with adaptive constraints
     */
    fun schedulePeriodicWork(
        workName: String,
        workClass: Class<out ListenableWorker>,
        repeatInterval: Long = 15,
        repeatIntervalTimeUnit: TimeUnit = TimeUnit.MINUTES,
        inputData: Data = Data.EMPTY
    ): Operation {
        
        val battery = _batteryState.value
        val resources = _resourceState.value
        
        // Adjust repeat interval based on battery level
        val adjustedInterval = when {
            battery.level < 10 -> repeatInterval * 4 // Much less frequent
            battery.level < 20 -> repeatInterval * 2 // Less frequent
            battery.level < 50 -> (repeatInterval * 1.5).toLong() // Slightly less frequent
            else -> repeatInterval // Normal frequency
        }
        
        val constraints = Constraints.Builder().apply {
            if (battery.level < 15) {
                setRequiresCharging(true)
            }
            if (resources.isOnMeteredConnection) {
                setRequiredNetworkType(NetworkType.UNMETERED)
            } else {
                setRequiredNetworkType(NetworkType.CONNECTED)
            }
            setRequiresBatteryNotLow(true)
            setRequiresStorageNotLow(true)
        }.build()
        
        val workRequest = PeriodicWorkRequestBuilder(workClass, adjustedInterval, repeatIntervalTimeUnit)
            .setConstraints(constraints)
            .setInputData(inputData)
            .setBackoffCriteria(
                BackoffPolicy.EXPONENTIAL,
                WorkRequest.MIN_BACKOFF_MILLIS,
                TimeUnit.MILLISECONDS
            )
            .build()
        
        Log.i(TAG, "Scheduling periodic work: $workName with interval ${adjustedInterval}${repeatIntervalTimeUnit.name}")
        return workManager.enqueueUniquePeriodicWork(
            workName,
            ExistingPeriodicWorkPolicy.REPLACE,
            workRequest
        )
    }

    /**
     * Optimizes memory usage by triggering garbage collection and cleanup
     */
    suspend fun optimizeMemoryUsage() {
        try {
            val beforeMemory = getMemoryUsage()
            
            // Suggest garbage collection
            System.gc()
            
            // Clear any cached data if memory pressure is high
            if (beforeMemory.usagePercent > 85) {
                clearCaches()
            }
            
            // Wait a moment for GC to complete
            delay(1000)
            
            val afterMemory = getMemoryUsage()
            val memoryFreed = beforeMemory.usedMB - afterMemory.usedMB
            
            Log.i(TAG, "Memory optimization completed: freed ${memoryFreed}MB")
            
            // Log the optimization event
            logOptimizationEvent("MEMORY_OPTIMIZATION", mapOf(
                "beforeUsageMB" to beforeMemory.usedMB,
                "afterUsageMB" to afterMemory.usedMB,
                "memoryFreedMB" to memoryFreed,
                "beforeUsagePercent" to beforeMemory.usagePercent,
                "afterUsagePercent" to afterMemory.usagePercent
            ))
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to optimize memory usage", e)
        }
    }

    /**
     * Implements intelligent batching based on current conditions
     */
    fun getOptimalBatchSize(baseBatchSize: Int, operationType: String): Int {
        val settings = _optimizationSettings.value
        val battery = _batteryState.value
        val resources = _resourceState.value
        
        var multiplier = settings.batchSizeMultiplier
        
        // Adjust based on specific conditions
        when {
            battery.level < 10 -> multiplier *= 0.5 // Smaller batches to reduce processing
            battery.level < 20 -> multiplier *= 0.7
            resources.memoryUsagePercent > 90 -> multiplier *= 0.6 // Smaller batches for memory pressure
            resources.memoryUsagePercent > 80 -> multiplier *= 0.8
            battery.isCharging && resources.memoryUsagePercent < 50 -> multiplier *= 1.5 // Larger batches when conditions are good
        }
        
        val optimalSize = (baseBatchSize * multiplier).toInt().coerceIn(1, baseBatchSize * 3)
        
        Log.d(TAG, "Optimal batch size for $operationType: $optimalSize (base: $baseBatchSize, multiplier: $multiplier)")
        return optimalSize
    }

    /**
     * Gets the optimal collection frequency multiplier
     */
    fun getOptimalCollectionFrequency(): Double {
        return _optimizationSettings.value.collectionFrequencyMultiplier
    }

    /**
     * Checks if intensive operations should be deferred
     */
    fun shouldDeferIntensiveOperations(): Boolean {
        val battery = _batteryState.value
        val resources = _resourceState.value
        
        return when {
            battery.level < 10 -> true
            battery.level < 20 && !battery.isCharging -> true
            resources.memoryUsagePercent > 90 -> true
            battery.temperature > 40.0 -> true // Battery too hot
            else -> false
        }
    }

    /**
     * Gets current optimization recommendations
     */
    suspend fun getOptimizationRecommendations(): List<OptimizationRecommendation> {
        val battery = _batteryState.value
        val resources = _resourceState.value
        val recommendations = mutableListOf<OptimizationRecommendation>()
        
        // Battery recommendations
        if (battery.level < 10) {
            recommendations.add(OptimizationRecommendation(
                type = OptimizationType.BATTERY,
                priority = RecommendationPriority.CRITICAL,
                title = "Critical Battery Level",
                description = "Battery level is critically low (${battery.level}%)",
                action = "Enabling emergency power mode and deferring all non-essential operations",
                estimatedImpact = "Extends battery life by 2-4 hours"
            ))
        } else if (battery.level < 20) {
            recommendations.add(OptimizationRecommendation(
                type = OptimizationType.BATTERY,
                priority = RecommendationPriority.HIGH,
                title = "Low Battery Level",
                description = "Battery level is low (${battery.level}%)",
                action = "Reducing collection frequency and batch sizes",
                estimatedImpact = "Extends battery life by 30-60 minutes"
            ))
        }
        
        // Memory recommendations
        if (resources.memoryUsagePercent > 90) {
            recommendations.add(OptimizationRecommendation(
                type = OptimizationType.MEMORY,
                priority = RecommendationPriority.HIGH,
                title = "Critical Memory Usage",
                description = "Memory usage is critically high (${resources.memoryUsagePercent.toInt()}%)",
                action = "Triggering garbage collection and clearing caches",
                estimatedImpact = "Frees 10-50MB of memory"
            ))
        } else if (resources.memoryUsagePercent > 80) {
            recommendations.add(OptimizationRecommendation(
                type = OptimizationType.MEMORY,
                priority = RecommendationPriority.MEDIUM,
                title = "High Memory Usage",
                description = "Memory usage is high (${resources.memoryUsagePercent.toInt()}%)",
                action = "Reducing batch sizes and optimizing data structures",
                estimatedImpact = "Reduces memory pressure by 10-20%"
            ))
        }
        
        // Network recommendations
        if (resources.isOnMeteredConnection) {
            recommendations.add(OptimizationRecommendation(
                type = OptimizationType.NETWORK,
                priority = RecommendationPriority.MEDIUM,
                title = "Metered Connection Detected",
                description = "Device is on a metered network connection",
                action = "Deferring uploads until Wi-Fi is available",
                estimatedImpact = "Saves mobile data usage"
            ))
        }
        
        return recommendations
    }

    // Helper methods

    private fun isWifiConnected(): Boolean {
        return try {
            val connectivityManager = context.getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager
            val network = connectivityManager.activeNetwork
            val capabilities = connectivityManager.getNetworkCapabilities(network)
            capabilities?.hasTransport(NetworkCapabilities.TRANSPORT_WIFI) == true
        } catch (e: Exception) {
            Log.w(TAG, "Failed to check Wi-Fi connection", e)
            false
        }
    }

    private fun isOnMeteredConnection(): Boolean {
        return try {
            val connectivityManager = context.getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager
            val network = connectivityManager.activeNetwork
            val capabilities = connectivityManager.getNetworkCapabilities(network)
            !connectivityManager.isActiveNetworkMetered.not()
        } catch (e: Exception) {
            Log.w(TAG, "Failed to check metered connection", e)
            false
        }
    }

    private fun calculateBatchSizeMultiplier(battery: BatteryState, resources: ResourceState): Double {
        return when {
            battery.level < 10 -> 0.3
            battery.level < 20 -> 0.5
            resources.memoryUsagePercent > 90 -> 0.4
            resources.memoryUsagePercent > 80 -> 0.7
            battery.isCharging && resources.memoryUsagePercent < 50 -> 1.5
            else -> 1.0
        }
    }

    private fun calculateCollectionFrequencyMultiplier(battery: BatteryState, resources: ResourceState): Double {
        return when {
            battery.level < 10 -> 0.2
            battery.level < 20 -> 0.5
            battery.level < 50 -> 0.8
            resources.memoryUsagePercent > 90 -> 0.6
            else -> 1.0
        }
    }

    private suspend fun clearCaches() {
        try {
            // Clear any in-memory caches
            // This would be implemented based on specific cache implementations
            Log.i(TAG, "Clearing caches to free memory")
        } catch (e: Exception) {
            Log.w(TAG, "Failed to clear caches", e)
        }
    }

    private fun getMemoryUsage(): MemoryUsageSnapshot {
        val runtime = Runtime.getRuntime()
        val totalMemory = runtime.totalMemory()
        val freeMemory = runtime.freeMemory()
        val maxMemory = runtime.maxMemory()
        val usedMemory = totalMemory - freeMemory
        
        return MemoryUsageSnapshot(
            usedMB = (usedMemory / (1024 * 1024)).toDouble(),
            totalMB = (totalMemory / (1024 * 1024)).toDouble(),
            maxMB = (maxMemory / (1024 * 1024)).toDouble(),
            usagePercent = (usedMemory.toDouble() / maxMemory.toDouble()) * 100
        )
    }

    private suspend fun logOptimizationEvent(eventType: String, details: Map<String, Any>) {
        try {
            val database = AppDatabase.getInstance(context)
            val auditEntry = AuditLogEntity(
                id = java.util.UUID.randomUUID().toString(),
                timestamp = System.currentTimeMillis(),
                eventType = "OPTIMIZATION_$eventType",
                details = com.google.gson.Gson().toJson(details),
                userId = null
            )
            
            database.auditLogDao().insert(auditEntry)
        } catch (e: Exception) {
            Log.w(TAG, "Failed to log optimization event", e)
        }
    }

    fun cleanup() {
        scope.cancel()
    }

    // Data classes

    data class BatteryState(
        val level: Int = 100,
        val isCharging: Boolean = false,
        val isPlugged: Boolean = false,
        val temperature: Double = 25.0,
        val timestamp: Long = System.currentTimeMillis()
    )

    data class ResourceState(
        val memoryUsagePercent: Double = 0.0,
        val totalMemoryMB: Double = 0.0,
        val usedMemoryMB: Double = 0.0,
        val freeMemoryMB: Double = 0.0,
        val maxMemoryMB: Double = 0.0,
        val isWifiConnected: Boolean = false,
        val isOnMeteredConnection: Boolean = false,
        val timestamp: Long = System.currentTimeMillis()
    )

    data class OptimizationSettings(
        val enableBatteryOptimization: Boolean = false,
        val enableMemoryOptimization: Boolean = false,
        val enableNetworkOptimization: Boolean = false,
        val preferWifiForUploads: Boolean = false,
        val requireChargingForIntensiveWork: Boolean = false,
        val reducedCollectionFrequency: Boolean = false,
        val batchSizeMultiplier: Double = 1.0,
        val collectionFrequencyMultiplier: Double = 1.0,
        val timestamp: Long = System.currentTimeMillis()
    )

    data class OptimizationRecommendation(
        val type: OptimizationType,
        val priority: RecommendationPriority,
        val title: String,
        val description: String,
        val action: String,
        val estimatedImpact: String
    )

    data class MemoryUsageSnapshot(
        val usedMB: Double,
        val totalMB: Double,
        val maxMB: Double,
        val usagePercent: Double
    )

    enum class OptimizationType {
        BATTERY, MEMORY, NETWORK, STORAGE, CPU
    }

    enum class RecommendationPriority {
        LOW, MEDIUM, HIGH, CRITICAL
    }
}