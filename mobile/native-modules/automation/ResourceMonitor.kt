package com.lifetwin.automation

import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.Debug
import android.os.Process
import android.app.ActivityManager
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*
import java.io.File
import java.io.RandomAccessFile
import kotlin.math.max

/**
 * ResourceMonitor - Comprehensive resource usage monitoring and adaptive behavior
 * 
 * Implements Requirements:
 * - 8.6: Self-monitoring of CPU, memory, and battery usage
 * - 8.7: Adaptive behavior based on resource constraints
 * - Battery usage statistics for user transparency
 */
class ResourceMonitor(
    private val context: Context
) {
    private val scope = CoroutineScope(Dispatchers.Default + SupervisorJob())
    private val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
    
    // Resource usage tracking
    private val _resourceUsage = MutableStateFlow(ResourceUsage())
    val resourceUsage: StateFlow<ResourceUsage> = _resourceUsage.asStateFlow()
    
    // Adaptive behavior state
    private val _adaptiveBehavior = MutableStateFlow(AdaptiveBehavior())
    val adaptiveBehavior: StateFlow<AdaptiveBehavior> = _adaptiveBehavior.asStateFlow()
    
    // Battery statistics
    private val batteryStats = mutableListOf<BatterySnapshot>()
    private var lastBatteryLevel = -1
    private var lastBatteryTime = 0L
    
    // CPU monitoring
    private var lastCpuTime = 0L
    private var lastAppCpuTime = 0L
    
    init {
        startResourceMonitoring()
    }
    
    /**
     * Start continuous resource monitoring
     */
    private fun startResourceMonitoring() {
        scope.launch {
            while (isActive) {
                try {
                    updateResourceUsage()
                    updateAdaptiveBehavior()
                    delay(5000) // Monitor every 5 seconds
                } catch (e: Exception) {
                    // Continue monitoring even if individual measurements fail
                    delay(10000) // Wait longer on error
                }
            }
        }
    }
    
    /**
     * Update current resource usage measurements
     */
    private suspend fun updateResourceUsage() = withContext(Dispatchers.IO) {
        val currentUsage = ResourceUsage(
            cpuUsagePercent = measureCpuUsage(),
            memoryUsageMB = measureMemoryUsage(),
            batteryLevel = measureBatteryLevel(),
            batteryTemperature = measureBatteryTemperature(),
            availableStorageMB = measureAvailableStorage(),
            networkUsageKB = measureNetworkUsage(),
            timestamp = System.currentTimeMillis()
        )
        
        _resourceUsage.value = currentUsage
        
        // Update battery statistics
        updateBatteryStatistics(currentUsage)
    }
    
    /**
     * Measure CPU usage percentage
     */
    private fun measureCpuUsage(): Double {
        return try {
            val currentTime = System.currentTimeMillis()
            val currentCpuTime = getTotalCpuTime()
            val currentAppCpuTime = getAppCpuTime()
            
            if (lastCpuTime > 0 && lastAppCpuTime > 0) {
                val totalCpuDelta = currentCpuTime - lastCpuTime
                val appCpuDelta = currentAppCpuTime - lastAppCpuTime
                
                val cpuUsage = if (totalCpuDelta > 0) {
                    (appCpuDelta.toDouble() / totalCpuDelta) * 100.0
                } else 0.0
                
                lastCpuTime = currentCpuTime
                lastAppCpuTime = currentAppCpuTime
                
                max(0.0, cpuUsage.coerceAtMost(100.0))
            } else {
                lastCpuTime = currentCpuTime
                lastAppCpuTime = currentAppCpuTime
                0.0
            }
        } catch (e: Exception) {
            0.0
        }
    }
    
    /**
     * Get total system CPU time from /proc/stat
     */
    private fun getTotalCpuTime(): Long {
        return try {
            val reader = RandomAccessFile("/proc/stat", "r")
            val load = reader.readLine()
            reader.close()
            
            val toks = load.split(" ")
            var totalTime = 0L
            for (i in 1..7) {
                if (i < toks.size) {
                    totalTime += toks[i].toLongOrNull() ?: 0L
                }
            }
            totalTime
        } catch (e: Exception) {
            0L
        }
    }
    
    /**
     * Get app-specific CPU time
     */
    private fun getAppCpuTime(): Long {
        return try {
            val pid = Process.myPid()
            val reader = RandomAccessFile("/proc/$pid/stat", "r")
            val load = reader.readLine()
            reader.close()
            
            val toks = load.split(" ")
            if (toks.size >= 15) {
                val utime = toks[13].toLongOrNull() ?: 0L
                val stime = toks[14].toLongOrNull() ?: 0L
                utime + stime
            } else 0L
        } catch (e: Exception) {
            0L
        }
    }
    
    /**
     * Measure memory usage in MB
     */
    private fun measureMemoryUsage(): Double {
        return try {
            val memInfo = ActivityManager.MemoryInfo()
            activityManager.getMemoryInfo(memInfo)
            
            val runtime = Runtime.getRuntime()
            val usedMemory = runtime.totalMemory() - runtime.freeMemory()
            
            usedMemory / (1024.0 * 1024.0) // Convert to MB
        } catch (e: Exception) {
            0.0
        }
    }
    
    /**
     * Measure current battery level
     */
    private fun measureBatteryLevel(): Int {
        return try {
            val batteryIntent = context.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
            val level = batteryIntent?.getIntExtra(BatteryManager.EXTRA_LEVEL, -1) ?: -1
            val scale = batteryIntent?.getIntExtra(BatteryManager.EXTRA_SCALE, -1) ?: -1
            
            if (level >= 0 && scale > 0) {
                (level * 100 / scale)
            } else -1
        } catch (e: Exception) {
            -1
        }
    }
    
    /**
     * Measure battery temperature
     */
    private fun measureBatteryTemperature(): Double {
        return try {
            val batteryIntent = context.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
            val temperature = batteryIntent?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, -1) ?: -1
            
            if (temperature > 0) {
                temperature / 10.0 // Convert from tenths of degrees
            } else -1.0
        } catch (e: Exception) {
            -1.0
        }
    }
    
    /**
     * Measure available storage in MB
     */
    private fun measureAvailableStorage(): Double {
        return try {
            val internalDir = context.filesDir
            val availableBytes = internalDir.freeSpace
            availableBytes / (1024.0 * 1024.0) // Convert to MB
        } catch (e: Exception) {
            0.0
        }
    }
    
    /**
     * Measure network usage (placeholder - requires more complex implementation)
     */
    private fun measureNetworkUsage(): Double {
        // Simplified implementation - would need TrafficStats for real measurement
        return 0.0
    }
    
    /**
     * Update battery usage statistics
     */
    private fun updateBatteryStatistics(usage: ResourceUsage) {
        val currentTime = System.currentTimeMillis()
        
        if (lastBatteryLevel >= 0 && lastBatteryTime > 0) {
            val timeDelta = currentTime - lastBatteryTime
            val levelDelta = lastBatteryLevel - usage.batteryLevel
            
            if (timeDelta > 0 && levelDelta > 0) {
                val drainRate = (levelDelta.toDouble() / timeDelta) * 3600000 // Per hour
                
                batteryStats.add(
                    BatterySnapshot(
                        timestamp = currentTime,
                        level = usage.batteryLevel,
                        temperature = usage.batteryTemperature,
                        drainRate = drainRate
                    )
                )
                
                // Keep only last 24 hours of data
                val cutoffTime = currentTime - 24 * 3600 * 1000
                batteryStats.removeAll { it.timestamp < cutoffTime }
            }
        }
        
        lastBatteryLevel = usage.batteryLevel
        lastBatteryTime = currentTime
    }
    
    /**
     * Update adaptive behavior based on resource constraints
     */
    private fun updateAdaptiveBehavior() {
        val usage = _resourceUsage.value
        val behavior = AdaptiveBehavior(
            processingFrequency = calculateProcessingFrequency(usage),
            batchSize = calculateOptimalBatchSize(usage),
            cacheSize = calculateOptimalCacheSize(usage),
            backgroundProcessing = shouldAllowBackgroundProcessing(usage),
            networkOperations = shouldAllowNetworkOperations(usage),
            recommendations = generateResourceRecommendations(usage)
        )
        
        _adaptiveBehavior.value = behavior
    }
    
    /**
     * Calculate optimal processing frequency based on resources
     */
    private fun calculateProcessingFrequency(usage: ResourceUsage): Double {
        var frequency = 1.0
        
        // Reduce frequency based on battery level
        when {
            usage.batteryLevel <= 10 -> frequency *= 0.1
            usage.batteryLevel <= 20 -> frequency *= 0.3
            usage.batteryLevel <= 30 -> frequency *= 0.5
        }
        
        // Reduce frequency based on CPU usage
        if (usage.cpuUsagePercent > 80) frequency *= 0.5
        if (usage.cpuUsagePercent > 90) frequency *= 0.3
        
        // Reduce frequency based on memory pressure
        if (usage.memoryUsageMB > 500) frequency *= 0.7
        if (usage.memoryUsageMB > 1000) frequency *= 0.4
        
        // Reduce frequency based on temperature
        if (usage.batteryTemperature > 40) frequency *= 0.6
        if (usage.batteryTemperature > 45) frequency *= 0.3
        
        return frequency.coerceAtLeast(0.05) // Minimum 5% frequency
    }
    
    /**
     * Calculate optimal batch size for operations
     */
    private fun calculateOptimalBatchSize(usage: ResourceUsage): Int {
        var batchSize = 50 // Default batch size
        
        // Reduce batch size under resource pressure
        if (usage.batteryLevel <= 20) batchSize /= 2
        if (usage.cpuUsagePercent > 80) batchSize /= 2
        if (usage.memoryUsageMB > 800) batchSize /= 2
        
        return batchSize.coerceAtLeast(5) // Minimum batch size
    }
    
    /**
     * Calculate optimal cache size
     */
    private fun calculateOptimalCacheSize(usage: ResourceUsage): Int {
        var cacheSize = 100 // Default cache entries
        
        // Reduce cache size based on memory usage
        if (usage.memoryUsageMB > 500) cacheSize /= 2
        if (usage.memoryUsageMB > 1000) cacheSize /= 4
        
        return cacheSize.coerceAtLeast(10) // Minimum cache size
    }
    
    /**
     * Determine if background processing should be allowed
     */
    private fun shouldAllowBackgroundProcessing(usage: ResourceUsage): Boolean {
        return usage.batteryLevel > 15 && 
               usage.cpuUsagePercent < 85 && 
               usage.batteryTemperature < 42
    }
    
    /**
     * Determine if network operations should be allowed
     */
    private fun shouldAllowNetworkOperations(usage: ResourceUsage): Boolean {
        return usage.batteryLevel > 10 && usage.batteryTemperature < 45
    }
    
    /**
     * Generate resource optimization recommendations
     */
    private fun generateResourceRecommendations(usage: ResourceUsage): List<String> {
        val recommendations = mutableListOf<String>()
        
        if (usage.batteryLevel <= 20) {
            recommendations.add("Low battery: Reducing automation frequency")
        }
        
        if (usage.cpuUsagePercent > 80) {
            recommendations.add("High CPU usage: Optimizing processing")
        }
        
        if (usage.memoryUsageMB > 800) {
            recommendations.add("High memory usage: Clearing caches")
        }
        
        if (usage.batteryTemperature > 40) {
            recommendations.add("Device heating: Reducing processing intensity")
        }
        
        if (usage.availableStorageMB < 100) {
            recommendations.add("Low storage: Consider clearing old logs")
        }
        
        return recommendations
    }
    
    /**
     * Get battery usage statistics for user transparency
     */
    fun getBatteryStatistics(): BatteryStatistics {
        val now = System.currentTimeMillis()
        val oneHourAgo = now - 3600000
        val oneDayAgo = now - 86400000
        
        val hourlyStats = batteryStats.filter { it.timestamp >= oneHourAgo }
        val dailyStats = batteryStats.filter { it.timestamp >= oneDayAgo }
        
        return BatteryStatistics(
            currentLevel = _resourceUsage.value.batteryLevel,
            currentTemperature = _resourceUsage.value.batteryTemperature,
            hourlyDrainRate = hourlyStats.map { it.drainRate }.average().takeIf { !it.isNaN() } ?: 0.0,
            dailyDrainRate = dailyStats.map { it.drainRate }.average().takeIf { !it.isNaN() } ?: 0.0,
            peakTemperature = dailyStats.maxOfOrNull { it.temperature } ?: 0.0,
            totalSamples = batteryStats.size
        )
    }
    
    /**
     * Get current resource usage summary
     */
    fun getResourceSummary(): ResourceSummary {
        val usage = _resourceUsage.value
        val behavior = _adaptiveBehavior.value
        
        return ResourceSummary(
            cpuUsage = usage.cpuUsagePercent,
            memoryUsage = usage.memoryUsageMB,
            batteryLevel = usage.batteryLevel,
            adaptiveFrequency = behavior.processingFrequency,
            recommendationsCount = behavior.recommendations.size,
            isOptimized = behavior.processingFrequency < 1.0 || 
                          behavior.batchSize < 50 || 
                          !behavior.backgroundProcessing
        )
    }
    
    /**
     * Clean up resources
     */
    fun cleanup() {
        scope.cancel()
        batteryStats.clear()
    }
}

/**
 * Current resource usage snapshot
 */
data class ResourceUsage(
    val cpuUsagePercent: Double = 0.0,
    val memoryUsageMB: Double = 0.0,
    val batteryLevel: Int = -1,
    val batteryTemperature: Double = -1.0,
    val availableStorageMB: Double = 0.0,
    val networkUsageKB: Double = 0.0,
    val timestamp: Long = System.currentTimeMillis()
)

/**
 * Adaptive behavior configuration
 */
data class AdaptiveBehavior(
    val processingFrequency: Double = 1.0,
    val batchSize: Int = 50,
    val cacheSize: Int = 100,
    val backgroundProcessing: Boolean = true,
    val networkOperations: Boolean = true,
    val recommendations: List<String> = emptyList()
)

/**
 * Battery usage snapshot for statistics
 */
data class BatterySnapshot(
    val timestamp: Long,
    val level: Int,
    val temperature: Double,
    val drainRate: Double // Percent per hour
)

/**
 * Battery usage statistics
 */
data class BatteryStatistics(
    val currentLevel: Int,
    val currentTemperature: Double,
    val hourlyDrainRate: Double,
    val dailyDrainRate: Double,
    val peakTemperature: Double,
    val totalSamples: Int
)

/**
 * Resource usage summary for UI
 */
data class ResourceSummary(
    val cpuUsage: Double,
    val memoryUsage: Double,
    val batteryLevel: Int,
    val adaptiveFrequency: Double,
    val recommendationsCount: Int,
    val isOptimized: Boolean
)