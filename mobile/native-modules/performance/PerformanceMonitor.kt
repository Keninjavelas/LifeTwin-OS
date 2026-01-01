package com.lifetwin.mlp.performance

import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.Debug
import android.util.Log
import com.lifetwin.mlp.db.*
import com.google.gson.Gson
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import java.util.concurrent.atomic.AtomicLong
import kotlin.system.measureTimeMillis

private const val TAG = "PerformanceMonitor"

/**
 * Performance monitoring and adaptive behavior system
 * - Implements performance metrics collection and reporting
 * - Adds adaptive collection frequency based on battery level
 * - Implements intelligent batching for all data operations
 * - Provides performance optimization recommendations
 */
class PerformanceMonitor(private val context: Context) {
    
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private val gson = Gson()
    
    // Performance metrics tracking
    private val _performanceMetrics = MutableStateFlow(PerformanceMetrics())
    val performanceMetrics: StateFlow<PerformanceMetrics> = _performanceMetrics.asStateFlow()
    
    // Adaptive behavior state
    private val _adaptiveBehavior = MutableStateFlow(AdaptiveBehaviorState())
    val adaptiveBehavior: StateFlow<AdaptiveBehaviorState> = _adaptiveBehavior.asStateFlow()
    
    // Operation counters
    private val databaseOperationCount = AtomicLong(0)
    private val collectionOperationCount = AtomicLong(0)
    private val batchOperationCount = AtomicLong(0)
    
    // Performance thresholds
    private val lowBatteryThreshold = 20 // 20%
    private val criticalBatteryThreshold = 10 // 10%
    private val maxMemoryUsageThreshold = 0.8 // 80% of available memory
    
    init {
        startPerformanceMonitoring()
    }

    /**
     * Starts continuous performance monitoring
     */
    private fun startPerformanceMonitoring() {
        scope.launch {
            while (isActive) {
                try {
                    updatePerformanceMetrics()
                    updateAdaptiveBehavior()
                    delay(30000) // Update every 30 seconds
                } catch (e: Exception) {
                    Log.e(TAG, "Error in performance monitoring", e)
                    delay(60000) // Wait longer on error
                }
            }
        }
    }

    /**
     * Updates current performance metrics
     */
    private suspend fun updatePerformanceMetrics() {
        val currentMetrics = _performanceMetrics.value
        
        val batteryLevel = getBatteryLevel()
        val memoryUsage = getMemoryUsage()
        val cpuUsage = getCpuUsage()
        val databasePerformance = measureDatabasePerformance()
        
        val updatedMetrics = currentMetrics.copy(
            timestamp = System.currentTimeMillis(),
            batteryLevel = batteryLevel,
            memoryUsage = memoryUsage,
            cpuUsage = cpuUsage,
            databaseOperationsPerSecond = calculateOperationsPerSecond(),
            averageOperationLatency = databasePerformance.averageLatency,
            totalDatabaseOperations = databaseOperationCount.get(),
            totalCollectionOperations = collectionOperationCount.get(),
            totalBatchOperations = batchOperationCount.get(),
            memoryPressure = memoryUsage.usedMemoryMB > (memoryUsage.totalMemoryMB * maxMemoryUsageThreshold)
        )
        
        _performanceMetrics.value = updatedMetrics
        
        // Log performance metrics to database
        logPerformanceMetrics(updatedMetrics)
    }

    /**
     * Updates adaptive behavior based on current conditions
     */
    private suspend fun updateAdaptiveBehavior() {
        val metrics = _performanceMetrics.value
        val currentBehavior = _adaptiveBehavior.value
        
        val newCollectionFrequency = calculateOptimalCollectionFrequency(metrics)
        val newBatchSize = calculateOptimalBatchSize(metrics)
        val shouldReduceOperations = shouldReduceOperations(metrics)
        
        val updatedBehavior = currentBehavior.copy(
            timestamp = System.currentTimeMillis(),
            collectionFrequencyMultiplier = newCollectionFrequency,
            batchSizeMultiplier = newBatchSize,
            reducedOperationsMode = shouldReduceOperations,
            batteryOptimizationActive = metrics.batteryLevel < lowBatteryThreshold,
            memoryOptimizationActive = metrics.memoryPressure,
            performanceLevel = calculatePerformanceLevel(metrics)
        )
        
        _adaptiveBehavior.value = updatedBehavior
        
        // Log behavior changes
        if (behaviorChanged(currentBehavior, updatedBehavior)) {
            logBehaviorChange(currentBehavior, updatedBehavior)
        }
    }

    /**
     * Records a database operation for performance tracking
     */
    suspend fun recordDatabaseOperation(operationType: String, durationMs: Long) {
        databaseOperationCount.incrementAndGet()
        
        // Update running average of operation latency
        val currentMetrics = _performanceMetrics.value
        val newAverage = (currentMetrics.averageOperationLatency + durationMs) / 2
        
        _performanceMetrics.value = currentMetrics.copy(
            averageOperationLatency = newAverage,
            lastOperationLatency = durationMs
        )
        
        // Log slow operations
        if (durationMs > 1000) { // Operations taking more than 1 second
            Log.w(TAG, "Slow database operation: $operationType took ${durationMs}ms")
            logSlowOperation(operationType, durationMs)
        }
    }

    /**
     * Records a data collection operation
     */
    fun recordCollectionOperation(collectorType: CollectorType, recordCount: Int) {
        collectionOperationCount.incrementAndGet()
        
        scope.launch {
            try {
                val database = AppDatabase.getInstance(context)
                val performanceEntry = PerformanceLogEntity(
                    id = java.util.UUID.randomUUID().toString(),
                    timestamp = System.currentTimeMillis(),
                    operationType = "DATA_COLLECTION",
                    collectorType = collectorType.name,
                    recordCount = recordCount,
                    batteryLevel = getBatteryLevel(),
                    memoryUsageMB = getMemoryUsage().usedMemoryMB
                )
                
                database.performanceLogDao().insert(performanceEntry)
            } catch (e: Exception) {
                Log.w(TAG, "Failed to log collection operation", e)
            }
        }
    }

    /**
     * Records a batch operation for optimization tracking
     */
    fun recordBatchOperation(batchSize: Int, durationMs: Long) {
        batchOperationCount.incrementAndGet()
        
        scope.launch {
            try {
                val database = AppDatabase.getInstance(context)
                val performanceEntry = PerformanceLogEntity(
                    id = java.util.UUID.randomUUID().toString(),
                    timestamp = System.currentTimeMillis(),
                    operationType = "BATCH_OPERATION",
                    recordCount = batchSize,
                    durationMs = durationMs,
                    batteryLevel = getBatteryLevel(),
                    memoryUsageMB = getMemoryUsage().usedMemoryMB
                )
                
                database.performanceLogDao().insert(performanceEntry)
            } catch (e: Exception) {
                Log.w(TAG, "Failed to log batch operation", e)
            }
        }
    }

    /**
     * Gets current performance recommendations
     */
    suspend fun getPerformanceRecommendations(): List<PerformanceRecommendation> {
        val metrics = _performanceMetrics.value
        val recommendations = mutableListOf<PerformanceRecommendation>()
        
        // Battery recommendations
        if (metrics.batteryLevel < criticalBatteryThreshold) {
            recommendations.add(PerformanceRecommendation(
                type = RecommendationType.CRITICAL,
                title = "Critical Battery Level",
                description = "Battery level is critically low (${metrics.batteryLevel}%)",
                action = "Consider enabling emergency power mode to disable non-essential data collection"
            ))
        } else if (metrics.batteryLevel < lowBatteryThreshold) {
            recommendations.add(PerformanceRecommendation(
                type = RecommendationType.WARNING,
                title = "Low Battery Level",
                description = "Battery level is low (${metrics.batteryLevel}%)",
                action = "Reducing data collection frequency to preserve battery"
            ))
        }
        
        // Memory recommendations
        if (metrics.memoryPressure) {
            recommendations.add(PerformanceRecommendation(
                type = RecommendationType.WARNING,
                title = "High Memory Usage",
                description = "Memory usage is high (${metrics.memoryUsage.usedMemoryMB}MB / ${metrics.memoryUsage.totalMemoryMB}MB)",
                action = "Increasing batch sizes and reducing in-memory data structures"
            ))
        }
        
        // Performance recommendations
        if (metrics.averageOperationLatency > 500) {
            recommendations.add(PerformanceRecommendation(
                type = RecommendationType.INFO,
                title = "Slow Database Operations",
                description = "Average database operation latency is ${metrics.averageOperationLatency}ms",
                action = "Consider optimizing database queries or increasing batch sizes"
            ))
        }
        
        return recommendations
    }

    /**
     * Gets performance statistics for a time period
     */
    suspend fun getPerformanceStatistics(
        startTime: Long = System.currentTimeMillis() - (24 * 60 * 60 * 1000L),
        endTime: Long = System.currentTimeMillis()
    ): PerformanceStatistics {
        return try {
            val database = AppDatabase.getInstance(context)
            val performanceLogs = database.performanceLogDao().getLogsByTimeRange(startTime, endTime)
            
            val collectionOperations = performanceLogs.filter { it.operationType == "DATA_COLLECTION" }
            val batchOperations = performanceLogs.filter { it.operationType == "BATCH_OPERATION" }
            val slowOperations = performanceLogs.filter { (it.durationMs ?: 0) > 1000 }
            
            PerformanceStatistics(
                timeRangeStart = startTime,
                timeRangeEnd = endTime,
                totalOperations = performanceLogs.size,
                collectionOperations = collectionOperations.size,
                batchOperations = batchOperations.size,
                slowOperations = slowOperations.size,
                averageBatteryLevel = performanceLogs.mapNotNull { it.batteryLevel }.average(),
                averageMemoryUsage = performanceLogs.mapNotNull { it.memoryUsageMB }.average(),
                averageOperationDuration = performanceLogs.mapNotNull { it.durationMs }.average(),
                peakMemoryUsage = performanceLogs.mapNotNull { it.memoryUsageMB }.maxOrNull() ?: 0.0,
                operationsByType = performanceLogs.groupBy { it.operationType }.mapValues { it.value.size }
            )
        } catch (e: Exception) {
            Log.e(TAG, "Failed to get performance statistics", e)
            PerformanceStatistics(
                timeRangeStart = startTime,
                timeRangeEnd = endTime,
                totalOperations = 0,
                collectionOperations = 0,
                batchOperations = 0,
                slowOperations = 0,
                averageBatteryLevel = 0.0,
                averageMemoryUsage = 0.0,
                averageOperationDuration = 0.0,
                peakMemoryUsage = 0.0,
                operationsByType = emptyMap()
            )
        }
    }

    // Helper methods

    private fun getBatteryLevel(): Int {
        return try {
            val batteryIntent = context.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
            val level = batteryIntent?.getIntExtra(BatteryManager.EXTRA_LEVEL, -1) ?: -1
            val scale = batteryIntent?.getIntExtra(BatteryManager.EXTRA_SCALE, -1) ?: -1
            
            if (level >= 0 && scale > 0) {
                (level * 100 / scale)
            } else {
                100 // Default to full battery if unable to read
            }
        } catch (e: Exception) {
            Log.w(TAG, "Failed to get battery level", e)
            100
        }
    }

    private fun getMemoryUsage(): MemoryUsage {
        return try {
            val runtime = Runtime.getRuntime()
            val totalMemory = runtime.totalMemory()
            val freeMemory = runtime.freeMemory()
            val usedMemory = totalMemory - freeMemory
            val maxMemory = runtime.maxMemory()
            
            MemoryUsage(
                usedMemoryMB = (usedMemory / (1024 * 1024)).toDouble(),
                totalMemoryMB = (totalMemory / (1024 * 1024)).toDouble(),
                maxMemoryMB = (maxMemory / (1024 * 1024)).toDouble(),
                freeMemoryMB = (freeMemory / (1024 * 1024)).toDouble()
            )
        } catch (e: Exception) {
            Log.w(TAG, "Failed to get memory usage", e)
            MemoryUsage(0.0, 0.0, 0.0, 0.0)
        }
    }

    private fun getCpuUsage(): Double {
        return try {
            // This is a simplified CPU usage calculation
            // In a real implementation, you might use more sophisticated methods
            val debugInfo = Debug.MemoryInfo()
            Debug.getMemoryInfo(debugInfo)
            
            // Return a normalized value between 0 and 1
            minOf(1.0, debugInfo.totalPss / 100000.0)
        } catch (e: Exception) {
            Log.w(TAG, "Failed to get CPU usage", e)
            0.0
        }
    }

    private suspend fun measureDatabasePerformance(): DatabasePerformance {
        return try {
            val database = AppDatabase.getInstance(context)
            
            val queryTime = measureTimeMillis {
                database.usageEventDao().getEventCount()
            }
            
            DatabasePerformance(
                averageLatency = queryTime.toDouble(),
                queryCount = 1
            )
        } catch (e: Exception) {
            Log.w(TAG, "Failed to measure database performance", e)
            DatabasePerformance(0.0, 0)
        }
    }

    private fun calculateOperationsPerSecond(): Double {
        val currentTime = System.currentTimeMillis()
        val timeWindow = 60000L // 1 minute
        val operations = databaseOperationCount.get()
        
        return operations.toDouble() / (timeWindow / 1000.0)
    }

    private fun calculateOptimalCollectionFrequency(metrics: PerformanceMetrics): Double {
        return when {
            metrics.batteryLevel < criticalBatteryThreshold -> 0.1 // 10% of normal frequency
            metrics.batteryLevel < lowBatteryThreshold -> 0.5 // 50% of normal frequency
            metrics.memoryPressure -> 0.7 // 70% of normal frequency
            else -> 1.0 // Normal frequency
        }
    }

    private fun calculateOptimalBatchSize(metrics: PerformanceMetrics): Double {
        return when {
            metrics.memoryPressure -> 0.5 // Smaller batches to reduce memory usage
            metrics.averageOperationLatency > 1000 -> 2.0 // Larger batches for slow operations
            else -> 1.0 // Normal batch size
        }
    }

    private fun shouldReduceOperations(metrics: PerformanceMetrics): Boolean {
        return metrics.batteryLevel < lowBatteryThreshold || 
               metrics.memoryPressure || 
               metrics.averageOperationLatency > 2000
    }

    private fun calculatePerformanceLevel(metrics: PerformanceMetrics): PerformanceLevel {
        return when {
            metrics.batteryLevel < criticalBatteryThreshold || metrics.memoryPressure -> PerformanceLevel.CRITICAL
            metrics.batteryLevel < lowBatteryThreshold || metrics.averageOperationLatency > 1000 -> PerformanceLevel.DEGRADED
            else -> PerformanceLevel.OPTIMAL
        }
    }

    private fun behaviorChanged(old: AdaptiveBehaviorState, new: AdaptiveBehaviorState): Boolean {
        return old.collectionFrequencyMultiplier != new.collectionFrequencyMultiplier ||
               old.batchSizeMultiplier != new.batchSizeMultiplier ||
               old.reducedOperationsMode != new.reducedOperationsMode ||
               old.performanceLevel != new.performanceLevel
    }

    private suspend fun logPerformanceMetrics(metrics: PerformanceMetrics) {
        try {
            val database = AppDatabase.getInstance(context)
            val performanceEntry = PerformanceLogEntity(
                id = java.util.UUID.randomUUID().toString(),
                timestamp = metrics.timestamp,
                operationType = "PERFORMANCE_METRICS",
                batteryLevel = metrics.batteryLevel,
                memoryUsageMB = metrics.memoryUsage.usedMemoryMB,
                cpuUsage = metrics.cpuUsage,
                durationMs = metrics.averageOperationLatency.toLong()
            )
            
            database.performanceLogDao().insert(performanceEntry)
        } catch (e: Exception) {
            Log.w(TAG, "Failed to log performance metrics", e)
        }
    }

    private suspend fun logBehaviorChange(old: AdaptiveBehaviorState, new: AdaptiveBehaviorState) {
        try {
            val database = AppDatabase.getInstance(context)
            val auditEntry = AuditLogEntity(
                id = java.util.UUID.randomUUID().toString(),
                timestamp = System.currentTimeMillis(),
                eventType = "ADAPTIVE_BEHAVIOR_CHANGED",
                details = gson.toJson(mapOf(
                    "oldBehavior" to old,
                    "newBehavior" to new,
                    "reason" to "performance_optimization"
                )),
                userId = null
            )
            
            database.auditLogDao().insert(auditEntry)
            
            Log.i(TAG, "Adaptive behavior changed: performance level ${new.performanceLevel}")
        } catch (e: Exception) {
            Log.w(TAG, "Failed to log behavior change", e)
        }
    }

    private suspend fun logSlowOperation(operationType: String, durationMs: Long) {
        try {
            val database = AppDatabase.getInstance(context)
            val performanceEntry = PerformanceLogEntity(
                id = java.util.UUID.randomUUID().toString(),
                timestamp = System.currentTimeMillis(),
                operationType = "SLOW_OPERATION",
                collectorType = operationType,
                durationMs = durationMs,
                batteryLevel = getBatteryLevel(),
                memoryUsageMB = getMemoryUsage().usedMemoryMB
            )
            
            database.performanceLogDao().insert(performanceEntry)
        } catch (e: Exception) {
            Log.w(TAG, "Failed to log slow operation", e)
        }
    }

    fun cleanup() {
        scope.cancel()
    }

    // Data classes

    data class PerformanceMetrics(
        val timestamp: Long = System.currentTimeMillis(),
        val batteryLevel: Int = 100,
        val memoryUsage: MemoryUsage = MemoryUsage(),
        val cpuUsage: Double = 0.0,
        val databaseOperationsPerSecond: Double = 0.0,
        val averageOperationLatency: Double = 0.0,
        val lastOperationLatency: Double = 0.0,
        val totalDatabaseOperations: Long = 0,
        val totalCollectionOperations: Long = 0,
        val totalBatchOperations: Long = 0,
        val memoryPressure: Boolean = false
    )

    data class MemoryUsage(
        val usedMemoryMB: Double = 0.0,
        val totalMemoryMB: Double = 0.0,
        val maxMemoryMB: Double = 0.0,
        val freeMemoryMB: Double = 0.0
    )

    data class AdaptiveBehaviorState(
        val timestamp: Long = System.currentTimeMillis(),
        val collectionFrequencyMultiplier: Double = 1.0,
        val batchSizeMultiplier: Double = 1.0,
        val reducedOperationsMode: Boolean = false,
        val batteryOptimizationActive: Boolean = false,
        val memoryOptimizationActive: Boolean = false,
        val performanceLevel: PerformanceLevel = PerformanceLevel.OPTIMAL
    )

    data class DatabasePerformance(
        val averageLatency: Double,
        val queryCount: Int
    )

    data class PerformanceRecommendation(
        val type: RecommendationType,
        val title: String,
        val description: String,
        val action: String
    )

    data class PerformanceStatistics(
        val timeRangeStart: Long,
        val timeRangeEnd: Long,
        val totalOperations: Int,
        val collectionOperations: Int,
        val batchOperations: Int,
        val slowOperations: Int,
        val averageBatteryLevel: Double,
        val averageMemoryUsage: Double,
        val averageOperationDuration: Double,
        val peakMemoryUsage: Double,
        val operationsByType: Map<String, Int>
    )

    enum class PerformanceLevel {
        OPTIMAL, DEGRADED, CRITICAL
    }

    enum class RecommendationType {
        INFO, WARNING, CRITICAL
    }
}