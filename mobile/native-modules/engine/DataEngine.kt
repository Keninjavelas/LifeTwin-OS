package com.lifetwin.mlp.engine

import android.content.Context
import android.util.Log
import com.lifetwin.mlp.db.*
import com.lifetwin.mlp.privacy.PrivacyManager
import com.lifetwin.mlp.performance.PerformanceMonitor
import com.lifetwin.mlp.performance.BatteryOptimizer
import com.lifetwin.mlp.export.DataExporter
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicBoolean

private const val TAG = "DataEngine"

/**
 * Central coordination engine for all data collection components
 * - Creates DataEngine for coordinating all collectors
 * - Implements unified permission management across components
 * - Adds comprehensive error handling and recovery mechanisms
 * - Provides centralized configuration and monitoring
 */
class DataEngine(private val context: Context) {
    
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    
    // Core components
    private val privacyManager = PrivacyManager(context)
    private val performanceMonitor = PerformanceMonitor(context)
    private val batteryOptimizer = BatteryOptimizer(context)
    private val dataExporter = DataExporter(context)
    
    // Component registry
    private val collectors = ConcurrentHashMap<CollectorType, DataCollector>()
    private val collectorStates = ConcurrentHashMap<CollectorType, CollectorState>()
    
    // Engine state
    private val _engineState = MutableStateFlow(EngineState())
    val engineState: StateFlow<EngineState> = _engineState.asStateFlow()
    
    private val isInitialized = AtomicBoolean(false)
    private val isRunning = AtomicBoolean(false)

    /**
     * Initializes the data engine and all components
     */
    suspend fun initialize(): Boolean {
        return try {
            if (isInitialized.get()) {
                Log.w(TAG, "DataEngine already initialized")
                return true
            }
            
            Log.i(TAG, "Initializing DataEngine...")
            
            // Initialize core components
            initializeComponents()
            
            // Register collectors
            registerCollectors()
            
            // Start monitoring and coordination
            startCoordination()
            
            isInitialized.set(true)
            updateEngineState { it.copy(initialized = true, status = EngineStatus.READY) }
            
            Log.i(TAG, "DataEngine initialization completed successfully")
            true
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize DataEngine", e)
            updateEngineState { it.copy(status = EngineStatus.ERROR, lastError = e.message) }
            false
        }
    }

    /**
     * Starts the data collection engine
     */
    suspend fun start(): Boolean {
        return try {
            if (!isInitialized.get()) {
                Log.e(TAG, "Cannot start DataEngine - not initialized")
                return false
            }
            
            if (isRunning.get()) {
                Log.w(TAG, "DataEngine already running")
                return true
            }
            
            Log.i(TAG, "Starting DataEngine...")
            
            // Start collectors based on privacy settings
            startEnabledCollectors()
            
            // Start background processing
            startBackgroundProcessing()
            
            isRunning.set(true)
            updateEngineState { it.copy(running = true, status = EngineStatus.RUNNING) }
            
            Log.i(TAG, "DataEngine started successfully")
            true
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to start DataEngine", e)
            updateEngineState { it.copy(status = EngineStatus.ERROR, lastError = e.message) }
            false
        }
    }

    /**
     * Stops the data collection engine
     */
    suspend fun stop(): Boolean {
        return try {
            if (!isRunning.get()) {
                Log.w(TAG, "DataEngine not running")
                return true
            }
            
            Log.i(TAG, "Stopping DataEngine...")
            
            // Stop all collectors
            stopAllCollectors()
            
            // Stop background processing
            stopBackgroundProcessing()
            
            isRunning.set(false)
            updateEngineState { it.copy(running = false, status = EngineStatus.STOPPED) }
            
            Log.i(TAG, "DataEngine stopped successfully")
            true
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to stop DataEngine", e)
            updateEngineState { it.copy(status = EngineStatus.ERROR, lastError = e.message) }
            false
        }
    }

    /**
     * Restarts the data collection engine
     */
    suspend fun restart(): Boolean {
        Log.i(TAG, "Restarting DataEngine...")
        return stop() && start()
    }

    /**
     * Registers a data collector with the engine
     */
    fun registerCollector(collectorType: CollectorType, collector: DataCollector) {
        collectors[collectorType] = collector
        collectorStates[collectorType] = CollectorState(
            type = collectorType,
            status = CollectorStatus.REGISTERED,
            lastUpdate = System.currentTimeMillis()
        )
        
        Log.i(TAG, "Registered collector: $collectorType")
        updateCollectorCounts()
    }

    /**
     * Unregisters a data collector from the engine
     */
    suspend fun unregisterCollector(collectorType: CollectorType) {
        collectors[collectorType]?.let { collector ->
            try {
                if (collector.isCollectionActive()) {
                    collector.stopCollection()
                }
            } catch (e: Exception) {
                Log.w(TAG, "Error stopping collector during unregistration: $collectorType", e)
            }
        }
        
        collectors.remove(collectorType)
        collectorStates.remove(collectorType)
        
        Log.i(TAG, "Unregistered collector: $collectorType")
        updateCollectorCounts()
    }

    /**
     * Enables or disables a specific collector
     */
    suspend fun setCollectorEnabled(collectorType: CollectorType, enabled: Boolean): Boolean {
        return try {
            // Update privacy settings
            privacyManager.setCollectorEnabled(collectorType, enabled)
            
            // Start or stop the collector
            val collector = collectors[collectorType]
            if (collector != null) {
                if (enabled && isRunning.get()) {
                    startCollector(collectorType, collector)
                } else {
                    stopCollector(collectorType, collector)
                }
                true
            } else {
                Log.w(TAG, "Collector not registered: $collectorType")
                false
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to set collector enabled state: $collectorType", e)
            false
        }
    }

    /**
     * Gets the current status of all collectors
     */
    fun getCollectorStates(): Map<CollectorType, CollectorState> {
        return collectorStates.toMap()
    }

    /**
     * Gets comprehensive engine statistics
     */
    suspend fun getEngineStatistics(): EngineStatistics {
        return try {
            val performanceStats = performanceMonitor.getPerformanceStatistics()
            val privacySettings = privacyManager.getPrivacySettings()
            val batteryState = batteryOptimizer.batteryState.value
            val resourceState = batteryOptimizer.resourceState.value
            
            val collectorStats = collectors.mapValues { (type, collector) ->
                CollectorStatistics(
                    type = type,
                    isActive = collector.isCollectionActive(),
                    dataCount = collector.getCollectedDataCount(),
                    lastCollection = getLastCollectionTime(type),
                    errorCount = getCollectorErrorCount(type)
                )
            }
            
            EngineStatistics(
                engineState = _engineState.value,
                collectorStatistics = collectorStats,
                performanceStatistics = performanceStats,
                privacySettings = privacySettings,
                batteryState = batteryState,
                resourceState = resourceState,
                generatedAt = System.currentTimeMillis()
            )
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to get engine statistics", e)
            EngineStatistics(
                engineState = _engineState.value,
                collectorStatistics = emptyMap(),
                performanceStatistics = com.lifetwin.mlp.performance.PerformanceMonitor.PerformanceStatistics(
                    timeRangeStart = 0,
                    timeRangeEnd = 0,
                    totalOperations = 0,
                    collectionOperations = 0,
                    batchOperations = 0,
                    slowOperations = 0,
                    averageBatteryLevel = 0.0,
                    averageMemoryUsage = 0.0,
                    averageOperationDuration = 0.0,
                    peakMemoryUsage = 0.0,
                    operationsByType = emptyMap()
                ),
                privacySettings = com.lifetwin.mlp.privacy.PrivacyManager.PrivacySettings(
                    privacyLevel = com.lifetwin.mlp.privacy.PrivacyManager.PrivacyLevel.STANDARD,
                    enabledCollectors = emptySet(),
                    dataRetentionDays = 7,
                    anonymizationOptions = com.lifetwin.mlp.privacy.PrivacyManager.AnonymizationOptions(),
                    dataSharingControls = com.lifetwin.mlp.privacy.PrivacyManager.DataSharingControls()
                ),
                batteryState = BatteryOptimizer.BatteryState(),
                resourceState = BatteryOptimizer.ResourceState(),
                generatedAt = System.currentTimeMillis()
            )
        }
    }

    /**
     * Handles system events (device restart, permission changes, etc.)
     */
    suspend fun handleSystemEvent(event: SystemEvent) {
        try {
            Log.i(TAG, "Handling system event: ${event.type}")
            
            when (event.type) {
                SystemEventType.DEVICE_RESTART -> {
                    Log.i(TAG, "Device restart detected - reinitializing collectors")
                    restart()
                }
                SystemEventType.PERMISSION_GRANTED -> {
                    val collectorType = event.data["collectorType"] as? CollectorType
                    if (collectorType != null) {
                        Log.i(TAG, "Permission granted for $collectorType - starting collector")
                        setCollectorEnabled(collectorType, true)
                    }
                }
                SystemEventType.PERMISSION_REVOKED -> {
                    val collectorType = event.data["collectorType"] as? CollectorType
                    if (collectorType != null) {
                        Log.i(TAG, "Permission revoked for $collectorType - stopping collector")
                        setCollectorEnabled(collectorType, false)
                    }
                }
                SystemEventType.LOW_BATTERY -> {
                    Log.i(TAG, "Low battery detected - enabling power saving mode")
                    enablePowerSavingMode()
                }
                SystemEventType.CHARGING_STARTED -> {
                    Log.i(TAG, "Charging started - resuming normal operations")
                    disablePowerSavingMode()
                }
                SystemEventType.MEMORY_PRESSURE -> {
                    Log.i(TAG, "Memory pressure detected - optimizing memory usage")
                    batteryOptimizer.optimizeMemoryUsage()
                }
            }
            
            // Log the event
            logSystemEvent(event)
            
            // Notify listeners
            notifySystemEvent(event)
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to handle system event: ${event.type}", e)
        }
    }

    // Private methods

    private suspend fun initializeComponents() {
        // Components are already initialized in constructor
        Log.d(TAG, "Core components initialized")
    }

    private fun registerCollectors() {
        // Collectors would be registered by their respective managers
        // This is a placeholder for the registration process
        Log.d(TAG, "Collector registration completed")
    }

    private fun startCoordination() {
        scope.launch {
            // Monitor privacy settings changes
            launch {
                privacyManager.getPrivacySettings() // This would be a flow in real implementation
                // React to privacy setting changes
            }
            
            // Monitor performance metrics
            launch {
                performanceMonitor.performanceMetrics.collect { metrics ->
                    // React to performance changes
                    if (metrics.memoryPressure) {
                        batteryOptimizer.optimizeMemoryUsage()
                    }
                }
            }
            
            // Monitor battery state
            launch {
                batteryOptimizer.batteryState.collect { batteryState ->
                    // React to battery changes
                    if (batteryState.level < 15 && !batteryState.isCharging) {
                        enablePowerSavingMode()
                    }
                }
            }
        }
    }

    private suspend fun startEnabledCollectors() {
        val privacySettings = privacyManager.getPrivacySettings()
        
        collectors.forEach { (type, collector) ->
            if (type in privacySettings.enabledCollectors) {
                startCollector(type, collector)
            }
        }
    }

    private suspend fun startCollector(type: CollectorType, collector: DataCollector) {
        try {
            if (!collector.isCollectionActive()) {
                collector.startCollection()
                updateCollectorState(type) { it.copy(status = CollectorStatus.ACTIVE) }
                Log.i(TAG, "Started collector: $type")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to start collector: $type", e)
            updateCollectorState(type) { it.copy(status = CollectorStatus.ERROR, lastError = e.message) }
        }
    }

    private suspend fun stopCollector(type: CollectorType, collector: DataCollector) {
        try {
            if (collector.isCollectionActive()) {
                collector.stopCollection()
                updateCollectorState(type) { it.copy(status = CollectorStatus.STOPPED) }
                Log.i(TAG, "Stopped collector: $type")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to stop collector: $type", e)
            updateCollectorState(type) { it.copy(status = CollectorStatus.ERROR, lastError = e.message) }
        }
    }

    private suspend fun stopAllCollectors() {
        collectors.forEach { (type, collector) ->
            stopCollector(type, collector)
        }
    }

    private fun startBackgroundProcessing() {
        // Start background tasks like summary generation, cleanup, etc.
        Log.d(TAG, "Background processing started")
    }

    private fun stopBackgroundProcessing() {
        // Stop background tasks
        Log.d(TAG, "Background processing stopped")
    }

    private suspend fun enablePowerSavingMode() {
        try {
            // Reduce collection frequency for all active collectors
            collectors.forEach { (type, collector) ->
                if (collector.isCollectionActive()) {
                    // Implement power saving logic for each collector
                    updateCollectorState(type) { it.copy(powerSavingMode = true) }
                }
            }
            
            updateEngineState { it.copy(powerSavingMode = true) }
            Log.i(TAG, "Power saving mode enabled")
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to enable power saving mode", e)
        }
    }

    private suspend fun disablePowerSavingMode() {
        try {
            // Resume normal collection frequency
            collectors.forEach { (type, collector) ->
                if (collector.isCollectionActive()) {
                    updateCollectorState(type) { it.copy(powerSavingMode = false) }
                }
            }
            
            updateEngineState { it.copy(powerSavingMode = false) }
            Log.i(TAG, "Power saving mode disabled")
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to disable power saving mode", e)
        }
    }

    private suspend fun getLastCollectionTime(type: CollectorType): Long? {
        return try {
            val database = AppDatabase.getInstance(context)
            when (type) {
                CollectorType.USAGE_STATS -> database.usageEventDao().getLatestEvent()?.startTime
                CollectorType.NOTIFICATIONS -> database.notificationEventDao().getLatestEvent()?.timestamp
                CollectorType.SCREEN_EVENTS -> database.screenSessionDao().getLatestSession()?.startTime
                CollectorType.INTERACTIONS -> database.interactionMetricsDao().getLatestMetrics()?.timestamp
                CollectorType.SENSORS -> database.activityContextDao().getLatestContext()?.timestamp
            }
        } catch (e: Exception) {
            Log.w(TAG, "Failed to get last collection time for $type", e)
            null
        }
    }

    private suspend fun getCollectorErrorCount(type: CollectorType): Int {
        return try {
            val database = AppDatabase.getInstance(context)
            val last24Hours = System.currentTimeMillis() - (24 * 60 * 60 * 1000L)
            database.auditLogDao().getLogsByTypeAndTimeRange(
                "COLLECTOR_ERROR_$type",
                last24Hours,
                System.currentTimeMillis()
            ).size
        } catch (e: Exception) {
            Log.w(TAG, "Failed to get error count for $type", e)
            0
        }
    }

    private suspend fun logSystemEvent(event: SystemEvent) {
        try {
            val database = AppDatabase.getInstance(context)
            val auditEntry = AuditLogEntity(
                id = java.util.UUID.randomUUID().toString(),
                timestamp = System.currentTimeMillis(),
                eventType = "SYSTEM_EVENT_${event.type.name}",
                details = com.google.gson.Gson().toJson(event.data),
                userId = null
            )
            
            database.auditLogDao().insert(auditEntry)
        } catch (e: Exception) {
            Log.w(TAG, "Failed to log system event", e)
        }
    }

    private fun updateEngineState(update: (EngineState) -> EngineState) {
        val newState = update(_engineState.value)
        _engineState.value = newState
        
        // Notify listeners asynchronously
        scope.launch {
            notifyEngineStateChanged(newState)
        }
    }

    private fun updateCollectorState(type: CollectorType, update: (CollectorState) -> CollectorState) {
        collectorStates[type]?.let { currentState ->
            val newState = update(currentState).copy(lastUpdate = System.currentTimeMillis())
            collectorStates[type] = newState
            
            // Notify listeners asynchronously
            scope.launch {
                notifyCollectorStateChanged(type, newState)
            }
        }
    }

    private fun updateCollectorCounts() {
        val totalCollectors = collectors.size
        val activeCollectors = collectors.count { it.value.isCollectionActive() }
        
        updateEngineState { 
            it.copy(
                totalCollectors = totalCollectors,
                activeCollectors = activeCollectors
            )
        }
    }

    fun cleanup() {
        scope.cancel()
        performanceMonitor.cleanup()
        batteryOptimizer.cleanup()
    }

    // Data classes

    data class EngineState(
        val initialized: Boolean = false,
        val running: Boolean = false,
        val status: EngineStatus = EngineStatus.INITIALIZING,
        val totalCollectors: Int = 0,
        val activeCollectors: Int = 0,
        val powerSavingMode: Boolean = false,
        val lastUpdate: Long = System.currentTimeMillis(),
        val lastError: String? = null
    )

    data class CollectorState(
        val type: CollectorType,
        val status: CollectorStatus,
        val powerSavingMode: Boolean = false,
        val lastUpdate: Long,
        val lastError: String? = null
    )

    data class CollectorStatistics(
        val type: CollectorType,
        val isActive: Boolean,
        val dataCount: Int,
        val lastCollection: Long?,
        val errorCount: Int
    )

    data class EngineStatistics(
        val engineState: EngineState,
        val collectorStatistics: Map<CollectorType, CollectorStatistics>,
        val performanceStatistics: com.lifetwin.mlp.performance.PerformanceMonitor.PerformanceStatistics,
        val privacySettings: com.lifetwin.mlp.privacy.PrivacyManager.PrivacySettings,
        val batteryState: BatteryOptimizer.BatteryState,
        val resourceState: BatteryOptimizer.ResourceState,
        val generatedAt: Long
    )

    data class SystemEvent(
        val type: SystemEventType,
        val data: Map<String, Any> = emptyMap(),
        val timestamp: Long = System.currentTimeMillis()
    )

    enum class EngineStatus {
        INITIALIZING, READY, RUNNING, STOPPED, ERROR
    }

    enum class CollectorStatus {
        REGISTERED, ACTIVE, STOPPED, ERROR
    }

    enum class SystemEventType {
        DEVICE_RESTART,
        PERMISSION_GRANTED,
        PERMISSION_REVOKED,
        LOW_BATTERY,
        CHARGING_STARTED,
        MEMORY_PRESSURE
    }

    /**
     * Interface for components that need to be notified of engine events
     */
    interface EngineEventListener {
        suspend fun onEngineStateChanged(state: EngineState)
        suspend fun onCollectorStateChanged(type: CollectorType, state: CollectorState)
        suspend fun onSystemEvent(event: SystemEvent)
    }

    // Event listeners
    private val eventListeners = mutableListOf<EngineEventListener>()

    /**
     * Registers an event listener
     */
    fun addEventListener(listener: EngineEventListener) {
        eventListeners.add(listener)
    }

    /**
     * Unregisters an event listener
     */
    fun removeEventListener(listener: EngineEventListener) {
        eventListeners.remove(listener)
    }

    private suspend fun notifyEngineStateChanged(state: EngineState) {
        eventListeners.forEach { listener ->
            try {
                listener.onEngineStateChanged(state)
            } catch (e: Exception) {
                Log.w(TAG, "Error notifying engine state change", e)
            }
        }
    }

    private suspend fun notifyCollectorStateChanged(type: CollectorType, state: CollectorState) {
        eventListeners.forEach { listener ->
            try {
                listener.onCollectorStateChanged(type, state)
            } catch (e: Exception) {
                Log.w(TAG, "Error notifying collector state change", e)
            }
        }
    }

    private suspend fun notifySystemEvent(event: SystemEvent) {
        eventListeners.forEach { listener ->
            try {
                listener.onSystemEvent(event)
            } catch (e: Exception) {
                Log.w(TAG, "Error notifying system event", e)
            }
        }
    }
}