package com.lifetwin.automation

import android.content.Context
import com.lifetwin.engine.DataEngine
import com.lifetwin.ml.ModelInferenceManager
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*
import kotlinx.serialization.Serializable

/**
 * AutomationSystemIntegrator - Complete system integration with existing components
 * 
 * Implements Requirements:
 * - 10.1: Integration with DataEngine, ModelInferenceManager, and simulation engine
 * - 10.2: Compatibility with existing privacy and performance systems
 * - 10.3: API endpoints for future component integration
 * - 10.4: System coordination and lifecycle management
 * - 10.7: Comprehensive automation system orchestration
 */
class AutomationSystemIntegrator(
    private val context: Context
) {
    private val scope = CoroutineScope(Dispatchers.Default + SupervisorJob())
    
    // Core automation components
    private lateinit var automationEngine: AutomationEngine
    private lateinit var ruleBasedSystem: RuleBasedSystem
    private lateinit var androidIntegration: AndroidIntegration
    private lateinit var resourceMonitor: ResourceMonitor
    private lateinit var privacyController: PrivacyController
    private lateinit var dataRetentionManager: DataRetentionManager
    private lateinit var systemHealthMonitor: SystemHealthMonitor
    private lateinit var abTestingFramework: ABTestingFramework
    
    // External system integrations
    private lateinit var dataEngine: DataEngine
    private lateinit var modelInferenceManager: ModelInferenceManager
    
    // System state
    private val _systemState = MutableStateFlow(SystemState.INITIALIZING)
    val systemState: StateFlow<SystemState> = _systemState.asStateFlow()
    
    // Integration status
    private val _integrationStatus = MutableStateFlow(IntegrationStatus())
    val integrationStatus: StateFlow<IntegrationStatus> = _integrationStatus.asStateFlow()
    
    // API endpoints for external access
    private val apiEndpoints = mutableMapOf<String, suspend (Map<String, Any>) -> Any>()
    
    init {
        initializeSystem()
    }
    
    /**
     * Initialize the complete automation system
     */
    private fun initializeSystem() {
        scope.launch {
            try {
                _systemState.value = SystemState.INITIALIZING
                
                // Initialize external dependencies
                initializeExternalDependencies()
                
                // Initialize core automation components
                initializeAutomationComponents()
                
                // Set up component integrations
                setupComponentIntegrations()
                
                // Register API endpoints
                registerApiEndpoints()
                
                // Start system coordination
                startSystemCoordination()
                
                _systemState.value = SystemState.RUNNING
                updateIntegrationStatus(true, "System successfully initialized")
                
            } catch (e: Exception) {
                _systemState.value = SystemState.ERROR
                updateIntegrationStatus(false, "System initialization failed: ${e.message}")
                systemHealthMonitor.recordError("SystemIntegrator", "Initialization failed", e)
            }
        }
    }
    
    /**
     * Initialize external system dependencies
     */
    private suspend fun initializeExternalDependencies() = withContext(Dispatchers.IO) {
        // Initialize DataEngine integration
        dataEngine = DataEngine(context)
        
        // Initialize ML inference manager
        modelInferenceManager = ModelInferenceManager(context)
        
        // Verify external systems are ready
        if (!dataEngine.isInitialized()) {
            throw IllegalStateException("DataEngine not properly initialized")
        }
        
        if (!modelInferenceManager.isReady()) {
            throw IllegalStateException("ModelInferenceManager not ready")
        }
    }
    
    /**
     * Initialize core automation components
     */
    private suspend fun initializeAutomationComponents() = withContext(Dispatchers.IO) {
        // Initialize privacy controller first (required by others)
        privacyController = PrivacyController(context)
        
        // Initialize resource monitoring
        resourceMonitor = ResourceMonitor(context)
        
        // Initialize Android system integration
        androidIntegration = AndroidIntegration(context)
        
        // Initialize rule-based system
        ruleBasedSystem = RuleBasedSystem(context, dataEngine)
        
        // Initialize automation engine
        automationEngine = AutomationEngine(
            context = context,
            dataEngine = dataEngine,
            ruleBasedSystem = ruleBasedSystem,
            androidIntegration = androidIntegration,
            resourceMonitor = resourceMonitor,
            privacyController = privacyController
        )
        
        // Initialize data retention manager
        dataRetentionManager = DataRetentionManager(
            context = context,
            automationLog = automationEngine.automationLog,
            privacyController = privacyController
        )
        
        // Initialize A/B testing framework
        abTestingFramework = ABTestingFramework(
            context = context,
            automationEngine = automationEngine,
            dataEngine = dataEngine
        )
        
        // Initialize system health monitor (last, as it monitors others)
        systemHealthMonitor = SystemHealthMonitor(
            context = context,
            automationEngine = automationEngine,
            resourceMonitor = resourceMonitor,
            privacyController = privacyController
        )
    }
    
    /**
     * Set up integrations between components
     */
    private suspend fun setupComponentIntegrations() = withContext(Dispatchers.IO) {
        // Connect automation engine to data engine
        automationEngine.setDataSource(dataEngine)
        
        // Connect rule-based system to ML predictions
        ruleBasedSystem.setMLPredictor { behavioralContext ->
            modelInferenceManager.predictNextApp(behavioralContext.toMLFeatures())
        }
        
        // Connect resource monitor to automation engine for adaptive behavior
        resourceMonitor.adaptiveBehavior.collect { behavior ->
            automationEngine.updateAdaptiveBehavior(behavior)
        }
        
        // Connect privacy controller to all data processing components
        privacyController.privacySettings.collect { settings ->
            automationEngine.updatePrivacySettings(settings)
            dataRetentionManager.updateRetentionPolicy(
                dataRetentionManager.retentionPolicy.value.copy(
                    anonymizeTrainingData = settings.anonymizeData
                )
            )
        }
        
        // Connect system health monitor to all components
        systemHealthMonitor.systemHealth.collect { health ->
            if (health.overallStatus == HealthStatus.CRITICAL) {
                automationEngine.enterSafeMode()
            }
        }
    }
    
    /**
     * Register API endpoints for external integration
     */
    private fun registerApiEndpoints() {
        // Automation control endpoints
        apiEndpoints["automation/start"] = { params ->
            automationEngine.start()
            mapOf("status" to "started", "timestamp" to System.currentTimeMillis())
        }
        
        apiEndpoints["automation/stop"] = { params ->
            automationEngine.stop()
            mapOf("status" to "stopped", "timestamp" to System.currentTimeMillis())
        }
        
        apiEndpoints["automation/status"] = { params ->
            mapOf(
                "isRunning" to automationEngine.isRunning(),
                "systemHealth" to systemHealthMonitor.systemHealth.value,
                "resourceUsage" to resourceMonitor.resourceUsage.value
            )
        }
        
        // Configuration endpoints
        apiEndpoints["automation/configure"] = { params ->
            val settings = params["settings"] as? Map<String, Any>
            if (settings != null) {
                updateSystemConfiguration(settings)
                mapOf("status" to "configured")
            } else {
                mapOf("error" to "Invalid settings")
            }
        }
        
        // Data endpoints
        apiEndpoints["automation/logs"] = { params ->
            val limit = (params["limit"] as? Number)?.toInt() ?: 100
            automationEngine.automationLog.getRecentLogs(limit)
        }
        
        apiEndpoints["automation/metrics"] = { params ->
            mapOf(
                "performance" to systemHealthMonitor.getHealthReport().performanceMetrics,
                "errors" to systemHealthMonitor.getHealthReport().topErrors,
                "battery" to resourceMonitor.getBatteryStatistics()
            )
        }
        
        // Diagnostic endpoints
        apiEndpoints["automation/diagnostics"] = { params ->
            systemHealthMonitor.runDiagnostics()
        }
        
        apiEndpoints["automation/health"] = { params ->
            systemHealthMonitor.getHealthReport()
        }
        
        // Privacy endpoints
        apiEndpoints["automation/privacy/status"] = { params ->
            privacyController.getPrivacyComplianceReport()
        }
        
        apiEndpoints["automation/privacy/export"] = { params ->
            val dataTypes = (params["dataTypes"] as? List<String>)?.map { 
                DataType.valueOf(it.uppercase()) 
            } ?: listOf(DataType.AUTOMATION_LOGS)
            
            dataRetentionManager.exportUserData(dataTypes)
        }
        
        // A/B testing endpoints
        apiEndpoints["automation/ab-test/status"] = { params ->
            abTestingFramework.getCurrentExperiments()
        }
        
        apiEndpoints["automation/ab-test/results"] = { params ->
            val experimentId = params["experimentId"] as? String
            if (experimentId != null) {
                abTestingFramework.getExperimentResults(experimentId)
            } else {
                mapOf("error" to "Missing experimentId")
            }
        }
    }
    
    /**
     * Start system coordination and monitoring
     */
    private fun startSystemCoordination() {
        scope.launch {
            while (isActive) {
                try {
                    coordinateSystemOperations()
                    delay(60000) // Coordinate every minute
                } catch (e: Exception) {
                    systemHealthMonitor.recordError("SystemIntegrator", "Coordination error", e)
                    delay(120000) // Wait longer on error
                }
            }
        }
    }
    
    /**
     * Coordinate operations between system components
     */
    private suspend fun coordinateSystemOperations() = withContext(Dispatchers.IO) {
        // Check if any component needs attention
        val health = systemHealthMonitor.systemHealth.value
        
        if (health.overallStatus == HealthStatus.WARNING) {
            // Reduce system load
            automationEngine.reduceProcessingFrequency(0.5)
            resourceMonitor.adaptiveBehavior.value.let { behavior ->
                if (!behavior.backgroundProcessing) {
                    androidIntegration.pauseBackgroundOperations()
                }
            }
        }
        
        if (health.overallStatus == HealthStatus.HEALTHY) {
            // Restore normal operations
            automationEngine.restoreNormalFrequency()
            androidIntegration.resumeBackgroundOperations()
        }
        
        // Coordinate data retention
        val privacySettings = privacyController.privacySettings.value
        if (privacySettings.dataRetentionDays < 90) {
            dataRetentionManager.performScheduledCleanup()
        }
        
        // Update integration status
        updateIntegrationStatus(
            success = health.overallStatus != HealthStatus.CRITICAL,
            message = "System coordination completed"
        )
    }
    
    /**
     * Update system configuration
     */
    private suspend fun updateSystemConfiguration(settings: Map<String, Any>) = withContext(Dispatchers.IO) {
        try {
            // Update automation settings
            settings["automationEnabled"]?.let { enabled ->
                if (enabled as Boolean) {
                    automationEngine.start()
                } else {
                    automationEngine.stop()
                }
            }
            
            // Update privacy settings
            settings["privacySettings"]?.let { privacyMap ->
                val privacySettings = parsePrivacySettings(privacyMap as Map<String, Any>)
                privacyController.updatePrivacySettings(privacySettings)
            }
            
            // Update resource monitoring settings
            settings["resourceMonitoring"]?.let { enabled ->
                if (!(enabled as Boolean)) {
                    resourceMonitor.cleanup()
                }
            }
            
        } catch (e: Exception) {
            systemHealthMonitor.recordError("SystemIntegrator", "Configuration update failed", e)
            throw e
        }
    }
    
    /**
     * Parse privacy settings from map
     */
    private fun parsePrivacySettings(settingsMap: Map<String, Any>): PrivacySettings {
        return PrivacySettings(
            allowDataCollection = settingsMap["allowDataCollection"] as? Boolean ?: true,
            allowBehavioralAnalysis = settingsMap["allowBehavioralAnalysis"] as? Boolean ?: true,
            allowRLLearning = settingsMap["allowRLLearning"] as? Boolean ?: true,
            anonymizeData = settingsMap["anonymizeData"] as? Boolean ?: true,
            dataRetentionDays = (settingsMap["dataRetentionDays"] as? Number)?.toInt() ?: 90
        )
    }
    
    /**
     * Call API endpoint
     */
    suspend fun callApiEndpoint(endpoint: String, params: Map<String, Any> = emptyMap()): Any? {
        return try {
            apiEndpoints[endpoint]?.invoke(params)
        } catch (e: Exception) {
            systemHealthMonitor.recordError("SystemIntegrator", "API call failed: $endpoint", e)
            mapOf("error" to e.message)
        }
    }
    
    /**
     * Get available API endpoints
     */
    fun getAvailableEndpoints(): List<String> {
        return apiEndpoints.keys.toList()
    }
    
    /**
     * Update integration status
     */
    private fun updateIntegrationStatus(success: Boolean, message: String) {
        _integrationStatus.value = IntegrationStatus(
            isIntegrated = success,
            lastUpdate = System.currentTimeMillis(),
            message = message,
            componentCount = 8, // Number of integrated components
            healthyComponents = if (success) 8 else 0
        )
    }
    
    /**
     * Graceful system shutdown
     */
    suspend fun shutdown() = withContext(Dispatchers.IO) {
        try {
            _systemState.value = SystemState.SHUTTING_DOWN
            
            // Stop automation engine
            automationEngine.stop()
            
            // Clean up components
            systemHealthMonitor.cleanup()
            resourceMonitor.cleanup()
            dataRetentionManager.cleanup()
            privacyController.cleanup()
            
            // Cancel coordination scope
            scope.cancel()
            
            _systemState.value = SystemState.STOPPED
            
        } catch (e: Exception) {
            _systemState.value = SystemState.ERROR
            throw e
        }
    }
    
    /**
     * Get system status summary
     */
    fun getSystemStatus(): SystemStatusSummary {
        return SystemStatusSummary(
            state = _systemState.value,
            integrationStatus = _integrationStatus.value,
            systemHealth = systemHealthMonitor.systemHealth.value,
            resourceUsage = resourceMonitor.resourceUsage.value,
            privacyCompliance = privacyController.getPrivacyComplianceReport(),
            availableEndpoints = apiEndpoints.keys.toList()
        )
    }
}

/**
 * System state enumeration
 */
enum class SystemState {
    INITIALIZING,
    RUNNING,
    SHUTTING_DOWN,
    STOPPED,
    ERROR
}

/**
 * Integration status
 */
data class IntegrationStatus(
    val isIntegrated: Boolean = false,
    val lastUpdate: Long = System.currentTimeMillis(),
    val message: String = "",
    val componentCount: Int = 0,
    val healthyComponents: Int = 0
)

/**
 * System status summary
 */
data class SystemStatusSummary(
    val state: SystemState,
    val integrationStatus: IntegrationStatus,
    val systemHealth: SystemHealth,
    val resourceUsage: ResourceUsage,
    val privacyCompliance: PrivacyComplianceReport,
    val availableEndpoints: List<String>
)

/**
 * Extension function to convert BehavioralContext to ML features
 */
private fun BehavioralContext.toMLFeatures(): Map<String, Double> {
    return mapOf(
        "hour_of_day" to this.timeContext.hourOfDay.toDouble(),
        "day_of_week" to this.timeContext.dayOfWeek.toDouble(),
        "app_usage_duration" to this.usageSnapshot.totalUsageTime.toDouble(),
        "notification_count" to this.usageSnapshot.notificationCount.toDouble(),
        "screen_interactions" to this.usageSnapshot.screenInteractions.toDouble(),
        "battery_level" to this.environmentContext.batteryLevel.toDouble(),
        "is_charging" to if (this.environmentContext.isCharging) 1.0 else 0.0,
        "wifi_connected" to if (this.environmentContext.wifiConnected) 1.0 else 0.0
    )
}