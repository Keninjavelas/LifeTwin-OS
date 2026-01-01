package com.lifetwin.automation

import android.content.Context
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import kotlinx.serialization.encodeToString
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicLong
import kotlin.math.max

/**
 * SystemHealthMonitor - Comprehensive system health monitoring and diagnostics
 * 
 * Implements Requirements:
 * - 8.6: Comprehensive error tracking and reporting
 * - System health checks and status monitoring
 * - Diagnostic tools for troubleshooting automation issues
 */
class SystemHealthMonitor(
    private val context: Context,
    private val automationEngine: AutomationEngine,
    private val resourceMonitor: ResourceMonitor,
    private val privacyController: PrivacyController
) {
    private val scope = CoroutineScope(Dispatchers.Default + SupervisorJob())
    
    // Health status tracking
    private val _systemHealth = MutableStateFlow(SystemHealth())
    val systemHealth: StateFlow<SystemHealth> = _systemHealth.asStateFlow()
    
    // Error tracking
    private val errorCounts = ConcurrentHashMap<String, AtomicLong>()
    private val recentErrors = mutableListOf<ErrorReport>()
    private val maxRecentErrors = 100
    
    // Performance metrics
    private val performanceMetrics = ConcurrentHashMap<String, PerformanceMetric>()
    
    // Component health status
    private val componentHealth = ConcurrentHashMap<String, ComponentHealth>()
    
    // Diagnostic results
    private val _diagnosticResults = MutableStateFlow<List<DiagnosticResult>>(emptyList())
    val diagnosticResults: StateFlow<List<DiagnosticResult>> = _diagnosticResults.asStateFlow()
    
    init {
        startHealthMonitoring()
        initializeComponentHealth()
    }
    
    /**
     * Start continuous system health monitoring
     */
    private fun startHealthMonitoring() {
        scope.launch {
            while (isActive) {
                try {
                    updateSystemHealth()
                    checkComponentHealth()
                    cleanupOldErrors()
                    delay(30000) // Check every 30 seconds
                } catch (e: Exception) {
                    recordError("HealthMonitor", "Monitoring loop error", e)
                    delay(60000) // Wait longer on error
                }
            }
        }
    }
    
    /**
     * Initialize component health tracking
     */
    private fun initializeComponentHealth() {
        val components = listOf(
            "AutomationEngine",
            "RuleBasedSystem", 
            "AndroidIntegration",
            "ResourceMonitor",
            "PrivacyController",
            "DataRetentionManager",
            "PerformanceOptimizer"
        )
        
        components.forEach { component ->
            componentHealth[component] = ComponentHealth(
                name = component,
                status = HealthStatus.UNKNOWN,
                lastCheck = System.currentTimeMillis(),
                errorCount = 0,
                uptime = 0L
            )
        }
    }
    
    /**
     * Record an error for tracking and analysis
     */
    fun recordError(component: String, message: String, exception: Throwable? = null) {
        val errorKey = "$component:${exception?.javaClass?.simpleName ?: "Unknown"}"
        errorCounts.computeIfAbsent(errorKey) { AtomicLong(0) }.incrementAndGet()
        
        val errorReport = ErrorReport(
            timestamp = System.currentTimeMillis(),
            component = component,
            message = message,
            exceptionType = exception?.javaClass?.simpleName,
            stackTrace = exception?.stackTraceToString(),
            severity = determineSeverity(exception)
        )
        
        synchronized(recentErrors) {
            recentErrors.add(errorReport)
            if (recentErrors.size > maxRecentErrors) {
                recentErrors.removeAt(0)
            }
        }
        
        // Update component health
        updateComponentHealth(component, HealthStatus.ERROR)
    }
    
    /**
     * Record performance metric
     */
    fun recordPerformanceMetric(operation: String, duration: Long, success: Boolean = true) {
        val metric = performanceMetrics.computeIfAbsent(operation) { 
            PerformanceMetric(operation = operation)
        }
        
        synchronized(metric) {
            metric.totalCalls++
            metric.totalDuration += duration
            metric.averageDuration = metric.totalDuration / metric.totalCalls
            metric.maxDuration = max(metric.maxDuration, duration)
            
            if (success) {
                metric.successCount++
            } else {
                metric.failureCount++
            }
            
            metric.successRate = (metric.successCount.toDouble() / metric.totalCalls) * 100.0
            metric.lastUpdated = System.currentTimeMillis()
        }
    }
    
    /**
     * Update overall system health status
     */
    private suspend fun updateSystemHealth() = withContext(Dispatchers.IO) {
        val currentTime = System.currentTimeMillis()
        
        // Calculate error rates
        val recentErrorCount = synchronized(recentErrors) {
            recentErrors.count { currentTime - it.timestamp < 3600000 } // Last hour
        }
        
        val criticalErrorCount = synchronized(recentErrors) {
            recentErrors.count { 
                it.severity == ErrorSeverity.CRITICAL && 
                currentTime - it.timestamp < 3600000 
            }
        }
        
        // Get resource status
        val resourceUsage = resourceMonitor.resourceUsage.value
        
        // Determine overall health status
        val overallStatus = when {
            criticalErrorCount > 0 -> HealthStatus.CRITICAL
            recentErrorCount > 10 -> HealthStatus.ERROR
            resourceUsage.batteryLevel < 10 -> HealthStatus.WARNING
            resourceUsage.cpuUsagePercent > 90 -> HealthStatus.WARNING
            else -> HealthStatus.HEALTHY
        }
        
        val health = SystemHealth(
            overallStatus = overallStatus,
            uptime = currentTime - getSystemStartTime(),
            errorRate = recentErrorCount.toDouble(),
            criticalErrors = criticalErrorCount,
            componentCount = componentHealth.size,
            healthyComponents = componentHealth.values.count { it.status == HealthStatus.HEALTHY },
            lastUpdate = currentTime,
            memoryUsage = resourceUsage.memoryUsageMB,
            cpuUsage = resourceUsage.cpuUsagePercent,
            batteryLevel = resourceUsage.batteryLevel
        )
        
        _systemHealth.value = health
    }
    
    /**
     * Check health of individual components
     */
    private suspend fun checkComponentHealth() = withContext(Dispatchers.IO) {
        componentHealth.forEach { (name, health) ->
            try {
                val status = when (name) {
                    "AutomationEngine" -> checkAutomationEngineHealth()
                    "ResourceMonitor" -> checkResourceMonitorHealth()
                    "PrivacyController" -> checkPrivacyControllerHealth()
                    else -> checkGenericComponentHealth(name)
                }
                
                updateComponentHealth(name, status)
            } catch (e: Exception) {
                recordError("HealthMonitor", "Failed to check $name health", e)
                updateComponentHealth(name, HealthStatus.ERROR)
            }
        }
    }
    
    /**
     * Check AutomationEngine health
     */
    private suspend fun checkAutomationEngineHealth(): HealthStatus {
        return try {
            val isRunning = automationEngine.isRunning()
            val hasRecentActivity = automationEngine.getLastActivityTime() > 
                                  System.currentTimeMillis() - 300000 // 5 minutes
            
            when {
                !isRunning -> HealthStatus.CRITICAL
                !hasRecentActivity -> HealthStatus.WARNING
                else -> HealthStatus.HEALTHY
            }
        } catch (e: Exception) {
            HealthStatus.ERROR
        }
    }
    
    /**
     * Check ResourceMonitor health
     */
    private suspend fun checkResourceMonitorHealth(): HealthStatus {
        return try {
            val usage = resourceMonitor.resourceUsage.value
            val isRecent = System.currentTimeMillis() - usage.timestamp < 60000 // 1 minute
            
            when {
                !isRecent -> HealthStatus.ERROR
                usage.batteryLevel < 5 -> HealthStatus.CRITICAL
                usage.cpuUsagePercent > 95 -> HealthStatus.WARNING
                else -> HealthStatus.HEALTHY
            }
        } catch (e: Exception) {
            HealthStatus.ERROR
        }
    }
    
    /**
     * Check PrivacyController health
     */
    private suspend fun checkPrivacyControllerHealth(): HealthStatus {
        return try {
            val report = privacyController.getPrivacyComplianceReport()
            
            when {
                !report.dataLocalProcessing -> HealthStatus.CRITICAL
                !report.encryptionActive -> HealthStatus.CRITICAL
                report.complianceScore < 70 -> HealthStatus.WARNING
                else -> HealthStatus.HEALTHY
            }
        } catch (e: Exception) {
            HealthStatus.ERROR
        }
    }
    
    /**
     * Check generic component health based on error rates
     */
    private fun checkGenericComponentHealth(componentName: String): HealthStatus {
        val recentErrors = synchronized(recentErrors) {
            recentErrors.count { 
                it.component == componentName && 
                System.currentTimeMillis() - it.timestamp < 3600000 
            }
        }
        
        return when {
            recentErrors > 5 -> HealthStatus.ERROR
            recentErrors > 2 -> HealthStatus.WARNING
            else -> HealthStatus.HEALTHY
        }
    }
    
    /**
     * Update component health status
     */
    private fun updateComponentHealth(componentName: String, status: HealthStatus) {
        val currentHealth = componentHealth[componentName] ?: return
        
        val updatedHealth = currentHealth.copy(
            status = status,
            lastCheck = System.currentTimeMillis(),
            errorCount = if (status == HealthStatus.ERROR || status == HealthStatus.CRITICAL) {
                currentHealth.errorCount + 1
            } else currentHealth.errorCount
        )
        
        componentHealth[componentName] = updatedHealth
    }
    
    /**
     * Run comprehensive system diagnostics
     */
    suspend fun runDiagnostics(): List<DiagnosticResult> = withContext(Dispatchers.IO) {
        val results = mutableListOf<DiagnosticResult>()
        
        // Memory diagnostic
        results.add(runMemoryDiagnostic())
        
        // Performance diagnostic
        results.add(runPerformanceDiagnostic())
        
        // Error pattern diagnostic
        results.add(runErrorPatternDiagnostic())
        
        // Component connectivity diagnostic
        results.add(runConnectivityDiagnostic())
        
        // Privacy compliance diagnostic
        results.add(runPrivacyDiagnostic())
        
        // Battery usage diagnostic
        results.add(runBatteryDiagnostic())
        
        _diagnosticResults.value = results
        results
    }
    
    /**
     * Run memory usage diagnostic
     */
    private fun runMemoryDiagnostic(): DiagnosticResult {
        val usage = resourceMonitor.resourceUsage.value
        val memoryUsage = usage.memoryUsageMB
        
        val status = when {
            memoryUsage > 1000 -> DiagnosticStatus.CRITICAL
            memoryUsage > 500 -> DiagnosticStatus.WARNING
            else -> DiagnosticStatus.HEALTHY
        }
        
        val recommendations = when (status) {
            DiagnosticStatus.CRITICAL -> listOf(
                "Restart the application to free memory",
                "Clear application cache",
                "Reduce automation frequency"
            )
            DiagnosticStatus.WARNING -> listOf(
                "Monitor memory usage closely",
                "Consider clearing cache"
            )
            else -> emptyList()
        }
        
        return DiagnosticResult(
            name = "Memory Usage",
            status = status,
            message = "Memory usage: ${memoryUsage.toInt()}MB",
            details = "Current memory consumption and optimization recommendations",
            recommendations = recommendations,
            timestamp = System.currentTimeMillis()
        )
    }
    
    /**
     * Run performance diagnostic
     */
    private fun runPerformanceDiagnostic(): DiagnosticResult {
        val slowOperations = performanceMetrics.values.filter { 
            it.averageDuration > 1000 // Slower than 1 second
        }
        
        val status = when {
            slowOperations.any { it.averageDuration > 5000 } -> DiagnosticStatus.CRITICAL
            slowOperations.isNotEmpty() -> DiagnosticStatus.WARNING
            else -> DiagnosticStatus.HEALTHY
        }
        
        val recommendations = if (slowOperations.isNotEmpty()) {
            listOf(
                "Optimize slow operations: ${slowOperations.joinToString { it.operation }}",
                "Consider increasing batch sizes",
                "Review database query performance"
            )
        } else {
            emptyList()
        }
        
        return DiagnosticResult(
            name = "Performance",
            status = status,
            message = "Found ${slowOperations.size} slow operations",
            details = "Performance analysis of system operations",
            recommendations = recommendations,
            timestamp = System.currentTimeMillis()
        )
    }
    
    /**
     * Run error pattern diagnostic
     */
    private fun runErrorPatternDiagnostic(): DiagnosticResult {
        val currentTime = System.currentTimeMillis()
        val recentErrors = synchronized(recentErrors) {
            recentErrors.filter { currentTime - it.timestamp < 3600000 }
        }
        
        val errorsByComponent = recentErrors.groupBy { it.component }
        val problematicComponents = errorsByComponent.filter { it.value.size > 3 }
        
        val status = when {
            problematicComponents.isNotEmpty() -> DiagnosticStatus.WARNING
            recentErrors.size > 10 -> DiagnosticStatus.WARNING
            else -> DiagnosticStatus.HEALTHY
        }
        
        val recommendations = if (problematicComponents.isNotEmpty()) {
            listOf(
                "Investigate components with high error rates: ${problematicComponents.keys.joinToString()}",
                "Check logs for recurring error patterns",
                "Consider component restart or reconfiguration"
            )
        } else {
            emptyList()
        }
        
        return DiagnosticResult(
            name = "Error Patterns",
            status = status,
            message = "${recentErrors.size} errors in last hour",
            details = "Analysis of recent error patterns and frequencies",
            recommendations = recommendations,
            timestamp = System.currentTimeMillis()
        )
    }
    
    /**
     * Run component connectivity diagnostic
     */
    private fun runConnectivityDiagnostic(): DiagnosticResult {
        val unhealthyComponents = componentHealth.values.filter { 
            it.status == HealthStatus.ERROR || it.status == HealthStatus.CRITICAL 
        }
        
        val status = when {
            unhealthyComponents.any { it.status == HealthStatus.CRITICAL } -> DiagnosticStatus.CRITICAL
            unhealthyComponents.isNotEmpty() -> DiagnosticStatus.WARNING
            else -> DiagnosticStatus.HEALTHY
        }
        
        val recommendations = if (unhealthyComponents.isNotEmpty()) {
            listOf(
                "Restart unhealthy components: ${unhealthyComponents.joinToString { it.name }}",
                "Check component dependencies",
                "Verify system permissions"
            )
        } else {
            emptyList()
        }
        
        return DiagnosticResult(
            name = "Component Connectivity",
            status = status,
            message = "${unhealthyComponents.size} unhealthy components",
            details = "Health status of system components",
            recommendations = recommendations,
            timestamp = System.currentTimeMillis()
        )
    }
    
    /**
     * Run privacy compliance diagnostic
     */
    private fun runPrivacyDiagnostic(): DiagnosticResult {
        val report = privacyController.getPrivacyComplianceReport()
        
        val status = when {
            !report.dataLocalProcessing || !report.encryptionActive -> DiagnosticStatus.CRITICAL
            report.complianceScore < 80 -> DiagnosticStatus.WARNING
            else -> DiagnosticStatus.HEALTHY
        }
        
        val recommendations = when {
            !report.dataLocalProcessing -> listOf("Ensure all data processing remains local")
            !report.encryptionActive -> listOf("Enable data encryption")
            report.complianceScore < 80 -> listOf("Review privacy settings", "Update consent preferences")
            else -> emptyList()
        }
        
        return DiagnosticResult(
            name = "Privacy Compliance",
            status = status,
            message = "Compliance score: ${report.complianceScore.toInt()}%",
            details = "Privacy and data protection compliance status",
            recommendations = recommendations,
            timestamp = System.currentTimeMillis()
        )
    }
    
    /**
     * Run battery usage diagnostic
     */
    private fun runBatteryDiagnostic(): DiagnosticResult {
        val usage = resourceMonitor.resourceUsage.value
        val batteryStats = resourceMonitor.getBatteryStatistics()
        
        val status = when {
            usage.batteryLevel < 10 -> DiagnosticStatus.CRITICAL
            usage.batteryLevel < 20 || batteryStats.hourlyDrainRate > 20 -> DiagnosticStatus.WARNING
            else -> DiagnosticStatus.HEALTHY
        }
        
        val recommendations = when (status) {
            DiagnosticStatus.CRITICAL -> listOf(
                "Enable battery saver mode",
                "Reduce automation frequency",
                "Disable non-essential features"
            )
            DiagnosticStatus.WARNING -> listOf(
                "Monitor battery usage",
                "Consider reducing processing frequency"
            )
            else -> emptyList()
        }
        
        return DiagnosticResult(
            name = "Battery Usage",
            status = status,
            message = "Battery: ${usage.batteryLevel}%, Drain: ${batteryStats.hourlyDrainRate.toInt()}%/hr",
            details = "Battery level and consumption analysis",
            recommendations = recommendations,
            timestamp = System.currentTimeMillis()
        )
    }
    
    /**
     * Get system health report
     */
    fun getHealthReport(): SystemHealthReport {
        val health = _systemHealth.value
        val components = componentHealth.values.toList()
        val recentErrorsCount = synchronized(recentErrors) {
            recentErrors.count { System.currentTimeMillis() - it.timestamp < 3600000 }
        }
        
        return SystemHealthReport(
            overallHealth = health,
            componentHealth = components,
            recentErrorCount = recentErrorsCount,
            performanceMetrics = performanceMetrics.values.toList(),
            topErrors = getTopErrors(),
            recommendations = generateHealthRecommendations(health, components)
        )
    }
    
    /**
     * Get top error types
     */
    private fun getTopErrors(): List<Pair<String, Long>> {
        return errorCounts.entries
            .sortedByDescending { it.value.get() }
            .take(5)
            .map { it.key to it.value.get() }
    }
    
    /**
     * Generate health recommendations
     */
    private fun generateHealthRecommendations(
        health: SystemHealth, 
        components: List<ComponentHealth>
    ): List<String> {
        val recommendations = mutableListOf<String>()
        
        if (health.overallStatus != HealthStatus.HEALTHY) {
            recommendations.add("System health requires attention")
        }
        
        if (health.errorRate > 5) {
            recommendations.add("High error rate detected - investigate recent errors")
        }
        
        val unhealthyComponents = components.filter { it.status != HealthStatus.HEALTHY }
        if (unhealthyComponents.isNotEmpty()) {
            recommendations.add("Restart unhealthy components: ${unhealthyComponents.joinToString { it.name }}")
        }
        
        if (health.memoryUsage > 800) {
            recommendations.add("High memory usage - consider clearing cache")
        }
        
        if (health.batteryLevel < 20) {
            recommendations.add("Low battery - enable power saving mode")
        }
        
        return recommendations
    }
    
    /**
     * Determine error severity
     */
    private fun determineSeverity(exception: Throwable?): ErrorSeverity {
        return when (exception) {
            is OutOfMemoryError -> ErrorSeverity.CRITICAL
            is SecurityException -> ErrorSeverity.CRITICAL
            is IllegalStateException -> ErrorSeverity.HIGH
            is RuntimeException -> ErrorSeverity.MEDIUM
            else -> ErrorSeverity.LOW
        }
    }
    
    /**
     * Get system start time (placeholder)
     */
    private fun getSystemStartTime(): Long {
        // This would track actual system start time
        return System.currentTimeMillis() - 3600000 // Placeholder: 1 hour ago
    }
    
    /**
     * Clean up old errors
     */
    private fun cleanupOldErrors() {
        val cutoffTime = System.currentTimeMillis() - 86400000 // 24 hours
        
        synchronized(recentErrors) {
            recentErrors.removeAll { it.timestamp < cutoffTime }
        }
    }
    
    /**
     * Clean up resources
     */
    fun cleanup() {
        scope.cancel()
        errorCounts.clear()
        recentErrors.clear()
        performanceMetrics.clear()
        componentHealth.clear()
    }
}

/**
 * System health status
 */
data class SystemHealth(
    val overallStatus: HealthStatus = HealthStatus.UNKNOWN,
    val uptime: Long = 0L,
    val errorRate: Double = 0.0,
    val criticalErrors: Int = 0,
    val componentCount: Int = 0,
    val healthyComponents: Int = 0,
    val lastUpdate: Long = System.currentTimeMillis(),
    val memoryUsage: Double = 0.0,
    val cpuUsage: Double = 0.0,
    val batteryLevel: Int = -1
)

/**
 * Component health status
 */
data class ComponentHealth(
    val name: String,
    val status: HealthStatus,
    val lastCheck: Long,
    val errorCount: Int,
    val uptime: Long
)

/**
 * Health status enumeration
 */
enum class HealthStatus {
    HEALTHY,
    WARNING,
    ERROR,
    CRITICAL,
    UNKNOWN
}

/**
 * Error report
 */
@Serializable
data class ErrorReport(
    val timestamp: Long,
    val component: String,
    val message: String,
    val exceptionType: String?,
    val stackTrace: String?,
    val severity: ErrorSeverity
)

/**
 * Error severity levels
 */
@Serializable
enum class ErrorSeverity {
    LOW,
    MEDIUM,
    HIGH,
    CRITICAL
}

/**
 * Performance metric tracking
 */
data class PerformanceMetric(
    val operation: String,
    var totalCalls: Long = 0,
    var successCount: Long = 0,
    var failureCount: Long = 0,
    var totalDuration: Long = 0,
    var averageDuration: Long = 0,
    var maxDuration: Long = 0,
    var successRate: Double = 0.0,
    var lastUpdated: Long = System.currentTimeMillis()
)

/**
 * Diagnostic result
 */
data class DiagnosticResult(
    val name: String,
    val status: DiagnosticStatus,
    val message: String,
    val details: String,
    val recommendations: List<String>,
    val timestamp: Long
)

/**
 * Diagnostic status
 */
enum class DiagnosticStatus {
    HEALTHY,
    WARNING,
    CRITICAL
}

/**
 * Comprehensive system health report
 */
data class SystemHealthReport(
    val overallHealth: SystemHealth,
    val componentHealth: List<ComponentHealth>,
    val recentErrorCount: Int,
    val performanceMetrics: List<PerformanceMetric>,
    val topErrors: List<Pair<String, Long>>,
    val recommendations: List<String>
)