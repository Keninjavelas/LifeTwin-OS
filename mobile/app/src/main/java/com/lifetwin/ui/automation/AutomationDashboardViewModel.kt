package com.lifetwin.ui.automation

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.lifetwin.automation.*
import kotlinx.coroutines.flow.*
import kotlinx.coroutines.launch
import javax.inject.Inject
import dagger.hilt.android.lifecycle.HiltViewModel

/**
 * AutomationDashboardViewModel - ViewModel for automation dashboard UI
 * 
 * Manages dashboard state, metrics, and real-time updates
 */
@HiltViewModel
class AutomationDashboardViewModel @Inject constructor(
    private val automationEngine: AutomationEngine,
    private val automationLog: AutomationLog,
    private val resourceMonitor: ResourceMonitor
) : ViewModel() {
    
    // Dashboard state
    private val _dashboardState = MutableStateFlow(DashboardState())
    val dashboardState: StateFlow<DashboardState> = _dashboardState.asStateFlow()
    
    // Recent interventions
    private val _recentInterventions = MutableStateFlow<List<InterventionSummary>>(emptyList())
    val recentInterventions: StateFlow<List<InterventionSummary>> = _recentInterventions.asStateFlow()
    
    // Effectiveness metrics
    private val _effectivenessMetrics = MutableStateFlow(EffectivenessMetrics())
    val effectivenessMetrics: StateFlow<EffectivenessMetrics> = _effectivenessMetrics.asStateFlow()
    
    init {
        startDashboardUpdates()
    }
    
    /**
     * Start real-time dashboard updates
     */
    private fun startDashboardUpdates() {
        viewModelScope.launch {
            // Combine multiple data sources for dashboard updates
            combine(
                automationEngine.automationState,
                automationLog.recentLogs,
                resourceMonitor.resourceUsage
            ) { automationState, logs, resourceUsage ->
                updateDashboardState(automationState, logs, resourceUsage)
            }.collect()
        }
        
        // Update metrics periodically
        viewModelScope.launch {
            while (true) {
                updateEffectivenessMetrics()
                updateRecentInterventions()
                kotlinx.coroutines.delay(30000) // Update every 30 seconds
            }
        }
    }
    
    /**
     * Update dashboard state with latest data
     */
    private suspend fun updateDashboardState(
        automationState: AutomationState,
        logs: List<AutomationLogEntry>,
        resourceUsage: ResourceUsage
    ) {
        val todayStart = System.currentTimeMillis() - (24 * 60 * 60 * 1000)
        val todayInterventions = logs.count { it.timestamp >= todayStart }
        
        val effectiveness = calculateOverallEffectiveness(logs)
        val batteryImpact = calculateBatteryImpact(resourceUsage)
        
        val activityTimeline = generateActivityTimeline(logs)
        val weeklySummary = generateWeeklySummary(logs)
        
        _dashboardState.value = DashboardState(
            automationEnabled = automationState.enabled,
            lastUpdateTime = System.currentTimeMillis(),
            todayInterventions = todayInterventions,
            overallEffectiveness = effectiveness,
            batteryImpact = batteryImpact,
            activityTimeline = activityTimeline,
            weeklySummary = weeklySummary
        )
    }
    
    /**
     * Update effectiveness metrics
     */
    private suspend fun updateEffectivenessMetrics() {
        val logs = automationLog.getLogsInRange(
            System.currentTimeMillis() - (7 * 24 * 60 * 60 * 1000), // Last 7 days
            System.currentTimeMillis()
        )
        
        val successRate = calculateSuccessRate(logs)
        val userSatisfaction = calculateUserSatisfaction(logs)
        val avgResponseTime = calculateAverageResponseTime(logs)
        
        val weeklyTrends = calculateWeeklyTrends(logs)
        
        _effectivenessMetrics.value = EffectivenessMetrics(
            successRate = successRate,
            successRateTrend = calculateTrend(logs, "success_rate"),
            userSatisfaction = userSatisfaction,
            satisfactionTrend = calculateTrend(logs, "satisfaction"),
            avgResponseTimeMs = avgResponseTime,
            responseTimeTrend = calculateTrend(logs, "response_time"),
            weeklyTrends = weeklyTrends
        )
    }
    
    /**
     * Update recent interventions list
     */
    private suspend fun updateRecentInterventions() {
        val logs = automationLog.getRecentLogs(20) // Get last 20 interventions
        
        val interventions = logs.map { log ->
            InterventionSummary(
                id = log.id,
                type = mapToInterventionType(log.interventionType),
                title = generateInterventionTitle(log),
                description = log.description,
                timestamp = log.timestamp,
                effectiveness = log.effectiveness,
                userFeedback = log.userFeedback?.let { 
                    UserFeedback(it.rating, it.comment) 
                }
            )
        }
        
        _recentInterventions.value = interventions
    }
    
    /**
     * Calculate overall effectiveness from logs
     */
    private fun calculateOverallEffectiveness(logs: List<AutomationLogEntry>): Double {
        if (logs.isEmpty()) return 0.0
        
        return logs.map { it.effectiveness }.average()
    }
    
    /**
     * Calculate battery impact percentage
     */
    private fun calculateBatteryImpact(resourceUsage: ResourceUsage): Double {
        // Estimate battery impact based on resource usage
        val cpuImpact = resourceUsage.cpuUsagePercent * 0.4
        val memoryImpact = (resourceUsage.memoryUsageMB / 1000.0) * 0.3
        val baseImpact = 0.5 // Base automation overhead
        
        return (cpuImpact + memoryImpact + baseImpact).coerceAtMost(5.0)
    }
    
    /**
     * Generate activity timeline from logs
     */
    private fun generateActivityTimeline(logs: List<AutomationLogEntry>): List<ActivityEvent> {
        return logs.take(10).map { log ->
            ActivityEvent(
                type = "intervention",
                description = generateTimelineDescription(log),
                timestamp = log.timestamp
            )
        }
    }
    
    /**
     * Generate weekly summary from logs
     */
    private fun generateWeeklySummary(logs: List<AutomationLogEntry>): WeeklySummary {
        val weekStart = System.currentTimeMillis() - (7 * 24 * 60 * 60 * 1000)
        val weekLogs = logs.filter { it.timestamp >= weekStart }
        
        val totalInterventions = weekLogs.size
        val successRate = if (weekLogs.isNotEmpty()) {
            weekLogs.map { it.effectiveness }.average()
        } else 0.0
        
        val timeSaved = weekLogs.sumOf { it.timeSavedMinutes ?: 0 }
        
        return WeeklySummary(
            totalInterventions = totalInterventions,
            successRate = successRate,
            timeSavedMinutes = timeSaved
        )
    }
    
    /**
     * Calculate success rate from logs
     */
    private fun calculateSuccessRate(logs: List<AutomationLogEntry>): Double {
        if (logs.isEmpty()) return 0.0
        
        val successfulInterventions = logs.count { it.effectiveness >= 0.7 }
        return successfulInterventions.toDouble() / logs.size
    }
    
    /**
     * Calculate user satisfaction from feedback
     */
    private fun calculateUserSatisfaction(logs: List<AutomationLogEntry>): Double {
        val logsWithFeedback = logs.filter { it.userFeedback != null }
        if (logsWithFeedback.isEmpty()) return 0.0
        
        val avgRating = logsWithFeedback.mapNotNull { it.userFeedback?.rating }.average()
        return avgRating / 5.0 // Convert to 0-1 scale
    }
    
    /**
     * Calculate average response time
     */
    private fun calculateAverageResponseTime(logs: List<AutomationLogEntry>): Long {
        val responseTimes = logs.mapNotNull { it.responseTimeMs }
        return if (responseTimes.isNotEmpty()) {
            responseTimes.average().toLong()
        } else 0L
    }
    
    /**
     * Calculate weekly trends for visualization
     */
    private fun calculateWeeklyTrends(logs: List<AutomationLogEntry>): List<Double> {
        val trends = mutableListOf<Double>()
        val now = System.currentTimeMillis()
        
        for (i in 6 downTo 0) {
            val dayStart = now - (i * 24 * 60 * 60 * 1000)
            val dayEnd = dayStart + (24 * 60 * 60 * 1000)
            
            val dayLogs = logs.filter { it.timestamp in dayStart..dayEnd }
            val dayEffectiveness = if (dayLogs.isNotEmpty()) {
                dayLogs.map { it.effectiveness }.average()
            } else 0.0
            
            trends.add(dayEffectiveness)
        }
        
        return trends
    }
    
    /**
     * Calculate trend for a specific metric
     */
    private fun calculateTrend(logs: List<AutomationLogEntry>, metric: String): Double {
        if (logs.size < 2) return 0.0
        
        val now = System.currentTimeMillis()
        val halfPoint = now - (3.5 * 24 * 60 * 60 * 1000) // 3.5 days ago
        
        val recentLogs = logs.filter { it.timestamp >= halfPoint }
        val olderLogs = logs.filter { it.timestamp < halfPoint }
        
        if (recentLogs.isEmpty() || olderLogs.isEmpty()) return 0.0
        
        val recentValue = when (metric) {
            "success_rate" -> recentLogs.count { it.effectiveness >= 0.7 }.toDouble() / recentLogs.size
            "satisfaction" -> recentLogs.mapNotNull { it.userFeedback?.rating }.average() / 5.0
            "response_time" -> recentLogs.mapNotNull { it.responseTimeMs }.average()
            else -> 0.0
        }
        
        val olderValue = when (metric) {
            "success_rate" -> olderLogs.count { it.effectiveness >= 0.7 }.toDouble() / olderLogs.size
            "satisfaction" -> olderLogs.mapNotNull { it.userFeedback?.rating }.average() / 5.0
            "response_time" -> olderLogs.mapNotNull { it.responseTimeMs }.average()
            else -> 0.0
        }
        
        return if (olderValue > 0) {
            (recentValue - olderValue) / olderValue
        } else 0.0
    }
    
    /**
     * Map automation log intervention type to UI intervention type
     */
    private fun mapToInterventionType(type: String): InterventionType {
        return when (type.lowercase()) {
            "notification_limit" -> InterventionType.NOTIFICATION_LIMIT
            "break_reminder" -> InterventionType.BREAK_REMINDER
            "focus_mode" -> InterventionType.FOCUS_MODE
            "bedtime_reminder" -> InterventionType.BEDTIME_REMINDER
            "app_limit" -> InterventionType.APP_LIMIT
            else -> InterventionType.CUSTOM
        }
    }
    
    /**
     * Generate user-friendly intervention title
     */
    private fun generateInterventionTitle(log: AutomationLogEntry): String {
        return when (log.interventionType.lowercase()) {
            "notification_limit" -> "Notification Limit Reached"
            "break_reminder" -> "Break Time Reminder"
            "focus_mode" -> "Focus Mode Activated"
            "bedtime_reminder" -> "Bedtime Reminder"
            "app_limit" -> "App Usage Limit"
            else -> "Automation Intervention"
        }
    }
    
    /**
     * Generate timeline description
     */
    private fun generateTimelineDescription(log: AutomationLogEntry): String {
        val title = generateInterventionTitle(log)
        val effectiveness = (log.effectiveness * 100).toInt()
        return "$title (${effectiveness}% effective)"
    }
    
    /**
     * Refresh dashboard data manually
     */
    fun refreshDashboard() {
        viewModelScope.launch {
            updateEffectivenessMetrics()
            updateRecentInterventions()
        }
    }
    
    /**
     * Toggle automation on/off
     */
    fun toggleAutomation() {
        viewModelScope.launch {
            val currentState = automationEngine.automationState.value
            if (currentState.enabled) {
                automationEngine.disableAutomation()
            } else {
                automationEngine.enableAutomation()
            }
        }
    }
}