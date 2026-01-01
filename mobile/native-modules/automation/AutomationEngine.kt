package com.lifetwin.mlp.automation

import android.content.Context
import android.util.Log
import com.lifetwin.mlp.db.*
import com.lifetwin.mlp.ml.ModelInferenceManager
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*
import org.json.JSONObject
import java.util.concurrent.ConcurrentHashMap
import kotlin.math.min

private const val TAG = "AutomationEngine"

/**
 * Core orchestrator for the LifeTwin automation system.
 * Combines rule-based automation with RL policy recommendations for personalized interventions.
 */
class AutomationEngine(private val context: Context) {
    
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    
    // Core components
    private lateinit var ruleBasedSystem: RuleBasedSystem
    private lateinit var androidIntegration: AndroidIntegration
    private lateinit var automationLog: AutomationLog
    private lateinit var safetyWrapper: SafetyWrapper
    
    // State management
    private val _automationStatus = MutableStateFlow(AutomationStatus.STOPPED)
    val automationStatus: StateFlow<AutomationStatus> = _automationStatus.asStateFlow()
    
    private val _recentInterventions = MutableStateFlow<List<InterventionResult>>(emptyList())
    val recentInterventions: StateFlow<List<InterventionResult>> = _recentInterventions.asStateFlow()
    
    // User preferences
    private val userPreferences = ConcurrentHashMap<String, Any>()
    private var isPaused = false
    private var pauseEndTime = 0L
    
    /**
     * Initialize the automation engine and all its components
     */
    suspend fun initialize(): Boolean {
        return try {
            Log.i(TAG, "Initializing AutomationEngine...")
            
            // Initialize core components
            ruleBasedSystem = RuleBasedSystem(context)
            androidIntegration = AndroidIntegration(context)
            automationLog = AutomationLog(context)
            safetyWrapper = SafetyWrapper()
            
            // Initialize components
            val initResults = listOf(
                ruleBasedSystem.initialize(),
                androidIntegration.initialize(),
                automationLog.initialize(),
                safetyWrapper.initialize()
            )
            
            if (initResults.all { it }) {
                _automationStatus.value = AutomationStatus.RUNNING
                startAutomationLoop()
                Log.i(TAG, "AutomationEngine initialized successfully")
                true
            } else {
                Log.e(TAG, "Failed to initialize some automation components")
                false
            }
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize AutomationEngine", e)
            false
        }
    }
    
    /**
     * Main automation loop that continuously evaluates and executes interventions
     */
    private fun startAutomationLoop() {
        scope.launch {
            while (_automationStatus.value == AutomationStatus.RUNNING) {
                try {
                    // Check if automation is paused
                    if (isPaused && System.currentTimeMillis() < pauseEndTime) {
                        delay(60000) // Check every minute during pause
                        continue
                    } else if (isPaused) {
                        isPaused = false
                        Log.i(TAG, "Automation pause period ended")
                    }
                    
                    // Evaluate potential interventions
                    val interventions = evaluateInterventions()
                    
                    // Execute approved interventions
                    for (intervention in interventions) {
                        val result = executeIntervention(intervention)
                        updateRecentInterventions(result)
                    }
                    
                    // Wait before next evaluation (adaptive based on activity)
                    val delayMs = calculateNextEvaluationDelay()
                    delay(delayMs)
                    
                } catch (e: Exception) {
                    Log.e(TAG, "Error in automation loop", e)
                    delay(30000) // Wait 30 seconds before retrying
                }
            }
        }
    }
    
    /**
     * Evaluate potential interventions based on current behavioral context
     */
    suspend fun evaluateInterventions(): List<InterventionRecommendation> {
        val startTime = System.currentTimeMillis()
        
        try {
            // Get current behavioral context
            val context = getCurrentBehavioralContext()
            
            // Get rule-based recommendations
            val ruleRecommendations = ruleBasedSystem.evaluateRules(context)
            
            // TODO: Add RL policy recommendations when implemented
            // val rlRecommendations = rlPolicy.getRecommendations(context)
            
            // Combine and prioritize recommendations
            val allRecommendations = ruleRecommendations.toMutableList()
            
            // Apply safety constraints and user preferences
            val safeRecommendations = allRecommendations.filter { recommendation ->
                safetyWrapper.validateRecommendation(recommendation, context) &&
                isAllowedByUserPreferences(recommendation)
            }
            
            // Limit to prevent overwhelming the user
            val finalRecommendations = safeRecommendations.take(2)
            
            val processingTime = System.currentTimeMillis() - startTime
            Log.d(TAG, "Evaluated ${finalRecommendations.size} interventions in ${processingTime}ms")
            
            return finalRecommendations
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to evaluate interventions", e)
            return emptyList()
        }
    }
    
    /**
     * Execute a specific intervention recommendation
     */
    suspend fun executeIntervention(intervention: InterventionRecommendation): InterventionResult {
        val startTime = System.currentTimeMillis()
        
        return try {
            Log.i(TAG, "Executing intervention: ${intervention.type} - ${intervention.reasoning}")
            
            // Execute through Android integration
            val success = androidIntegration.executeIntervention(intervention)
            
            val result = InterventionResult(
                interventionId = intervention.id,
                executed = success,
                userResponse = UserResponse.PENDING,
                actualImpact = null,
                executionTime = System.currentTimeMillis() - startTime
            )
            
            // Log the intervention
            automationLog.logIntervention(intervention, result)
            
            result
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to execute intervention: ${intervention.id}", e)
            
            val result = InterventionResult(
                interventionId = intervention.id,
                executed = false,
                userResponse = UserResponse.ERROR,
                actualImpact = null,
                executionTime = System.currentTimeMillis() - startTime
            )
            
            automationLog.logIntervention(intervention, result)
            result
        }
    }
    
    /**
     * Update user feedback for a specific intervention
     */
    fun updateUserFeedback(interventionId: String, feedback: UserFeedback) {
        scope.launch {
            try {
                automationLog.updateFeedback(interventionId, feedback)
                
                // Update recent interventions list
                val updated = _recentInterventions.value.map { result ->
                    if (result.interventionId == interventionId) {
                        result.copy(userResponse = when (feedback.helpful) {
                            true -> UserResponse.ACCEPTED
                            false -> UserResponse.DISMISSED
                        })
                    } else {
                        result
                    }
                }
                _recentInterventions.value = updated
                
                Log.d(TAG, "Updated feedback for intervention $interventionId: helpful=${feedback.helpful}")
                
            } catch (e: Exception) {
                Log.e(TAG, "Failed to update user feedback", e)
            }
        }
    }
    
    /**
     * Pause automation for a specified duration
     */
    fun pauseAutomation(durationMinutes: Int) {
        isPaused = true
        pauseEndTime = System.currentTimeMillis() + (durationMinutes * 60 * 1000L)
        Log.i(TAG, "Automation paused for $durationMinutes minutes")
    }
    
    /**
     * Resume automation immediately
     */
    fun resumeAutomation() {
        isPaused = false
        pauseEndTime = 0L
        Log.i(TAG, "Automation resumed")
    }
    
    /**
     * Update user preferences for automation behavior
     */
    fun updateUserPreferences(preferences: Map<String, Any>) {
        userPreferences.clear()
        userPreferences.putAll(preferences)
        Log.d(TAG, "Updated user preferences: ${preferences.keys}")
    }
    
    /**
     * Get current automation status and metrics
     */
    fun getAutomationStatus(): AutomationStatus {
        return _automationStatus.value
    }
    
    /**
     * Get automation effectiveness metrics
     */
    suspend fun getEffectivenessMetrics(): AutomationMetrics {
        return try {
            automationLog.getEffectivenessMetrics()
        } catch (e: Exception) {
            Log.e(TAG, "Failed to get effectiveness metrics", e)
            AutomationMetrics()
        }
    }
    
    /**
     * Cleanup resources
     */
    fun cleanup() {
        _automationStatus.value = AutomationStatus.STOPPED
        scope.cancel()
        Log.i(TAG, "AutomationEngine cleaned up")
    }
    
    // Private helper methods
    
    private suspend fun getCurrentBehavioralContext(): BehavioralContext {
        val database = AppDatabase.getInstance(context)
        val currentTime = System.currentTimeMillis()
        
        // Get recent usage data
        val recentUsage = database.usageEventDao().getEventsByTimeRange(
            currentTime - (2 * 60 * 60 * 1000), // Last 2 hours
            currentTime
        )
        
        // Get recent notifications
        val recentNotifications = database.notificationEventDao().getEventsByTimeRange(
            currentTime - (60 * 60 * 1000), // Last hour
            currentTime
        )
        
        // Get today's summary
        val todaySummary = database.dailySummaryDao().getSummaryByDate(currentTime)
        
        return BehavioralContext(
            currentUsage = UsageSnapshot.fromEvents(recentUsage),
            recentPatterns = UsagePatterns.fromSummary(todaySummary),
            timeContext = TimeContext.fromTimestamp(currentTime),
            environmentContext = EnvironmentContext.getCurrent(context),
            userState = UserState.fromPreferences(userPreferences)
        )
    }
    
    private fun isAllowedByUserPreferences(recommendation: InterventionRecommendation): Boolean {
        // Check if intervention type is enabled
        val typeEnabled = userPreferences["${recommendation.type.name.lowercase()}_enabled"] as? Boolean ?: true
        
        // Check quiet hours
        val quietHoursEnabled = userPreferences["quiet_hours_enabled"] as? Boolean ?: false
        if (quietHoursEnabled) {
            val currentHour = java.util.Calendar.getInstance().get(java.util.Calendar.HOUR_OF_DAY)
            val quietStart = userPreferences["quiet_hours_start"] as? Int ?: 22
            val quietEnd = userPreferences["quiet_hours_end"] as? Int ?: 7
            
            if (currentHour >= quietStart || currentHour <= quietEnd) {
                return false
            }
        }
        
        return typeEnabled
    }
    
    private fun calculateNextEvaluationDelay(): Long {
        // Adaptive delay based on recent activity and time of day
        val baseDelay = 5 * 60 * 1000L // 5 minutes
        val currentHour = java.util.Calendar.getInstance().get(java.util.Calendar.HOUR_OF_DAY)
        
        return when {
            currentHour in 23..6 -> baseDelay * 4 // Less frequent at night
            currentHour in 9..17 -> baseDelay // Normal frequency during work hours
            else -> baseDelay * 2 // Moderate frequency during evening
        }
    }
    
    private fun updateRecentInterventions(result: InterventionResult) {
        val current = _recentInterventions.value.toMutableList()
        current.add(0, result) // Add to beginning
        
        // Keep only last 20 interventions
        if (current.size > 20) {
            current.removeAt(current.size - 1)
        }
        
        _recentInterventions.value = current
    }
}

// Data classes and enums

enum class AutomationStatus {
    STOPPED, RUNNING, PAUSED, ERROR
}

enum class InterventionType {
    BREAK_SUGGESTION,
    DND_ENABLE,
    APP_LIMIT_SUGGESTION,
    FOCUS_MODE_ENABLE,
    NOTIFICATION_REDUCTION,
    ACTIVITY_SUGGESTION
}

enum class UserResponse {
    PENDING, ACCEPTED, DISMISSED, IGNORED, ERROR
}

data class BehavioralContext(
    val currentUsage: UsageSnapshot,
    val recentPatterns: UsagePatterns,
    val timeContext: TimeContext,
    val environmentContext: EnvironmentContext,
    val userState: UserState
)

data class InterventionRecommendation(
    val id: String = java.util.UUID.randomUUID().toString(),
    val type: InterventionType,
    val trigger: String,
    val confidence: Float,
    val reasoning: String,
    val suggestedTiming: Long = System.currentTimeMillis(),
    val expectedImpact: ImpactPrediction? = null
)

data class InterventionResult(
    val interventionId: String,
    val executed: Boolean,
    val userResponse: UserResponse,
    val actualImpact: ImpactMeasurement?,
    val executionTime: Long
)

data class UserFeedback(
    val interventionId: String,
    val rating: Int, // 1-5 scale
    val helpful: Boolean,
    val timing: TimingFeedback,
    val comments: String? = null
)

enum class TimingFeedback {
    TOO_EARLY, PERFECT, TOO_LATE
}

data class ImpactPrediction(
    val energyChange: Float,
    val focusChange: Float,
    val moodChange: Float
)

data class ImpactMeasurement(
    val actualEnergyChange: Float,
    val actualFocusChange: Float,
    val actualMoodChange: Float
)

data class AutomationMetrics(
    val totalInterventions: Int = 0,
    val acceptanceRate: Float = 0f,
    val averageRating: Float = 0f,
    val effectivenessScore: Float = 0f
)

// Helper data classes for behavioral context

data class UsageSnapshot(
    val totalScreenTime: Long,
    val socialUsage: Long,
    val workUsage: Long,
    val notificationCount: Int,
    val appSwitches: Int
) {
    companion object {
        fun fromEvents(events: List<UsageEventEntity>): UsageSnapshot {
            val totalTime = events.sumOf { it.totalTimeInForeground }
            val socialTime = events.filter { isSocialApp(it.packageName) }.sumOf { it.totalTimeInForeground }
            val workTime = events.filter { isWorkApp(it.packageName) }.sumOf { it.totalTimeInForeground }
            
            return UsageSnapshot(
                totalScreenTime = totalTime,
                socialUsage = socialTime,
                workUsage = workTime,
                notificationCount = 0, // Will be filled from notification events
                appSwitches = events.size
            )
        }
        
        private fun isSocialApp(packageName: String): Boolean {
            return packageName.contains("facebook") || 
                   packageName.contains("instagram") || 
                   packageName.contains("twitter") ||
                   packageName.contains("tiktok") ||
                   packageName.contains("snapchat")
        }
        
        private fun isWorkApp(packageName: String): Boolean {
            return packageName.contains("office") ||
                   packageName.contains("gmail") ||
                   packageName.contains("slack") ||
                   packageName.contains("teams") ||
                   packageName.contains("zoom")
        }
    }
}

data class UsagePatterns(
    val averageDailyUsage: Long,
    val peakUsageHour: Int,
    val weekendVsWeekday: Float
) {
    companion object {
        fun fromSummary(summary: DailySummaryEntity?): UsagePatterns {
            return UsagePatterns(
                averageDailyUsage = summary?.totalScreenTime ?: 0L,
                peakUsageHour = summary?.mostCommonHour ?: 12,
                weekendVsWeekday = 1.0f // TODO: Calculate from historical data
            )
        }
    }
}

data class TimeContext(
    val hourOfDay: Int,
    val dayOfWeek: Int,
    val isWeekend: Boolean,
    val isWorkHour: Boolean
) {
    companion object {
        fun fromTimestamp(timestamp: Long): TimeContext {
            val calendar = java.util.Calendar.getInstance()
            calendar.timeInMillis = timestamp
            
            val hour = calendar.get(java.util.Calendar.HOUR_OF_DAY)
            val dayOfWeek = calendar.get(java.util.Calendar.DAY_OF_WEEK)
            val isWeekend = dayOfWeek == java.util.Calendar.SATURDAY || dayOfWeek == java.util.Calendar.SUNDAY
            val isWorkHour = hour in 9..17 && !isWeekend
            
            return TimeContext(hour, dayOfWeek, isWeekend, isWorkHour)
        }
    }
}

data class EnvironmentContext(
    val batteryLevel: Float,
    val isCharging: Boolean,
    val wifiConnected: Boolean
) {
    companion object {
        fun getCurrent(context: Context): EnvironmentContext {
            // TODO: Implement actual environment detection
            return EnvironmentContext(
                batteryLevel = 0.8f,
                isCharging = false,
                wifiConnected = true
            )
        }
    }
}

data class UserState(
    val currentMood: Float,
    val energyLevel: Float,
    val focusLevel: Float,
    val stressLevel: Float
) {
    companion object {
        fun fromPreferences(preferences: Map<String, Any>): UserState {
            return UserState(
                currentMood = preferences["current_mood"] as? Float ?: 0.5f,
                energyLevel = preferences["energy_level"] as? Float ?: 0.5f,
                focusLevel = preferences["focus_level"] as? Float ?: 0.5f,
                stressLevel = preferences["stress_level"] as? Float ?: 0.5f
            )
        }
    }
}