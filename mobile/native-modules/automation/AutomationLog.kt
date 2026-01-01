package com.lifetwin.mlp.automation

import android.content.Context
import android.util.Log
import com.lifetwin.mlp.db.AppDatabase
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.text.SimpleDateFormat
import java.util.*
import kotlin.math.roundToInt

private const val TAG = "AutomationLog"

/**
 * Comprehensive logging system for all automation activities and outcomes.
 */
class AutomationLog(private val context: Context) {
    
    private lateinit var database: AppDatabase
    
    suspend fun initialize(): Boolean {
        return try {
            Log.i(TAG, "Initializing AutomationLog...")
            database = AppDatabase.getInstance(context)
            Log.i(TAG, "AutomationLog initialized successfully")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize AutomationLog", e)
            false
        }
    }
    
    /**
     * Log an intervention and its result
     */
    suspend fun logIntervention(
        intervention: InterventionRecommendation,
        result: InterventionResult
    ) {
        withContext(Dispatchers.IO) {
            try {
                val logEntry = AutomationLogEntity(
                    id = java.util.UUID.randomUUID().toString(),
                    interventionId = intervention.id,
                    timestamp = System.currentTimeMillis(),
                    interventionType = intervention.type.name,
                    trigger = intervention.trigger,
                    reasoning = intervention.reasoning,
                    confidence = intervention.confidence,
                    executed = result.executed,
                    userResponse = result.userResponse.name,
                    executionTimeMs = result.executionTime,
                    feedbackRating = null,
                    feedbackComments = null,
                    helpful = null
                )
                
                database.automationLogDao().insert(logEntry)
                Log.d(TAG, "Logged intervention: ${intervention.type} (${intervention.id})")
                
            } catch (e: Exception) {
                Log.e(TAG, "Failed to log intervention", e)
            }
        }
    }
    
    /**
     * Update user feedback for a specific intervention
     */
    suspend fun updateFeedback(interventionId: String, feedback: UserFeedback) {
        withContext(Dispatchers.IO) {
            try {
                database.automationLogDao().updateFeedback(
                    interventionId = interventionId,
                    rating = feedback.rating,
                    comments = feedback.comments,
                    helpful = feedback.helpful
                )
                
                Log.d(TAG, "Updated feedback for intervention $interventionId: rating=${feedback.rating}")
                
            } catch (e: Exception) {
                Log.e(TAG, "Failed to update feedback", e)
            }
        }
    }
    
    /**
     * Get automation effectiveness metrics
     */
    suspend fun getEffectivenessMetrics(): AutomationMetrics {
        return withContext(Dispatchers.IO) {
            try {
                val logs = database.automationLogDao().getAllLogs()
                
                if (logs.isEmpty()) {
                    return@withContext AutomationMetrics()
                }
                
                val totalInterventions = logs.size
                val acceptedCount = logs.count { it.userResponse == UserResponse.ACCEPTED.name }
                val acceptanceRate = acceptedCount.toFloat() / totalInterventions.toFloat()
                
                val ratingsWithFeedback = logs.mapNotNull { it.feedbackRating }
                val averageRating = if (ratingsWithFeedback.isNotEmpty()) {
                    ratingsWithFeedback.average().toFloat()
                } else 0f
                
                // Calculate effectiveness score based on acceptance rate and ratings
                val effectivenessScore = (acceptanceRate * 0.6f) + (averageRating / 5f * 0.4f)
                
                AutomationMetrics(
                    totalInterventions = totalInterventions,
                    acceptanceRate = acceptanceRate,
                    averageRating = averageRating,
                    effectivenessScore = effectivenessScore
                )
                
            } catch (e: Exception) {
                Log.e(TAG, "Failed to calculate effectiveness metrics", e)
                AutomationMetrics()
            }
        }
    }
    
    /**
     * Get daily summary of automation activity
     */
    suspend fun getDailySummary(date: Long = System.currentTimeMillis()): DailyAutomationSummary {
        return withContext(Dispatchers.IO) {
            try {
                val calendar = Calendar.getInstance()
                calendar.timeInMillis = date
                calendar.set(Calendar.HOUR_OF_DAY, 0)
                calendar.set(Calendar.MINUTE, 0)
                calendar.set(Calendar.SECOND, 0)
                calendar.set(Calendar.MILLISECOND, 0)
                val dayStart = calendar.timeInMillis
                
                calendar.add(Calendar.DAY_OF_MONTH, 1)
                val dayEnd = calendar.timeInMillis
                
                val dayLogs = database.automationLogDao().getLogsByTimeRange(dayStart, dayEnd)
                
                val interventionsByType = dayLogs.groupBy { it.interventionType }
                val mostCommonType = interventionsByType.maxByOrNull { it.value.size }?.key
                
                val acceptedCount = dayLogs.count { it.userResponse == UserResponse.ACCEPTED.name }
                val dismissedCount = dayLogs.count { it.userResponse == UserResponse.DISMISSED.name }
                
                DailyAutomationSummary(
                    date = dayStart,
                    totalInterventions = dayLogs.size,
                    acceptedInterventions = acceptedCount,
                    dismissedInterventions = dismissedCount,
                    mostCommonType = mostCommonType,
                    averageConfidence = dayLogs.map { it.confidence }.average().toFloat(),
                    interventionsByHour = calculateInterventionsByHour(dayLogs)
                )
                
            } catch (e: Exception) {
                Log.e(TAG, "Failed to generate daily summary", e)
                DailyAutomationSummary(date = date)
            }
        }
    }
    
    /**
     * Get weekly summary of automation activity
     */
    suspend fun getWeeklySummary(weekStart: Long = getWeekStart()): WeeklyAutomationSummary {
        return withContext(Dispatchers.IO) {
            try {
                val weekEnd = weekStart + (7 * 24 * 60 * 60 * 1000L)
                val weekLogs = database.automationLogDao().getLogsByTimeRange(weekStart, weekEnd)
                
                val dailySummaries = mutableListOf<DailyAutomationSummary>()
                for (i in 0..6) {
                    val dayStart = weekStart + (i * 24 * 60 * 60 * 1000L)
                    dailySummaries.add(getDailySummary(dayStart))
                }
                
                val totalInterventions = weekLogs.size
                val acceptanceRate = if (totalInterventions > 0) {
                    weekLogs.count { it.userResponse == UserResponse.ACCEPTED.name }.toFloat() / totalInterventions
                } else 0f
                
                val improvementTrend = calculateImprovementTrend(dailySummaries)
                
                WeeklyAutomationSummary(
                    weekStart = weekStart,
                    totalInterventions = totalInterventions,
                    averageAcceptanceRate = acceptanceRate,
                    improvementTrend = improvementTrend,
                    dailySummaries = dailySummaries,
                    topTriggers = getTopTriggers(weekLogs)
                )
                
            } catch (e: Exception) {
                Log.e(TAG, "Failed to generate weekly summary", e)
                WeeklyAutomationSummary(weekStart = weekStart)
            }
        }
    }
    
    /**
     * Identify patterns in successful vs unsuccessful interventions
     */
    suspend fun getInterventionPatterns(): InterventionPatterns {
        return withContext(Dispatchers.IO) {
            try {
                val logs = database.automationLogDao().getAllLogs()
                val successful = logs.filter { it.userResponse == UserResponse.ACCEPTED.name }
                val unsuccessful = logs.filter { it.userResponse == UserResponse.DISMISSED.name }
                
                val successfulByType = successful.groupBy { it.interventionType }
                val unsuccessfulByType = unsuccessful.groupBy { it.interventionType }
                
                val successfulByHour = successful.groupBy { getHourFromTimestamp(it.timestamp) }
                val unsuccessfulByHour = unsuccessful.groupBy { getHourFromTimestamp(it.timestamp) }
                
                val bestTypes = successfulByType.keys.toList()
                val worstTypes = unsuccessfulByType.keys.toList()
                
                val bestHours = successfulByHour.keys.sorted()
                val worstHours = unsuccessfulByHour.keys.sorted()
                
                InterventionPatterns(
                    mostSuccessfulTypes = bestTypes,
                    leastSuccessfulTypes = worstTypes,
                    bestTimeHours = bestHours,
                    worstTimeHours = worstHours,
                    averageSuccessfulConfidence = successful.map { it.confidence }.average().toFloat(),
                    averageUnsuccessfulConfidence = unsuccessful.map { it.confidence }.average().toFloat()
                )
                
            } catch (e: Exception) {
                Log.e(TAG, "Failed to analyze intervention patterns", e)
                InterventionPatterns()
            }
        }
    }
    
    /**
     * Export automation data for analysis while preserving privacy
     */
    suspend fun exportAnonymizedData(): String {
        return withContext(Dispatchers.IO) {
            try {
                val logs = database.automationLogDao().getAllLogs()
                val exportData = JSONObject()
                
                exportData.put("export_timestamp", System.currentTimeMillis())
                exportData.put("total_interventions", logs.size)
                
                val anonymizedLogs = logs.map { log ->
                    JSONObject().apply {
                        put("type", log.interventionType)
                        put("trigger", log.trigger)
                        put("confidence", log.confidence)
                        put("executed", log.executed)
                        put("response", log.userResponse)
                        put("hour_of_day", getHourFromTimestamp(log.timestamp))
                        put("day_of_week", getDayOfWeekFromTimestamp(log.timestamp))
                        put("rating", log.feedbackRating ?: -1)
                    }
                }
                
                exportData.put("interventions", anonymizedLogs)
                exportData.toString(2)
                
            } catch (e: Exception) {
                Log.e(TAG, "Failed to export anonymized data", e)
                "{\"error\": \"Export failed\"}"
            }
        }
    }
    
    /**
     * Clean up old logs based on retention policy
     */
    suspend fun cleanupOldLogs(retentionDays: Int = 90) {
        withContext(Dispatchers.IO) {
            try {
                val cutoffTime = System.currentTimeMillis() - (retentionDays * 24 * 60 * 60 * 1000L)
                val deletedCount = database.automationLogDao().deleteLogsOlderThan(cutoffTime)
                Log.i(TAG, "Cleaned up $deletedCount old log entries (older than $retentionDays days)")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to cleanup old logs", e)
            }
        }
    }
    
    // Helper methods
    
    private fun calculateInterventionsByHour(logs: List<AutomationLogEntity>): Map<Int, Int> {
        return logs.groupBy { getHourFromTimestamp(it.timestamp) }
            .mapValues { it.value.size }
    }
    
    private fun getHourFromTimestamp(timestamp: Long): Int {
        val calendar = Calendar.getInstance()
        calendar.timeInMillis = timestamp
        return calendar.get(Calendar.HOUR_OF_DAY)
    }
    
    private fun getDayOfWeekFromTimestamp(timestamp: Long): Int {
        val calendar = Calendar.getInstance()
        calendar.timeInMillis = timestamp
        return calendar.get(Calendar.DAY_OF_WEEK)
    }
    
    private fun getWeekStart(date: Long = System.currentTimeMillis()): Long {
        val calendar = Calendar.getInstance()
        calendar.timeInMillis = date
        calendar.set(Calendar.DAY_OF_WEEK, Calendar.MONDAY)
        calendar.set(Calendar.HOUR_OF_DAY, 0)
        calendar.set(Calendar.MINUTE, 0)
        calendar.set(Calendar.SECOND, 0)
        calendar.set(Calendar.MILLISECOND, 0)
        return calendar.timeInMillis
    }
    
    private fun calculateImprovementTrend(dailySummaries: List<DailyAutomationSummary>): Float {
        if (dailySummaries.size < 2) return 0f
        
        val firstHalf = dailySummaries.take(dailySummaries.size / 2)
        val secondHalf = dailySummaries.drop(dailySummaries.size / 2)
        
        val firstHalfAcceptance = firstHalf.map { 
            if (it.totalInterventions > 0) it.acceptedInterventions.toFloat() / it.totalInterventions else 0f 
        }.average().toFloat()
        
        val secondHalfAcceptance = secondHalf.map { 
            if (it.totalInterventions > 0) it.acceptedInterventions.toFloat() / it.totalInterventions else 0f 
        }.average().toFloat()
        
        return secondHalfAcceptance - firstHalfAcceptance
    }
    
    private fun getTopTriggers(logs: List<AutomationLogEntity>): List<String> {
        return logs.groupBy { it.trigger }
            .toList()
            .sortedByDescending { it.second.size }
            .take(5)
            .map { it.first }
    }
}

// Data classes for summaries and patterns

data class DailyAutomationSummary(
    val date: Long,
    val totalInterventions: Int = 0,
    val acceptedInterventions: Int = 0,
    val dismissedInterventions: Int = 0,
    val mostCommonType: String? = null,
    val averageConfidence: Float = 0f,
    val interventionsByHour: Map<Int, Int> = emptyMap()
)

data class WeeklyAutomationSummary(
    val weekStart: Long,
    val totalInterventions: Int = 0,
    val averageAcceptanceRate: Float = 0f,
    val improvementTrend: Float = 0f,
    val dailySummaries: List<DailyAutomationSummary> = emptyList(),
    val topTriggers: List<String> = emptyList()
)

data class InterventionPatterns(
    val mostSuccessfulTypes: List<String> = emptyList(),
    val leastSuccessfulTypes: List<String> = emptyList(),
    val bestTimeHours: List<Int> = emptyList(),
    val worstTimeHours: List<Int> = emptyList(),
    val averageSuccessfulConfidence: Float = 0f,
    val averageUnsuccessfulConfidence: Float = 0f
)

// Database entity for automation logs is defined in Entities.kt