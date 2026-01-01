package com.lifetwin.ui.automation

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.lifetwin.automation.*
import kotlinx.coroutines.flow.*
import kotlinx.coroutines.launch
import javax.inject.Inject
import dagger.hilt.android.lifecycle.HiltViewModel

/**
 * FeedbackSystemViewModel - ViewModel for user feedback collection and analysis
 * 
 * Manages feedback collection, analytics, and insights generation
 */
@HiltViewModel
class FeedbackSystemViewModel @Inject constructor(
    private val automationLog: AutomationLog,
    private val automationEngine: AutomationEngine
) : ViewModel() {
    
    // Feedback system state
    private val _feedbackState = MutableStateFlow(FeedbackSystemState())
    val feedbackState: StateFlow<FeedbackSystemState> = _feedbackState.asStateFlow()
    
    // Pending feedback requests
    private val _pendingFeedback = MutableStateFlow<List<InterventionSummary>>(emptyList())
    val pendingFeedback: StateFlow<List<InterventionSummary>> = _pendingFeedback.asStateFlow()
    
    // Feedback history
    private val _feedbackHistory = MutableStateFlow<List<FeedbackEntry>>(emptyList())
    val feedbackHistory: StateFlow<List<FeedbackEntry>> = _feedbackHistory.asStateFlow()
    
    // Feedback analytics
    private val _feedbackAnalytics = MutableStateFlow(FeedbackAnalytics())
    val feedbackAnalytics: StateFlow<FeedbackAnalytics> = _feedbackAnalytics.asStateFlow()
    
    init {
        startFeedbackMonitoring()
        loadFeedbackHistory()
        updateFeedbackAnalytics()
    }
    
    /**
     * Start monitoring for feedback opportunities
     */
    private fun startFeedbackMonitoring() {
        viewModelScope.launch {
            automationLog.recentLogs.collect { logs ->
                updatePendingFeedback(logs)
            }
        }
    }
    
    /**
     * Update pending feedback based on recent interventions
     */
    private suspend fun updatePendingFeedback(logs: List<AutomationLogEntry>) {
        val currentTime = System.currentTimeMillis()
        val feedbackDelay = _feedbackState.value.feedbackPromptDelay
        
        // Find interventions that need feedback
        val needsFeedback = logs.filter { log ->
            log.userFeedback == null && // No feedback yet
            (currentTime - log.timestamp) >= feedbackDelay && // Enough time has passed
            (currentTime - log.timestamp) <= (feedbackDelay * 4) // Not too old
        }
        
        val pendingInterventions = needsFeedback.map { log ->
            InterventionSummary(
                id = log.id,
                type = mapToInterventionType(log.interventionType),
                title = generateInterventionTitle(log),
                description = log.description,
                timestamp = log.timestamp,
                effectiveness = log.effectiveness
            )
        }
        
        _pendingFeedback.value = pendingInterventions.take(3) // Limit to 3 pending
    }
    
    /**
     * Submit user feedback for an intervention
     */
    fun submitFeedback(interventionId: String, rating: Int, comment: String) {
        viewModelScope.launch {
            val feedback = UserFeedback(
                rating = rating,
                comment = comment,
                timestamp = System.currentTimeMillis()
            )
            
            // Update automation log with feedback
            automationLog.updateFeedback(interventionId, feedback)
            
            // Add to feedback history
            val entry = FeedbackEntry(
                id = generateFeedbackId(),
                interventionId = interventionId,
                interventionTitle = getPendingInterventionTitle(interventionId),
                rating = rating,
                comment = comment,
                timestamp = System.currentTimeMillis()
            )
            
            val currentHistory = _feedbackHistory.value.toMutableList()
            currentHistory.add(0, entry) // Add to beginning
            _feedbackHistory.value = currentHistory
            
            // Remove from pending
            _pendingFeedback.value = _pendingFeedback.value.filter { it.id != interventionId }
            
            // Update analytics
            updateFeedbackAnalytics()
            
            // Apply feedback to improve automation
            applyFeedbackLearning(interventionId, rating, comment)
        }
    }
    
    /**
     * Dismiss feedback request without rating
     */
    fun dismissFeedback(interventionId: String) {
        _pendingFeedback.value = _pendingFeedback.value.filter { it.id != interventionId }
    }
    
    /**
     * Load feedback history from storage
     */
    private fun loadFeedbackHistory() {
        viewModelScope.launch {
            val logs = automationLog.getLogsWithFeedback()
            
            val history = logs.mapNotNull { log ->
                log.userFeedback?.let { feedback ->
                    FeedbackEntry(
                        id = "${log.id}_feedback",
                        interventionId = log.id,
                        interventionTitle = generateInterventionTitle(log),
                        rating = feedback.rating,
                        comment = feedback.comment,
                        timestamp = feedback.timestamp
                    )
                }
            }.sortedByDescending { it.timestamp }
            
            _feedbackHistory.value = history
        }
    }
    
    /**
     * Update feedback analytics and insights
     */
    private fun updateFeedbackAnalytics() {
        viewModelScope.launch {
            val history = _feedbackHistory.value
            
            if (history.isEmpty()) {
                _feedbackAnalytics.value = FeedbackAnalytics()
                return@launch
            }
            
            val totalInterventions = automationLog.getTotalLogCount()
            val totalFeedback = history.size
            
            val averageRating = history.map { it.rating }.average()
            val responseRate = if (totalInterventions > 0) {
                totalFeedback.toDouble() / totalInterventions
            } else 0.0
            
            val ratingDistribution = (1..5).map { rating ->
                history.count { it.rating == rating }
            }
            
            val insights = generateInsights(history)
            val recommendations = generateRecommendations(history, insights)
            
            _feedbackAnalytics.value = FeedbackAnalytics(
                averageRating = averageRating,
                responseRate = responseRate,
                totalFeedback = totalFeedback,
                ratingDistribution = ratingDistribution,
                insights = insights,
                recommendations = recommendations
            )
        }
    }
    
    /**
     * Generate insights from feedback patterns
     */
    private fun generateInsights(history: List<FeedbackEntry>): List<FeedbackInsight> {
        val insights = mutableListOf<FeedbackInsight>()
        
        if (history.size < 5) return insights
        
        // Trend analysis
        val recentFeedback = history.take(10)
        val olderFeedback = history.drop(10).take(10)
        
        if (recentFeedback.isNotEmpty() && olderFeedback.isNotEmpty()) {
            val recentAvg = recentFeedback.map { it.rating }.average()
            val olderAvg = olderFeedback.map { it.rating }.average()
            
            when {
                recentAvg > olderAvg + 0.5 -> {
                    insights.add(
                        FeedbackInsight(
                            type = "trend_positive",
                            message = "Feedback ratings have improved recently (${String.format("%.1f", recentAvg)} vs ${String.format("%.1f", olderAvg)})",
                            confidence = 0.8
                        )
                    )
                }
                recentAvg < olderAvg - 0.5 -> {
                    insights.add(
                        FeedbackInsight(
                            type = "trend_negative",
                            message = "Feedback ratings have declined recently (${String.format("%.1f", recentAvg)} vs ${String.format("%.1f", olderAvg)})",
                            confidence = 0.8
                        )
                    )
                }
            }
        }
        
        // Rating distribution analysis
        val highRatings = history.count { it.rating >= 4 }
        val lowRatings = history.count { it.rating <= 2 }
        val highRatingPercentage = (highRatings.toDouble() / history.size) * 100
        
        when {
            highRatingPercentage >= 80 -> {
                insights.add(
                    FeedbackInsight(
                        type = "pattern",
                        message = "${highRatingPercentage.toInt()}% of interventions are rated highly (4-5 stars)",
                        confidence = 0.9
                    )
                )
            }
            lowRatings > highRatings -> {
                insights.add(
                    FeedbackInsight(
                        type = "pattern",
                        message = "More interventions receive low ratings than high ratings",
                        confidence = 0.8
                    )
                )
            }
        }
        
        // Comment analysis
        val commentsWithContent = history.filter { it.comment.isNotEmpty() }
        if (commentsWithContent.size >= 5) {
            val commonWords = analyzeCommonWords(commentsWithContent.map { it.comment })
            if (commonWords.isNotEmpty()) {
                insights.add(
                    FeedbackInsight(
                        type = "pattern",
                        message = "Common feedback themes: ${commonWords.take(3).joinToString(", ")}",
                        confidence = 0.7
                    )
                )
            }
        }
        
        return insights
    }
    
    /**
     * Generate recommendations based on feedback analysis
     */
    private fun generateRecommendations(
        history: List<FeedbackEntry>,
        insights: List<FeedbackInsight>
    ): List<String> {
        val recommendations = mutableListOf<String>()
        
        val averageRating = history.map { it.rating }.average()
        
        when {
            averageRating < 2.5 -> {
                recommendations.add("Consider reducing intervention frequency or adjusting sensitivity")
                recommendations.add("Review intervention timing to avoid disrupting important activities")
            }
            averageRating < 3.5 -> {
                recommendations.add("Fine-tune intervention thresholds based on user behavior patterns")
                recommendations.add("Provide more context in intervention messages")
            }
            averageRating >= 4.0 -> {
                recommendations.add("Current automation settings are working well")
                recommendations.add("Consider gradually increasing intervention sophistication")
            }
        }
        
        // Trend-based recommendations
        insights.forEach { insight ->
            when (insight.type) {
                "trend_negative" -> {
                    recommendations.add("Review recent changes to automation settings")
                    recommendations.add("Consider reverting to previous configuration")
                }
                "trend_positive" -> {
                    recommendations.add("Current improvements are working - maintain current approach")
                }
            }
        }
        
        // Response rate recommendations
        val analytics = _feedbackAnalytics.value
        if (analytics.responseRate < 0.3) {
            recommendations.add("Consider simplifying feedback collection process")
            recommendations.add("Reduce feedback request frequency")
        }
        
        return recommendations.distinct()
    }
    
    /**
     * Analyze common words in feedback comments
     */
    private fun analyzeCommonWords(comments: List<String>): List<String> {
        val words = comments
            .flatMap { it.lowercase().split(Regex("\\W+")) }
            .filter { it.length > 3 }
            .filter { it !in listOf("this", "that", "with", "from", "they", "were", "been", "have", "their") }
        
        return words
            .groupingBy { it }
            .eachCount()
            .toList()
            .sortedByDescending { it.second }
            .take(5)
            .map { it.first }
    }
    
    /**
     * Apply feedback learning to improve automation
     */
    private suspend fun applyFeedbackLearning(interventionId: String, rating: Int, comment: String) {
        // Get intervention details
        val intervention = automationLog.getLogById(interventionId)
        if (intervention != null) {
            // Adjust automation based on feedback
            when {
                rating <= 2 -> {
                    // Negative feedback - reduce similar interventions
                    automationEngine.adjustInterventionWeight(intervention.interventionType, -0.1)
                }
                rating >= 4 -> {
                    // Positive feedback - encourage similar interventions
                    automationEngine.adjustInterventionWeight(intervention.interventionType, 0.1)
                }
            }
            
            // Analyze comment for specific improvements
            if (comment.isNotEmpty()) {
                analyzeCommentForImprovements(comment, intervention)
            }
        }
    }
    
    /**
     * Analyze feedback comment for specific improvements
     */
    private suspend fun analyzeCommentForImprovements(comment: String, intervention: AutomationLogEntry) {
        val lowerComment = comment.lowercase()
        
        when {
            "too frequent" in lowerComment || "too many" in lowerComment -> {
                automationEngine.adjustInterventionFrequency(intervention.interventionType, -0.2)
            }
            "too late" in lowerComment || "timing" in lowerComment -> {
                automationEngine.adjustInterventionTiming(intervention.interventionType, -300000) // -5 minutes
            }
            "not relevant" in lowerComment || "wrong time" in lowerComment -> {
                automationEngine.adjustContextSensitivity(intervention.interventionType, 0.1)
            }
            "helpful" in lowerComment || "good" in lowerComment -> {
                automationEngine.adjustInterventionWeight(intervention.interventionType, 0.05)
            }
        }
    }
    
    /**
     * Helper functions
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
    
    private fun getPendingInterventionTitle(interventionId: String): String {
        return _pendingFeedback.value.find { it.id == interventionId }?.title ?: "Unknown Intervention"
    }
    
    private fun generateFeedbackId(): String {
        return "feedback_${System.currentTimeMillis()}_${(1000..9999).random()}"
    }
}