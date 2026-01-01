package com.lifetwin.automation

import android.content.Context
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*
import kotlinx.serialization.Serializable
import kotlin.math.*

/**
 * RLEnvironment - Reinforcement Learning environment and observation space
 * 
 * Implements Requirements:
 * - 3.1: Comprehensive observation space with 50+ behavioral features
 * - 4.2: Temporal features, context features, and wellbeing metrics
 * - Observation vector generation from BehavioralContext
 */
class RLEnvironment(
    private val context: Context,
    private val behavioralContext: BehavioralContext,
    private val resourceMonitor: ResourceMonitor
) {
    companion object {
        const val OBSERVATION_SPACE_SIZE = 52 // 50+ features as required
        const val ACTION_SPACE_SIZE = 8 // Discrete action space
    }
    
    // Current observation state
    private val _currentObservation = MutableStateFlow(FloatArray(OBSERVATION_SPACE_SIZE))
    val currentObservation: StateFlow<FloatArray> = _currentObservation.asStateFlow()
    
    // Environment state tracking
    private var episodeStep = 0
    private var totalReward = 0.0
    private var lastActionTime = 0L
    
    /**
     * Generate comprehensive observation vector from current behavioral context
     */
    fun generateObservation(): FloatArray {
        val observation = FloatArray(OBSERVATION_SPACE_SIZE)
        var index = 0
        
        // Temporal features (12 features)
        observation[index++] = behavioralContext.timeContext.hourOfDay.toFloat() / 24f
        observation[index++] = behavioralContext.timeContext.dayOfWeek.toFloat() / 7f
        observation[index++] = behavioralContext.timeContext.dayOfMonth.toFloat() / 31f
        observation[index++] = behavioralContext.timeContext.weekOfYear.toFloat() / 52f
        observation[index++] = if (behavioralContext.timeContext.isWeekend) 1f else 0f
        observation[index++] = if (behavioralContext.timeContext.isWorkingHours) 1f else 0f
        observation[index++] = if (behavioralContext.timeContext.isLateNight) 1f else 0f
        observation[index++] = if (behavioralContext.timeContext.isEarlyMorning) 1f else 0f
        observation[index++] = behavioralContext.timeContext.timeSinceLastBreak.toFloat() / 3600000f // Hours
        observation[index++] = behavioralContext.timeContext.timeSinceWakeup.toFloat() / 86400000f // Days
        observation[index++] = behavioralContext.timeContext.timeUntilBedtime.toFloat() / 86400000f // Days
        observation[index++] = behavioralContext.timeContext.sessionDuration.toFloat() / 3600000f // Hours
        
        // Usage snapshot features (15 features)
        observation[index++] = normalizeUsageTime(behavioralContext.usageSnapshot.totalUsageTime)
        observation[index++] = normalizeUsageTime(behavioralContext.usageSnapshot.socialMediaTime)
        observation[index++] = normalizeUsageTime(behavioralContext.usageSnapshot.productivityTime)
        observation[index++] = normalizeUsageTime(behavioralContext.usageSnapshot.entertainmentTime)
        observation[index++] = normalizeUsageTime(behavioralContext.usageSnapshot.communicationTime)
        observation[index++] = behavioralContext.usageSnapshot.appSwitchCount.toFloat() / 100f
        observation[index++] = behavioralContext.usageSnapshot.notificationCount.toFloat() / 50f
        observation[index++] = behavioralContext.usageSnapshot.screenInteractions.toFloat() / 1000f
        observation[index++] = behavioralContext.usageSnapshot.averageSessionLength.toFloat() / 1800000f // 30 min
        observation[index++] = behavioralContext.usageSnapshot.longestSession.toFloat() / 7200000f // 2 hours
        observation[index++] = behavioralContext.usageSnapshot.shortestSession.toFloat() / 60000f // 1 minute
        observation[index++] = behavioralContext.usageSnapshot.uniqueAppsUsed.toFloat() / 50f
        observation[index++] = behavioralContext.usageSnapshot.backgroundAppTime.toFloat() / 3600000f // Hours
        observation[index++] = behavioralContext.usageSnapshot.foregroundTransitions.toFloat() / 100f
        observation[index++] = behavioralContext.usageSnapshot.multitaskingScore.toFloat()
        
        // Environment context features (8 features)
        observation[index++] = behavioralContext.environmentContext.batteryLevel.toFloat() / 100f
        observation[index++] = if (behavioralContext.environmentContext.isCharging) 1f else 0f
        observation[index++] = if (behavioralContext.environmentContext.wifiConnected) 1f else 0f
        observation[index++] = if (behavioralContext.environmentContext.bluetoothConnected) 1f else 0f
        observation[index++] = behavioralContext.environmentContext.brightnessLevel.toFloat() / 255f
        observation[index++] = behavioralContext.environmentContext.volumeLevel.toFloat() / 15f
        observation[index++] = if (behavioralContext.environmentContext.headphonesConnected) 1f else 0f
        observation[index++] = behavioralContext.environmentContext.locationContext.toFloat()
        
        // User state features (7 features)
        observation[index++] = behavioralContext.userState.stressLevel.toFloat()
        observation[index++] = behavioralContext.userState.focusLevel.toFloat()
        observation[index++] = behavioralContext.userState.energyLevel.toFloat()
        observation[index++] = behavioralContext.userState.moodScore.toFloat()
        observation[index++] = behavioralContext.userState.productivityScore.toFloat()
        observation[index++] = behavioralContext.userState.wellbeingScore.toFloat()
        observation[index++] = behavioralContext.userState.engagementLevel.toFloat()
        
        // Resource monitoring features (5 features)
        val resourceUsage = resourceMonitor.resourceUsage.value
        observation[index++] = (resourceUsage.cpuUsagePercent / 100.0).toFloat()
        observation[index++] = (resourceUsage.memoryUsageMB / 1000.0).toFloat() // Normalize to GB
        observation[index++] = resourceUsage.batteryLevel.toFloat() / 100f
        observation[index++] = (resourceUsage.batteryTemperature / 50.0).toFloat() // Normalize to reasonable range
        observation[index++] = (resourceUsage.availableStorageMB / 10000.0).toFloat() // Normalize to 10GB
        
        // Historical patterns (5 features)
        observation[index++] = calculateUsageTrend()
        observation[index++] = calculateInterventionEffectiveness()
        observation[index++] = calculateHabitStrength()
        observation[index++] = calculateContextSimilarity()
        observation[index++] = calculateWellbeingTrend()
        
        _currentObservation.value = observation
        return observation
    }
    
    /**
     * Normalize usage time to 0-1 range (assuming max 8 hours)
     */
    private fun normalizeUsageTime(timeMs: Long): Float {
        return (timeMs.toFloat() / (8 * 3600 * 1000)).coerceIn(0f, 1f)
    }
    
    /**
     * Calculate usage trend over recent history
     */
    private fun calculateUsageTrend(): Float {
        // Simplified trend calculation - would use historical data in real implementation
        val recentUsage = behavioralContext.usageSnapshot.totalUsageTime
        val averageUsage = 4 * 3600 * 1000L // 4 hours average
        
        return ((recentUsage - averageUsage).toFloat() / averageUsage).coerceIn(-1f, 1f)
    }
    
    /**
     * Calculate intervention effectiveness based on user responses
     */
    private fun calculateInterventionEffectiveness(): Float {
        // Simplified calculation - would use actual intervention history
        return 0.7f // 70% effectiveness placeholder
    }
    
    /**
     * Calculate habit strength based on usage patterns
     */
    private fun calculateHabitStrength(): Float {
        val consistency = behavioralContext.usageSnapshot.averageSessionLength.toFloat() / 
                         behavioralContext.usageSnapshot.longestSession.toFloat()
        return consistency.coerceIn(0f, 1f)
    }
    
    /**
     * Calculate similarity to previous contexts
     */
    private fun calculateContextSimilarity(): Float {
        // Simplified similarity calculation
        val timeOfDaySimilarity = 1f - abs(behavioralContext.timeContext.hourOfDay - 12f) / 12f
        val dayTypeSimilarity = if (behavioralContext.timeContext.isWeekend) 0.3f else 0.7f
        
        return (timeOfDaySimilarity + dayTypeSimilarity) / 2f
    }
    
    /**
     * Calculate wellbeing trend
     */
    private fun calculateWellbeingTrend(): Float {
        val currentWellbeing = behavioralContext.userState.wellbeingScore
        val targetWellbeing = 0.8f // Target 80% wellbeing
        
        return (currentWellbeing - targetWellbeing).coerceIn(-1f, 1f)
    }
    
    /**
     * Reset environment for new episode
     */
    fun reset(): FloatArray {
        episodeStep = 0
        totalReward = 0.0
        lastActionTime = System.currentTimeMillis()
        
        return generateObservation()
    }
    
    /**
     * Execute action in environment and return next observation, reward, done
     */
    fun step(action: Int): StepResult {
        episodeStep++
        
        // Calculate reward based on action and current state
        val reward = calculateReward(action)
        totalReward += reward
        
        // Update environment state based on action
        updateEnvironmentState(action)
        
        // Generate next observation
        val nextObservation = generateObservation()
        
        // Check if episode is done (e.g., after 24 hours or 100 steps)
        val done = episodeStep >= 100 || isEpisodeComplete()
        
        lastActionTime = System.currentTimeMillis()
        
        return StepResult(
            observation = nextObservation,
            reward = reward,
            done = done,
            info = mapOf(
                "episode_step" to episodeStep,
                "total_reward" to totalReward,
                "action_taken" to action
            )
        )
    }
    
    /**
     * Calculate reward for taken action
     */
    private fun calculateReward(action: Int): Double {
        var reward = 0.0
        
        // Base reward for taking any action
        reward += 0.1
        
        // Reward based on action appropriateness
        reward += calculateActionAppropriatenessReward(action)
        
        // Reward based on wellbeing improvement
        reward += calculateWellbeingReward()
        
        // Penalty for inappropriate timing
        reward -= calculateTimingPenalty(action)
        
        // Reward for user engagement
        reward += calculateEngagementReward()
        
        return reward.coerceIn(-1.0, 1.0)
    }
    
    /**
     * Calculate reward based on action appropriateness
     */
    private fun calculateActionAppropriatenessReward(action: Int): Double {
        val currentUsage = behavioralContext.usageSnapshot.totalUsageTime
        val socialMediaUsage = behavioralContext.usageSnapshot.socialMediaTime
        val isLateNight = behavioralContext.timeContext.isLateNight
        val stressLevel = behavioralContext.userState.stressLevel
        
        return when (action) {
            0 -> 0.0 // No intervention
            1 -> if (socialMediaUsage > 2 * 3600 * 1000) 0.5 else -0.2 // Social media break
            2 -> if (isLateNight) 0.6 else -0.1 // Bedtime suggestion
            3 -> if (currentUsage > 4 * 3600 * 1000) 0.4 else -0.1 // General break
            4 -> if (stressLevel > 0.7) 0.5 else 0.0 // Stress reduction
            5 -> if (behavioralContext.timeContext.isWorkingHours) 0.3 else -0.1 // Focus mode
            6 -> 0.2 // Positive reinforcement
            7 -> if (behavioralContext.userState.energyLevel < 0.3) 0.4 else 0.0 // Energy boost
            else -> -0.5 // Invalid action
        }
    }
    
    /**
     * Calculate reward based on wellbeing improvement
     */
    private fun calculateWellbeingReward(): Double {
        val wellbeingScore = behavioralContext.userState.wellbeingScore
        val targetWellbeing = 0.8
        
        return if (wellbeingScore > targetWellbeing) 0.3 else 0.0
    }
    
    /**
     * Calculate penalty for poor timing
     */
    private fun calculateTimingPenalty(action: Int): Double {
        val timeSinceLastAction = System.currentTimeMillis() - lastActionTime
        val minInterval = 15 * 60 * 1000L // 15 minutes
        
        return if (timeSinceLastAction < minInterval && action != 0) 0.3 else 0.0
    }
    
    /**
     * Calculate reward based on user engagement
     */
    private fun calculateEngagementReward(): Double {
        val engagementLevel = behavioralContext.userState.engagementLevel
        return engagementLevel * 0.2
    }
    
    /**
     * Update environment state based on action
     */
    private fun updateEnvironmentState(action: Int) {
        // This would update the behavioral context based on the action taken
        // For now, this is a placeholder that would integrate with the actual system
    }
    
    /**
     * Check if episode is complete
     */
    private fun isEpisodeComplete(): Boolean {
        // Episode completes after 24 hours or when user goes to sleep
        val sessionDuration = behavioralContext.timeContext.sessionDuration
        return sessionDuration > 24 * 3600 * 1000 || behavioralContext.timeContext.isLateNight
    }
    
    /**
     * Get observation space information
     */
    fun getObservationSpace(): ObservationSpace {
        return ObservationSpace(
            shape = intArrayOf(OBSERVATION_SPACE_SIZE),
            low = FloatArray(OBSERVATION_SPACE_SIZE) { 0f },
            high = FloatArray(OBSERVATION_SPACE_SIZE) { 1f },
            dtype = "float32"
        )
    }
    
    /**
     * Get action space information
     */
    fun getActionSpace(): ActionSpace {
        return ActionSpace(
            n = ACTION_SPACE_SIZE,
            actions = listOf(
                "no_intervention",
                "social_media_break",
                "bedtime_suggestion", 
                "general_break",
                "stress_reduction",
                "focus_mode",
                "positive_reinforcement",
                "energy_boost"
            )
        )
    }
    
    /**
     * Get current environment statistics
     */
    fun getEnvironmentStats(): EnvironmentStats {
        return EnvironmentStats(
            episodeStep = episodeStep,
            totalReward = totalReward,
            averageReward = if (episodeStep > 0) totalReward / episodeStep else 0.0,
            observationSpaceSize = OBSERVATION_SPACE_SIZE,
            actionSpaceSize = ACTION_SPACE_SIZE,
            lastActionTime = lastActionTime
        )
    }
}

/**
 * Result of environment step
 */
data class StepResult(
    val observation: FloatArray,
    val reward: Double,
    val done: Boolean,
    val info: Map<String, Any>
)

/**
 * Observation space definition
 */
@Serializable
data class ObservationSpace(
    val shape: IntArray,
    val low: FloatArray,
    val high: FloatArray,
    val dtype: String
)

/**
 * Action space definition
 */
@Serializable
data class ActionSpace(
    val n: Int,
    val actions: List<String>
)

/**
 * Environment statistics
 */
data class EnvironmentStats(
    val episodeStep: Int,
    val totalReward: Double,
    val averageReward: Double,
    val observationSpaceSize: Int,
    val actionSpaceSize: Int,
    val lastActionTime: Long
)