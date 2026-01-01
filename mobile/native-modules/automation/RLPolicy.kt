package com.lifetwin.automation

import android.content.Context
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*
import kotlinx.serialization.Serializable
import kotlin.math.*
import kotlin.random.Random

/**
 * RLPolicy - Reinforcement Learning policy interface and implementation
 * 
 * Implements Requirements:
 * - 4.3: Discrete action space for intervention types, timing, intensity
 * - 3.4: RLPolicy class for on-device inference
 * - Policy action validation and bounds checking
 */
class RLPolicy(
    private val context: Context,
    private val modelPath: String? = null
) {
    companion object {
        const val ACTION_SPACE_SIZE = 8
        const val OBSERVATION_SPACE_SIZE = 52
    }
    
    // Policy state
    private var isLoaded = false
    private var policyVersion = "1.0.0"
    private var lastInferenceTime = 0L
    
    // Action definitions
    private val actionDefinitions = mapOf(
        0 to ActionDefinition(
            id = 0,
            name = "no_intervention",
            description = "Take no action, continue monitoring",
            interventionType = InterventionType.NONE,
            intensity = 0.0,
            minInterval = 0L,
            maxFrequency = Int.MAX_VALUE
        ),
        1 to ActionDefinition(
            id = 1,
            name = "social_media_break",
            description = "Suggest taking a break from social media",
            interventionType = InterventionType.USAGE_REMINDER,
            intensity = 0.6,
            minInterval = 30 * 60 * 1000L, // 30 minutes
            maxFrequency = 8 // per day
        ),
        2 to ActionDefinition(
            id = 2,
            name = "bedtime_suggestion",
            description = "Suggest winding down for bedtime",
            interventionType = InterventionType.BEDTIME_SUGGESTION,
            intensity = 0.7,
            minInterval = 60 * 60 * 1000L, // 1 hour
            maxFrequency = 3 // per day
        ),
        3 to ActionDefinition(
            id = 3,
            name = "general_break",
            description = "Suggest taking a general screen break",
            interventionType = InterventionType.ACTIVITY_ENCOURAGEMENT,
            intensity = 0.5,
            minInterval = 45 * 60 * 1000L, // 45 minutes
            maxFrequency = 6 // per day
        ),
        4 to ActionDefinition(
            id = 4,
            name = "stress_reduction",
            description = "Suggest stress reduction activities",
            interventionType = InterventionType.WELLNESS_SUGGESTION,
            intensity = 0.8,
            minInterval = 20 * 60 * 1000L, // 20 minutes
            maxFrequency = 5 // per day
        ),
        5 to ActionDefinition(
            id = 5,
            name = "focus_mode",
            description = "Suggest enabling focus mode",
            interventionType = InterventionType.FOCUS_PROTECTION,
            intensity = 0.9,
            minInterval = 2 * 60 * 60 * 1000L, // 2 hours
            maxFrequency = 4 // per day
        ),
        6 to ActionDefinition(
            id = 6,
            name = "positive_reinforcement",
            description = "Provide positive feedback on behavior",
            interventionType = InterventionType.POSITIVE_REINFORCEMENT,
            intensity = 0.3,
            minInterval = 15 * 60 * 1000L, // 15 minutes
            maxFrequency = 10 // per day
        ),
        7 to ActionDefinition(
            id = 7,
            name = "energy_boost",
            description = "Suggest activities to boost energy",
            interventionType = InterventionType.ACTIVITY_ENCOURAGEMENT,
            intensity = 0.6,
            minInterval = 30 * 60 * 1000L, // 30 minutes
            maxFrequency = 4 // per day
        )
    )
    
    // Action history for frequency tracking
    private val actionHistory = mutableListOf<ActionExecution>()
    
    // Policy parameters (would be loaded from trained model)
    private var policyWeights: FloatArray = FloatArray(OBSERVATION_SPACE_SIZE * ACTION_SPACE_SIZE)
    private var policyBias: FloatArray = FloatArray(ACTION_SPACE_SIZE)
    
    init {
        initializePolicy()
    }
    
    /**
     * Initialize the policy (load model or use default)
     */
    private fun initializePolicy() {
        if (modelPath != null) {
            loadTrainedModel(modelPath)
        } else {
            initializeDefaultPolicy()
        }
        isLoaded = true
    }
    
    /**
     * Load trained model from file
     */
    private fun loadTrainedModel(path: String) {
        try {
            // This would load actual trained model weights
            // For now, initialize with reasonable defaults
            initializeDefaultPolicy()
            policyVersion = "trained_1.0.0"
        } catch (e: Exception) {
            // Fallback to default policy
            initializeDefaultPolicy()
        }
    }
    
    /**
     * Initialize default rule-based policy
     */
    private fun initializeDefaultPolicy() {
        // Initialize with rule-based heuristics
        policyWeights = FloatArray(OBSERVATION_SPACE_SIZE * ACTION_SPACE_SIZE) { 
            Random.nextFloat() * 0.1f - 0.05f // Small random weights
        }
        
        policyBias = floatArrayOf(
            0.5f,  // no_intervention (default bias)
            -0.2f, // social_media_break
            -0.3f, // bedtime_suggestion
            -0.1f, // general_break
            -0.4f, // stress_reduction
            -0.5f, // focus_mode
            0.1f,  // positive_reinforcement
            -0.2f  // energy_boost
        )
        
        policyVersion = "default_1.0.0"
    }
    
    /**
     * Predict action given observation
     */
    fun predict(observation: FloatArray): PolicyPrediction {
        require(observation.size == OBSERVATION_SPACE_SIZE) {
            "Observation size must be $OBSERVATION_SPACE_SIZE, got ${observation.size}"
        }
        
        val actionScores = calculateActionScores(observation)
        val actionProbabilities = softmax(actionScores)
        
        // Select action based on probabilities (with exploration)
        val selectedAction = selectAction(actionProbabilities)
        
        // Validate action
        val validatedAction = validateAction(selectedAction, observation)
        
        lastInferenceTime = System.currentTimeMillis()
        
        return PolicyPrediction(
            action = validatedAction,
            actionProbabilities = actionProbabilities,
            confidence = actionProbabilities[validatedAction],
            rawScores = actionScores,
            isExploration = validatedAction != selectedAction,
            inferenceTime = System.currentTimeMillis() - lastInferenceTime
        )
    }
    
    /**
     * Calculate action scores using policy network
     */
    private fun calculateActionScores(observation: FloatArray): FloatArray {
        val scores = FloatArray(ACTION_SPACE_SIZE)
        
        // Simple linear policy: scores = observation * weights + bias
        for (actionIdx in 0 until ACTION_SPACE_SIZE) {
            var score = policyBias[actionIdx]
            
            for (obsIdx in 0 until OBSERVATION_SPACE_SIZE) {
                val weightIdx = actionIdx * OBSERVATION_SPACE_SIZE + obsIdx
                score += observation[obsIdx] * policyWeights[weightIdx]
            }
            
            scores[actionIdx] = score
        }
        
        // Apply rule-based adjustments
        applyRuleBasedAdjustments(scores, observation)
        
        return scores
    }
    
    /**
     * Apply rule-based adjustments to action scores
     */
    private fun applyRuleBasedAdjustments(scores: FloatArray, observation: FloatArray) {
        // Extract key features from observation
        val hourOfDay = (observation[0] * 24).toInt()
        val isLateNight = observation[6] > 0.5f
        val socialMediaUsage = observation[13]
        val stressLevel = observation[35]
        val batteryLevel = observation[27]
        
        // Rule-based adjustments
        if (isLateNight) {
            scores[2] += 1.0f // Boost bedtime suggestion
            scores[1] += 0.5f // Boost social media break
        }
        
        if (socialMediaUsage > 0.7f) {
            scores[1] += 0.8f // Boost social media break
        }
        
        if (stressLevel > 0.7f) {
            scores[4] += 0.6f // Boost stress reduction
        }
        
        if (batteryLevel < 0.2f) {
            scores[0] += 0.3f // Prefer no intervention when battery low
        }
        
        // Working hours adjustments
        if (hourOfDay in 9..17) {
            scores[5] += 0.4f // Boost focus mode during work hours
            scores[2] -= 0.5f // Reduce bedtime suggestions
        }
    }
    
    /**
     * Apply softmax to convert scores to probabilities
     */
    private fun softmax(scores: FloatArray): FloatArray {
        val maxScore = scores.maxOrNull() ?: 0f
        val expScores = scores.map { exp(it - maxScore) }.toFloatArray()
        val sumExp = expScores.sum()
        
        return expScores.map { it / sumExp }.toFloatArray()
    }
    
    /**
     * Select action based on probabilities with exploration
     */
    private fun selectAction(probabilities: FloatArray, explorationRate: Double = 0.1): Int {
        return if (Random.nextDouble() < explorationRate) {
            // Exploration: random action
            Random.nextInt(ACTION_SPACE_SIZE)
        } else {
            // Exploitation: action with highest probability
            probabilities.indices.maxByOrNull { probabilities[it] } ?: 0
        }
    }
    
    /**
     * Validate action against constraints and history
     */
    private fun validateAction(action: Int, observation: FloatArray): Int {
        val actionDef = actionDefinitions[action] ?: return 0
        
        // Check if action violates frequency constraints
        if (violatesFrequencyConstraints(action)) {
            return findAlternativeAction(action, observation)
        }
        
        // Check if action violates timing constraints
        if (violatesTimingConstraints(action)) {
            return findAlternativeAction(action, observation)
        }
        
        // Check if action is contextually appropriate
        if (!isContextuallyAppropriate(action, observation)) {
            return findAlternativeAction(action, observation)
        }
        
        return action
    }
    
    /**
     * Check if action violates frequency constraints
     */
    private fun violatesFrequencyConstraints(action: Int): Boolean {
        val actionDef = actionDefinitions[action] ?: return true
        val now = System.currentTimeMillis()
        val oneDayAgo = now - 24 * 60 * 60 * 1000L
        
        val recentActions = actionHistory.count { 
            it.action == action && it.timestamp > oneDayAgo 
        }
        
        return recentActions >= actionDef.maxFrequency
    }
    
    /**
     * Check if action violates timing constraints
     */
    private fun violatesTimingConstraints(action: Int): Boolean {
        val actionDef = actionDefinitions[action] ?: return true
        val now = System.currentTimeMillis()
        
        val lastSameAction = actionHistory
            .filter { it.action == action }
            .maxByOrNull { it.timestamp }
        
        return lastSameAction != null && 
               (now - lastSameAction.timestamp) < actionDef.minInterval
    }
    
    /**
     * Check if action is contextually appropriate
     */
    private fun isContextuallyAppropriate(action: Int, observation: FloatArray): Boolean {
        val hourOfDay = (observation[0] * 24).toInt()
        val batteryLevel = observation[27]
        val isCharging = observation[28] > 0.5f
        
        return when (action) {
            2 -> hourOfDay >= 20 || hourOfDay <= 6 // Bedtime suggestion only at night
            5 -> hourOfDay in 8..18 // Focus mode during reasonable hours
            4 -> batteryLevel > 0.1f || isCharging // Stress reduction needs battery
            else -> true // Other actions are generally appropriate
        }
    }
    
    /**
     * Find alternative action when primary action is invalid
     */
    private fun findAlternativeAction(originalAction: Int, observation: FloatArray): Int {
        val scores = calculateActionScores(observation)
        
        // Try actions in order of preference, skipping invalid ones
        val sortedActions = scores.indices.sortedByDescending { scores[it] }
        
        for (action in sortedActions) {
            if (action != originalAction && 
                !violatesFrequencyConstraints(action) &&
                !violatesTimingConstraints(action) &&
                isContextuallyAppropriate(action, observation)) {
                return action
            }
        }
        
        // Fallback to no intervention
        return 0
    }
    
    /**
     * Execute action and record in history
     */
    fun executeAction(action: Int, context: BehavioralContext): ActionResult {
        val actionDef = actionDefinitions[action] ?: return ActionResult(
            success = false,
            message = "Invalid action: $action"
        )
        
        // Record action execution
        val execution = ActionExecution(
            action = action,
            timestamp = System.currentTimeMillis(),
            context = context,
            intensity = actionDef.intensity
        )
        
        actionHistory.add(execution)
        
        // Clean up old history (keep last 7 days)
        val cutoffTime = System.currentTimeMillis() - 7 * 24 * 60 * 60 * 1000L
        actionHistory.removeAll { it.timestamp < cutoffTime }
        
        return ActionResult(
            success = true,
            message = "Executed ${actionDef.name}: ${actionDef.description}",
            actionDefinition = actionDef,
            execution = execution
        )
    }
    
    /**
     * Get action definition by ID
     */
    fun getActionDefinition(actionId: Int): ActionDefinition? {
        return actionDefinitions[actionId]
    }
    
    /**
     * Get all action definitions
     */
    fun getAllActionDefinitions(): Map<Int, ActionDefinition> {
        return actionDefinitions.toMap()
    }
    
    /**
     * Get action space information
     */
    fun getActionSpace(): ActionSpace {
        return ActionSpace(
            n = ACTION_SPACE_SIZE,
            actions = actionDefinitions.values.map { it.name }
        )
    }
    
    /**
     * Get policy information
     */
    fun getPolicyInfo(): PolicyInfo {
        return PolicyInfo(
            version = policyVersion,
            isLoaded = isLoaded,
            observationSpaceSize = OBSERVATION_SPACE_SIZE,
            actionSpaceSize = ACTION_SPACE_SIZE,
            lastInferenceTime = lastInferenceTime,
            totalExecutions = actionHistory.size,
            modelPath = modelPath
        )
    }
    
    /**
     * Get action execution history
     */
    fun getActionHistory(limit: Int = 100): List<ActionExecution> {
        return actionHistory.takeLast(limit)
    }
    
    /**
     * Update policy with new weights (for online learning)
     */
    fun updatePolicy(newWeights: FloatArray, newBias: FloatArray) {
        require(newWeights.size == policyWeights.size) {
            "Weight size mismatch: expected ${policyWeights.size}, got ${newWeights.size}"
        }
        require(newBias.size == policyBias.size) {
            "Bias size mismatch: expected ${policyBias.size}, got ${newBias.size}"
        }
        
        policyWeights = newWeights.copyOf()
        policyBias = newBias.copyOf()
        policyVersion = "${policyVersion}_updated_${System.currentTimeMillis()}"
    }
    
    /**
     * Reset policy to default state
     */
    fun resetPolicy() {
        initializeDefaultPolicy()
        actionHistory.clear()
        lastInferenceTime = 0L
    }
}

/**
 * Action definition with constraints
 */
@Serializable
data class ActionDefinition(
    val id: Int,
    val name: String,
    val description: String,
    val interventionType: InterventionType,
    val intensity: Double, // 0.0 to 1.0
    val minInterval: Long, // Minimum time between same actions (ms)
    val maxFrequency: Int // Maximum times per day
)

/**
 * Intervention types
 */
@Serializable
enum class InterventionType {
    NONE,
    USAGE_REMINDER,
    BEDTIME_SUGGESTION,
    ACTIVITY_ENCOURAGEMENT,
    WELLNESS_SUGGESTION,
    FOCUS_PROTECTION,
    POSITIVE_REINFORCEMENT
}

/**
 * Policy prediction result
 */
data class PolicyPrediction(
    val action: Int,
    val actionProbabilities: FloatArray,
    val confidence: Float,
    val rawScores: FloatArray,
    val isExploration: Boolean,
    val inferenceTime: Long
)

/**
 * Action execution record
 */
@Serializable
data class ActionExecution(
    val action: Int,
    val timestamp: Long,
    val context: BehavioralContext,
    val intensity: Double
)

/**
 * Action execution result
 */
data class ActionResult(
    val success: Boolean,
    val message: String,
    val actionDefinition: ActionDefinition? = null,
    val execution: ActionExecution? = null
)

/**
 * Policy information
 */
data class PolicyInfo(
    val version: String,
    val isLoaded: Boolean,
    val observationSpaceSize: Int,
    val actionSpaceSize: Int,
    val lastInferenceTime: Long,
    val totalExecutions: Int,
    val modelPath: String?
)