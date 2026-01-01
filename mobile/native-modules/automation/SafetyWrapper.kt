package com.lifetwin.automation

import android.util.Log
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*
import kotlinx.serialization.Serializable
import java.util.concurrent.ConcurrentHashMap
import kotlin.math.*

private const val TAG = "SafetyWrapper"

/**
 * Enhanced Safety wrapper that ensures RL policy actions remain within safe and acceptable boundaries.
 * Prevents harmful or annoying automation behaviors through comprehensive rule-based constraints.
 * 
 * Implements Requirements:
 * - 3.7: Safety constraint validation for all RL actions
 * - 5.1: Intervention frequency limits and context-aware restrictions
 * - 5.5: Violation reporting and policy adjustment mechanisms
 */
class SafetyWrapper {
    
    // Safety constraint configuration with multiple constraint types
    private val safetyConstraints = SafetyConstraints()
    
    // Comprehensive tracking systems
    private val recentInterventions = mutableListOf<InterventionRecord>()
    private val violationHistory = mutableListOf<SafetyViolation>()
    private val constraintViolationCounts = ConcurrentHashMap<String, Int>()
    private val safetyStatistics = SafetyStatistics()
    
    // Context and state tracking
    private var isEmergencyMode = false
    private var isInImportantCall = false
    private var userDefinedQuietPeriods = mutableListOf<QuietPeriod>()
    private var safetyCooldownUntil = 0L
    private var consecutiveViolations = 0
    
    // Safety monitoring flows
    private val _safetyEvents = MutableSharedFlow<SafetyEvent>()
    val safetyEvents: SharedFlow<SafetyEvent> = _safetyEvents.asSharedFlow()
    
    private val _safetyStatisticsFlow = MutableStateFlow(safetyStatistics)
    val safetyStatisticsFlow: StateFlow<SafetyStatistics> = _safetyStatisticsFlow.asStateFlow()
    
    
    suspend fun initialize(): Boolean {
        return try {
            Log.i(TAG, "Initializing enhanced SafetyWrapper...")
            
            // Load default safety constraints
            loadDefaultConstraints()
            
            // Initialize safety monitoring
            initializeSafetyMonitoring()
            
            // Start background safety checks
            startSafetyMonitoring()
            
            Log.i(TAG, "Enhanced SafetyWrapper initialized successfully")
            true
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize SafetyWrapper", e)
            false
        }
    }
    
    /**
     * Initialize safety monitoring systems
     */
    private fun initializeSafetyMonitoring() {
        safetyStatistics.apply {
            totalValidations = 0
            totalViolations = 0
            violationRate = 0.0
            lastViolationTime = 0L
            averageViolationsPerHour = 0.0
            mostCommonViolationType = ""
            safetyScore = 1.0
        }
        
        // Initialize violation counters
        ConstraintType.values().forEach { type ->
            constraintViolationCounts[type.name] = 0
        }
    }
    
    /**
     * Start background safety monitoring
     */
    private fun startSafetyMonitoring() {
        CoroutineScope(Dispatchers.Default).launch {
            while (true) {
                delay(60000) // Check every minute
                
                // Clean up old records
                cleanupOldRecords()
                
                // Update safety statistics
                updateSafetyStatistics()
                
                // Check for safety cooldown expiration
                checkSafetyCooldown()
                
                // Emit safety statistics update
                _safetyStatisticsFlow.value = safetyStatistics.copy()
            }
        }
    }
    
    /**
     * Enhanced validation with comprehensive constraint checking
     */
    fun validateRecommendation(
        recommendation: InterventionRecommendation,
        context: BehavioralContext
    ): SafetyValidationResult {
        return try {
            safetyStatistics.totalValidations++
            
            // Check if in safety cooldown
            if (isInSafetyCooldown()) {
                return createViolationResult(
                    recommendation,
                    ConstraintType.COOLDOWN,
                    "Safety cooldown active until ${safetyCooldownUntil - System.currentTimeMillis()}ms"
                )
            }
            
            // Comprehensive constraint validation
            val validationResults = listOf(
                checkFrequencyConstraints(recommendation),
                checkTimingConstraints(recommendation, context),
                checkSequenceConstraints(recommendation),
                checkContextualConstraints(recommendation, context),
                checkEmergencyConstraints(recommendation, context),
                checkUserPreferenceConstraints(recommendation),
                checkResourceConstraints(recommendation, context),
                checkEscalationPrevention(recommendation),
                checkCumulativeImpact(recommendation)
            )
            
            val violations = validationResults.filter { !it.isValid }
            
            if (violations.isNotEmpty()) {
                // Record violations
                violations.forEach { violation ->
                    recordViolation(recommendation, violation.constraintType, violation.reason)
                }
                
                return SafetyValidationResult(
                    isValid = false,
                    violations = violations,
                    alternativeAction = findSafeAlternative(recommendation, context),
                    safetyScore = calculateSafetyScore(violations),
                    recommendedCooldown = calculateRecommendedCooldown(violations)
                )
            } else {
                // Record successful intervention
                recordIntervention(recommendation)
                
                return SafetyValidationResult(
                    isValid = true,
                    violations = emptyList(),
                    alternativeAction = null,
                    safetyScore = 1.0,
                    recommendedCooldown = 0L
                )
            }
            
        } catch (e: Exception) {
            Log.e(TAG, "Error validating recommendation", e)
            // Fail safe - block if validation fails
            return createViolationResult(
                recommendation,
                ConstraintType.SYSTEM_ERROR,
                "Validation system error: ${e.message}"
            )
        }
    }
    
    /**
     * Enhanced public API methods
     */
    
    /**
     * Update safety constraints with enhanced configuration
     */
    fun updateSafetyConstraints(constraints: SafetyConstraints) {
        safetyConstraints.apply {
            maxInterventionsPerHour = constraints.maxInterventionsPerHour
            maxInterventionsPerDay = constraints.maxInterventionsPerDay
            minIntervalBetweenInterventions = constraints.minIntervalBetweenInterventions
            maxConsecutiveStrongInterventions = constraints.maxConsecutiveStrongInterventions
            maxCumulativeInterventionsPerHour = constraints.maxCumulativeInterventionsPerHour
            maxConsecutiveViolations = constraints.maxConsecutiveViolations
            safetyCooldownDuration = constraints.safetyCooldownDuration
            emergencyModeEnabled = constraints.emergencyModeEnabled
            respectCallState = constraints.respectCallState
        }
        
        Log.d(TAG, "Updated enhanced safety constraints")
        
        CoroutineScope(Dispatchers.Default).launch {
            _safetyEvents.emit(SafetyEvent.ConstraintsUpdated(constraints))
        }
    }
    
    /**
     * Report a safety violation with enhanced tracking
     */
    fun reportViolation(action: InterventionRecommendation, constraintType: ConstraintType, reason: String) {
        recordViolation(action, constraintType, reason)
        
        Log.w(TAG, "External safety violation reported: ${constraintType.name} - $reason")
    }
    
    /**
     * Get comprehensive safety statistics
     */
    fun getSafetyStatistics(): SafetyStatistics {
        updateSafetyStatistics()
        return safetyStatistics.copy()
    }
    
    /**
     * Get violation history with filtering
     */
    fun getViolationHistory(
        constraintType: ConstraintType? = null,
        limit: Int = 100,
        since: Long = 0L
    ): List<SafetyViolation> {
        return violationHistory
            .filter { violation ->
                (constraintType == null || violation.constraintType == constraintType) &&
                violation.timestamp >= since
            }
            .sortedByDescending { it.timestamp }
            .take(limit)
    }
    
    /**
     * Get constraint violation counts
     */
    fun getConstraintViolationCounts(): Map<String, Int> {
        return constraintViolationCounts.toMap()
    }
    
    /**
     * Check if system is in safety cooldown
     */
    fun isInCooldown(): Boolean {
        return isInSafetyCooldown()
    }
    
    /**
     * Get remaining cooldown time
     */
    fun getRemainingCooldownTime(): Long {
        return if (isInSafetyCooldown()) {
            safetyCooldownUntil - System.currentTimeMillis()
        } else 0L
    }
    
    /**
     * Manually trigger safety cooldown (for testing or emergency)
     */
    fun triggerManualCooldown(duration: Long = safetyConstraints.safetyCooldownDuration) {
        safetyCooldownUntil = System.currentTimeMillis() + duration
        
        CoroutineScope(Dispatchers.Default).launch {
            _safetyEvents.emit(SafetyEvent.ManualCooldownTriggered(duration))
        }
        
        Log.i(TAG, "Manual safety cooldown triggered for ${duration / 1000}s")
    }
    
    /**
     * Clear safety cooldown
     */
    fun clearCooldown() {
        safetyCooldownUntil = 0L
        consecutiveViolations = 0
        
        CoroutineScope(Dispatchers.Default).launch {
            _safetyEvents.emit(SafetyEvent.CooldownCleared)
        }
        
        Log.i(TAG, "Safety cooldown cleared")
    }
    
    /**
     * Reset safety statistics and history
     */
    fun resetSafetyData() {
        violationHistory.clear()
        constraintViolationCounts.clear()
        consecutiveViolations = 0
        safetyCooldownUntil = 0L
        
        safetyStatistics.apply {
            totalValidations = 0
            totalViolations = 0
            violationRate = 0.0
            lastViolationTime = 0L
            averageViolationsPerHour = 0.0
            mostCommonViolationType = ""
            safetyScore = 1.0
        }
        
        CoroutineScope(Dispatchers.Default).launch {
            _safetyEvents.emit(SafetyEvent.SafetyDataReset)
        }
        
        Log.i(TAG, "Safety data reset")
    }
    
    /**
     * Set emergency mode to block all non-critical interventions
     */
    fun setEmergencyMode(enabled: Boolean) {
        isEmergencyMode = enabled
        Log.i(TAG, "Emergency mode ${if (enabled) "enabled" else "disabled"}")
    }
    
    /**
     * Set important call state to block interrupting interventions
     */
    fun setImportantCallState(inCall: Boolean) {
        isInImportantCall = inCall
        Log.d(TAG, "Important call state: $inCall")
    }
    
    /**
     * Add user-defined quiet periods
     */
    fun addQuietPeriod(period: QuietPeriod) {
        userDefinedQuietPeriods.add(period)
        Log.d(TAG, "Added quiet period: ${period.name}")
    }
    
    /**
     * Remove user-defined quiet periods
     */
    fun removeQuietPeriod(periodId: String) {
        userDefinedQuietPeriods.removeAll { it.id == periodId }
        Log.d(TAG, "Removed quiet period: $periodId")
    }
    
    /**
     * Enhanced constraint checking methods
     */
    
    private fun checkFrequencyConstraints(recommendation: InterventionRecommendation): ConstraintValidationResult {
        val currentTime = System.currentTimeMillis()
        val oneHourAgo = currentTime - (60 * 60 * 1000L)
        val oneDayAgo = currentTime - (24 * 60 * 60 * 1000L)
        
        // Check hourly limit
        val recentHourlyCount = recentInterventions.count { 
            it.timestamp > oneHourAgo && it.type == recommendation.type 
        }
        if (recentHourlyCount >= safetyConstraints.maxInterventionsPerHour) {
            return ConstraintValidationResult(
                false, 
                ConstraintType.FREQUENCY_HOURLY,
                "Hourly intervention limit exceeded ($recentHourlyCount/${safetyConstraints.maxInterventionsPerHour})"
            )
        }
        
        // Check daily limit
        val recentDailyCount = recentInterventions.count { 
            it.timestamp > oneDayAgo && it.type == recommendation.type 
        }
        if (recentDailyCount >= safetyConstraints.maxInterventionsPerDay) {
            return ConstraintValidationResult(
                false,
                ConstraintType.FREQUENCY_DAILY,
                "Daily intervention limit exceeded ($recentDailyCount/${safetyConstraints.maxInterventionsPerDay})"
            )
        }
        
        return ConstraintValidationResult(true, ConstraintType.FREQUENCY_HOURLY, "Frequency limits OK")
    }
    
    private fun checkTimingConstraints(
        recommendation: InterventionRecommendation,
        context: BehavioralContext
    ): ConstraintValidationResult {
        val currentTime = System.currentTimeMillis()
        
        // Check minimum interval between same intervention types
        val lastSameTypeIntervention = recentInterventions
            .filter { it.type == recommendation.type }
            .maxByOrNull { it.timestamp }
        
        if (lastSameTypeIntervention != null) {
            val timeSinceLastIntervention = currentTime - lastSameTypeIntervention.timestamp
            if (timeSinceLastIntervention < safetyConstraints.minIntervalBetweenInterventions) {
                val remainingTime = (safetyConstraints.minIntervalBetweenInterventions - timeSinceLastIntervention) / 1000
                return ConstraintValidationResult(
                    false,
                    ConstraintType.TIMING_INTERVAL,
                    "Minimum interval not met (${remainingTime}s remaining)"
                )
            }
        }
        
        // Check quiet periods
        for (quietPeriod in userDefinedQuietPeriods) {
            if (quietPeriod.isActive(currentTime, context.timeContext.hourOfDay, context.timeContext.dayOfWeek)) {
                if (recommendation.type !in quietPeriod.allowedInterventionTypes) {
                    return ConstraintValidationResult(
                        false,
                        ConstraintType.TIMING_QUIET_PERIOD,
                        "Intervention blocked during quiet period: ${quietPeriod.name}"
                    )
                }
            }
        }
        
        return ConstraintValidationResult(true, ConstraintType.TIMING_INTERVAL, "Timing constraints OK")
    }
    
    private fun checkSequenceConstraints(recommendation: InterventionRecommendation): ConstraintValidationResult {
        val recentSequence = recentInterventions
            .filter { it.timestamp > System.currentTimeMillis() - (2 * 60 * 60 * 1000L) } // Last 2 hours
            .sortedByDescending { it.timestamp }
            .take(5)
        
        // Check for repetitive patterns
        val sameTypeCount = recentSequence.count { it.type == recommendation.type }
        if (sameTypeCount >= 3) {
            return ConstraintValidationResult(
                false,
                ConstraintType.SEQUENCE_REPETITIVE,
                "Too many similar interventions in sequence ($sameTypeCount/3)"
            )
        }
        
        // Check for escalating intensity
        val strongInterventions = recentSequence.count { it.isStrongIntervention }
        if (strongInterventions >= safetyConstraints.maxConsecutiveStrongInterventions && 
            isStrongInterventionType(recommendation.type)) {
            return ConstraintValidationResult(
                false,
                ConstraintType.SEQUENCE_ESCALATION,
                "Too many consecutive strong interventions"
            )
        }
        
        return ConstraintValidationResult(true, ConstraintType.SEQUENCE_REPETITIVE, "Sequence constraints OK")
    }
    
    private fun checkContextualConstraints(
        recommendation: InterventionRecommendation,
        context: BehavioralContext
    ): ConstraintValidationResult {
        // Emergency mode check
        if (isEmergencyMode) {
            val criticalTypes = listOf(
                InterventionType.DND_ENABLE,
                InterventionType.ACTIVITY_SUGGESTION
            )
            
            if (recommendation.type !in criticalTypes) {
                return ConstraintValidationResult(
                    false,
                    ConstraintType.CONTEXT_EMERGENCY,
                    "Non-critical intervention blocked during emergency mode"
                )
            }
        }
        
        // Important call check
        if (isInImportantCall && recommendation.type != InterventionType.DND_ENABLE) {
            return ConstraintValidationResult(
                false,
                ConstraintType.CONTEXT_CALL,
                "Intervention blocked during important call"
            )
        }
        
        // Context-specific restrictions
        val hour = context.timeContext.hourOfDay
        when (recommendation.type) {
            InterventionType.ACTIVITY_SUGGESTION -> {
                if (hour >= 23 || hour <= 5) {
                    return ConstraintValidationResult(
                        false,
                        ConstraintType.CONTEXT_TIME,
                        "Activity suggestion blocked during sleep hours"
                    )
                }
            }
            InterventionType.FOCUS_MODE_ENABLE -> {
                if (context.environmentContext.batteryLevel < 20 && !context.environmentContext.isCharging) {
                    return ConstraintValidationResult(
                        false,
                        ConstraintType.CONTEXT_BATTERY,
                        "Focus mode blocked due to low battery"
                    )
                }
            }
        }
        
        return ConstraintValidationResult(true, ConstraintType.CONTEXT_EMERGENCY, "Contextual constraints OK")
    }
    
    private fun checkEmergencyConstraints(
        recommendation: InterventionRecommendation,
        context: BehavioralContext
    ): ConstraintValidationResult {
        // Check for emergency indicators in context
        if (context.userState.stressLevel > 0.9) {
            // Only allow stress reduction interventions during high stress
            if (recommendation.type != InterventionType.BREAK_SUGGESTION &&
                recommendation.type != InterventionType.ACTIVITY_SUGGESTION) {
                return ConstraintValidationResult(
                    false,
                    ConstraintType.CONTEXT_EMERGENCY,
                    "Only stress-relief interventions allowed during high stress"
                )
            }
        }
        
        return ConstraintValidationResult(true, ConstraintType.CONTEXT_EMERGENCY, "Emergency constraints OK")
    }
    
    private fun checkUserPreferenceConstraints(recommendation: InterventionRecommendation): ConstraintValidationResult {
        // TODO: Implement user-specific preferences and blacklists
        // For now, no user preference violations
        return ConstraintValidationResult(true, ConstraintType.USER_PREFERENCE, "User preferences OK")
    }
    
    private fun checkResourceConstraints(
        recommendation: InterventionRecommendation,
        context: BehavioralContext
    ): ConstraintValidationResult {
        // Check battery level for resource-intensive interventions
        if (context.environmentContext.batteryLevel < 10 && !context.environmentContext.isCharging) {
            val resourceIntensiveTypes = listOf(
                InterventionType.FOCUS_MODE_ENABLE,
                InterventionType.NOTIFICATION_REDUCTION
            )
            
            if (recommendation.type in resourceIntensiveTypes) {
                return ConstraintValidationResult(
                    false,
                    ConstraintType.RESOURCE_BATTERY,
                    "Resource-intensive intervention blocked due to critical battery level"
                )
            }
        }
        
        return ConstraintValidationResult(true, ConstraintType.RESOURCE_BATTERY, "Resource constraints OK")
    }
    
    private fun checkEscalationPrevention(recommendation: InterventionRecommendation): ConstraintValidationResult {
        // Check for escalating intervention intensity
        val recentStrongInterventions = recentInterventions
            .filter { it.timestamp > System.currentTimeMillis() - (60 * 60 * 1000L) }
            .filter { it.isStrongIntervention }
            .size
        
        if (recentStrongInterventions >= safetyConstraints.maxConsecutiveStrongInterventions) {
            val isStrongIntervention = isStrongInterventionType(recommendation.type)
            if (isStrongIntervention) {
                return ConstraintValidationResult(
                    false,
                    ConstraintType.ESCALATION_PREVENTION,
                    "Escalation prevention: too many strong interventions"
                )
            }
        }
        
        return ConstraintValidationResult(true, ConstraintType.ESCALATION_PREVENTION, "Escalation prevention OK")
    }
    
    private fun checkCumulativeImpact(recommendation: InterventionRecommendation): ConstraintValidationResult {
        val recentInterventionCount = recentInterventions.count { 
            it.timestamp > System.currentTimeMillis() - (60 * 60 * 1000L) 
        }
        
        // Check cumulative intervention load
        if (recentInterventionCount >= safetyConstraints.maxCumulativeInterventionsPerHour) {
            return ConstraintValidationResult(
                false,
                ConstraintType.CUMULATIVE_IMPACT,
                "Cumulative intervention limit exceeded ($recentInterventionCount/${safetyConstraints.maxCumulativeInterventionsPerHour})"
            )
        }
        
        return ConstraintValidationResult(true, ConstraintType.CUMULATIVE_IMPACT, "Cumulative impact OK")
    }
    
    /**
     * Enhanced utility and management methods
     */
    
    private fun recordViolation(
        recommendation: InterventionRecommendation,
        constraintType: ConstraintType,
        reason: String
    ) {
        val violation = SafetyViolation(
            interventionId = recommendation.id,
            timestamp = System.currentTimeMillis(),
            constraintType = constraintType,
            reason = reason,
            interventionType = recommendation.type
        )
        
        violationHistory.add(violation)
        constraintViolationCounts[constraintType.name] = 
            (constraintViolationCounts[constraintType.name] ?: 0) + 1
        
        safetyStatistics.totalViolations++
        safetyStatistics.lastViolationTime = violation.timestamp
        
        consecutiveViolations++
        
        // Trigger safety cooldown if too many consecutive violations
        if (consecutiveViolations >= safetyConstraints.maxConsecutiveViolations) {
            triggerSafetyCooldown()
        }
        
        // Emit safety event
        CoroutineScope(Dispatchers.Default).launch {
            _safetyEvents.emit(SafetyEvent.ViolationDetected(violation))
        }
        
        Log.w(TAG, "Safety violation: ${constraintType.name} - $reason")
    }
    
    private fun recordIntervention(recommendation: InterventionRecommendation) {
        val record = InterventionRecord(
            interventionId = recommendation.id,
            timestamp = System.currentTimeMillis(),
            type = recommendation.type,
            isStrongIntervention = isStrongInterventionType(recommendation.type)
        )
        
        recentInterventions.add(record)
        consecutiveViolations = 0 // Reset violation counter on successful intervention
        
        // Emit safety event
        CoroutineScope(Dispatchers.Default).launch {
            _safetyEvents.emit(SafetyEvent.InterventionApproved(record))
        }
    }
    
    private fun createViolationResult(
        recommendation: InterventionRecommendation,
        constraintType: ConstraintType,
        reason: String
    ): SafetyValidationResult {
        recordViolation(recommendation, constraintType, reason)
        
        return SafetyValidationResult(
            isValid = false,
            violations = listOf(ConstraintValidationResult(false, constraintType, reason)),
            alternativeAction = null,
            safetyScore = 0.0,
            recommendedCooldown = calculateRecommendedCooldown(
                listOf(ConstraintValidationResult(false, constraintType, reason))
            )
        )
    }
    
    private fun findSafeAlternative(
        recommendation: InterventionRecommendation,
        context: BehavioralContext
    ): InterventionRecommendation? {
        // Try to find a less intensive alternative
        val alternatives = when (recommendation.type) {
            InterventionType.FOCUS_MODE_ENABLE -> listOf(InterventionType.NOTIFICATION_REDUCTION)
            InterventionType.APP_LIMIT_SUGGESTION -> listOf(InterventionType.BREAK_SUGGESTION)
            InterventionType.DND_ENABLE -> listOf(InterventionType.NOTIFICATION_REDUCTION)
            else -> emptyList()
        }
        
        for (altType in alternatives) {
            val altRecommendation = recommendation.copy(type = altType)
            val altValidation = validateRecommendation(altRecommendation, context)
            if (altValidation.isValid) {
                return altRecommendation
            }
        }
        
        return null
    }
    
    private fun calculateSafetyScore(violations: List<ConstraintValidationResult>): Double {
        if (violations.isEmpty()) return 1.0
        
        val severityWeights = mapOf(
            ConstraintType.SYSTEM_ERROR to 1.0,
            ConstraintType.CONTEXT_EMERGENCY to 0.9,
            ConstraintType.ESCALATION_PREVENTION to 0.8,
            ConstraintType.FREQUENCY_DAILY to 0.7,
            ConstraintType.SEQUENCE_ESCALATION to 0.6,
            ConstraintType.FREQUENCY_HOURLY to 0.5,
            ConstraintType.TIMING_INTERVAL to 0.4,
            ConstraintType.CONTEXT_CALL to 0.3,
            ConstraintType.TIMING_QUIET_PERIOD to 0.2,
            ConstraintType.USER_PREFERENCE to 0.1
        )
        
        val totalSeverity = violations.sumOf { violation ->
            severityWeights[violation.constraintType] ?: 0.5
        }
        
        return max(0.0, 1.0 - (totalSeverity / violations.size))
    }
    
    private fun calculateRecommendedCooldown(violations: List<ConstraintValidationResult>): Long {
        val baseCooldown = 5 * 60 * 1000L // 5 minutes
        val severityMultiplier = violations.maxOfOrNull { violation ->
            when (violation.constraintType) {
                ConstraintType.SYSTEM_ERROR -> 6.0
                ConstraintType.CONTEXT_EMERGENCY -> 4.0
                ConstraintType.ESCALATION_PREVENTION -> 3.0
                ConstraintType.FREQUENCY_DAILY -> 2.0
                else -> 1.0
            }
        } ?: 1.0
        
        return (baseCooldown * severityMultiplier).toLong()
    }
    
    private fun triggerSafetyCooldown() {
        val cooldownDuration = safetyConstraints.safetyCooldownDuration
        safetyCooldownUntil = System.currentTimeMillis() + cooldownDuration
        
        CoroutineScope(Dispatchers.Default).launch {
            _safetyEvents.emit(SafetyEvent.CooldownTriggered(cooldownDuration))
        }
        
        Log.w(TAG, "Safety cooldown triggered for ${cooldownDuration / 1000}s due to consecutive violations")
    }
    
    private fun isInSafetyCooldown(): Boolean {
        return System.currentTimeMillis() < safetyCooldownUntil
    }
    
    private fun checkSafetyCooldown() {
        if (safetyCooldownUntil > 0 && System.currentTimeMillis() >= safetyCooldownUntil) {
            safetyCooldownUntil = 0L
            consecutiveViolations = 0
            
            CoroutineScope(Dispatchers.Default).launch {
                _safetyEvents.emit(SafetyEvent.CooldownExpired)
            }
        }
    }
    
    private fun cleanupOldRecords() {
        val cutoffTime = System.currentTimeMillis() - (7 * 24 * 60 * 60 * 1000L) // 7 days
        
        recentInterventions.removeAll { it.timestamp < cutoffTime }
        violationHistory.removeAll { it.timestamp < cutoffTime }
    }
    
    private fun updateSafetyStatistics() {
        val totalValidations = safetyStatistics.totalValidations
        val totalViolations = safetyStatistics.totalViolations
        
        safetyStatistics.violationRate = if (totalValidations > 0) {
            totalViolations.toDouble() / totalValidations
        } else 0.0
        
        // Calculate violations per hour over last 24 hours
        val oneDayAgo = System.currentTimeMillis() - (24 * 60 * 60 * 1000L)
        val recentViolations = violationHistory.count { it.timestamp > oneDayAgo }
        safetyStatistics.averageViolationsPerHour = recentViolations / 24.0
        
        // Find most common violation type
        safetyStatistics.mostCommonViolationType = constraintViolationCounts
            .maxByOrNull { it.value }?.key ?: ""
        
        // Calculate overall safety score
        safetyStatistics.safetyScore = calculateOverallSafetyScore()
    }
    
    private fun calculateOverallSafetyScore(): Double {
        val violationRate = safetyStatistics.violationRate
        val recentViolationRate = calculateRecentViolationRate()
        val cooldownPenalty = if (isInSafetyCooldown()) 0.2 else 0.0
        
        val baseScore = 1.0 - violationRate
        val recentPenalty = recentViolationRate * 0.3
        
        return max(0.0, baseScore - recentPenalty - cooldownPenalty)
    }
    
    private fun calculateRecentViolationRate(): Double {
        val oneHourAgo = System.currentTimeMillis() - (60 * 60 * 1000L)
        val recentViolations = violationHistory.count { it.timestamp > oneHourAgo }
        val recentValidations = max(1, recentViolations + recentInterventions.count { it.timestamp > oneHourAgo })
        
        return recentViolations.toDouble() / recentValidations
    }
    
    private fun isStrongInterventionType(type: InterventionType): Boolean {
        return when (type) {
            InterventionType.APP_LIMIT_SUGGESTION -> true
            InterventionType.DND_ENABLE -> true
            InterventionType.FOCUS_MODE_ENABLE -> true
            InterventionType.BREAK_SUGGESTION -> false
            InterventionType.NOTIFICATION_REDUCTION -> false
            InterventionType.ACTIVITY_SUGGESTION -> false
        }
    }
    
    private fun loadDefaultConstraints() {
        safetyConstraints.apply {
            maxInterventionsPerHour = 2
            maxInterventionsPerDay = 12
            minIntervalBetweenInterventions = 30 * 60 * 1000L // 30 minutes
            maxConsecutiveStrongInterventions = 2
            maxCumulativeInterventionsPerHour = 5
            maxConsecutiveViolations = 3
            safetyCooldownDuration = 15 * 60 * 1000L // 15 minutes
            emergencyModeEnabled = true
            respectCallState = true
        }
    }
}

// Enhanced data classes for comprehensive safety management

/**
 * Enhanced safety constraints with multiple constraint types
 */
data class SafetyConstraints(
    var maxInterventionsPerHour: Int = 2,
    var maxInterventionsPerDay: Int = 12,
    var minIntervalBetweenInterventions: Long = 30 * 60 * 1000L, // 30 minutes
    var maxConsecutiveStrongInterventions: Int = 2,
    var maxCumulativeInterventionsPerHour: Int = 5,
    var maxConsecutiveViolations: Int = 3,
    var safetyCooldownDuration: Long = 15 * 60 * 1000L, // 15 minutes
    var emergencyModeEnabled: Boolean = true,
    var respectCallState: Boolean = true
)

/**
 * Constraint types for detailed violation tracking
 */
enum class ConstraintType {
    FREQUENCY_HOURLY,
    FREQUENCY_DAILY,
    TIMING_INTERVAL,
    TIMING_QUIET_PERIOD,
    SEQUENCE_REPETITIVE,
    SEQUENCE_ESCALATION,
    CONTEXT_EMERGENCY,
    CONTEXT_CALL,
    CONTEXT_TIME,
    CONTEXT_BATTERY,
    USER_PREFERENCE,
    RESOURCE_BATTERY,
    ESCALATION_PREVENTION,
    CUMULATIVE_IMPACT,
    COOLDOWN,
    SYSTEM_ERROR
}

/**
 * Enhanced validation result with constraint details
 */
data class ConstraintValidationResult(
    val isValid: Boolean,
    val constraintType: ConstraintType,
    val reason: String
)

/**
 * Comprehensive safety validation result
 */
data class SafetyValidationResult(
    val isValid: Boolean,
    val violations: List<ConstraintValidationResult>,
    val alternativeAction: InterventionRecommendation?,
    val safetyScore: Double, // 0.0 to 1.0
    val recommendedCooldown: Long // milliseconds
)

/**
 * Safety violation record
 */
@Serializable
data class SafetyViolation(
    val interventionId: String,
    val timestamp: Long,
    val constraintType: ConstraintType,
    val reason: String,
    val interventionType: InterventionType
)

/**
 * Enhanced intervention record
 */
@Serializable
data class InterventionRecord(
    val interventionId: String,
    val timestamp: Long,
    val type: InterventionType,
    val isStrongIntervention: Boolean
)

/**
 * Comprehensive safety statistics
 */
@Serializable
data class SafetyStatistics(
    var totalValidations: Int = 0,
    var totalViolations: Int = 0,
    var violationRate: Double = 0.0, // 0.0 to 1.0
    var lastViolationTime: Long = 0L,
    var averageViolationsPerHour: Double = 0.0,
    var mostCommonViolationType: String = "",
    var safetyScore: Double = 1.0 // 0.0 to 1.0, higher is better
)

/**
 * Safety events for monitoring and notifications
 */
sealed class SafetyEvent {
    data class ViolationDetected(val violation: SafetyViolation) : SafetyEvent()
    data class InterventionApproved(val intervention: InterventionRecord) : SafetyEvent()
    data class CooldownTriggered(val duration: Long) : SafetyEvent()
    data class ManualCooldownTriggered(val duration: Long) : SafetyEvent()
    object CooldownExpired : SafetyEvent()
    object CooldownCleared : SafetyEvent()
    data class ConstraintsUpdated(val constraints: SafetyConstraints) : SafetyEvent()
    object SafetyDataReset : SafetyEvent()
}

/**
 * Enhanced quiet period with more flexibility
 */
data class QuietPeriod(
    val id: String,
    val name: String,
    val startHour: Int,
    val endHour: Int,
    val daysOfWeek: List<Int>, // Calendar.SUNDAY, etc.
    val allowedInterventionTypes: List<InterventionType> = emptyList()
) {
    fun isActive(currentTime: Long, currentHour: Int, currentDayOfWeek: Int): Boolean {
        // Check if current day is in the quiet period
        if (currentDayOfWeek !in daysOfWeek) return false
        
        // Check if current hour is in the quiet period
        return if (startHour <= endHour) {
            currentHour in startHour..endHour
        } else {
            // Handles overnight periods (e.g., 23:00 to 06:00)
            currentHour >= startHour || currentHour <= endHour
        }
    }
}