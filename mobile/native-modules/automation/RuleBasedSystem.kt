package com.lifetwin.mlp.automation

import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.util.concurrent.ConcurrentHashMap

private const val TAG = "RuleBasedSystem"

/**
 * Rule-based automation system that evaluates behavioral triggers and suggests interventions.
 */
class RuleBasedSystem(private val context: Context) {
    
    private val activeRules = ConcurrentHashMap<String, AutomationRule>()
    private val appCategoryMapping = AppCategoryMapping()
    private lateinit var database: AppDatabase
    
    // User preferences and quiet hours
    private val userPreferences = ConcurrentHashMap<String, Any>()
    private var quietHoursEnabled = false
    private var quietHoursStart = 22 // 10 PM
    private var quietHoursEnd = 7   // 7 AM
    private var doNotDisturbActive = false
    
    /**
     * Update user preferences for automation behavior
     */
    fun updateUserPreferences(preferences: Map<String, Any>) {
        userPreferences.clear()
        userPreferences.putAll(preferences)
        
        // Update quiet hours settings
        quietHoursEnabled = preferences["quiet_hours_enabled"] as? Boolean ?: false
        quietHoursStart = preferences["quiet_hours_start"] as? Int ?: 22
        quietHoursEnd = preferences["quiet_hours_end"] as? Int ?: 7
        doNotDisturbActive = preferences["dnd_active"] as? Boolean ?: false
        
        // Update rule thresholds from preferences
        updateRuleThresholdsFromPreferences(preferences)
        
        Log.d(TAG, "Updated user preferences: quiet hours $quietHoursStart-$quietHoursEnd, enabled=$quietHoursEnabled")
    }
    
    /**
     * Check if current time is within quiet hours
     */
    fun isQuietHours(): Boolean {
        if (!quietHoursEnabled) return false
        
        val currentHour = java.util.Calendar.getInstance().get(java.util.Calendar.HOUR_OF_DAY)
        
        return if (quietHoursStart <= quietHoursEnd) {
            // Same day quiet hours (e.g., 14:00 - 18:00)
            currentHour in quietHoursStart..quietHoursEnd
        } else {
            // Overnight quiet hours (e.g., 22:00 - 07:00)
            currentHour >= quietHoursStart || currentHour <= quietHoursEnd
        }
    }
    
    /**
     * Check if a specific intervention type is enabled by user preferences
     */
    fun isInterventionTypeEnabled(type: InterventionType): Boolean {
        val prefKey = "${type.name.lowercase()}_enabled"
        return userPreferences[prefKey] as? Boolean ?: true
    }
    
    /**
     * Get custom threshold for a rule from user preferences
     */
    fun getCustomThreshold(ruleId: String): Any? {
        return userPreferences["${ruleId}_threshold"]
    }
    
    /**
     * Set quiet hours configuration
     */
    fun setQuietHours(enabled: Boolean, startHour: Int, endHour: Int) {
        quietHoursEnabled = enabled
        quietHoursStart = startHour
        quietHoursEnd = endHour
        
        userPreferences["quiet_hours_enabled"] = enabled
        userPreferences["quiet_hours_start"] = startHour
        userPreferences["quiet_hours_end"] = endHour
        
        Log.d(TAG, "Updated quiet hours: $startHour-$endHour, enabled=$enabled")
    }
    
    /**
     * Enable or disable Do Not Disturb mode
     */
    fun setDoNotDisturb(active: Boolean) {
        doNotDisturbActive = active
        userPreferences["dnd_active"] = active
        Log.d(TAG, "Do Not Disturb ${if (active) "enabled" else "disabled"}")
    }
    
    /**
     * Get current user preference settings
     */
    fun getUserPreferences(): Map<String, Any> {
        return userPreferences.toMap()
    }
    
    /**
     * Get intervention frequency limits from preferences
     */
    fun getInterventionLimits(): InterventionLimits {
        return InterventionLimits(
            maxPerHour = userPreferences["max_interventions_per_hour"] as? Int ?: 3,
            maxPerDay = userPreferences["max_interventions_per_day"] as? Int ?: 20,
            minIntervalMinutes = userPreferences["min_intervention_interval"] as? Int ?: 15,
            respectQuietHours = quietHoursEnabled,
            respectDnd = doNotDisturbActive
        )
    }
    
    /**
     * Check if intervention should be suppressed based on user preferences
     */
    fun shouldSuppressIntervention(recommendation: InterventionRecommendation): Boolean {
        // Check if intervention type is disabled
        if (!isInterventionTypeEnabled(recommendation.type)) {
            Log.d(TAG, "Suppressing ${recommendation.type} - disabled by user preference")
            return true
        }
        
        // Check quiet hours
        if (isQuietHours()) {
            Log.d(TAG, "Suppressing ${recommendation.type} - quiet hours active")
            return true
        }
        
        // Check Do Not Disturb
        if (doNotDisturbActive && recommendation.type != InterventionType.DND_ENABLE) {
            Log.d(TAG, "Suppressing ${recommendation.type} - DND active")
            return true
        }
        
        // Check confidence threshold
        val minConfidence = userPreferences["min_confidence_threshold"] as? Float ?: 0.3f
        if (recommendation.confidence < minConfidence) {
            Log.d(TAG, "Suppressing ${recommendation.type} - confidence ${recommendation.confidence} below threshold $minConfidence")
            return true
        }
        
        return false
    }
    
    private fun updateRuleThresholdsFromPreferences(preferences: Map<String, Any>) {
        val thresholdUpdates = mutableMapOf<String, Any>()
        
        // Extract rule-specific thresholds from preferences
        for ((key, value) in preferences) {
            if (key.endsWith("_threshold")) {
                val ruleId = key.removeSuffix("_threshold")
                thresholdUpdates[ruleId] = value
            }
        }
        
        if (thresholdUpdates.isNotEmpty()) {
            updateRuleThresholds(thresholdUpdates)
        }
    }
    
    suspend fun initialize(): Boolean {
        return try {
            Log.i(TAG, "Initializing RuleBasedSystem...")
            
            // Initialize database connection
            database = AppDatabase.getInstance(context)
            
            // Load default rules
            loadDefaultRules()
            
            // Initialize app category mapping
            appCategoryMapping.initialize()
            
            Log.i(TAG, "RuleBasedSystem initialized with ${activeRules.size} rules")
            true
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize RuleBasedSystem", e)
            false
        }
    }
    
    /**
     * Evaluate all active rules against the current behavioral context
     */
    suspend fun evaluateRules(context: BehavioralContext): List<InterventionRecommendation> {
        return withContext(Dispatchers.Default) {
            val recommendations = mutableListOf<InterventionRecommendation>()
            
            for (rule in activeRules.values) {
                if (rule.isEnabled) {
                    try {
                        val recommendation = rule.evaluate(context)
                        if (recommendation != null && !shouldSuppressIntervention(recommendation)) {
                            recommendations.add(recommendation)
                            Log.d(TAG, "Rule '${rule.name}' triggered: ${recommendation.reasoning}")
                        } else if (recommendation != null) {
                            Log.d(TAG, "Rule '${rule.name}' triggered but suppressed by user preferences")
                        }
                    } catch (e: Exception) {
                        Log.e(TAG, "Error evaluating rule '${rule.name}'", e)
                    }
                }
            }
            
            // Sort by confidence and priority, then by trigger type
            recommendations.sortedWith(
                compareByDescending<InterventionRecommendation> { it.confidence }
                    .thenBy { it.type.ordinal }
            )
        }
    }
    
    /**
     * Update rule thresholds based on user preferences
     */
    fun updateRuleThresholds(rules: Map<String, Any>) {
        for ((ruleId, threshold) in rules) {
            activeRules[ruleId]?.let { rule ->
                rule.updateThreshold(threshold)
                Log.d(TAG, "Updated threshold for rule '$ruleId': $threshold")
            }
        }
    }
    
    /**
     * Add a custom user-defined rule
     */
    fun addCustomRule(rule: CustomRule) {
        activeRules[rule.id] = rule
        Log.i(TAG, "Added custom rule: ${rule.name}")
    }
    
    /**
     * Get all currently active rules
     */
    fun getActiveRules(): List<AutomationRule> {
        return activeRules.values.toList()
    }
    
    /**
     * Compute app category usage from raw usage events
     */
    suspend fun computeAppCategoryUsage(
        startTime: Long,
        endTime: Long
    ): Map<AppCategory, Long> {
        return withContext(Dispatchers.IO) {
            try {
                val usageEvents = database.usageEventDao().getEventsByTimeRange(startTime, endTime)
                val cacheKey = "usage_${startTime}_${endTime}"
                
                val categoryUsage = appCategoryMapping.computeCategoryUsage(usageEvents, cacheKey)
                
                Log.d(TAG, "Computed category usage for ${usageEvents.size} events: $categoryUsage")
                categoryUsage
                
            } catch (e: Exception) {
                Log.e(TAG, "Failed to compute app category usage", e)
                emptyMap()
            }
        }
    }
    
    /**
     * Get category-based intervention triggers
     */
    suspend fun getCategoryTriggers(
        startTime: Long,
        endTime: Long,
        thresholds: Map<AppCategory, Long>
    ): List<CategoryTrigger> {
        return withContext(Dispatchers.IO) {
            try {
                val categoryUsage = computeAppCategoryUsage(startTime, endTime)
                appCategoryMapping.getCategoryTriggers(categoryUsage, thresholds)
            } catch (e: Exception) {
                Log.e(TAG, "Failed to get category triggers", e)
                emptyList()
            }
        }
    }
    
    /**
     * Get usage distribution by category
     */
    suspend fun getCategoryDistribution(
        startTime: Long,
        endTime: Long
    ): Map<AppCategory, Float> {
        return withContext(Dispatchers.IO) {
            try {
                val categoryUsage = computeAppCategoryUsage(startTime, endTime)
                appCategoryMapping.getCategoryDistribution(categoryUsage)
            } catch (e: Exception) {
                Log.e(TAG, "Failed to get category distribution", e)
                emptyMap()
            }
        }
    }
    
    /**
     * Get top apps by category
     */
    suspend fun getTopAppsByCategory(
        startTime: Long,
        endTime: Long,
        category: AppCategory,
        limit: Int = 5
    ): List<AppUsageInfo> {
        return withContext(Dispatchers.IO) {
            try {
                val usageEvents = database.usageEventDao().getEventsByTimeRange(startTime, endTime)
                appCategoryMapping.getTopAppsByCategory(usageEvents, category, limit)
            } catch (e: Exception) {
                Log.e(TAG, "Failed to get top apps by category", e)
                emptyList()
            }
        }
    }
    
    /**
     * Set custom category for an app
     */
    fun setCustomAppCategory(packageName: String, category: AppCategory) {
        appCategoryMapping.setCustomCategory(packageName, category)
        Log.d(TAG, "Set custom category for $packageName: $category")
    }
    
    /**
     * Get notification count from database for time range
     */
    suspend fun getNotificationCount(startTime: Long, endTime: Long): Int {
        return withContext(Dispatchers.IO) {
            try {
                database.notificationEventDao().getEventCountByTimeRange(startTime, endTime)
            } catch (e: Exception) {
                Log.e(TAG, "Failed to get notification count", e)
                0
            }
        }
    }
    
    /**
     * Enable or disable a specific rule
     */
    fun setRuleEnabled(ruleId: String, enabled: Boolean) {
        activeRules[ruleId]?.let { rule ->
            rule.isEnabled = enabled
            Log.d(TAG, "Rule '$ruleId' ${if (enabled) "enabled" else "disabled"}")
        }
    }
    
    /**
     * Get rule status and configuration
     */
    fun getRuleStatus(): Map<String, RuleStatus> {
        return activeRules.mapValues { (_, rule) ->
            RuleStatus(
                id = rule.id,
                name = rule.name,
                isEnabled = rule.isEnabled,
                description = rule.getDescription(),
                configuration = rule.getConfiguration()
            )
        }
    }
    
    private fun loadDefaultRules() {
        // Social media usage rule
        activeRules["social_usage_limit"] = SocialUsageRule(
            threshold = 90 * 60 * 1000L, // 90 minutes in milliseconds
            windowMinutes = 120 // 2 hour window
        )
        
        // Late night usage rule
        activeRules["late_night_usage"] = LateNightUsageRule(
            startHour = 23,
            endHour = 6
        )
        
        // High notification frequency rule
        activeRules["notification_overload"] = NotificationOverloadRule(
            threshold = 15, // notifications per hour
            focusHoursOnly = true
        )
        
        // Low work productivity rule
        activeRules["low_work_productivity"] = WorkProductivityRule(
            minWorkPercentage = 0.3f, // 30% minimum
            workHoursOnly = true
        )
        
        // Inactivity detection rule
        activeRules["inactivity_detection"] = InactivityRule(
            thresholdMinutes = 30,
            dayTimeOnly = true
        )
        
        // Focus time protection rule
        activeRules["focus_protection"] = FocusProtectionRule(
            maxInterruptionsPerHour = 3
        )
    }
}

/**
 * Base class for all automation rules
 */
abstract class AutomationRule(
    val id: String,
    val name: String,
    var isEnabled: Boolean = true
) {
    abstract fun evaluate(context: BehavioralContext): InterventionRecommendation?
    abstract fun updateThreshold(threshold: Any)
    abstract fun getDescription(): String
    
    /**
     * Get current configuration as a map
     */
    open fun getConfiguration(): Map<String, Any> {
        return mapOf(
            "id" to id,
            "name" to name,
            "enabled" to isEnabled
        )
    }
}

/**
 * Rule for detecting excessive social media usage
 */
class SocialUsageRule(
    private var threshold: Long,
    private val windowMinutes: Int
) : AutomationRule("social_usage_limit", "Social Media Usage Limit") {
    
    override fun evaluate(context: BehavioralContext): InterventionRecommendation? {
        val socialUsage = context.currentUsage.socialUsage
        
        return if (socialUsage > threshold) {
            val usageMinutes = socialUsage / (60 * 1000)
            InterventionRecommendation(
                type = InterventionType.BREAK_SUGGESTION,
                trigger = "social_usage_exceeded",
                confidence = minOf(1.0f, socialUsage.toFloat() / threshold.toFloat()),
                reasoning = "You've spent $usageMinutes minutes on social media in the last $windowMinutes minutes. Consider taking a break to recharge!",
                expectedImpact = ImpactPrediction(
                    energyChange = 0.2f,
                    focusChange = 0.3f,
                    moodChange = 0.1f
                )
            )
        } else null
    }
    
    override fun updateThreshold(threshold: Any) {
        if (threshold is Number) {
            this.threshold = threshold.toLong() * 60 * 1000L // Convert minutes to milliseconds
        }
    }
    
    override fun getDescription(): String {
        return "Suggests breaks when social media usage exceeds ${threshold / (60 * 1000)} minutes in $windowMinutes minutes"
    }
    
    override fun getConfiguration(): Map<String, Any> {
        return super.getConfiguration() + mapOf(
            "threshold_minutes" to (threshold / (60 * 1000)),
            "window_minutes" to windowMinutes
        )
    }
}

/**
 * Rule for detecting late night device usage
 */
class LateNightUsageRule(
    private val startHour: Int,
    private val endHour: Int
) : AutomationRule("late_night_usage", "Late Night Usage Detection") {
    
    override fun evaluate(context: BehavioralContext): InterventionRecommendation? {
        val currentHour = context.timeContext.hourOfDay
        
        return if (currentHour >= startHour || currentHour <= endHour) {
            InterventionRecommendation(
                type = InterventionType.DND_ENABLE,
                trigger = "late_night_usage",
                confidence = 0.8f,
                reasoning = "It's ${currentHour}:00 - consider enabling Do Not Disturb for better sleep quality."
            )
        } else null
    }
    
    override fun updateThreshold(threshold: Any) {
        // Late night hours are typically fixed, but could be customizable
    }
    
    override fun getDescription(): String {
        return "Suggests enabling Do Not Disturb between ${startHour}:00 and ${endHour}:00 for better sleep"
    }
    
    override fun getConfiguration(): Map<String, Any> {
        return super.getConfiguration() + mapOf(
            "start_hour" to startHour,
            "end_hour" to endHour
        )
    }
}

/**
 * Rule for detecting notification overload
 */
class NotificationOverloadRule(
    private var threshold: Int,
    private val focusHoursOnly: Boolean
) : AutomationRule("notification_overload", "Notification Overload Detection") {
    
    override fun evaluate(context: BehavioralContext): InterventionRecommendation? {
        val notificationCount = context.currentUsage.notificationCount
        val isWorkHour = context.timeContext.isWorkHour
        
        return if (notificationCount > threshold && (!focusHoursOnly || isWorkHour)) {
            InterventionRecommendation(
                type = InterventionType.NOTIFICATION_REDUCTION,
                trigger = "notification_overload",
                confidence = minOf(1.0f, notificationCount.toFloat() / threshold.toFloat()),
                reasoning = "You've received $notificationCount notifications in the last hour. Consider reducing notification frequency for better focus."
            )
        } else null
    }
    
    override fun updateThreshold(threshold: Any) {
        if (threshold is Number) {
            this.threshold = threshold.toInt()
        }
    }
    
    override fun getDescription(): String {
        val timeScope = if (focusHoursOnly) "during work hours" else "anytime"
        return "Suggests reducing notifications when count exceeds $threshold per hour $timeScope"
    }
    
    override fun getConfiguration(): Map<String, Any> {
        return super.getConfiguration() + mapOf(
            "threshold" to threshold,
            "focus_hours_only" to focusHoursOnly
        )
    }
}

/**
 * Rule for detecting low work productivity
 */
class WorkProductivityRule(
    private var minWorkPercentage: Float,
    private val workHoursOnly: Boolean
) : AutomationRule("low_work_productivity", "Work Productivity Monitor") {
    
    override fun evaluate(context: BehavioralContext): InterventionRecommendation? {
        if (workHoursOnly && !context.timeContext.isWorkHour) return null
        
        val totalUsage = context.currentUsage.totalScreenTime
        val workUsage = context.currentUsage.workUsage
        
        if (totalUsage == 0L) return null
        
        val workPercentage = workUsage.toFloat() / totalUsage.toFloat()
        
        return if (workPercentage < minWorkPercentage) {
            InterventionRecommendation(
                type = InterventionType.FOCUS_MODE_ENABLE,
                trigger = "low_work_productivity",
                confidence = 1.0f - workPercentage,
                reasoning = "Work apps account for only ${(workPercentage * 100).toInt()}% of your screen time during work hours. Consider enabling focus mode."
            )
        } else null
    }
    
    override fun updateThreshold(threshold: Any) {
        if (threshold is Number) {
            this.minWorkPercentage = threshold.toFloat()
        }
    }
    
    override fun getDescription(): String {
        val timeScope = if (workHoursOnly) "during work hours" else "anytime"
        return "Suggests focus mode when work app usage falls below ${(minWorkPercentage * 100).toInt()}% $timeScope"
    }
    
    override fun getConfiguration(): Map<String, Any> {
        return super.getConfiguration() + mapOf(
            "min_work_percentage" to minWorkPercentage,
            "work_hours_only" to workHoursOnly
        )
    }
}

/**
 * Rule for detecting extended inactivity periods
 */
class InactivityRule(
    private var thresholdMinutes: Int,
    private val dayTimeOnly: Boolean
) : AutomationRule("inactivity_detection", "Inactivity Detection") {
    
    private var lastActivityTime = System.currentTimeMillis()
    
    override fun evaluate(context: BehavioralContext): InterventionRecommendation? {
        val currentTime = System.currentTimeMillis()
        val inactiveMinutes = (currentTime - lastActivityTime) / (60 * 1000)
        val isNightTime = context.timeContext.hourOfDay < 7 || context.timeContext.hourOfDay > 22
        
        return if (inactiveMinutes > thresholdMinutes && (!dayTimeOnly || !isNightTime)) {
            InterventionRecommendation(
                type = InterventionType.ACTIVITY_SUGGESTION,
                trigger = "extended_inactivity",
                confidence = minOf(1.0f, inactiveMinutes.toFloat() / thresholdMinutes.toFloat()),
                reasoning = "You've been inactive for ${inactiveMinutes} minutes. Consider taking a walk or doing some light exercise!"
            )
        } else null
    }
    
    override fun updateThreshold(threshold: Any) {
        if (threshold is Number) {
            this.thresholdMinutes = threshold.toInt()
        }
    }
    
    override fun getDescription(): String {
        val timeScope = if (dayTimeOnly) "during daytime" else "anytime"
        return "Suggests physical activity after $thresholdMinutes minutes of inactivity $timeScope"
    }
    
    override fun getConfiguration(): Map<String, Any> {
        return super.getConfiguration() + mapOf(
            "threshold_minutes" to thresholdMinutes,
            "day_time_only" to dayTimeOnly
        )
    }
}

/**
 * Rule for protecting focus time from interruptions
 */
class FocusProtectionRule(
    private var maxInterruptionsPerHour: Int
) : AutomationRule("focus_protection", "Focus Time Protection") {
    
    private val recentInterruptions = mutableListOf<Long>()
    
    override fun evaluate(context: BehavioralContext): InterventionRecommendation? {
        val currentTime = System.currentTimeMillis()
        val oneHourAgo = currentTime - (60 * 60 * 1000)
        
        // Clean old interruptions
        recentInterruptions.removeAll { it < oneHourAgo }
        
        // Count app switches as interruptions
        val appSwitches = context.currentUsage.appSwitches
        
        return if (appSwitches > maxInterruptionsPerHour) {
            InterventionRecommendation(
                type = InterventionType.FOCUS_MODE_ENABLE,
                trigger = "focus_interruption_limit",
                confidence = minOf(1.0f, appSwitches.toFloat() / maxInterruptionsPerHour.toFloat()),
                reasoning = "You've switched between apps $appSwitches times in the last hour. Consider enabling focus mode to reduce distractions."
            )
        } else null
    }
    
    override fun updateThreshold(threshold: Any) {
        if (threshold is Number) {
            this.maxInterruptionsPerHour = threshold.toInt()
        }
    }
    
    override fun getDescription(): String {
        return "Suggests focus mode when app switches exceed $maxInterruptionsPerHour per hour"
    }
    
    override fun getConfiguration(): Map<String, Any> {
        return super.getConfiguration() + mapOf(
            "max_interruptions_per_hour" to maxInterruptionsPerHour
        )
    }
}

/**
 * Custom user-defined rule
 */
class CustomRule(
    id: String,
    name: String,
    private val condition: (BehavioralContext) -> Boolean,
    private val interventionType: InterventionType,
    private val message: String
) : AutomationRule(id, name) {
    
    override fun evaluate(context: BehavioralContext): InterventionRecommendation? {
        return if (condition(context)) {
            InterventionRecommendation(
                type = interventionType,
                trigger = "custom_rule_$id",
                confidence = 0.7f,
                reasoning = message
            )
        } else null
    }
    
    override fun updateThreshold(threshold: Any) {
        // Custom rules may have their own threshold logic
    }
    
    override fun getDescription(): String {
        return "Custom rule: $message"
    }
    
    override fun getConfiguration(): Map<String, Any> {
        return super.getConfiguration() + mapOf(
            "intervention_type" to interventionType.name,
            "message" to message
        )
    }
}

/**
 * App category mapping system with comprehensive categorization
 */
class AppCategoryMapping {
    
    private val categoryMap = mutableMapOf<String, AppCategory>()
    private val categoryUsageCache = mutableMapOf<String, Map<AppCategory, Long>>()
    
    fun initialize() {
        loadDefaultCategories()
        loadUserCustomCategories()
    }
    
    fun getCategory(packageName: String): AppCategory {
        return categoryMap[packageName] ?: categorizeByPackageName(packageName)
    }
    
    /**
     * Compute category usage from usage events with caching
     */
    suspend fun computeCategoryUsage(
        events: List<UsageEventEntity>,
        cacheKey: String? = null
    ): Map<AppCategory, Long> {
        // Check cache first
        cacheKey?.let { key ->
            categoryUsageCache[key]?.let { cached ->
                return cached
            }
        }
        
        val categoryUsage = mutableMapOf<AppCategory, Long>()
        
        for (event in events) {
            val category = getCategory(event.packageName)
            val currentUsage = categoryUsage[category] ?: 0L
            categoryUsage[category] = currentUsage + event.totalTimeInForeground
        }
        
        // Ensure all categories are represented
        AppCategory.values().forEach { category ->
            if (!categoryUsage.containsKey(category)) {
                categoryUsage[category] = 0L
            }
        }
        
        // Cache result if key provided
        cacheKey?.let { key ->
            categoryUsageCache[key] = categoryUsage
        }
        
        return categoryUsage
    }
    
    /**
     * Get category-based intervention triggers
     */
    fun getCategoryTriggers(
        categoryUsage: Map<AppCategory, Long>,
        thresholds: Map<AppCategory, Long>
    ): List<CategoryTrigger> {
        val triggers = mutableListOf<CategoryTrigger>()
        
        for ((category, usage) in categoryUsage) {
            val threshold = thresholds[category] ?: continue
            
            if (usage > threshold) {
                val severity = calculateSeverity(usage, threshold)
                triggers.add(
                    CategoryTrigger(
                        category = category,
                        usage = usage,
                        threshold = threshold,
                        severity = severity,
                        recommendedAction = getRecommendedAction(category, severity)
                    )
                )
            }
        }
        
        return triggers.sortedByDescending { it.severity }
    }
    
    /**
     * Add or update custom category mapping
     */
    fun setCustomCategory(packageName: String, category: AppCategory) {
        categoryMap[packageName] = category
        // Clear cache to force recalculation
        categoryUsageCache.clear()
    }
    
    /**
     * Get usage distribution as percentages
     */
    fun getCategoryDistribution(categoryUsage: Map<AppCategory, Long>): Map<AppCategory, Float> {
        val totalUsage = categoryUsage.values.sum()
        if (totalUsage == 0L) return AppCategory.values().associateWith { 0f }
        
        return categoryUsage.mapValues { (_, usage) ->
            usage.toFloat() / totalUsage.toFloat()
        }
    }
    
    /**
     * Get top apps by category
     */
    suspend fun getTopAppsByCategory(
        events: List<UsageEventEntity>,
        category: AppCategory,
        limit: Int = 5
    ): List<AppUsageInfo> {
        return events
            .filter { getCategory(it.packageName) == category }
            .groupBy { it.packageName }
            .mapValues { (_, events) -> events.sumOf { it.totalTimeInForeground } }
            .toList()
            .sortedByDescending { it.second }
            .take(limit)
            .map { (packageName, usage) ->
                AppUsageInfo(
                    packageName = packageName,
                    category = category,
                    totalUsage = usage,
                    appName = getAppName(packageName)
                )
            }
    }
    
    private fun loadDefaultCategories() {
        // Social media apps
        val socialApps = listOf(
            "com.facebook.katana", "com.instagram.android", "com.twitter.android",
            "com.snapchat.android", "com.zhiliaoapp.musically", "com.linkedin.android",
            "com.reddit.frontpage", "com.pinterest", "com.tumblr", "com.discord"
        )
        socialApps.forEach { categoryMap[it] = AppCategory.SOCIAL }
        
        // Productivity apps
        val productivityApps = listOf(
            "com.microsoft.office.outlook", "com.google.android.gm", "com.slack",
            "com.microsoft.teams", "us.zoom.videomeetings", "com.google.android.apps.docs",
            "com.microsoft.office.word", "com.microsoft.office.excel", "com.microsoft.office.powerpoint",
            "com.notion.id", "com.todoist", "com.any.do", "com.evernote"
        )
        productivityApps.forEach { categoryMap[it] = AppCategory.PRODUCTIVITY }
        
        // Entertainment apps
        val entertainmentApps = listOf(
            "com.google.android.youtube", "com.netflix.mediaclient", "com.spotify.music",
            "com.amazon.avod.thirdpartyclient", "com.hulu.plus", "com.disney.disneyplus",
            "com.twitch.android.app", "com.amazon.mp3", "com.pandora.android"
        )
        entertainmentApps.forEach { categoryMap[it] = AppCategory.ENTERTAINMENT }
        
        // Health and fitness apps
        val healthApps = listOf(
            "com.google.android.apps.fitness", "com.myfitnesspal.android", "com.strava",
            "com.nike.ntc", "com.headspace.android", "com.calm.android", "com.samsung.android.app.health"
        )
        healthApps.forEach { categoryMap[it] = AppCategory.HEALTH }
        
        // Communication apps
        val communicationApps = listOf(
            "com.whatsapp", "com.facebook.orca", "com.viber.voip", "com.skype.raider",
            "com.telegram.messenger", "com.google.android.apps.messaging", "com.android.mms"
        )
        communicationApps.forEach { categoryMap[it] = AppCategory.COMMUNICATION }
        
        // Shopping apps
        val shoppingApps = listOf(
            "com.amazon.mShop.android.shopping", "com.ebay.mobile", "com.etsy.android",
            "com.shopify.arrive", "com.target.ui", "com.walmart.android"
        )
        shoppingApps.forEach { categoryMap[it] = AppCategory.SHOPPING }
        
        // News apps
        val newsApps = listOf(
            "com.google.android.apps.magazines", "flipboard.app", "com.cnn.mobile.android.phone",
            "com.nytimes.android", "com.bbc.newsreader", "com.reuters.android"
        )
        newsApps.forEach { categoryMap[it] = AppCategory.NEWS }
    }
    
    private fun loadUserCustomCategories() {
        // TODO: Load user-defined category mappings from preferences/database
    }
    
    private fun categorizeByPackageName(packageName: String): AppCategory {
        return when {
            packageName.contains("social") || packageName.contains("chat") || 
            packageName.contains("facebook") || packageName.contains("instagram") -> AppCategory.SOCIAL
            
            packageName.contains("office") || packageName.contains("work") || 
            packageName.contains("productivity") || packageName.contains("docs") -> AppCategory.PRODUCTIVITY
            
            packageName.contains("game") || packageName.contains("video") || 
            packageName.contains("music") || packageName.contains("entertainment") -> AppCategory.ENTERTAINMENT
            
            packageName.contains("health") || packageName.contains("fitness") || 
            packageName.contains("medical") -> AppCategory.HEALTH
            
            packageName.contains("message") || packageName.contains("mail") || 
            packageName.contains("whatsapp") || packageName.contains("telegram") -> AppCategory.COMMUNICATION
            
            packageName.contains("shop") || packageName.contains("buy") || 
            packageName.contains("store") || packageName.contains("market") -> AppCategory.SHOPPING
            
            packageName.contains("news") || packageName.contains("magazine") || 
            packageName.contains("journal") -> AppCategory.NEWS
            
            else -> AppCategory.OTHER
        }
    }
    
    private fun calculateSeverity(usage: Long, threshold: Long): Float {
        return minOf(2.0f, usage.toFloat() / threshold.toFloat())
    }
    
    private fun getRecommendedAction(category: AppCategory, severity: Float): InterventionType {
        return when {
            severity > 1.5f -> when (category) {
                AppCategory.SOCIAL -> InterventionType.BREAK_SUGGESTION
                AppCategory.ENTERTAINMENT -> InterventionType.BREAK_SUGGESTION
                else -> InterventionType.APP_LIMIT_SUGGESTION
            }
            severity > 1.2f -> InterventionType.NOTIFICATION_REDUCTION
            else -> InterventionType.BREAK_SUGGESTION
        }
    }
    
    private fun getAppName(packageName: String): String {
        // TODO: Implement app name resolution from package manager
        return packageName.split(".").lastOrNull()?.replaceFirstChar { it.uppercase() } ?: packageName
    }
}

/**
 * Category-based trigger information
 */
data class CategoryTrigger(
    val category: AppCategory,
    val usage: Long,
    val threshold: Long,
    val severity: Float,
    val recommendedAction: InterventionType
)

/**
 * App usage information with category
 */
data class AppUsageInfo(
    val packageName: String,
    val category: AppCategory,
    val totalUsage: Long,
    val appName: String
)

enum class AppCategory {
    SOCIAL, PRODUCTIVITY, ENTERTAINMENT, HEALTH, COMMUNICATION, SHOPPING, NEWS, OTHER
}

/**
 * Status information for an automation rule
 */
data class RuleStatus(
    val id: String,
    val name: String,
    val isEnabled: Boolean,
    val description: String,
    val configuration: Map<String, Any> = emptyMap()
)

/**
 * Intervention frequency and timing limits
 */
data class InterventionLimits(
    val maxPerHour: Int,
    val maxPerDay: Int,
    val minIntervalMinutes: Int,
    val respectQuietHours: Boolean,
    val respectDnd: Boolean
)