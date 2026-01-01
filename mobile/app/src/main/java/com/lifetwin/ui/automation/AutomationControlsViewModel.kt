package com.lifetwin.ui.automation

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.lifetwin.automation.*
import kotlinx.coroutines.flow.*
import kotlinx.coroutines.launch
import javax.inject.Inject
import dagger.hilt.android.lifecycle.HiltViewModel

/**
 * AutomationControlsViewModel - ViewModel for automation control interfaces
 * 
 * Manages automation settings, profiles, and threshold configurations
 */
@HiltViewModel
class AutomationControlsViewModel @Inject constructor(
    private val automationEngine: AutomationEngine,
    private val ruleBasedSystem: RuleBasedSystem,
    private val privacyController: PrivacyController
) : ViewModel() {
    
    // Controls state
    private val _controlsState = MutableStateFlow(AutomationControlsState())
    val controlsState: StateFlow<AutomationControlsState> = _controlsState.asStateFlow()
    
    // Automation profiles
    private val _automationProfiles = MutableStateFlow(getDefaultProfiles())
    val automationProfiles: StateFlow<List<AutomationProfile>> = _automationProfiles.asStateFlow()
    
    // Threshold settings
    private val _thresholdSettings = MutableStateFlow(getDefaultThresholdSettings())
    val thresholdSettings: StateFlow<Map<String, Double>> = _thresholdSettings.asStateFlow()
    
    init {
        loadControlsState()
        observeAutomationState()
    }
    
    /**
     * Load current controls state
     */
    private fun loadControlsState() {
        viewModelScope.launch {
            val automationState = automationEngine.automationState.value
            val userPreferences = ruleBasedSystem.userPreferences.value
            
            val interventionControls = createInterventionControls(userPreferences)
            
            _controlsState.value = AutomationControlsState(
                masterEnabled = automationState.enabled,
                selectedProfile = userPreferences.selectedProfile,
                interventionControls = interventionControls
            )
        }
    }
    
    /**
     * Observe automation state changes
     */
    private fun observeAutomationState() {
        viewModelScope.launch {
            combine(
                automationEngine.automationState,
                ruleBasedSystem.userPreferences
            ) { automationState, userPreferences ->
                updateControlsState(automationState, userPreferences)
            }.collect()
        }
    }
    
    /**
     * Update controls state with latest data
     */
    private fun updateControlsState(
        automationState: AutomationState,
        userPreferences: UserPreferences
    ) {
        val interventionControls = createInterventionControls(userPreferences)
        
        _controlsState.value = _controlsState.value.copy(
            masterEnabled = automationState.enabled,
            selectedProfile = userPreferences.selectedProfile,
            interventionControls = interventionControls
        )
    }
    
    /**
     * Create intervention controls from user preferences
     */
    private fun createInterventionControls(preferences: UserPreferences): List<InterventionControl> {
        return listOf(
            InterventionControl(
                type = InterventionType.NOTIFICATION_LIMIT,
                name = "Notification Management",
                description = "Limit excessive notifications during focus periods and quiet hours",
                enabled = preferences.notificationLimitEnabled,
                hasThreshold = true,
                threshold = preferences.notificationThreshold.toDouble(),
                thresholdRange = 1.0..50.0,
                thresholdLabel = "Max notifications per hour",
                thresholdUnit = "notifications",
                examples = listOf(
                    "Pause non-essential notifications during work hours",
                    "Group similar notifications together",
                    "Suggest notification-free periods"
                )
            ),
            
            InterventionControl(
                type = InterventionType.BREAK_REMINDER,
                name = "Break Reminders",
                description = "Encourage healthy breaks during extended device usage",
                enabled = preferences.breakReminderEnabled,
                hasThreshold = true,
                threshold = preferences.breakReminderInterval.toDouble(),
                thresholdRange = 15.0..180.0,
                thresholdLabel = "Break reminder interval",
                thresholdUnit = "minutes",
                examples = listOf(
                    "Suggest 5-minute breaks every hour",
                    "Recommend eye rest exercises",
                    "Encourage physical movement"
                )
            ),
            
            InterventionControl(
                type = InterventionType.FOCUS_MODE,
                name = "Focus Protection",
                description = "Minimize distractions during important tasks and deep work",
                enabled = preferences.focusModeEnabled,
                hasThreshold = true,
                threshold = preferences.focusSessionDuration.toDouble(),
                thresholdRange = 15.0..240.0,
                thresholdLabel = "Focus session duration",
                thresholdUnit = "minutes",
                examples = listOf(
                    "Block distracting apps during work",
                    "Silence non-urgent notifications",
                    "Provide focus session summaries"
                )
            ),
            
            InterventionControl(
                type = InterventionType.BEDTIME_REMINDER,
                name = "Sleep Wellness",
                description = "Promote healthy sleep habits with bedtime reminders and wind-down suggestions",
                enabled = preferences.bedtimeReminderEnabled,
                hasThreshold = true,
                threshold = preferences.bedtimeHour.toDouble(),
                thresholdRange = 20.0..24.0,
                thresholdLabel = "Bedtime hour",
                thresholdUnit = "hour",
                examples = listOf(
                    "Remind to start winding down before bed",
                    "Suggest reducing screen brightness",
                    "Recommend sleep-friendly activities"
                )
            ),
            
            InterventionControl(
                type = InterventionType.APP_LIMIT,
                name = "App Usage Limits",
                description = "Set healthy boundaries for time spent in specific apps or categories",
                enabled = preferences.appLimitEnabled,
                hasThreshold = true,
                threshold = preferences.dailyAppLimitHours.toDouble(),
                thresholdRange = 0.5..8.0,
                thresholdLabel = "Daily app limit",
                thresholdUnit = "hours",
                examples = listOf(
                    "Warn when approaching daily social media limit",
                    "Suggest alternative activities",
                    "Track progress toward usage goals"
                )
            )
        )
    }
    
    /**
     * Toggle master automation control
     */
    fun toggleMasterControl() {
        viewModelScope.launch {
            val currentState = automationEngine.automationState.value
            if (currentState.enabled) {
                automationEngine.disableAutomation()
            } else {
                automationEngine.enableAutomation()
            }
        }
    }
    
    /**
     * Select automation profile
     */
    fun selectProfile(profileId: String) {
        viewModelScope.launch {
            val profile = _automationProfiles.value.find { it.id == profileId }
            if (profile != null) {
                applyProfile(profile)
                
                // Update user preferences
                val currentPreferences = ruleBasedSystem.userPreferences.value
                val updatedPreferences = currentPreferences.copy(
                    selectedProfile = profileId
                )
                ruleBasedSystem.updateUserPreferences(updatedPreferences)
            }
        }
    }
    
    /**
     * Apply automation profile settings
     */
    private suspend fun applyProfile(profile: AutomationProfile) {
        val settings = profile.settings
        val currentPreferences = ruleBasedSystem.userPreferences.value
        
        val updatedPreferences = when (profile.type) {
            "focus" -> currentPreferences.copy(
                notificationLimitEnabled = true,
                notificationThreshold = (settings["notification_threshold"] as? Double)?.toInt() ?: 5,
                focusModeEnabled = true,
                focusSessionDuration = (settings["focus_duration"] as? Double)?.toInt() ?: 90,
                breakReminderEnabled = true,
                breakReminderInterval = (settings["break_interval"] as? Double)?.toInt() ?: 60
            )
            
            "wellness" -> currentPreferences.copy(
                notificationLimitEnabled = true,
                notificationThreshold = (settings["notification_threshold"] as? Double)?.toInt() ?: 15,
                breakReminderEnabled = true,
                breakReminderInterval = (settings["break_interval"] as? Double)?.toInt() ?: 45,
                bedtimeReminderEnabled = true,
                bedtimeHour = (settings["bedtime_hour"] as? Double)?.toInt() ?: 22,
                appLimitEnabled = true,
                dailyAppLimitHours = (settings["app_limit_hours"] as? Double)?.toInt() ?: 3
            )
            
            "minimal" -> currentPreferences.copy(
                notificationLimitEnabled = true,
                notificationThreshold = (settings["notification_threshold"] as? Double)?.toInt() ?: 25,
                breakReminderEnabled = false,
                focusModeEnabled = false,
                bedtimeReminderEnabled = false,
                appLimitEnabled = false
            )
            
            else -> currentPreferences
        }
        
        ruleBasedSystem.updateUserPreferences(updatedPreferences)
    }
    
    /**
     * Toggle specific intervention type
     */
    fun toggleIntervention(type: InterventionType) {
        viewModelScope.launch {
            val currentPreferences = ruleBasedSystem.userPreferences.value
            
            val updatedPreferences = when (type) {
                InterventionType.NOTIFICATION_LIMIT -> currentPreferences.copy(
                    notificationLimitEnabled = !currentPreferences.notificationLimitEnabled
                )
                InterventionType.BREAK_REMINDER -> currentPreferences.copy(
                    breakReminderEnabled = !currentPreferences.breakReminderEnabled
                )
                InterventionType.FOCUS_MODE -> currentPreferences.copy(
                    focusModeEnabled = !currentPreferences.focusModeEnabled
                )
                InterventionType.BEDTIME_REMINDER -> currentPreferences.copy(
                    bedtimeReminderEnabled = !currentPreferences.bedtimeReminderEnabled
                )
                InterventionType.APP_LIMIT -> currentPreferences.copy(
                    appLimitEnabled = !currentPreferences.appLimitEnabled
                )
                else -> currentPreferences
            }
            
            ruleBasedSystem.updateUserPreferences(updatedPreferences)
        }
    }
    
    /**
     * Update threshold for specific intervention type
     */
    fun updateThreshold(type: InterventionType, threshold: Double) {
        viewModelScope.launch {
            val currentPreferences = ruleBasedSystem.userPreferences.value
            
            val updatedPreferences = when (type) {
                InterventionType.NOTIFICATION_LIMIT -> currentPreferences.copy(
                    notificationThreshold = threshold.toInt()
                )
                InterventionType.BREAK_REMINDER -> currentPreferences.copy(
                    breakReminderInterval = threshold.toInt()
                )
                InterventionType.FOCUS_MODE -> currentPreferences.copy(
                    focusSessionDuration = threshold.toInt()
                )
                InterventionType.BEDTIME_REMINDER -> currentPreferences.copy(
                    bedtimeHour = threshold.toInt()
                )
                InterventionType.APP_LIMIT -> currentPreferences.copy(
                    dailyAppLimitHours = threshold.toInt()
                )
                else -> currentPreferences
            }
            
            ruleBasedSystem.updateUserPreferences(updatedPreferences)
        }
    }
    
    /**
     * Update advanced setting
     */
    fun updateAdvancedSetting(key: String, value: Double) {
        viewModelScope.launch {
            val currentSettings = _thresholdSettings.value.toMutableMap()
            currentSettings[key] = value
            _thresholdSettings.value = currentSettings
            
            // Apply advanced setting to automation engine
            applyAdvancedSetting(key, value)
        }
    }
    
    /**
     * Apply advanced setting to automation system
     */
    private suspend fun applyAdvancedSetting(key: String, value: Double) {
        when (key) {
            "intervention_frequency" -> {
                // Update intervention frequency multiplier
                automationEngine.updateInterventionFrequency(value)
            }
            "sensitivity" -> {
                // Update detection sensitivity
                ruleBasedSystem.updateSensitivity(value)
            }
            "cooldown_period" -> {
                // Update cooldown period between interventions
                automationEngine.updateCooldownPeriod(value.toLong() * 60 * 1000) // Convert to ms
            }
            "learning_rate" -> {
                // Update RL learning rate (if RL is enabled)
                automationEngine.updateLearningRate(value)
            }
            "confidence_threshold" -> {
                // Update confidence threshold for interventions
                automationEngine.updateConfidenceThreshold(value)
            }
        }
    }
    
    /**
     * Reset all settings to defaults
     */
    fun resetToDefaults() {
        viewModelScope.launch {
            val defaultPreferences = UserPreferences()
            ruleBasedSystem.updateUserPreferences(defaultPreferences)
            
            _thresholdSettings.value = getDefaultThresholdSettings()
            
            // Reset to wellness profile
            selectProfile("wellness")
        }
    }
    
    /**
     * Export current settings
     */
    fun exportSettings() {
        viewModelScope.launch {
            // Implementation would export settings to file or share
            // This is a placeholder for the actual export functionality
        }
    }
    
    /**
     * Import settings from file
     */
    fun importSettings() {
        viewModelScope.launch {
            // Implementation would import settings from file
            // This is a placeholder for the actual import functionality
        }
    }
    
    /**
     * Get default automation profiles
     */
    private fun getDefaultProfiles(): List<AutomationProfile> {
        return listOf(
            AutomationProfile(
                id = "focus",
                name = "Focus Mode",
                description = "Maximize productivity with minimal distractions and regular breaks",
                type = "focus",
                settings = mapOf(
                    "notification_threshold" to 5.0,
                    "focus_duration" to 90.0,
                    "break_interval" to 60.0
                )
            ),
            
            AutomationProfile(
                id = "wellness",
                name = "Digital Wellness",
                description = "Balanced approach promoting healthy digital habits and mindful usage",
                type = "wellness",
                settings = mapOf(
                    "notification_threshold" to 15.0,
                    "break_interval" to 45.0,
                    "bedtime_hour" to 22.0,
                    "app_limit_hours" to 3.0
                )
            ),
            
            AutomationProfile(
                id = "minimal",
                name = "Minimal Intervention",
                description = "Light touch approach with only essential notifications and limits",
                type = "minimal",
                settings = mapOf(
                    "notification_threshold" to 25.0
                )
            ),
            
            AutomationProfile(
                id = "custom",
                name = "Custom Profile",
                description = "Fully customizable settings tailored to your specific needs",
                type = "custom",
                settings = emptyMap()
            )
        )
    }
    
    /**
     * Get default threshold settings
     */
    private fun getDefaultThresholdSettings(): Map<String, Double> {
        return mapOf(
            "intervention_frequency" to 1.0,
            "sensitivity" to 1.0,
            "cooldown_period" to 15.0,
            "learning_rate" to 0.1,
            "confidence_threshold" to 0.7
        )
    }
}