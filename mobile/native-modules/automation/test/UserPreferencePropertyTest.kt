package com.lifetwin.mlp.automation.test

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.lifetwin.mlp.automation.*
import io.kotest.property.Arb
import io.kotest.property.arbitrary.*
import io.kotest.property.checkAll
import kotlinx.coroutines.runBlocking
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import kotlin.test.assertTrue
import kotlin.test.assertEquals
import kotlin.test.assertFalse

@RunWith(AndroidJUnit4::class)
class UserPreferencePropertyTest {

    private lateinit var context: Context
    private lateinit var ruleBasedSystem: RuleBasedSystem

    @Before
    fun setup() {
        context = ApplicationProvider.getApplicationContext()
        ruleBasedSystem = RuleBasedSystem(context)
        runBlocking {
            ruleBasedSystem.initialize()
        }
    }

    /**
     * Property 6: User preference respect
     * Validates that user preferences are properly respected in rule evaluation
     */
    @Test
    fun `property test - user preference respect`() = runBlocking {
        checkAll(
            iterations = 30,
            Arb.userPreferences(),
            Arb.behavioralContext()
        ) { preferences, context ->
            // Apply user preferences
            ruleBasedSystem.updateUserPreferences(preferences)
            
            // Evaluate rules
            val recommendations = ruleBasedSystem.evaluateRules(context)
            
            // Verify that disabled intervention types are not recommended
            recommendations.forEach { recommendation ->
                val typeEnabled = preferences["${recommendation.type.name.lowercase()}_enabled"] as? Boolean ?: true
                assertTrue(
                    typeEnabled,
                    "Recommendation type ${recommendation.type} should be enabled in preferences"
                )
            }
            
            // Verify confidence threshold is respected
            val minConfidence = preferences["min_confidence_threshold"] as? Float ?: 0.3f
            recommendations.forEach { recommendation ->
                assertTrue(
                    recommendation.confidence >= minConfidence,
                    "Recommendation confidence ${recommendation.confidence} should meet minimum threshold $minConfidence"
                )
            }
        }
    }

    /**
     * Property 7: Quiet hours enforcement
     * Validates that quiet hours are properly enforced
     */
    @Test
    fun `property test - quiet hours enforcement`() = runBlocking {
        checkAll(
            iterations = 24, // Test all hours
            Arb.int(0, 23), // current hour
            Arb.quietHoursConfig()
        ) { currentHour, quietConfig ->
            // Set up quiet hours
            ruleBasedSystem.setQuietHours(
                enabled = quietConfig.enabled,
                startHour = quietConfig.startHour,
                endHour = quietConfig.endHour
            )
            
            // Create context with specific hour
            val context = createBehavioralContextWithHour(currentHour)
            
            // Check if current hour should be quiet
            val shouldBeQuiet = if (quietConfig.enabled) {
                if (quietConfig.startHour <= quietConfig.endHour) {
                    // Same day quiet hours
                    currentHour in quietConfig.startHour..quietConfig.endHour
                } else {
                    // Overnight quiet hours
                    currentHour >= quietConfig.startHour || currentHour <= quietConfig.endHour
                }
            } else {
                false
            }
            
            // Verify quiet hours detection
            assertEquals(
                shouldBeQuiet,
                ruleBasedSystem.isQuietHours(),
                "Quiet hours detection should be correct for hour $currentHour with config $quietConfig"
            )
            
            // Evaluate rules and check suppression
            val recommendations = ruleBasedSystem.evaluateRules(context)
            
            if (shouldBeQuiet) {
                // During quiet hours, most interventions should be suppressed
                // (except maybe DND_ENABLE which might be allowed)
                val nonDndRecommendations = recommendations.filter { it.type != InterventionType.DND_ENABLE }
                assertTrue(
                    nonDndRecommendations.isEmpty(),
                    "Non-DND interventions should be suppressed during quiet hours at $currentHour"
                )
            }
        }
    }

    /**
     * Property 8: Do Not Disturb mode enforcement
     * Validates that DND mode properly suppresses interventions
     */
    @Test
    fun `property test - dnd mode enforcement`() = runBlocking {
        checkAll(
            iterations = 20,
            Arb.boolean(), // DND active
            Arb.behavioralContext()
        ) { dndActive, context ->
            // Set DND mode
            ruleBasedSystem.setDoNotDisturb(dndActive)
            
            // Evaluate rules
            val recommendations = ruleBasedSystem.evaluateRules(context)
            
            if (dndActive) {
                // During DND, only DND_ENABLE interventions should be allowed
                recommendations.forEach { recommendation ->
                    assertTrue(
                        recommendation.type == InterventionType.DND_ENABLE,
                        "Only DND_ENABLE interventions should be allowed during DND mode, got ${recommendation.type}"
                    )
                }
            }
        }
    }

    /**
     * Property 9: Intervention type enablement
     * Validates that individual intervention types can be enabled/disabled
     */
    @Test
    fun `property test - intervention type enablement`() = runBlocking {
        checkAll(
            iterations = 30,
            Arb.interventionTypePreferences(),
            Arb.behavioralContext()
        ) { typePreferences, context ->
            // Apply intervention type preferences
            ruleBasedSystem.updateUserPreferences(typePreferences)
            
            // Evaluate rules
            val recommendations = ruleBasedSystem.evaluateRules(context)
            
            // Verify that only enabled intervention types are recommended
            recommendations.forEach { recommendation ->
                val isEnabled = ruleBasedSystem.isInterventionTypeEnabled(recommendation.type)
                assertTrue(
                    isEnabled,
                    "Intervention type ${recommendation.type} should be enabled"
                )
            }
            
            // Test each intervention type individually
            InterventionType.values().forEach { type ->
                val isEnabled = ruleBasedSystem.isInterventionTypeEnabled(type)
                val expectedEnabled = typePreferences["${type.name.lowercase()}_enabled"] as? Boolean ?: true
                
                assertEquals(
                    expectedEnabled,
                    isEnabled,
                    "Intervention type $type enablement should match preference"
                )
            }
        }
    }

    /**
     * Property 10: Custom threshold application
     * Validates that custom thresholds are properly applied to rules
     */
    @Test
    fun `property test - custom threshold application`() = runBlocking {
        checkAll(
            iterations = 20,
            Arb.customThresholds(),
            Arb.socialUsageContext()
        ) { thresholds, context ->
            // Apply custom thresholds
            ruleBasedSystem.updateUserPreferences(thresholds)
            
            // Evaluate rules
            val recommendations = ruleBasedSystem.evaluateRules(context)
            
            // Check social usage rule specifically
            val socialRecommendation = recommendations.find { it.trigger == "social_usage_exceeded" }
            val socialThreshold = thresholds["social_usage_limit_threshold"] as? Int
            
            if (socialThreshold != null) {
                val socialUsageMinutes = context.currentUsage.socialUsage / (60 * 1000)
                
                if (socialUsageMinutes > socialThreshold) {
                    // Should have recommendation if usage exceeds custom threshold
                    assertTrue(
                        socialRecommendation != null,
                        "Social usage rule should trigger with custom threshold $socialThreshold when usage is $socialUsageMinutes minutes"
                    )
                } else {
                    // Should not have recommendation if usage is below custom threshold
                    assertTrue(
                        socialRecommendation == null,
                        "Social usage rule should not trigger with custom threshold $socialThreshold when usage is $socialUsageMinutes minutes"
                    )
                }
            }
        }
    }

    /**
     * Property 11: Intervention limits enforcement
     * Validates that intervention frequency limits are respected
     */
    @Test
    fun `property test - intervention limits enforcement`() = runBlocking {
        checkAll(
            iterations = 20,
            Arb.interventionLimitsConfig()
        ) { limitsConfig ->
            // Apply intervention limits
            val preferences = mapOf(
                "max_interventions_per_hour" to limitsConfig.maxPerHour,
                "max_interventions_per_day" to limitsConfig.maxPerDay,
                "min_intervention_interval" to limitsConfig.minIntervalMinutes
            )
            ruleBasedSystem.updateUserPreferences(preferences)
            
            // Get intervention limits
            val limits = ruleBasedSystem.getInterventionLimits()
            
            // Verify limits are applied correctly
            assertEquals(
                limitsConfig.maxPerHour,
                limits.maxPerHour,
                "Max interventions per hour should match preference"
            )
            assertEquals(
                limitsConfig.maxPerDay,
                limits.maxPerDay,
                "Max interventions per day should match preference"
            )
            assertEquals(
                limitsConfig.minIntervalMinutes,
                limits.minIntervalMinutes,
                "Min intervention interval should match preference"
            )
        }
    }

    /**
     * Property 12: Preference persistence
     * Validates that user preferences are properly stored and retrieved
     */
    @Test
    fun `property test - preference persistence`() = runBlocking {
        checkAll(
            iterations = 20,
            Arb.userPreferences()
        ) { preferences ->
            // Set preferences
            ruleBasedSystem.updateUserPreferences(preferences)
            
            // Retrieve preferences
            val retrievedPreferences = ruleBasedSystem.getUserPreferences()
            
            // Verify all preferences are stored
            preferences.forEach { (key, value) ->
                assertTrue(
                    retrievedPreferences.containsKey(key),
                    "Preference key '$key' should be stored"
                )
                assertEquals(
                    value,
                    retrievedPreferences[key],
                    "Preference value for '$key' should match"
                )
            }
        }
    }

    // Helper functions

    private fun createBehavioralContextWithHour(hour: Int): BehavioralContext {
        return BehavioralContext(
            currentUsage = UsageSnapshot(
                totalScreenTime = 60 * 60 * 1000L,
                socialUsage = 30 * 60 * 1000L,
                workUsage = 20 * 60 * 1000L,
                notificationCount = 10,
                appSwitches = 5
            ),
            recentPatterns = UsagePatterns(
                averageDailyUsage = 4 * 60 * 60 * 1000L,
                peakUsageHour = 14,
                weekendVsWeekday = 1.2f
            ),
            timeContext = TimeContext(
                hourOfDay = hour,
                dayOfWeek = 3,
                isWeekend = false,
                isWorkHour = hour in 9..17
            ),
            environmentContext = EnvironmentContext(
                batteryLevel = 0.7f,
                isCharging = false,
                wifiConnected = true
            ),
            userState = UserState(
                currentMood = 0.6f,
                energyLevel = 0.7f,
                focusLevel = 0.5f,
                stressLevel = 0.4f
            )
        )
    }
}

// Arbitrary generators for user preferences

fun Arb.Companion.userPreferences(): Arb<Map<String, Any>> = arbitrary { rs ->
    mapOf(
        "break_suggestion_enabled" to boolean().bind(rs),
        "dnd_enable_enabled" to boolean().bind(rs),
        "app_limit_suggestion_enabled" to boolean().bind(rs),
        "focus_mode_enable_enabled" to boolean().bind(rs),
        "notification_reduction_enabled" to boolean().bind(rs),
        "activity_suggestion_enabled" to boolean().bind(rs),
        "min_confidence_threshold" to float(0.1f, 0.9f).bind(rs),
        "quiet_hours_enabled" to boolean().bind(rs),
        "quiet_hours_start" to int(20, 23).bind(rs),
        "quiet_hours_end" to int(5, 9).bind(rs),
        "max_interventions_per_hour" to int(1, 10).bind(rs),
        "max_interventions_per_day" to int(5, 50).bind(rs),
        "min_intervention_interval" to int(5, 60).bind(rs)
    )
}

fun Arb.Companion.quietHoursConfig(): Arb<QuietHoursConfig> = arbitrary { rs ->
    val enabled = boolean().bind(rs)
    val startHour = int(20, 23).bind(rs)
    val endHour = int(5, 9).bind(rs)
    
    QuietHoursConfig(enabled, startHour, endHour)
}

fun Arb.Companion.interventionTypePreferences(): Arb<Map<String, Any>> = arbitrary { rs ->
    InterventionType.values().associate { type ->
        "${type.name.lowercase()}_enabled" to boolean().bind(rs)
    }
}

fun Arb.Companion.customThresholds(): Arb<Map<String, Any>> = arbitrary { rs ->
    mapOf(
        "social_usage_limit_threshold" to int(30, 180), // minutes
        "notification_overload_threshold" to int(5, 30), // count
        "low_work_productivity_threshold" to float(0.1f, 0.8f), // percentage
        "inactivity_detection_threshold" to int(15, 120) // minutes
    ).mapValues { (_, arb) -> arb.bind(rs) }
}

fun Arb.Companion.interventionLimitsConfig(): Arb<InterventionLimitsConfig> = arbitrary { rs ->
    InterventionLimitsConfig(
        maxPerHour = int(1, 10).bind(rs),
        maxPerDay = int(5, 50).bind(rs),
        minIntervalMinutes = int(5, 60).bind(rs)
    )
}

// Data classes for test configuration

data class QuietHoursConfig(
    val enabled: Boolean,
    val startHour: Int,
    val endHour: Int
)

data class InterventionLimitsConfig(
    val maxPerHour: Int,
    val maxPerDay: Int,
    val minIntervalMinutes: Int
)