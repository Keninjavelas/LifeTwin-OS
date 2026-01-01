package com.lifetwin.mlp.automation.test

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.lifetwin.mlp.automation.*
import com.lifetwin.mlp.db.DailySummaryEntity
import com.lifetwin.mlp.db.UsageEventEntity
import io.kotest.property.Arb
import io.kotest.property.arbitrary.*
import io.kotest.property.checkAll
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import kotlin.test.assertTrue

/**
 * Property-based tests for BehavioralContext data models
 * Feature: automation-layer, Property 2: Behavioral context data completeness
 */
@RunWith(AndroidJUnit4::class)
class BehavioralContextPropertyTest {
    
    private lateinit var context: Context
    
    @Before
    fun setup() {
        context = ApplicationProvider.getApplicationContext()
    }
    
    /**
     * Property 2: Behavioral context data completeness
     * For any valid usage data, BehavioralContext should contain complete and consistent information
     * **Validates: Requirements 3.1**
     */
    @Test
    fun testBehavioralContextDataCompleteness() {
        val usageSnapshotArb = Arb.bind(
            Arb.long(0..24 * 60 * 60 * 1000), // Total screen time
            Arb.long(0..12 * 60 * 60 * 1000), // Social usage
            Arb.long(0..8 * 60 * 60 * 1000),  // Work usage
            Arb.long(0..6 * 60 * 60 * 1000),  // Entertainment usage
            Arb.int(0..100),                   // Notification count
            Arb.int(0..50),                    // App switches
            Arb.int(1..20)                     // Unique apps
        ) { totalTime, socialTime, workTime, entertainmentTime, notifications, switches, uniqueApps ->
            UsageSnapshot(
                totalScreenTime = totalTime,
                socialUsage = socialTime,
                workUsage = workTime,
                entertainmentUsage = entertainmentTime,
                communicationUsage = 0L,
                healthUsage = 0L,
                notificationCount = notifications,
                appSwitches = switches,
                uniqueAppsUsed = uniqueApps,
                longestSessionDuration = totalTime / 4,
                averageSessionDuration = totalTime / switches.coerceAtLeast(1),
                pickupCount = switches
            )
        }
        
        checkAll(usageSnapshotArb, iterations = 100) { usageSnapshot ->
            val timeContext = TimeContext.fromTimestamp(System.currentTimeMillis())
            val environmentContext = EnvironmentContext.getCurrent(context)
            val userState = UserState.fromBehavioralData(usageSnapshot, UsagePatterns.fromSummary(null))
            
            val behavioralContext = BehavioralContext(
                currentUsage = usageSnapshot,
                recentPatterns = UsagePatterns.fromSummary(null),
                timeContext = timeContext,
                environmentContext = environmentContext,
                userState = userState
            )
            
            // Test data completeness
            assertTrue(
                behavioralContext.currentUsage.totalScreenTime >= 0,
                "Total screen time should be non-negative"
            )
            
            assertTrue(
                behavioralContext.timeContext.hourOfDay in 0..23,
                "Hour of day should be valid (0-23)"
            )
            
            assertTrue(
                behavioralContext.environmentContext.batteryLevel in 0f..1f,
                "Battery level should be between 0 and 1"
            )
            
            assertTrue(
                behavioralContext.userState.stressLevel in 0f..1f,
                "Stress level should be between 0 and 1"
            )
            
            // Test data consistency
            val usageDistribution = behavioralContext.currentUsage.getUsageDistribution()
            val totalPercentage = usageDistribution.socialPercentage + 
                                usageDistribution.workPercentage + 
                                usageDistribution.entertainmentPercentage +
                                usageDistribution.communicationPercentage +
                                usageDistribution.healthPercentage
            
            assertTrue(
                totalPercentage <= 1.1f, // Allow small floating point errors
                "Usage distribution percentages should not exceed 100%"
            )
        }
    }
    
    /**
     * Property 3: Usage snapshot consistency
     * For any usage events, the generated UsageSnapshot should maintain data consistency
     * **Validates: Requirements 3.1**
     */
    @Test
    fun testUsageSnapshotConsistency() {
        val usageEventsArb = Arb.list(
            Arb.bind(
                Arb.string(5..20), // Package name
                Arb.long(0..60 * 60 * 1000), // Time in foreground
                Arb.long(System.currentTimeMillis() - 24 * 60 * 60 * 1000, System.currentTimeMillis()) // Timestamp
            ) { packageName, timeInForeground, timestamp ->
                UsageEventEntity(
                    id = 0,
                    packageName = packageName,
                    timestamp = timestamp,
                    totalTimeInForeground = timeInForeground,
                    lastTimeUsed = timestamp,
                    eventType = "usage"
                )
            },
            range = 0..50
        )
        
        checkAll(usageEventsArb, iterations = 100) { usageEvents ->
            val usageSnapshot = UsageSnapshot.fromEvents(usageEvents)
            
            // Test basic consistency
            assertTrue(
                usageSnapshot.totalScreenTime >= 0,
                "Total screen time should be non-negative"
            )
            
            assertTrue(
                usageSnapshot.socialUsage <= usageSnapshot.totalScreenTime,
                "Social usage should not exceed total screen time"
            )
            
            assertTrue(
                usageSnapshot.workUsage <= usageSnapshot.totalScreenTime,
                "Work usage should not exceed total screen time"
            )
            
            assertTrue(
                usageSnapshot.uniqueAppsUsed <= usageEvents.map { it.packageName }.distinct().size,
                "Unique apps used should not exceed actual unique apps in events"
            )
            
            assertTrue(
                usageSnapshot.appSwitches >= 0,
                "App switches should be non-negative"
            )
            
            // Test logical consistency
            if (usageEvents.isNotEmpty()) {
                assertTrue(
                    usageSnapshot.longestSessionDuration <= usageSnapshot.totalScreenTime,
                    "Longest session should not exceed total screen time"
                )
                
                if (usageSnapshot.appSwitches > 0) {
                    assertTrue(
                        usageSnapshot.averageSessionDuration <= usageSnapshot.totalScreenTime,
                        "Average session duration should not exceed total screen time"
                    )
                }
            }
        }
    }
    
    /**
     * Property 4: Time context validity
     * For any timestamp, TimeContext should generate valid time information
     * **Validates: Requirements 3.1**
     */
    @Test
    fun testTimeContextValidity() {
        val timestampArb = Arb.long(
            System.currentTimeMillis() - 365 * 24 * 60 * 60 * 1000L, // One year ago
            System.currentTimeMillis() + 365 * 24 * 60 * 60 * 1000L  // One year from now
        )
        
        checkAll(timestampArb, iterations = 100) { timestamp ->
            val timeContext = TimeContext.fromTimestamp(timestamp)
            
            // Test valid ranges
            assertTrue(
                timeContext.hourOfDay in 0..23,
                "Hour of day should be valid (0-23)"
            )
            
            assertTrue(
                timeContext.dayOfWeek in 1..7,
                "Day of week should be valid (1-7)"
            )
            
            assertTrue(
                timeContext.dayOfMonth in 1..31,
                "Day of month should be valid (1-31)"
            )
            
            assertTrue(
                timeContext.month in 0..11,
                "Month should be valid (0-11)"
            )
            
            assertTrue(
                timeContext.year > 1970,
                "Year should be reasonable (after 1970)"
            )
            
            // Test logical consistency
            val isWeekendExpected = timeContext.dayOfWeek == 1 || timeContext.dayOfWeek == 7 // Sunday or Saturday
            assertTrue(
                timeContext.isWeekend == isWeekendExpected,
                "Weekend flag should match day of week"
            )
            
            val isWorkHourExpected = timeContext.hourOfDay in 9..17 && !timeContext.isWeekend
            assertTrue(
                timeContext.isWorkHour == isWorkHourExpected,
                "Work hour flag should match hour and weekend status"
            )
            
            val isNightHourExpected = timeContext.hourOfDay >= 23 || timeContext.hourOfDay <= 6
            assertTrue(
                timeContext.isNightHour == isNightHourExpected,
                "Night hour flag should match hour"
            )
        }
    }
    
    /**
     * Property 5: User state inference consistency
     * For any behavioral data, UserState inference should produce consistent results
     * **Validates: Requirements 3.1**
     */
    @Test
    fun testUserStateInferenceConsistency() {
        val behavioralDataArb = Arb.bind(
            Arb.long(0..24 * 60 * 60 * 1000), // Total screen time
            Arb.int(0..100),                   // App switches
            Arb.int(0..50),                    // Notification count
            Arb.long(0..4 * 60 * 60 * 1000)   // Social usage
        ) { totalTime, switches, notifications, socialUsage ->
            val usageSnapshot = UsageSnapshot(
                totalScreenTime = totalTime,
                socialUsage = socialUsage,
                workUsage = totalTime - socialUsage,
                entertainmentUsage = 0L,
                communicationUsage = 0L,
                healthUsage = 0L,
                notificationCount = notifications,
                appSwitches = switches,
                uniqueAppsUsed = switches.coerceAtMost(20),
                longestSessionDuration = totalTime / 4,
                averageSessionDuration = if (switches > 0) totalTime / switches else 0L,
                pickupCount = switches
            )
            
            Pair(usageSnapshot, UsagePatterns.fromSummary(null))
        }
        
        checkAll(behavioralDataArb, iterations = 100) { (usageSnapshot, patterns) ->
            val userState = UserState.fromBehavioralData(usageSnapshot, patterns)
            
            // Test valid ranges for all state values
            assertTrue(
                userState.stressLevel in 0f..1f,
                "Stress level should be between 0 and 1"
            )
            
            assertTrue(
                userState.focusLevel in 0f..1f,
                "Focus level should be between 0 and 1"
            )
            
            assertTrue(
                userState.energyLevel in 0f..1f,
                "Energy level should be between 0 and 1"
            )
            
            assertTrue(
                userState.socialEngagement in 0f..1f,
                "Social engagement should be between 0 and 1"
            )
            
            // Test logical consistency
            if (usageSnapshot.appSwitches > 30) {
                assertTrue(
                    userState.stressLevel > 0.5f,
                    "High app switches should indicate higher stress"
                )
            }
            
            if (usageSnapshot.workUsage > usageSnapshot.totalScreenTime * 0.7f) {
                assertTrue(
                    userState.focusLevel > 0.5f,
                    "High work usage should indicate higher focus"
                )
            }
            
            // Test emotional state consistency
            when (userState.emotionalState) {
                EmotionalState.STRESSED -> assertTrue(
                    userState.stressLevel > 0.6f,
                    "STRESSED emotional state should have high stress level"
                )
                EmotionalState.FOCUSED -> assertTrue(
                    userState.focusLevel > 0.6f,
                    "FOCUSED emotional state should have high focus level"
                )
                EmotionalState.ENERGETIC -> assertTrue(
                    userState.energyLevel > 0.6f,
                    "ENERGETIC emotional state should have high energy level"
                )
                else -> {
                    // NEUTRAL and other states are valid for any levels
                }
            }
        }
    }
    
    /**
     * Property 6: JSON serialization round-trip consistency
     * For any BehavioralContext, JSON serialization should preserve essential information
     * **Validates: Requirements 6.1, 6.2**
     */
    @Test
    fun testJsonSerializationConsistency() {
        val behavioralContextArb = Arb.bind(
            Arb.long(0..24 * 60 * 60 * 1000), // Screen time
            Arb.int(0..50),                    // App switches
            Arb.float(0f..1f),                 // Battery level
            Arb.float(0f..1f)                  // Stress level
        ) { screenTime, switches, batteryLevel, stressLevel ->
            BehavioralContext(
                currentUsage = UsageSnapshot(
                    totalScreenTime = screenTime,
                    socialUsage = screenTime / 4,
                    workUsage = screenTime / 2,
                    entertainmentUsage = screenTime / 4,
                    communicationUsage = 0L,
                    healthUsage = 0L,
                    notificationCount = 10,
                    appSwitches = switches,
                    uniqueAppsUsed = switches.coerceAtMost(20),
                    longestSessionDuration = screenTime / 4,
                    averageSessionDuration = if (switches > 0) screenTime / switches else 0L,
                    pickupCount = switches
                ),
                recentPatterns = UsagePatterns.fromSummary(null),
                timeContext = TimeContext.fromTimestamp(System.currentTimeMillis()),
                environmentContext = EnvironmentContext(
                    batteryLevel = batteryLevel,
                    isCharging = false,
                    wifiConnected = true,
                    mobileDataConnected = false,
                    locationContext = LocationContext.HOME,
                    deviceOrientation = DeviceOrientation.PORTRAIT,
                    ambientLightLevel = LightLevel.MEDIUM,
                    noiseLevel = NoiseLevel.MEDIUM,
                    isInMotion = false,
                    proximityToUser = ProximityLevel.NEAR
                ),
                userState = UserState(
                    currentMood = 0.5f,
                    energyLevel = 0.5f,
                    focusLevel = 0.5f,
                    stressLevel = stressLevel,
                    motivationLevel = 0.5f,
                    socialEngagement = 0.5f,
                    physicalActivity = 0.5f,
                    sleepQuality = 0.5f,
                    cognitiveLoad = stressLevel,
                    emotionalState = EmotionalState.NEUTRAL
                )
            )
        }
        
        checkAll(behavioralContextArb, iterations = 100) { originalContext ->
            val json = originalContext.toJson()
            
            // Test that JSON contains expected keys
            assertTrue(
                json.has("current_usage"),
                "JSON should contain current_usage"
            )
            
            assertTrue(
                json.has("time_context"),
                "JSON should contain time_context"
            )
            
            assertTrue(
                json.has("environment_context"),
                "JSON should contain environment_context"
            )
            
            assertTrue(
                json.has("user_state"),
                "JSON should contain user_state"
            )
            
            // Test that nested objects contain expected data
            val usageJson = json.getJSONObject("current_usage")
            assertTrue(
                usageJson.has("total_screen_time"),
                "Usage JSON should contain total_screen_time"
            )
            
            assertTrue(
                usageJson.getLong("total_screen_time") == originalContext.currentUsage.totalScreenTime,
                "JSON should preserve total screen time value"
            )
            
            val environmentJson = json.getJSONObject("environment_context")
            assertTrue(
                kotlin.math.abs(environmentJson.getDouble("battery_level") - originalContext.environmentContext.batteryLevel) < 0.01,
                "JSON should preserve battery level value"
            )
            
            val userStateJson = json.getJSONObject("user_state")
            assertTrue(
                kotlin.math.abs(userStateJson.getDouble("stress_level") - originalContext.userState.stressLevel) < 0.01,
                "JSON should preserve stress level value"
            )
        }
    }
}