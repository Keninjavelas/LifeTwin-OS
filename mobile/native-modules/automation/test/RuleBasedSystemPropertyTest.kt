package com.lifetwin.mlp.automation.test

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.lifetwin.mlp.automation.*
import com.lifetwin.mlp.db.*
import io.kotest.property.Arb
import io.kotest.property.arbitrary.*
import io.kotest.property.checkAll
import kotlinx.coroutines.runBlocking
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import kotlin.test.assertTrue
import kotlin.test.assertNotNull
import kotlin.test.assertEquals

@RunWith(AndroidJUnit4::class)
class RuleBasedSystemPropertyTest {

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
     * Property 4: Rule-based intervention consistency
     * Validates that rule evaluations are deterministic and consistent
     */
    @Test
    fun `property test - rule evaluation consistency`() = runBlocking {
        checkAll(
            iterations = 50,
            Arb.behavioralContext()
        ) { context ->
            // Evaluate rules multiple times with same context
            val evaluation1 = ruleBasedSystem.evaluateRules(context)
            val evaluation2 = ruleBasedSystem.evaluateRules(context)
            
            // Results should be identical for same input
            assertEquals(
                evaluation1.size, 
                evaluation2.size,
                "Rule evaluation should be deterministic - same context should produce same number of recommendations"
            )
            
            // Check that recommendations are identical
            evaluation1.zip(evaluation2).forEach { (rec1, rec2) ->
                assertEquals(rec1.type, rec2.type, "Intervention types should be identical")
                assertEquals(rec1.trigger, rec2.trigger, "Triggers should be identical")
                assertEquals(rec1.confidence, rec2.confidence, 0.001f, "Confidence should be identical")
                assertEquals(rec1.reasoning, rec2.reasoning, "Reasoning should be identical")
            }
        }
    }

    /**
     * Property 5: Rule threshold validation
     * Validates that rules respect their configured thresholds
     */
    @Test
    fun `property test - rule threshold validation`() = runBlocking {
        checkAll(
            iterations = 30,
            Arb.socialUsageContext(),
            Arb.int(30, 180) // threshold in minutes
        ) { context, thresholdMinutes ->
            // Update social usage rule threshold
            ruleBasedSystem.updateRuleThresholds(mapOf(
                "social_usage_limit" to thresholdMinutes
            ))
            
            val recommendations = ruleBasedSystem.evaluateRules(context)
            val socialRecommendation = recommendations.find { it.trigger == "social_usage_exceeded" }
            
            val socialUsageMinutes = context.currentUsage.socialUsage / (60 * 1000)
            
            if (socialUsageMinutes > thresholdMinutes) {
                assertNotNull(
                    socialRecommendation,
                    "Social usage rule should trigger when usage ($socialUsageMinutes min) exceeds threshold ($thresholdMinutes min)"
                )
                assertEquals(
                    InterventionType.BREAK_SUGGESTION,
                    socialRecommendation.type,
                    "Social usage rule should suggest breaks"
                )
            } else {
                assertTrue(
                    socialRecommendation == null || socialRecommendation.trigger != "social_usage_exceeded",
                    "Social usage rule should not trigger when usage ($socialUsageMinutes min) is below threshold ($thresholdMinutes min)"
                )
            }
        }
    }

    /**
     * Property 6: Late night usage detection
     * Validates that late night rules trigger correctly based on time
     */
    @Test
    fun `property test - late night usage detection`() = runBlocking {
        checkAll(
            iterations = 24, // Test all hours
            Arb.int(0, 23) // hour of day
        ) { hour ->
            val context = createBehavioralContextWithHour(hour)
            val recommendations = ruleBasedSystem.evaluateRules(context)
            val lateNightRecommendation = recommendations.find { it.trigger == "late_night_usage" }
            
            val isLateNight = hour >= 23 || hour <= 6
            
            if (isLateNight) {
                assertNotNull(
                    lateNightRecommendation,
                    "Late night rule should trigger at hour $hour"
                )
                assertEquals(
                    InterventionType.DND_ENABLE,
                    lateNightRecommendation.type,
                    "Late night rule should suggest DND"
                )
            } else {
                assertTrue(
                    lateNightRecommendation == null,
                    "Late night rule should not trigger at hour $hour"
                )
            }
        }
    }

    /**
     * Property 7: Notification overload detection
     * Validates that notification rules trigger based on count thresholds
     */
    @Test
    fun `property test - notification overload detection`() = runBlocking {
        checkAll(
            iterations = 30,
            Arb.int(0, 50), // notification count
            Arb.int(5, 25)  // threshold
        ) { notificationCount, threshold ->
            // Update notification overload threshold
            ruleBasedSystem.updateRuleThresholds(mapOf(
                "notification_overload" to threshold
            ))
            
            val context = createBehavioralContextWithNotifications(notificationCount)
            val recommendations = ruleBasedSystem.evaluateRules(context)
            val notificationRecommendation = recommendations.find { it.trigger == "notification_overload" }
            
            if (notificationCount > threshold) {
                assertNotNull(
                    notificationRecommendation,
                    "Notification overload rule should trigger when count ($notificationCount) exceeds threshold ($threshold)"
                )
                assertEquals(
                    InterventionType.NOTIFICATION_REDUCTION,
                    notificationRecommendation.type,
                    "Notification overload should suggest reduction"
                )
            } else {
                assertTrue(
                    notificationRecommendation == null,
                    "Notification overload rule should not trigger when count ($notificationCount) is below threshold ($threshold)"
                )
            }
        }
    }

    /**
     * Property 8: Work productivity monitoring
     * Validates that work productivity rules trigger based on work percentage
     */
    @Test
    fun `property test - work productivity monitoring`() = runBlocking {
        checkAll(
            iterations = 30,
            Arb.float(0.0f, 1.0f), // work percentage
            Arb.float(0.1f, 0.8f)  // minimum threshold
        ) { workPercentage, minThreshold ->
            // Update work productivity threshold
            ruleBasedSystem.updateRuleThresholds(mapOf(
                "low_work_productivity" to minThreshold
            ))
            
            val context = createBehavioralContextWithWorkUsage(workPercentage)
            val recommendations = ruleBasedSystem.evaluateRules(context)
            val productivityRecommendation = recommendations.find { it.trigger == "low_work_productivity" }
            
            if (workPercentage < minThreshold && context.timeContext.isWorkHour) {
                assertNotNull(
                    productivityRecommendation,
                    "Work productivity rule should trigger when percentage ($workPercentage) is below threshold ($minThreshold) during work hours"
                )
                assertEquals(
                    InterventionType.FOCUS_MODE_ENABLE,
                    productivityRecommendation.type,
                    "Low work productivity should suggest focus mode"
                )
            } else {
                assertTrue(
                    productivityRecommendation == null,
                    "Work productivity rule should not trigger when percentage ($workPercentage) meets threshold ($minThreshold) or outside work hours"
                )
            }
        }
    }

    /**
     * Property 9: Rule confidence bounds
     * Validates that all rule confidence values are within valid bounds
     */
    @Test
    fun `property test - rule confidence bounds`() = runBlocking {
        checkAll(
            iterations = 50,
            Arb.behavioralContext()
        ) { context ->
            val recommendations = ruleBasedSystem.evaluateRules(context)
            
            recommendations.forEach { recommendation ->
                assertTrue(
                    recommendation.confidence >= 0.0f,
                    "Rule confidence should be >= 0.0, got ${recommendation.confidence} for ${recommendation.trigger}"
                )
                assertTrue(
                    recommendation.confidence <= 1.0f,
                    "Rule confidence should be <= 1.0, got ${recommendation.confidence} for ${recommendation.trigger}"
                )
            }
        }
    }

    /**
     * Property 10: Rule priority ordering
     * Validates that recommendations are properly ordered by confidence and priority
     */
    @Test
    fun `property test - rule priority ordering`() = runBlocking {
        checkAll(
            iterations = 30,
            Arb.behavioralContext()
        ) { context ->
            val recommendations = ruleBasedSystem.evaluateRules(context)
            
            if (recommendations.size > 1) {
                // Check that recommendations are sorted by confidence (descending)
                for (i in 0 until recommendations.size - 1) {
                    val current = recommendations[i]
                    val next = recommendations[i + 1]
                    
                    assertTrue(
                        current.confidence >= next.confidence,
                        "Recommendations should be ordered by confidence: ${current.confidence} >= ${next.confidence}"
                    )
                }
            }
        }
    }

    // Helper functions for creating test contexts

    private fun createBehavioralContextWithHour(hour: Int): BehavioralContext {
        return BehavioralContext(
            currentUsage = UsageSnapshot(
                totalScreenTime = 60 * 60 * 1000L, // 1 hour
                socialUsage = 30 * 60 * 1000L,     // 30 minutes
                workUsage = 20 * 60 * 1000L,       // 20 minutes
                notificationCount = 10,
                appSwitches = 5
            ),
            recentPatterns = UsagePatterns(
                averageDailyUsage = 4 * 60 * 60 * 1000L, // 4 hours
                peakUsageHour = 14,
                weekendVsWeekday = 1.2f
            ),
            timeContext = TimeContext(
                hourOfDay = hour,
                dayOfWeek = 3, // Wednesday
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

    private fun createBehavioralContextWithNotifications(notificationCount: Int): BehavioralContext {
        return BehavioralContext(
            currentUsage = UsageSnapshot(
                totalScreenTime = 60 * 60 * 1000L,
                socialUsage = 30 * 60 * 1000L,
                workUsage = 20 * 60 * 1000L,
                notificationCount = notificationCount,
                appSwitches = 5
            ),
            recentPatterns = UsagePatterns(
                averageDailyUsage = 4 * 60 * 60 * 1000L,
                peakUsageHour = 14,
                weekendVsWeekday = 1.2f
            ),
            timeContext = TimeContext(
                hourOfDay = 14,
                dayOfWeek = 3,
                isWeekend = false,
                isWorkHour = true
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

    private fun createBehavioralContextWithWorkUsage(workPercentage: Float): BehavioralContext {
        val totalTime = 2 * 60 * 60 * 1000L // 2 hours
        val workTime = (totalTime * workPercentage).toLong()
        
        return BehavioralContext(
            currentUsage = UsageSnapshot(
                totalScreenTime = totalTime,
                socialUsage = (totalTime - workTime) / 2,
                workUsage = workTime,
                notificationCount = 10,
                appSwitches = 8
            ),
            recentPatterns = UsagePatterns(
                averageDailyUsage = 4 * 60 * 60 * 1000L,
                peakUsageHour = 14,
                weekendVsWeekday = 1.2f
            ),
            timeContext = TimeContext(
                hourOfDay = 14,
                dayOfWeek = 3,
                isWeekend = false,
                isWorkHour = true
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

// Arbitrary generators for property-based testing

fun Arb.Companion.behavioralContext(): Arb<BehavioralContext> = arbitrary { rs ->
    BehavioralContext(
        currentUsage = usageSnapshot().bind(rs),
        recentPatterns = usagePatterns().bind(rs),
        timeContext = timeContext().bind(rs),
        environmentContext = environmentContext().bind(rs),
        userState = userState().bind(rs)
    )
}

fun Arb.Companion.socialUsageContext(): Arb<BehavioralContext> = arbitrary { rs ->
    val socialUsage = long(0L, 4 * 60 * 60 * 1000L).bind(rs) // 0-4 hours
    val totalUsage = maxOf(socialUsage, long(socialUsage, 6 * 60 * 60 * 1000L).bind(rs))
    
    BehavioralContext(
        currentUsage = UsageSnapshot(
            totalScreenTime = totalUsage,
            socialUsage = socialUsage,
            workUsage = (totalUsage - socialUsage) / 2,
            notificationCount = int(0, 50).bind(rs),
            appSwitches = int(0, 20).bind(rs)
        ),
        recentPatterns = usagePatterns().bind(rs),
        timeContext = timeContext().bind(rs),
        environmentContext = environmentContext().bind(rs),
        userState = userState().bind(rs)
    )
}

fun Arb.Companion.usageSnapshot(): Arb<UsageSnapshot> = arbitrary { rs ->
    val totalTime = long(0L, 8 * 60 * 60 * 1000L).bind(rs) // 0-8 hours
    val socialTime = long(0L, totalTime).bind(rs)
    val workTime = long(0L, totalTime - socialTime).bind(rs)
    
    UsageSnapshot(
        totalScreenTime = totalTime,
        socialUsage = socialTime,
        workUsage = workTime,
        notificationCount = int(0, 100).bind(rs),
        appSwitches = int(0, 50).bind(rs)
    )
}

fun Arb.Companion.usagePatterns(): Arb<UsagePatterns> = arbitrary { rs ->
    UsagePatterns(
        averageDailyUsage = long(1 * 60 * 60 * 1000L, 12 * 60 * 60 * 1000L).bind(rs), // 1-12 hours
        peakUsageHour = int(0, 23).bind(rs),
        weekendVsWeekday = float(0.5f, 2.0f).bind(rs)
    )
}

fun Arb.Companion.timeContext(): Arb<TimeContext> = arbitrary { rs ->
    val hour = int(0, 23).bind(rs)
    val dayOfWeek = int(1, 7).bind(rs)
    val isWeekend = dayOfWeek == 1 || dayOfWeek == 7 // Sunday = 1, Saturday = 7
    val isWorkHour = hour in 9..17 && !isWeekend
    
    TimeContext(
        hourOfDay = hour,
        dayOfWeek = dayOfWeek,
        isWeekend = isWeekend,
        isWorkHour = isWorkHour
    )
}

fun Arb.Companion.environmentContext(): Arb<EnvironmentContext> = arbitrary { rs ->
    EnvironmentContext(
        batteryLevel = float(0.0f, 1.0f).bind(rs),
        isCharging = boolean().bind(rs),
        wifiConnected = boolean().bind(rs)
    )
}

fun Arb.Companion.userState(): Arb<UserState> = arbitrary { rs ->
    UserState(
        currentMood = float(0.0f, 1.0f).bind(rs),
        energyLevel = float(0.0f, 1.0f).bind(rs),
        focusLevel = float(0.0f, 1.0f).bind(rs),
        stressLevel = float(0.0f, 1.0f).bind(rs)
    )
}