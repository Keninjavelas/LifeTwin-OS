package com.lifetwin.automation.test

import com.lifetwin.automation.RLEnvironment
import com.lifetwin.automation.BehavioralContext
import com.lifetwin.automation.ResourceMonitor
import com.lifetwin.automation.TimeContext
import com.lifetwin.automation.UsageSnapshot
import com.lifetwin.automation.EnvironmentContext
import com.lifetwin.automation.UserState
import io.kotest.core.spec.style.StringSpec
import io.kotest.matchers.shouldBe
import io.kotest.matchers.shouldNotBe
import io.kotest.matchers.ints.shouldBeGreaterThan
import io.kotest.matchers.doubles.shouldBeBetween
import io.kotest.matchers.floats.shouldBeBetween
import io.kotest.matchers.collections.shouldHaveSize
import io.kotest.property.Arb
import io.kotest.property.arbitrary.*
import io.kotest.property.checkAll
import kotlinx.coroutines.runBlocking
import org.mockito.kotlin.mock
import org.mockito.kotlin.whenever
import android.content.Context

/**
 * Property-based tests for RLEnvironment
 * 
 * Property 10: RL observation space completeness
 * Validates: Requirements 3.1 (comprehensive observation space), 4.2 (temporal/context features)
 */
class RLEnvironmentPropertyTest : StringSpec({

    val mockContext = mock<Context>()
    val mockResourceMonitor = mock<ResourceMonitor>()
    
    "Property 10.1: Observation space always contains exactly 52 features" {
        checkAll(
            Arb.int(0..23), // hour of day
            Arb.int(1..7), // day of week
            Arb.long(0..86400000), // usage time
            Arb.int(0..100) // battery level
        ) { hour, dayOfWeek, usageTime, batteryLevel ->
            val behavioralContext = createMockBehavioralContext(hour, dayOfWeek, usageTime, batteryLevel)
            val environment = RLEnvironment(mockContext, behavioralContext, mockResourceMonitor)
            
            val observation = environment.generateObservation()
            
            // Observation must have exactly 52 features
            observation.size shouldBe RLEnvironment.OBSERVATION_SPACE_SIZE
            observation.size shouldBe 52
        }
    }
    
    "Property 10.2: All observation values are normalized between 0 and 1" {
        checkAll(
            Arb.int(0..23), // hour of day
            Arb.long(0..28800000), // usage time (0-8 hours)
            Arb.float(0f..1f), // stress level
            Arb.int(0..100) // battery level
        ) { hour, usageTime, stressLevel, batteryLevel ->
            val behavioralContext = createMockBehavioralContext(hour, 1, usageTime, batteryLevel, stressLevel)
            val environment = RLEnvironment(mockContext, behavioralContext, mockResourceMonitor)
            
            val observation = environment.generateObservation()
            
            // All values should be normalized between 0 and 1
            observation.forEach { value ->
                value shouldBeBetween 0f..1f
            }
        }
    }
    
    "Property 10.3: Temporal features are correctly normalized" {
        checkAll(
            Arb.int(0..23), // hour of day
            Arb.int(1..7), // day of week
            Arb.int(1..31), // day of month
            Arb.int(1..52) // week of year
        ) { hour, dayOfWeek, dayOfMonth, weekOfYear ->
            val timeContext = TimeContext(
                hourOfDay = hour,
                dayOfWeek = dayOfWeek,
                dayOfMonth = dayOfMonth,
                weekOfYear = weekOfYear,
                isWeekend = dayOfWeek in 6..7,
                isWorkingHours = hour in 9..17,
                isLateNight = hour >= 22 || hour <= 6,
                isEarlyMorning = hour in 5..8,
                timeSinceLastBreak = 3600000L, // 1 hour
                timeSinceWakeup = hour * 3600000L,
                timeUntilBedtime = (24 - hour) * 3600000L,
                sessionDuration = 1800000L // 30 minutes
            )
            
            val behavioralContext = createBehavioralContextWithTime(timeContext)
            val environment = RLEnvironment(mockContext, behavioralContext, mockResourceMonitor)
            
            val observation = environment.generateObservation()
            
            // Check temporal feature normalization
            observation[0] shouldBeBetween 0f..(hour.toFloat() / 24f + 0.01f) // Hour of day
            observation[1] shouldBeBetween 0f..(dayOfWeek.toFloat() / 7f + 0.01f) // Day of week
            observation[2] shouldBeBetween 0f..(dayOfMonth.toFloat() / 31f + 0.01f) // Day of month
            observation[3] shouldBeBetween 0f..(weekOfYear.toFloat() / 52f + 0.01f) // Week of year
        }
    }
    
    "Property 10.4: Usage snapshot features reflect actual usage patterns" {
        checkAll(
            Arb.long(0..28800000), // total usage (0-8 hours)
            Arb.long(0..14400000), // social media (0-4 hours)
            Arb.int(0..100), // app switches
            Arb.int(0..50) // notifications
        ) { totalUsage, socialMedia, appSwitches, notifications ->
            val usageSnapshot = UsageSnapshot(
                totalUsageTime = totalUsage,
                socialMediaTime = socialMedia,
                productivityTime = totalUsage / 4,
                entertainmentTime = totalUsage / 4,
                communicationTime = totalUsage / 4,
                appSwitchCount = appSwitches,
                notificationCount = notifications,
                screenInteractions = appSwitches * 10,
                averageSessionLength = if (appSwitches > 0) totalUsage / appSwitches else 0L,
                longestSession = totalUsage,
                shortestSession = 60000L, // 1 minute
                uniqueAppsUsed = minOf(appSwitches, 20),
                backgroundAppTime = totalUsage / 10,
                foregroundTransitions = appSwitches,
                multitaskingScore = 0.5f
            )
            
            val behavioralContext = createBehavioralContextWithUsage(usageSnapshot)
            val environment = RLEnvironment(mockContext, behavioralContext, mockResourceMonitor)
            
            val observation = environment.generateObservation()
            
            // Usage features should be properly normalized
            val expectedTotalUsage = (totalUsage.toFloat() / (8 * 3600 * 1000)).coerceIn(0f, 1f)
            val expectedSocialMedia = (socialMedia.toFloat() / (8 * 3600 * 1000)).coerceIn(0f, 1f)
            
            observation[12] shouldBeBetween (expectedTotalUsage - 0.01f)..(expectedTotalUsage + 0.01f)
            observation[13] shouldBeBetween (expectedSocialMedia - 0.01f)..(expectedSocialMedia + 0.01f)
        }
    }
    
    "Property 10.5: Environment context features are properly encoded" {
        checkAll(
            Arb.int(0..100), // battery level
            Arb.boolean(), // charging status
            Arb.boolean(), // wifi connected
            Arb.int(0..255) // brightness level
        ) { batteryLevel, isCharging, wifiConnected, brightness ->
            val environmentContext = EnvironmentContext(
                batteryLevel = batteryLevel,
                isCharging = isCharging,
                wifiConnected = wifiConnected,
                bluetoothConnected = false,
                brightnessLevel = brightness,
                volumeLevel = 7,
                headphonesConnected = false,
                locationContext = 0.5f
            )
            
            val behavioralContext = createBehavioralContextWithEnvironment(environmentContext)
            val environment = RLEnvironment(mockContext, behavioralContext, mockResourceMonitor)
            
            val observation = environment.generateObservation()
            
            // Environment features should be properly encoded
            observation[27] shouldBeBetween (batteryLevel.toFloat() / 100f - 0.01f)..(batteryLevel.toFloat() / 100f + 0.01f)
            observation[28] shouldBe if (isCharging) 1f else 0f
            observation[29] shouldBe if (wifiConnected) 1f else 0f
            observation[31] shouldBeBetween (brightness.toFloat() / 255f - 0.01f)..(brightness.toFloat() / 255f + 0.01f)
        }
    }
    
    "Property 10.6: User state features capture wellbeing metrics" {
        checkAll(
            Arb.float(0f..1f), // stress level
            Arb.float(0f..1f), // focus level
            Arb.float(0f..1f), // energy level
            Arb.float(0f..1f) // mood score
        ) { stressLevel, focusLevel, energyLevel, moodScore ->
            val userState = UserState(
                stressLevel = stressLevel,
                focusLevel = focusLevel,
                energyLevel = energyLevel,
                moodScore = moodScore,
                productivityScore = 0.7f,
                wellbeingScore = (1f - stressLevel + focusLevel + energyLevel + moodScore) / 4f,
                engagementLevel = 0.6f
            )
            
            val behavioralContext = createBehavioralContextWithUserState(userState)
            val environment = RLEnvironment(mockContext, behavioralContext, mockResourceMonitor)
            
            val observation = environment.generateObservation()
            
            // User state features should match input values
            observation[35] shouldBeBetween (stressLevel - 0.01f)..(stressLevel + 0.01f)
            observation[36] shouldBeBetween (focusLevel - 0.01f)..(focusLevel + 0.01f)
            observation[37] shouldBeBetween (energyLevel - 0.01f)..(energyLevel + 0.01f)
            observation[38] shouldBeBetween (moodScore - 0.01f)..(moodScore + 0.01f)
        }
    }
    
    "Property 10.7: Observation space metadata is consistent" {
        val behavioralContext = createMockBehavioralContext(12, 3, 7200000L, 80)
        val environment = RLEnvironment(mockContext, behavioralContext, mockResourceMonitor)
        
        val observationSpace = environment.getObservationSpace()
        
        // Observation space should have correct metadata
        observationSpace.shape shouldHaveSize 1
        observationSpace.shape[0] shouldBe 52
        observationSpace.low shouldHaveSize 52
        observationSpace.high shouldHaveSize 52
        observationSpace.dtype shouldBe "float32"
        
        // All bounds should be 0 to 1
        observationSpace.low.forEach { it shouldBe 0f }
        observationSpace.high.forEach { it shouldBe 1f }
    }
    
    "Property 10.8: Action space is properly defined" {
        val behavioralContext = createMockBehavioralContext(12, 3, 7200000L, 80)
        val environment = RLEnvironment(mockContext, behavioralContext, mockResourceMonitor)
        
        val actionSpace = environment.getActionSpace()
        
        // Action space should have 8 discrete actions
        actionSpace.n shouldBe 8
        actionSpace.actions shouldHaveSize 8
        
        // Should contain expected action types
        actionSpace.actions.contains("no_intervention") shouldBe true
        actionSpace.actions.contains("social_media_break") shouldBe true
        actionSpace.actions.contains("bedtime_suggestion") shouldBe true
        actionSpace.actions.contains("general_break") shouldBe true
    }
    
    "Property 10.9: Environment step produces valid results" {
        checkAll(
            Arb.int(0..7) // valid action
        ) { action ->
            val behavioralContext = createMockBehavioralContext(12, 3, 7200000L, 80)
            val environment = RLEnvironment(mockContext, behavioralContext, mockResourceMonitor)
            
            environment.reset()
            val stepResult = environment.step(action)
            
            // Step result should be valid
            stepResult.observation shouldHaveSize 52
            stepResult.reward shouldBeBetween -1.0..1.0
            stepResult.done shouldNotBe null
            stepResult.info.isNotEmpty() shouldBe true
            
            // Info should contain expected keys
            stepResult.info.containsKey("episode_step") shouldBe true
            stepResult.info.containsKey("total_reward") shouldBe true
            stepResult.info.containsKey("action_taken") shouldBe true
        }
    }
    
    "Property 10.10: Environment statistics are tracked correctly" {
        checkAll(
            Arb.list(Arb.int(0..7), 1..10) // sequence of actions
        ) { actions ->
            val behavioralContext = createMockBehavioralContext(12, 3, 7200000L, 80)
            val environment = RLEnvironment(mockContext, behavioralContext, mockResourceMonitor)
            
            environment.reset()
            
            actions.forEach { action ->
                environment.step(action)
            }
            
            val stats = environment.getEnvironmentStats()
            
            // Statistics should be consistent
            stats.episodeStep shouldBe actions.size
            stats.observationSpaceSize shouldBe 52
            stats.actionSpaceSize shouldBe 8
            stats.lastActionTime shouldBeGreaterThan 0L
            
            if (actions.isNotEmpty()) {
                stats.averageReward shouldBeBetween -1.0..1.0
            }
        }
    }
    
    // Helper functions to create mock behavioral contexts
    private fun createMockBehavioralContext(
        hour: Int, 
        dayOfWeek: Int, 
        usageTime: Long, 
        batteryLevel: Int,
        stressLevel: Float = 0.5f
    ): BehavioralContext {
        return BehavioralContext(
            timeContext = TimeContext(
                hourOfDay = hour,
                dayOfWeek = dayOfWeek,
                dayOfMonth = 15,
                weekOfYear = 26,
                isWeekend = dayOfWeek in 6..7,
                isWorkingHours = hour in 9..17,
                isLateNight = hour >= 22 || hour <= 6,
                isEarlyMorning = hour in 5..8,
                timeSinceLastBreak = 3600000L,
                timeSinceWakeup = hour * 3600000L,
                timeUntilBedtime = (24 - hour) * 3600000L,
                sessionDuration = 1800000L
            ),
            usageSnapshot = UsageSnapshot(
                totalUsageTime = usageTime,
                socialMediaTime = usageTime / 3,
                productivityTime = usageTime / 3,
                entertainmentTime = usageTime / 3,
                communicationTime = usageTime / 6,
                appSwitchCount = 20,
                notificationCount = 15,
                screenInteractions = 200,
                averageSessionLength = usageTime / 10,
                longestSession = usageTime / 2,
                shortestSession = 60000L,
                uniqueAppsUsed = 12,
                backgroundAppTime = usageTime / 10,
                foregroundTransitions = 25,
                multitaskingScore = 0.6f
            ),
            environmentContext = EnvironmentContext(
                batteryLevel = batteryLevel,
                isCharging = false,
                wifiConnected = true,
                bluetoothConnected = false,
                brightnessLevel = 128,
                volumeLevel = 7,
                headphonesConnected = false,
                locationContext = 0.5f
            ),
            userState = UserState(
                stressLevel = stressLevel,
                focusLevel = 0.6f,
                energyLevel = 0.7f,
                moodScore = 0.8f,
                productivityScore = 0.7f,
                wellbeingScore = 0.75f,
                engagementLevel = 0.6f
            )
        )
    }
    
    private fun createBehavioralContextWithTime(timeContext: TimeContext): BehavioralContext {
        return createMockBehavioralContext(12, 3, 7200000L, 80).copy(timeContext = timeContext)
    }
    
    private fun createBehavioralContextWithUsage(usageSnapshot: UsageSnapshot): BehavioralContext {
        return createMockBehavioralContext(12, 3, 7200000L, 80).copy(usageSnapshot = usageSnapshot)
    }
    
    private fun createBehavioralContextWithEnvironment(environmentContext: EnvironmentContext): BehavioralContext {
        return createMockBehavioralContext(12, 3, 7200000L, 80).copy(environmentContext = environmentContext)
    }
    
    private fun createBehavioralContextWithUserState(userState: UserState): BehavioralContext {
        return createMockBehavioralContext(12, 3, 7200000L, 80).copy(userState = userState)
    }
})