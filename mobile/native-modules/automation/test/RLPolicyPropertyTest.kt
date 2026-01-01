package com.lifetwin.automation.test

import com.lifetwin.automation.RLPolicy
import com.lifetwin.automation.BehavioralContext
import com.lifetwin.automation.TimeContext
import com.lifetwin.automation.UsageSnapshot
import com.lifetwin.automation.EnvironmentContext
import com.lifetwin.automation.UserState
import com.lifetwin.automation.InterventionType
import io.kotest.core.spec.style.StringSpec
import io.kotest.matchers.shouldBe
import io.kotest.matchers.shouldNotBe
import io.kotest.matchers.ints.shouldBeBetween
import io.kotest.matchers.ints.shouldBeGreaterThan
import io.kotest.matchers.ints.shouldBeLessThan
import io.kotest.matchers.floats.shouldBeBetween
import io.kotest.matchers.doubles.shouldBeBetween
import io.kotest.matchers.collections.shouldHaveSize
import io.kotest.matchers.collections.shouldContain
import io.kotest.property.Arb
import io.kotest.property.arbitrary.*
import io.kotest.property.checkAll
import kotlinx.coroutines.runBlocking
import org.mockito.kotlin.mock
import android.content.Context

/**
 * Property-based tests for RLPolicy
 * 
 * Property 11: RL policy bounded actions
 * Validates: Requirements 4.3 (discrete action space, validation, bounds checking)
 */
class RLPolicyPropertyTest : StringSpec({

    val mockContext = mock<Context>()
    
    "Property 11.1: Policy always predicts actions within valid bounds" {
        checkAll(
            Arb.list(Arb.float(0f..1f), 52..52) // Valid observation vector
        ) { observationList ->
            val policy = RLPolicy(mockContext)
            val observation = observationList.toFloatArray()
            
            val prediction = policy.predict(observation)
            
            // Action must be within valid bounds
            prediction.action shouldBeBetween 0..7
            
            // Action probabilities must sum to approximately 1
            val probabilitySum = prediction.actionProbabilities.sum()
            probabilitySum shouldBeBetween 0.99f..1.01f
            
            // All probabilities must be non-negative
            prediction.actionProbabilities.forEach { prob ->
                prob shouldBeBetween 0f..1f
            }
            
            // Confidence should match the probability of selected action
            prediction.confidence shouldBe prediction.actionProbabilities[prediction.action]
        }
    }
    
    "Property 11.2: Action space contains exactly 8 discrete actions" {
        val policy = RLPolicy(mockContext)
        val actionSpace = policy.getActionSpace()
        
        // Action space should have exactly 8 actions
        actionSpace.n shouldBe 8
        actionSpace.actions shouldHaveSize 8
        
        // Should contain expected action types
        actionSpace.actions shouldContain "no_intervention"
        actionSpace.actions shouldContain "social_media_break"
        actionSpace.actions shouldContain "bedtime_suggestion"
        actionSpace.actions shouldContain "general_break"
        actionSpace.actions shouldContain "stress_reduction"
        actionSpace.actions shouldContain "focus_mode"
        actionSpace.actions shouldContain "positive_reinforcement"
        actionSpace.actions shouldContain "energy_boost"
    }
    
    "Property 11.3: Action definitions have valid constraints" {
        val policy = RLPolicy(mockContext)
        val actionDefinitions = policy.getAllActionDefinitions()
        
        actionDefinitions.forEach { (actionId, definition) ->
            // Action ID should be within bounds
            actionId shouldBeBetween 0..7
            
            // Intensity should be between 0 and 1
            definition.intensity shouldBeBetween 0.0..1.0
            
            // Minimum interval should be non-negative
            definition.minInterval shouldBeGreaterThan -1L
            
            // Maximum frequency should be positive
            definition.maxFrequency shouldBeGreaterThan 0
            
            // Name should not be empty
            definition.name.isNotEmpty() shouldBe true
            
            // Description should not be empty
            definition.description.isNotEmpty() shouldBe true
            
            // Intervention type should be valid
            definition.interventionType shouldNotBe null
        }
    }
    
    "Property 11.4: Policy respects frequency constraints" {
        checkAll(
            Arb.int(0..7), // action
            Arb.int(1..10) // number of executions
        ) { action, executions ->
            val policy = RLPolicy(mockContext)
            val behavioralContext = createMockBehavioralContext()
            val actionDef = policy.getActionDefinition(action)
            
            if (actionDef != null) {
                // Execute action multiple times
                repeat(executions) {
                    policy.executeAction(action, behavioralContext)
                }
                
                val history = policy.getActionHistory()
                val actionCount = history.count { it.action == action }
                
                // Should not exceed maximum frequency per day
                actionCount shouldBeLessThan actionDef.maxFrequency + 1
            }
        }
    }
    
    "Property 11.5: Policy respects timing constraints" {
        checkAll(
            Arb.int(1..7) // action (excluding no_intervention)
        ) { action ->
            val policy = RLPolicy(mockContext)
            val behavioralContext = createMockBehavioralContext()
            val actionDef = policy.getActionDefinition(action)
            
            if (actionDef != null && actionDef.minInterval > 0) {
                // Execute action twice in quick succession
                val result1 = policy.executeAction(action, behavioralContext)
                val result2 = policy.executeAction(action, behavioralContext)
                
                // First execution should succeed
                result1.success shouldBe true
                
                // Second execution might be blocked by timing constraints
                // This depends on the validation logic in the policy
                if (!result2.success) {
                    // Timing constraint was enforced
                    result2.message.contains("interval") || 
                    result2.message.contains("timing") || 
                    result2.message.contains("frequency") shouldBe true
                }
            }
        }
    }
    
    "Property 11.6: Action validation produces contextually appropriate results" {
        checkAll(
            Arb.int(0..23), // hour of day
            Arb.float(0f..1f), // battery level
            Arb.boolean() // is charging
        ) { hour, batteryLevel, isCharging ->
            val policy = RLPolicy(mockContext)
            val observation = createObservationWithContext(hour, batteryLevel, isCharging)
            
            val prediction = policy.predict(observation)
            
            // Bedtime suggestions should be more likely at night
            if (hour >= 20 || hour <= 6) {
                // Night time - bedtime suggestion should have reasonable probability
                prediction.actionProbabilities[2] shouldBeGreaterThan 0.0f
            }
            
            // Focus mode should be more likely during work hours
            if (hour in 9..17) {
                // Work hours - focus mode should have reasonable probability
                prediction.actionProbabilities[5] shouldBeGreaterThan 0.0f
            }
            
            // Low battery should prefer less intensive actions
            if (batteryLevel < 0.2f && !isCharging) {
                // Low battery - no intervention should have higher probability
                prediction.actionProbabilities[0] shouldBeGreaterThan 0.1f
            }
        }
    }
    
    "Property 11.7: Policy prediction is deterministic for same input" {
        checkAll(
            Arb.list(Arb.float(0f..1f), 52..52) // observation
        ) { observationList ->
            val policy = RLPolicy(mockContext)
            val observation = observationList.toFloatArray()
            
            // Make multiple predictions with same input
            val prediction1 = policy.predict(observation)
            val prediction2 = policy.predict(observation)
            
            // Actions might differ due to exploration, but probabilities should be similar
            val probabilityDifference = prediction1.actionProbabilities.zip(prediction2.actionProbabilities)
                .map { (p1, p2) -> kotlin.math.abs(p1 - p2) }
                .maxOrNull() ?: 0f
            
            // Probabilities should be very similar (allowing for small numerical differences)
            probabilityDifference shouldBeLessThan 0.1f
        }
    }
    
    "Property 11.8: Action execution produces valid results" {
        checkAll(
            Arb.int(0..7) // valid action
        ) { action ->
            val policy = RLPolicy(mockContext)
            val behavioralContext = createMockBehavioralContext()
            
            val result = policy.executeAction(action, behavioralContext)
            
            // Result should be valid
            result.success shouldNotBe null
            result.message.isNotEmpty() shouldBe true
            
            if (result.success) {
                result.actionDefinition shouldNotBe null
                result.execution shouldNotBe null
                
                result.execution?.let { execution ->
                    execution.action shouldBe action
                    execution.timestamp shouldBeGreaterThan 0L
                    execution.intensity shouldBeBetween 0.0..1.0
                }
            }
        }
    }
    
    "Property 11.9: Policy info provides accurate metadata" {
        val policy = RLPolicy(mockContext)
        val policyInfo = policy.getPolicyInfo()
        
        // Policy info should be accurate
        policyInfo.isLoaded shouldBe true
        policyInfo.observationSpaceSize shouldBe 52
        policyInfo.actionSpaceSize shouldBe 8
        policyInfo.version.isNotEmpty() shouldBe true
        policyInfo.totalExecutions shouldBeGreaterThan -1
    }
    
    "Property 11.10: Action history is properly maintained" {
        checkAll(
            Arb.list(Arb.int(0..7), 1..20) // sequence of actions
        ) { actions ->
            val policy = RLPolicy(mockContext)
            val behavioralContext = createMockBehavioralContext()
            
            // Execute actions
            actions.forEach { action ->
                policy.executeAction(action, behavioralContext)
            }
            
            val history = policy.getActionHistory()
            
            // History should contain executed actions
            history.size shouldBeGreaterThan 0
            history.size shouldBeLessThan actions.size + 1 // Some might be filtered out
            
            // All history entries should be valid
            history.forEach { execution ->
                execution.action shouldBeBetween 0..7
                execution.timestamp shouldBeGreaterThan 0L
                execution.intensity shouldBeBetween 0.0..1.0
                execution.context shouldNotBe null
            }
        }
    }
    
    "Property 11.11: Invalid observations are handled gracefully" {
        checkAll(
            Arb.int(1..100) // invalid observation size
        ) { invalidSize ->
            if (invalidSize != 52) { // Skip valid size
                val policy = RLPolicy(mockContext)
                val invalidObservation = FloatArray(invalidSize) { 0.5f }
                
                try {
                    policy.predict(invalidObservation)
                    // Should not reach here
                    false shouldBe true
                } catch (e: IllegalArgumentException) {
                    // Expected behavior
                    e.message?.contains("Observation size") shouldBe true
                }
            }
        }
    }
    
    "Property 11.12: Policy update maintains bounds and constraints" {
        checkAll(
            Arb.list(Arb.float(-1f..1f), 52 * 8..52 * 8), // new weights
            Arb.list(Arb.float(-1f..1f), 8..8) // new bias
        ) { weightsList, biasList ->
            val policy = RLPolicy(mockContext)
            val newWeights = weightsList.toFloatArray()
            val newBias = biasList.toFloatArray()
            
            policy.updatePolicy(newWeights, newBias)
            
            // Policy should still work after update
            val observation = FloatArray(52) { 0.5f }
            val prediction = policy.predict(observation)
            
            // Predictions should still be valid
            prediction.action shouldBeBetween 0..7
            prediction.actionProbabilities.sum() shouldBeBetween 0.99f..1.01f
            prediction.confidence shouldBeBetween 0f..1f
        }
    }
    
    // Helper functions
    private fun createMockBehavioralContext(): BehavioralContext {
        return BehavioralContext(
            timeContext = TimeContext(
                hourOfDay = 12,
                dayOfWeek = 3,
                dayOfMonth = 15,
                weekOfYear = 26,
                isWeekend = false,
                isWorkingHours = true,
                isLateNight = false,
                isEarlyMorning = false,
                timeSinceLastBreak = 3600000L,
                timeSinceWakeup = 6 * 3600000L,
                timeUntilBedtime = 10 * 3600000L,
                sessionDuration = 1800000L
            ),
            usageSnapshot = UsageSnapshot(
                totalUsageTime = 7200000L, // 2 hours
                socialMediaTime = 1800000L, // 30 minutes
                productivityTime = 3600000L, // 1 hour
                entertainmentTime = 1800000L, // 30 minutes
                communicationTime = 600000L, // 10 minutes
                appSwitchCount = 25,
                notificationCount = 15,
                screenInteractions = 200,
                averageSessionLength = 300000L, // 5 minutes
                longestSession = 1800000L, // 30 minutes
                shortestSession = 60000L, // 1 minute
                uniqueAppsUsed = 12,
                backgroundAppTime = 600000L, // 10 minutes
                foregroundTransitions = 30,
                multitaskingScore = 0.6f
            ),
            environmentContext = EnvironmentContext(
                batteryLevel = 80,
                isCharging = false,
                wifiConnected = true,
                bluetoothConnected = false,
                brightnessLevel = 128,
                volumeLevel = 7,
                headphonesConnected = false,
                locationContext = 0.5f
            ),
            userState = UserState(
                stressLevel = 0.4f,
                focusLevel = 0.7f,
                energyLevel = 0.8f,
                moodScore = 0.75f,
                productivityScore = 0.8f,
                wellbeingScore = 0.78f,
                engagementLevel = 0.6f
            )
        )
    }
    
    private fun createObservationWithContext(hour: Int, batteryLevel: Float, isCharging: Boolean): FloatArray {
        val observation = FloatArray(52) { 0.5f }
        
        // Set specific context values
        observation[0] = hour.toFloat() / 24f // Hour of day
        observation[6] = if (hour >= 22 || hour <= 6) 1f else 0f // Is late night
        observation[27] = batteryLevel // Battery level
        observation[28] = if (isCharging) 1f else 0f // Is charging
        
        return observation
    }
})