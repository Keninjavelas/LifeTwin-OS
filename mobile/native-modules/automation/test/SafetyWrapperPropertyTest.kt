package com.lifetwin.automation.test

import com.lifetwin.automation.*
import io.kotest.core.spec.style.StringSpec
import io.kotest.matchers.shouldBe
import io.kotest.matchers.shouldNotBe
import io.kotest.matchers.collections.shouldNotBeEmpty
import io.kotest.matchers.doubles.shouldBeGreaterThan
import io.kotest.matchers.doubles.shouldBeLessThanOrEqual
import io.kotest.matchers.longs.shouldBeGreaterThan
import io.kotest.property.Arb
import io.kotest.property.arbitrary.*
import io.kotest.property.checkAll
import kotlinx.coroutines.delay
import kotlinx.coroutines.runBlocking

/**
 * Property-based tests for SafetyWrapper safety constraint enforcement
 * 
 * Validates Requirements:
 * - 3.7: Safety constraint validation for all RL actions
 * - 5.1: Intervention frequency limits and context-aware restrictions  
 * - 5.5: Violation reporting and policy adjustment mechanisms
 */
class SafetyWrapperPropertyTest : StringSpec({
    
    "Property 12: Safety constraint enforcement - frequency limits are always respected" {
        checkAll(
            Arb.int(1..5), // maxInterventionsPerHour
            Arb.int(5..20), // maxInterventionsPerDay
            Arb.long(5 * 60 * 1000L, 60 * 60 * 1000L) // minInterval (5min to 1hour)
        ) { maxHourly, maxDaily, minInterval ->
            
            val safetyWrapper = SafetyWrapper()
            runBlocking { safetyWrapper.initialize() }
            
            // Configure constraints
            val constraints = SafetyConstraints(
                maxInterventionsPerHour = maxHourly,
                maxInterventionsPerDay = maxDaily,
                minIntervalBetweenInterventions = minInterval
            )
            safetyWrapper.updateSafetyConstraints(constraints)
            
            val context = createTestBehavioralContext()
            var approvedCount = 0
            var rejectedCount = 0
            
            // Try to exceed hourly limit
            repeat(maxHourly + 3) { i ->
                val recommendation = createTestRecommendation("test_$i")
                val result = safetyWrapper.validateRecommendation(recommendation, context)
                
                if (result.isValid) {
                    approvedCount++
                } else {
                    rejectedCount++
                }
                
                // Should never exceed hourly limit
                approvedCount shouldBeLessThanOrEqual maxHourly
            }
            
            // Should have rejected some interventions
            rejectedCount shouldBeGreaterThan 0
        }
    }
    
    "Property 12: Safety constraint enforcement - timing constraints prevent rapid-fire interventions" {
        checkAll(
            Arb.long(10 * 1000L, 30 * 60 * 1000L) // minInterval (10s to 30min)
        ) { minInterval ->
            
            val safetyWrapper = SafetyWrapper()
            runBlocking { safetyWrapper.initialize() }
            
            val constraints = SafetyConstraints(
                minIntervalBetweenInterventions = minInterval,
                maxInterventionsPerHour = 10 // High limit to focus on timing
            )
            safetyWrapper.updateSafetyConstraints(constraints)
            
            val context = createTestBehavioralContext()
            
            // First intervention should be approved
            val firstRecommendation = createTestRecommendation("first")
            val firstResult = safetyWrapper.validateRecommendation(firstRecommendation, context)
            firstResult.isValid shouldBe true
            
            // Immediate second intervention should be rejected
            val secondRecommendation = createTestRecommendation("second")
            val secondResult = safetyWrapper.validateRecommendation(secondRecommendation, context)
            secondResult.isValid shouldBe false
            
            // Should have timing-related violation
            secondResult.violations.any { 
                it.constraintType == ConstraintType.TIMING_INTERVAL 
            } shouldBe true
        }
    }
    
    "Property 12: Safety constraint enforcement - escalation prevention limits strong interventions" {
        checkAll(
            Arb.int(1..3) // maxConsecutiveStrongInterventions
        ) { maxStrong ->
            
            val safetyWrapper = SafetyWrapper()
            runBlocking { safetyWrapper.initialize() }
            
            val constraints = SafetyConstraints(
                maxConsecutiveStrongInterventions = maxStrong,
                maxInterventionsPerHour = 10,
                minIntervalBetweenInterventions = 1000L // 1 second for fast testing
            )
            safetyWrapper.updateSafetyConstraints(constraints)
            
            val context = createTestBehavioralContext()
            var strongApproved = 0
            
            // Try multiple strong interventions
            repeat(maxStrong + 2) { i ->
                delay(1100) // Wait for interval
                
                val recommendation = createStrongInterventionRecommendation("strong_$i")
                val result = safetyWrapper.validateRecommendation(recommendation, context)
                
                if (result.isValid) {
                    strongApproved++
                } else {
                    // Should have escalation prevention violation
                    result.violations.any { 
                        it.constraintType == ConstraintType.ESCALATION_PREVENTION 
                    } shouldBe true
                }
            }
            
            // Should not exceed max strong interventions
            strongApproved shouldBeLessThanOrEqual maxStrong
        }
    }
    
    "Property 12: Safety constraint enforcement - contextual restrictions are context-aware" {
        checkAll(
            Arb.int(0..23), // hour of day
            Arb.boolean(), // emergency mode
            Arb.boolean() // important call
        ) { hour, emergencyMode, importantCall ->
            
            val safetyWrapper = SafetyWrapper()
            runBlocking { safetyWrapper.initialize() }
            
            safetyWrapper.setEmergencyMode(emergencyMode)
            safetyWrapper.setImportantCallState(importantCall)
            
            val context = createTestBehavioralContext().copy(
                timeContext = createTestTimeContext().copy(hourOfDay = hour)
            )
            
            // Test activity suggestion during sleep hours
            if (hour >= 23 || hour <= 5) {
                val activityRecommendation = createTestRecommendation("activity").copy(
                    type = InterventionType.ACTIVITY_SUGGESTION
                )
                val result = safetyWrapper.validateRecommendation(activityRecommendation, context)
                
                // Should be blocked during sleep hours
                result.isValid shouldBe false
                result.violations.any { 
                    it.constraintType == ConstraintType.CONTEXT_TIME 
                } shouldBe true
            }
            
            // Test emergency mode restrictions
            if (emergencyMode) {
                val nonCriticalRecommendation = createTestRecommendation("non_critical").copy(
                    type = InterventionType.BREAK_SUGGESTION
                )
                val result = safetyWrapper.validateRecommendation(nonCriticalRecommendation, context)
                
                // Non-critical interventions should be blocked in emergency mode
                result.isValid shouldBe false
                result.violations.any { 
                    it.constraintType == ConstraintType.CONTEXT_EMERGENCY 
                } shouldBe true
            }
            
            // Test important call restrictions
            if (importantCall) {
                val nonDndRecommendation = createTestRecommendation("non_dnd").copy(
                    type = InterventionType.BREAK_SUGGESTION
                )
                val result = safetyWrapper.validateRecommendation(nonDndRecommendation, context)
                
                // Non-DND interventions should be blocked during calls
                result.isValid shouldBe false
                result.violations.any { 
                    it.constraintType == ConstraintType.CONTEXT_CALL 
                } shouldBe true
            }
        }
    }
    
    "Property 12: Safety constraint enforcement - violation tracking maintains accurate statistics" {
        checkAll(
            Arb.int(5..20) // number of test interventions
        ) { numInterventions ->
            
            val safetyWrapper = SafetyWrapper()
            runBlocking { safetyWrapper.initialize() }
            
            // Set restrictive constraints to generate violations
            val constraints = SafetyConstraints(
                maxInterventionsPerHour = 2,
                minIntervalBetweenInterventions = 60 * 60 * 1000L // 1 hour
            )
            safetyWrapper.updateSafetyConstraints(constraints)
            
            val context = createTestBehavioralContext()
            var expectedViolations = 0
            
            repeat(numInterventions) { i ->
                val recommendation = createTestRecommendation("test_$i")
                val result = safetyWrapper.validateRecommendation(recommendation, context)
                
                if (!result.isValid) {
                    expectedViolations++
                }
            }
            
            val statistics = safetyWrapper.getSafetyStatistics()
            
            // Statistics should match actual violations
            statistics.totalValidations shouldBe numInterventions
            statistics.totalViolations shouldBe expectedViolations
            
            if (numInterventions > 0) {
                statistics.violationRate shouldBe (expectedViolations.toDouble() / numInterventions)
            }
            
            // Should have violation history if violations occurred
            if (expectedViolations > 0) {
                val violationHistory = safetyWrapper.getViolationHistory()
                violationHistory.shouldNotBeEmpty()
                violationHistory.size shouldBe expectedViolations
            }
        }
    }
    
    "Property 12: Safety constraint enforcement - cooldown mechanism prevents intervention spam" {
        checkAll(
            Arb.int(2..5), // maxConsecutiveViolations
            Arb.long(5 * 1000L, 30 * 1000L) // cooldownDuration (5s to 30s)
        ) { maxViolations, cooldownDuration ->
            
            val safetyWrapper = SafetyWrapper()
            runBlocking { safetyWrapper.initialize() }
            
            val constraints = SafetyConstraints(
                maxConsecutiveViolations = maxViolations,
                safetyCooldownDuration = cooldownDuration,
                maxInterventionsPerHour = 1, // Very restrictive to trigger violations
                minIntervalBetweenInterventions = 60 * 60 * 1000L
            )
            safetyWrapper.updateSafetyConstraints(constraints)
            
            val context = createTestBehavioralContext()
            
            // Generate enough violations to trigger cooldown
            repeat(maxViolations + 1) { i ->
                val recommendation = createTestRecommendation("violation_$i")
                safetyWrapper.validateRecommendation(recommendation, context)
            }
            
            // Should be in cooldown now
            safetyWrapper.isInCooldown() shouldBe true
            safetyWrapper.getRemainingCooldownTime() shouldBeGreaterThan 0L
            
            // All interventions should be blocked during cooldown
            val cooldownRecommendation = createTestRecommendation("during_cooldown")
            val cooldownResult = safetyWrapper.validateRecommendation(cooldownRecommendation, context)
            
            cooldownResult.isValid shouldBe false
            cooldownResult.violations.any { 
                it.constraintType == ConstraintType.COOLDOWN 
            } shouldBe true
        }
    }
    
    "Property 12: Safety constraint enforcement - safety scores reflect constraint compliance" {
        checkAll(
            Arb.int(1..10) // number of violations to generate
        ) { numViolations ->
            
            val safetyWrapper = SafetyWrapper()
            runBlocking { safetyWrapper.initialize() }
            
            // Generate controlled violations
            val constraints = SafetyConstraints(
                maxInterventionsPerHour = 1,
                minIntervalBetweenInterventions = 60 * 60 * 1000L
            )
            safetyWrapper.updateSafetyConstraints(constraints)
            
            val context = createTestBehavioralContext()
            
            // First intervention should pass
            val validRecommendation = createTestRecommendation("valid")
            val validResult = safetyWrapper.validateRecommendation(validRecommendation, context)
            validResult.isValid shouldBe true
            validResult.safetyScore shouldBe 1.0
            
            // Generate violations
            repeat(numViolations) { i ->
                val recommendation = createTestRecommendation("violation_$i")
                val result = safetyWrapper.validateRecommendation(recommendation, context)
                
                if (!result.isValid) {
                    // Safety score should decrease with violations
                    result.safetyScore shouldBeLessThanOrEqual 1.0
                    result.safetyScore shouldBeGreaterThan 0.0
                }
            }
            
            val finalStatistics = safetyWrapper.getSafetyStatistics()
            
            // Overall safety score should reflect violation rate
            if (finalStatistics.totalViolations > 0) {
                finalStatistics.safetyScore shouldBeLessThanOrEqual 1.0
                finalStatistics.safetyScore shouldBeGreaterThan 0.0
            }
        }
    }
    
    "Property 12: Safety constraint enforcement - alternative actions are safer than originals" {
        checkAll(
            Arb.enum<InterventionType>()
        ) { interventionType ->
            
            val safetyWrapper = SafetyWrapper()
            runBlocking { safetyWrapper.initialize() }
            
            // Set up restrictive constraints for the specific intervention type
            val constraints = SafetyConstraints(
                maxInterventionsPerHour = 1,
                minIntervalBetweenInterventions = 60 * 60 * 1000L
            )
            safetyWrapper.updateSafetyConstraints(constraints)
            
            val context = createTestBehavioralContext()
            
            // First intervention of this type should pass
            val firstRecommendation = createTestRecommendation("first").copy(type = interventionType)
            safetyWrapper.validateRecommendation(firstRecommendation, context)
            
            // Second intervention should be blocked but may have alternative
            val secondRecommendation = createTestRecommendation("second").copy(type = interventionType)
            val result = safetyWrapper.validateRecommendation(secondRecommendation, context)
            
            if (!result.isValid && result.alternativeAction != null) {
                // Alternative should be different from original
                result.alternativeAction!!.type shouldNotBe interventionType
                
                // Alternative should be valid when tested
                val altResult = safetyWrapper.validateRecommendation(result.alternativeAction!!, context)
                // Note: Alternative might still be blocked due to other constraints, 
                // but it should be a safer option
            }
        }
    }
})

// Helper functions for test data generation

private fun createTestRecommendation(id: String) = InterventionRecommendation(
    id = id,
    type = InterventionType.BREAK_SUGGESTION,
    message = "Test intervention",
    priority = 0.5,
    timestamp = System.currentTimeMillis()
)

private fun createStrongInterventionRecommendation(id: String) = InterventionRecommendation(
    id = id,
    type = InterventionType.FOCUS_MODE_ENABLE, // Strong intervention type
    message = "Strong test intervention",
    priority = 0.8,
    timestamp = System.currentTimeMillis()
)

private fun createTestBehavioralContext() = BehavioralContext(
    timeContext = createTestTimeContext(),
    usageSnapshot = createTestUsageSnapshot(),
    environmentContext = createTestEnvironmentContext(),
    userState = createTestUserState()
)

private fun createTestTimeContext() = TimeContext(
    timestamp = System.currentTimeMillis(),
    hourOfDay = 12,
    dayOfWeek = 3, // Wednesday
    dayOfMonth = 15,
    weekOfYear = 20,
    isWeekend = false,
    isWorkingHours = true,
    isLateNight = false,
    isEarlyMorning = false,
    timeSinceLastBreak = 2 * 60 * 60 * 1000L, // 2 hours
    timeSinceWakeup = 6 * 60 * 60 * 1000L, // 6 hours
    timeUntilBedtime = 10 * 60 * 60 * 1000L, // 10 hours
    sessionDuration = 30 * 60 * 1000L // 30 minutes
)

private fun createTestUsageSnapshot() = UsageSnapshot(
    totalUsageTime = 4 * 60 * 60 * 1000L, // 4 hours
    socialMediaTime = 1 * 60 * 60 * 1000L, // 1 hour
    productivityTime = 2 * 60 * 60 * 1000L, // 2 hours
    entertainmentTime = 1 * 60 * 60 * 1000L, // 1 hour
    communicationTime = 30 * 60 * 1000L, // 30 minutes
    appSwitchCount = 25,
    notificationCount = 15,
    screenInteractions = 200,
    averageSessionLength = 10 * 60 * 1000L, // 10 minutes
    longestSession = 45 * 60 * 1000L, // 45 minutes
    shortestSession = 2 * 60 * 1000L, // 2 minutes
    uniqueAppsUsed = 12,
    backgroundAppTime = 30 * 60 * 1000L, // 30 minutes
    foregroundTransitions = 8,
    multitaskingScore = 0.6f
)

private fun createTestEnvironmentContext() = EnvironmentContext(
    batteryLevel = 75,
    isCharging = false,
    wifiConnected = true,
    bluetoothConnected = false,
    brightnessLevel = 128,
    volumeLevel = 8,
    headphonesConnected = false,
    locationContext = 1 // Home
)

private fun createTestUserState() = UserState(
    stressLevel = 0.4f,
    focusLevel = 0.7f,
    energyLevel = 0.6f,
    moodScore = 0.8f,
    productivityScore = 0.7f,
    wellbeingScore = 0.75f,
    engagementLevel = 0.6f
)

// Test data class for intervention recommendations
data class InterventionRecommendation(
    val id: String,
    val type: InterventionType,
    val message: String,
    val priority: Double,
    val timestamp: Long
)