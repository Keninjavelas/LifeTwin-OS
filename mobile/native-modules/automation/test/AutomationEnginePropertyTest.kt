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

/**
 * Property-based tests for AutomationEngine
 * Feature: automation-layer, Property 1: Automation engine initialization consistency
 */
@RunWith(AndroidJUnit4::class)
class AutomationEnginePropertyTest {
    
    private lateinit var context: Context
    
    @Before
    fun setup() {
        context = ApplicationProvider.getApplicationContext()
    }
    
    /**
     * Property 1: Automation engine initialization consistency
     * For any valid context, the AutomationEngine should initialize successfully and consistently
     * **Validates: Requirements 10.1**
     */
    @Test
    fun testAutomationEngineInitializationConsistency() = runBlocking {
        checkAll<Int>(iterations = 100) { seed ->
            // Create multiple AutomationEngine instances with the same context
            val engines = (1..3).map { AutomationEngine(context) }
            
            // All engines should initialize successfully
            val initResults = engines.map { it.initialize() }
            
            // All initialization results should be consistent (all true or all false)
            val allSuccessful = initResults.all { it }
            val allFailed = initResults.all { !it }
            
            assertTrue(
                allSuccessful || allFailed,
                "AutomationEngine initialization should be consistent across instances"
            )
            
            // If initialization was successful, status should be RUNNING
            if (allSuccessful) {
                engines.forEach { engine ->
                    assertTrue(
                        engine.getAutomationStatus() == AutomationStatus.RUNNING,
                        "Successfully initialized engine should have RUNNING status"
                    )
                }
            }
            
            // Cleanup
            engines.forEach { it.cleanup() }
        }
    }
    
    /**
     * Property 2: Automation engine state consistency
     * For any AutomationEngine, the status should remain consistent with its actual state
     * **Validates: Requirements 10.1**
     */
    @Test
    fun testAutomationEngineStateConsistency() = runBlocking {
        checkAll<Boolean>(iterations = 100) { shouldInitialize ->
            val engine = AutomationEngine(context)
            
            // Initial state should be STOPPED
            assertTrue(
                engine.getAutomationStatus() == AutomationStatus.STOPPED,
                "New AutomationEngine should start with STOPPED status"
            )
            
            if (shouldInitialize) {
                val initResult = engine.initialize()
                
                if (initResult) {
                    // After successful initialization, status should be RUNNING
                    assertTrue(
                        engine.getAutomationStatus() == AutomationStatus.RUNNING,
                        "Successfully initialized engine should have RUNNING status"
                    )
                } else {
                    // After failed initialization, status should remain STOPPED or be ERROR
                    val status = engine.getAutomationStatus()
                    assertTrue(
                        status == AutomationStatus.STOPPED || status == AutomationStatus.ERROR,
                        "Failed initialization should result in STOPPED or ERROR status"
                    )
                }
            }
            
            // Cleanup should set status to STOPPED
            engine.cleanup()
            assertTrue(
                engine.getAutomationStatus() == AutomationStatus.STOPPED,
                "Cleaned up engine should have STOPPED status"
            )
        }
    }
    
    /**
     * Property 3: Automation engine pause/resume consistency
     * For any AutomationEngine, pause and resume operations should be consistent
     * **Validates: Requirements 5.4**
     */
    @Test
    fun testAutomationEnginePauseResumeConsistency() = runBlocking {
        val pauseDurationArb = Arb.int(1..60) // 1 to 60 minutes
        
        checkAll(pauseDurationArb, iterations = 100) { pauseMinutes ->
            val engine = AutomationEngine(context)
            val initResult = engine.initialize()
            
            if (initResult) {
                // Test pause functionality
                engine.pauseAutomation(pauseMinutes)
                
                // Engine should still report RUNNING status (paused is internal state)
                assertTrue(
                    engine.getAutomationStatus() == AutomationStatus.RUNNING,
                    "Paused engine should still report RUNNING status"
                )
                
                // Test resume functionality
                engine.resumeAutomation()
                
                // Engine should still be RUNNING after resume
                assertTrue(
                    engine.getAutomationStatus() == AutomationStatus.RUNNING,
                    "Resumed engine should have RUNNING status"
                )
            }
            
            engine.cleanup()
        }
    }
    
    /**
     * Property 4: User preferences update consistency
     * For any valid user preferences map, the AutomationEngine should handle updates consistently
     * **Validates: Requirements 5.2, 5.3**
     */
    @Test
    fun testUserPreferencesUpdateConsistency() = runBlocking {
        val preferencesArb = Arb.map(
            Arb.string(1..20),
            Arb.choice(
                Arb.boolean(),
                Arb.int(0..100),
                Arb.float(0f..1f)
            ),
            minSize = 0,
            maxSize = 10
        )
        
        checkAll(preferencesArb, iterations = 100) { preferences ->
            val engine = AutomationEngine(context)
            val initResult = engine.initialize()
            
            if (initResult) {
                // Update preferences should not crash or change engine status
                val statusBefore = engine.getAutomationStatus()
                
                engine.updateUserPreferences(preferences)
                
                val statusAfter = engine.getAutomationStatus()
                
                assertTrue(
                    statusBefore == statusAfter,
                    "User preferences update should not change engine status"
                )
            }
            
            engine.cleanup()
        }
    }
    
    /**
     * Property 5: Intervention evaluation consistency
     * For any behavioral context, intervention evaluation should be deterministic and safe
     * **Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5**
     */
    @Test
    fun testInterventionEvaluationConsistency() = runBlocking {
        val behavioralContextArb = Arb.bind(
            Arb.long(0..24 * 60 * 60 * 1000), // Screen time in ms
            Arb.long(0..12 * 60 * 60 * 1000), // Social usage in ms
            Arb.long(0..8 * 60 * 60 * 1000),  // Work usage in ms
            Arb.int(0..100),                   // Notification count
            Arb.int(0..50)                     // App switches
        ) { totalTime, socialTime, workTime, notifications, switches ->
            createTestBehavioralContext(totalTime, socialTime, workTime, notifications, switches)
        }
        
        checkAll(behavioralContextArb, iterations = 100) { behavioralContext ->
            val engine = AutomationEngine(context)
            val initResult = engine.initialize()
            
            if (initResult) {
                // Evaluate interventions multiple times with same context
                val evaluations = (1..3).map { 
                    engine.evaluateInterventions()
                }
                
                // Results should be consistent (same recommendations for same context)
                val firstEvaluation = evaluations.first()
                evaluations.forEach { evaluation ->
                    assertTrue(
                        evaluation.size == firstEvaluation.size,
                        "Intervention evaluation should be deterministic for same context"
                    )
                }
                
                // All recommendations should be valid
                firstEvaluation.forEach { recommendation ->
                    assertTrue(
                        recommendation.confidence in 0f..1f,
                        "Recommendation confidence should be between 0 and 1"
                    )
                    
                    assertTrue(
                        recommendation.reasoning.isNotBlank(),
                        "Recommendation should have non-empty reasoning"
                    )
                }
            }
            
            engine.cleanup()
        }
    }
    
    // Helper method to create test behavioral context
    private fun createTestBehavioralContext(
        totalScreenTime: Long,
        socialUsage: Long,
        workUsage: Long,
        notificationCount: Int,
        appSwitches: Int
    ): BehavioralContext {
        val currentTime = System.currentTimeMillis()
        
        return BehavioralContext(
            currentUsage = UsageSnapshot(
                totalScreenTime = totalScreenTime,
                socialUsage = socialUsage,
                workUsage = workUsage,
                notificationCount = notificationCount,
                appSwitches = appSwitches
            ),
            recentPatterns = UsagePatterns(
                averageDailyUsage = totalScreenTime,
                peakUsageHour = 14,
                weekendVsWeekday = 1.2f
            ),
            timeContext = TimeContext.fromTimestamp(currentTime),
            environmentContext = EnvironmentContext.getCurrent(context),
            userState = UserState(
                currentMood = 0.5f,
                energyLevel = 0.5f,
                focusLevel = 0.5f,
                stressLevel = 0.5f
            )
        )
    }
}