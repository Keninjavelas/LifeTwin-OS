package com.lifetwin.mlp.automation.test

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.lifetwin.mlp.automation.*
import com.lifetwin.mlp.db.AppDatabase
import io.kotest.property.Arb
import io.kotest.property.arbitrary.*
import io.kotest.property.checkAll
import kotlinx.coroutines.runBlocking
import org.junit.After
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/**
 * Property-based tests for AutomationLog system
 * Feature: automation-layer, Property 3: Automation logging completeness
 */
@RunWith(AndroidJUnit4::class)
class AutomationLogPropertyTest {
    
    private lateinit var context: Context
    private lateinit var database: AppDatabase
    private lateinit var automationLog: AutomationLog
    
    @Before
    fun setup() {
        context = ApplicationProvider.getApplicationContext()
        database = AppDatabase.getInstance(context)
        automationLog = AutomationLog(context)
        runBlocking {
            automationLog.initialize()
        }
    }
    
    @After
    fun cleanup() {
        database.clearAllTables()
        AppDatabase.clearInstance()
    }
    
    /**
     * Property 3: Automation logging completeness
     * For any intervention that is executed, there should be a corresponding log entry with all required metadata
     * **Validates: Requirements 6.1**
     */
    @Test
    fun testAutomationLoggingCompleteness() = runBlocking {
        val interventionArb = Arb.bind(
            Arb.enum<InterventionType>(),
            Arb.string(5..50), // Trigger
            Arb.float(0f..1f), // Confidence
            Arb.string(10..200) // Reasoning
        ) { type, trigger, confidence, reasoning ->
            InterventionRecommendation(
                type = type,
                trigger = trigger,
                confidence = confidence,
                reasoning = reasoning
            )
        }
        
        val resultArb = Arb.bind(
            Arb.boolean(), // Executed
            Arb.enum<UserResponse>(), // User response
            Arb.long(0..5000) // Execution time
        ) { executed, userResponse, executionTime ->
            InterventionResult(
                interventionId = "", // Will be set from intervention
                executed = executed,
                userResponse = userResponse,
                actualImpact = null,
                executionTime = executionTime
            )
        }
        
        checkAll(interventionArb, resultArb, iterations = 100) { intervention, result ->
            val updatedResult = result.copy(interventionId = intervention.id)
            
            // Log the intervention
            automationLog.logIntervention(intervention, updatedResult)
            
            // Verify log entry exists
            val logEntry = database.automationLogDao().getLogByInterventionId(intervention.id)
            
            assertNotNull(logEntry, "Log entry should exist for intervention ${intervention.id}")
            
            // Verify all required fields are present and correct
            assertEquals(intervention.id, logEntry.interventionId, "Intervention ID should match")
            assertEquals(intervention.type.name, logEntry.interventionType, "Intervention type should match")
            assertEquals(intervention.trigger, logEntry.trigger, "Trigger should match")
            assertEquals(intervention.reasoning, logEntry.reasoning, "Reasoning should match")
            assertEquals(intervention.confidence, logEntry.confidence, 0.001f, "Confidence should match")
            assertEquals(updatedResult.executed, logEntry.executed, "Executed status should match")
            assertEquals(updatedResult.userResponse.name, logEntry.userResponse, "User response should match")
            assertEquals(updatedResult.executionTime, logEntry.executionTimeMs, "Execution time should match")
            
            // Verify timestamps are reasonable
            assertTrue(
                logEntry.timestamp > 0,
                "Timestamp should be positive"
            )
            
            assertTrue(
                logEntry.createdAt > 0,
                "Created at timestamp should be positive"
            )
            
            assertTrue(
                kotlin.math.abs(logEntry.timestamp - System.currentTimeMillis()) < 10000,
                "Timestamp should be recent (within 10 seconds)"
            )
        }
    }
    
    /**
     * Property 4: User feedback integration consistency
     * For any intervention with user feedback, the feedback should be properly stored and retrievable
     * **Validates: Requirements 6.2**
     */
    @Test
    fun testUserFeedbackIntegrationConsistency() = runBlocking {
        val feedbackArb = Arb.bind(
            Arb.int(1..5), // Rating
            Arb.boolean(), // Helpful
            Arb.enum<TimingFeedback>(), // Timing
            Arb.string(0..100).orNull() // Comments (optional)
        ) { rating, helpful, timing, comments ->
            UserFeedback(
                interventionId = "", // Will be set later
                rating = rating,
                helpful = helpful,
                timing = timing,
                comments = comments
            )
        }
        
        checkAll(feedbackArb, iterations = 100) { feedback ->
            // Create and log an intervention first
            val intervention = InterventionRecommendation(
                type = InterventionType.BREAK_SUGGESTION,
                trigger = "test_trigger",
                confidence = 0.8f,
                reasoning = "Test reasoning"
            )
            
            val result = InterventionResult(
                interventionId = intervention.id,
                executed = true,
                userResponse = UserResponse.PENDING,
                actualImpact = null,
                executionTime = 100L
            )
            
            automationLog.logIntervention(intervention, result)
            
            // Update feedback
            val updatedFeedback = feedback.copy(interventionId = intervention.id)
            automationLog.updateFeedback(intervention.id, updatedFeedback)
            
            // Verify feedback was stored
            val logEntry = database.automationLogDao().getLogByInterventionId(intervention.id)
            
            assertNotNull(logEntry, "Log entry should exist")
            assertEquals(updatedFeedback.rating, logEntry.feedbackRating, "Feedback rating should match")
            assertEquals(updatedFeedback.helpful, logEntry.helpful, "Helpful flag should match")
            assertEquals(updatedFeedback.comments, logEntry.feedbackComments, "Comments should match")
        }
    }
    
    /**
     * Property 5: Automation metrics calculation consistency
     * For any set of logged interventions, metrics should be calculated consistently
     * **Validates: Requirements 6.3**
     */
    @Test
    fun testAutomationMetricsCalculationConsistency() = runBlocking {
        val interventionListArb = Arb.list(
            Arb.bind(
                Arb.enum<InterventionType>(),
                Arb.enum<UserResponse>(),
                Arb.float(0f..1f), // Confidence
                Arb.int(1..5).orNull() // Rating (optional)
            ) { type, response, confidence, rating ->
                Triple(
                    InterventionRecommendation(
                        type = type,
                        trigger = "test_trigger",
                        confidence = confidence,
                        reasoning = "Test reasoning"
                    ),
                    InterventionResult(
                        interventionId = "", // Will be set
                        executed = true,
                        userResponse = response,
                        actualImpact = null,
                        executionTime = 100L
                    ),
                    rating
                )
            },
            range = 1..20
        )
        
        checkAll(interventionListArb, iterations = 50) { interventionData ->
            // Clear previous data
            database.clearAllTables()
            
            // Log all interventions
            for ((intervention, result, rating) in interventionData) {
                val updatedResult = result.copy(interventionId = intervention.id)
                automationLog.logIntervention(intervention, updatedResult)
                
                // Add feedback if rating is provided
                if (rating != null) {
                    val feedback = UserFeedback(
                        interventionId = intervention.id,
                        rating = rating,
                        helpful = rating >= 3,
                        timing = TimingFeedback.PERFECT
                    )
                    automationLog.updateFeedback(intervention.id, feedback)
                }
            }
            
            // Calculate metrics
            val metrics = automationLog.getEffectivenessMetrics()
            
            // Verify metrics consistency
            assertEquals(
                interventionData.size,
                metrics.totalInterventions,
                "Total interventions should match logged count"
            )
            
            val expectedAcceptedCount = interventionData.count { it.second.userResponse == UserResponse.ACCEPTED }
            val expectedAcceptanceRate = if (interventionData.isNotEmpty()) {
                expectedAcceptedCount.toFloat() / interventionData.size.toFloat()
            } else 0f
            
            assertEquals(
                expectedAcceptanceRate,
                metrics.acceptanceRate,
                0.001f,
                "Acceptance rate should be calculated correctly"
            )
            
            val ratingsWithFeedback = interventionData.mapNotNull { it.third }
            val expectedAverageRating = if (ratingsWithFeedback.isNotEmpty()) {
                ratingsWithFeedback.average().toFloat()
            } else 0f
            
            assertEquals(
                expectedAverageRating,
                metrics.averageRating,
                0.001f,
                "Average rating should be calculated correctly"
            )
            
            // Effectiveness score should be between 0 and 1
            assertTrue(
                metrics.effectivenessScore in 0f..1f,
                "Effectiveness score should be between 0 and 1"
            )
        }
    }
    
    /**
     * Property 6: Daily summary generation consistency
     * For any day with logged interventions, daily summary should contain accurate aggregated data
     * **Validates: Requirements 6.3**
     */
    @Test
    fun testDailySummaryGenerationConsistency() = runBlocking {
        val dailyInterventionsArb = Arb.list(
            Arb.bind(
                Arb.enum<InterventionType>(),
                Arb.enum<UserResponse>(),
                Arb.int(0..23), // Hour of day
                Arb.float(0f..1f) // Confidence
            ) { type, response, hour, confidence ->
                val baseTime = System.currentTimeMillis()
                val dayStart = baseTime - (baseTime % (24 * 60 * 60 * 1000))
                val timestamp = dayStart + (hour * 60 * 60 * 1000)
                
                Pair(
                    InterventionRecommendation(
                        type = type,
                        trigger = "test_trigger",
                        confidence = confidence,
                        reasoning = "Test reasoning",
                        suggestedTiming = timestamp
                    ),
                    InterventionResult(
                        interventionId = "", // Will be set
                        executed = true,
                        userResponse = response,
                        actualImpact = null,
                        executionTime = 100L
                    )
                )
            },
            range = 1..15
        )
        
        checkAll(dailyInterventionsArb, iterations = 50) { interventionData ->
            // Clear previous data
            database.clearAllTables()
            
            // Log all interventions with specific timestamps
            for ((intervention, result) in interventionData) {
                val updatedResult = result.copy(interventionId = intervention.id)
                
                // Manually insert with specific timestamp
                val logEntry = com.lifetwin.mlp.db.AutomationLogEntity(
                    id = java.util.UUID.randomUUID().toString(),
                    interventionId = intervention.id,
                    timestamp = intervention.suggestedTiming,
                    interventionType = intervention.type.name,
                    trigger = intervention.trigger,
                    reasoning = intervention.reasoning,
                    confidence = intervention.confidence,
                    executed = updatedResult.executed,
                    userResponse = updatedResult.userResponse.name,
                    executionTimeMs = updatedResult.executionTime,
                    feedbackRating = null,
                    feedbackComments = null,
                    helpful = null
                )
                
                database.automationLogDao().insert(logEntry)
            }
            
            // Generate daily summary
            val summary = automationLog.getDailySummary()
            
            // Verify summary accuracy
            assertEquals(
                interventionData.size,
                summary.totalInterventions,
                "Total interventions in summary should match logged count"
            )
            
            val expectedAccepted = interventionData.count { it.second.userResponse == UserResponse.ACCEPTED }
            assertEquals(
                expectedAccepted,
                summary.acceptedInterventions,
                "Accepted interventions should match"
            )
            
            val expectedDismissed = interventionData.count { it.second.userResponse == UserResponse.DISMISSED }
            assertEquals(
                expectedDismissed,
                summary.dismissedInterventions,
                "Dismissed interventions should match"
            )
            
            val expectedAverageConfidence = interventionData.map { it.first.confidence }.average().toFloat()
            assertEquals(
                expectedAverageConfidence,
                summary.averageConfidence,
                0.001f,
                "Average confidence should be calculated correctly"
            )
            
            // Verify interventions by hour
            val expectedByHour = interventionData.groupBy { 
                val hour = ((it.first.suggestedTiming % (24 * 60 * 60 * 1000)) / (60 * 60 * 1000)).toInt()
                hour
            }.mapValues { it.value.size }
            
            for ((hour, count) in expectedByHour) {
                assertEquals(
                    count,
                    summary.interventionsByHour[hour] ?: 0,
                    "Interventions by hour should match for hour $hour"
                )
            }
        }
    }
    
    /**
     * Property 7: Log cleanup consistency
     * For any retention period, old logs should be properly cleaned up while preserving recent data
     * **Validates: Requirements 9.6**
     */
    @Test
    fun testLogCleanupConsistency() = runBlocking {
        val retentionDaysArb = Arb.int(1..30)
        
        checkAll(retentionDaysArb, iterations = 50) { retentionDays ->
            // Clear previous data
            database.clearAllTables()
            
            val currentTime = System.currentTimeMillis()
            val cutoffTime = currentTime - (retentionDays * 24 * 60 * 60 * 1000L)
            
            // Create interventions both before and after cutoff
            val oldInterventions = (1..5).map { i ->
                val oldTimestamp = cutoffTime - (i * 24 * 60 * 60 * 1000L) // Days before cutoff
                createTestLogEntry("old_$i", oldTimestamp)
            }
            
            val recentInterventions = (1..5).map { i ->
                val recentTimestamp = cutoffTime + (i * 60 * 60 * 1000L) // Hours after cutoff
                createTestLogEntry("recent_$i", recentTimestamp)
            }
            
            // Insert all log entries
            val allEntries = oldInterventions + recentInterventions
            for (entry in allEntries) {
                database.automationLogDao().insert(entry)
            }
            
            // Verify all entries exist before cleanup
            val beforeCleanup = database.automationLogDao().getAllLogs()
            assertEquals(10, beforeCleanup.size, "Should have 10 entries before cleanup")
            
            // Perform cleanup
            automationLog.cleanupOldLogs(retentionDays)
            
            // Verify only recent entries remain
            val afterCleanup = database.automationLogDao().getAllLogs()
            assertEquals(5, afterCleanup.size, "Should have 5 entries after cleanup")
            
            // Verify all remaining entries are recent
            for (entry in afterCleanup) {
                assertTrue(
                    entry.timestamp >= cutoffTime,
                    "All remaining entries should be after cutoff time"
                )
                
                assertTrue(
                    entry.interventionId.startsWith("recent_"),
                    "All remaining entries should be recent interventions"
                )
            }
        }
    }
    
    // Helper method to create test log entries
    private fun createTestLogEntry(interventionId: String, timestamp: Long): com.lifetwin.mlp.db.AutomationLogEntity {
        return com.lifetwin.mlp.db.AutomationLogEntity(
            id = java.util.UUID.randomUUID().toString(),
            interventionId = interventionId,
            timestamp = timestamp,
            interventionType = InterventionType.BREAK_SUGGESTION.name,
            trigger = "test_trigger",
            reasoning = "Test reasoning",
            confidence = 0.8f,
            executed = true,
            userResponse = UserResponse.ACCEPTED.name,
            executionTimeMs = 100L,
            feedbackRating = null,
            feedbackComments = null,
            helpful = null
        )
    }
}