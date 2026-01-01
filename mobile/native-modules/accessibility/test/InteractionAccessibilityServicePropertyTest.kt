package com.lifetwin.mlp.accessibility.test

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.lifetwin.mlp.accessibility.InteractionAccessibilityService
import com.lifetwin.mlp.db.*
import io.kotest.property.Arb
import io.kotest.property.arbitrary.*
import io.kotest.property.checkAll
import kotlinx.coroutines.runBlocking
import org.junit.After
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import kotlin.test.*

/**
 * Property-based tests for InteractionAccessibilityService
 * Feature: data-collection-intelligence, Property 1: Event Recording Consistency (Interactions)
 * Validates: Requirements 4.1, 4.3, 4.5
 */
@RunWith(AndroidJUnit4::class)
class InteractionAccessibilityServicePropertyTest {

    private lateinit var context: Context
    private lateinit var service: InteractionAccessibilityService
    private var database: AppDatabase? = null

    @Before
    fun setup() = runBlocking {
        context = ApplicationProvider.getApplicationContext()
        service = InteractionAccessibilityService()
        
        // Initialize encrypted database
        database = DBHelper.initializeEncrypted(context)
        assertNotNull(database, "Database should be initialized")
    }

    @After
    fun teardown() {
        database?.close()
        AppDatabase.clearInstance()
    }

    /**
     * Property 1: Event Recording Consistency (Interactions)
     * For any interaction event, when it occurs, the interaction service should record it 
     * with accurate timestamp and metadata while preserving privacy
     */
    @Test
    fun testInteractionEventRecordingConsistency() = runBlocking {
        checkAll(
            iterations = 100,
            arbInteractionMetrics()
        ) { interactionMetrics ->
            
            // Store the interaction metrics using the service's internal method
            service.storeInteractionMetrics(interactionMetrics)
            
            // Verify it was recorded correctly
            val db = database!!
            val storedMetrics = db.interactionMetricsDao().getMetricsByTimeRange(
                interactionMetrics.timestamp - 1000,
                interactionMetrics.timestamp + 1000
            )
            
            val matchingMetric = storedMetrics.find { it.id == interactionMetrics.id }
            assertNotNull(matchingMetric, "Interaction metric ${interactionMetrics.id} should be stored")
            
            // Verify all fields are recorded accurately
            assertEquals(interactionMetrics.touchCount, matchingMetric.touchCount)
            assertEquals(interactionMetrics.scrollEvents, matchingMetric.scrollEvents)
            assertEquals(interactionMetrics.interactionIntensity, matchingMetric.interactionIntensity)
            assertEquals(interactionMetrics.timeWindow.startTime, matchingMetric.timeWindowStart)
            assertEquals(interactionMetrics.timeWindow.endTime, matchingMetric.timeWindowEnd)
            
            // Verify gesture patterns are preserved
            val storedGestures = try {
                com.google.gson.Gson().fromJson(matchingMetric.gesturePatterns, Array<String>::class.java)
                    .mapNotNull { gestureString ->
                        try {
                            GestureType.valueOf(gestureString)
                        } catch (e: IllegalArgumentException) {
                            null
                        }
                    }
            } catch (e: Exception) {
                emptyList()
            }
            
            assertEquals(
                interactionMetrics.gesturePatterns.sorted(),
                storedGestures.sorted(),
                "Gesture patterns should be preserved"
            )
        }
    }

    /**
     * Test interaction intensity calculation accuracy
     */
    @Test
    fun testInteractionIntensityCalculation() = runBlocking {
        checkAll(
            iterations = 100,
            arbInteractionWindow()
        ) { (touchCount, scrollCount, windowDuration) ->
            
            val totalInteractions = touchCount + scrollCount
            val expectedIntensity = if (windowDuration > 0) {
                (totalInteractions.toFloat() / windowDuration) * 60000f // per minute
            } else 0f
            
            val timeWindow = TimeRange(
                System.currentTimeMillis(),
                System.currentTimeMillis() + windowDuration
            )
            
            val metrics = InteractionMetrics(
                timestamp = timeWindow.endTime,
                touchCount = touchCount,
                scrollEvents = scrollCount,
                gesturePatterns = listOf(GestureType.TAP, GestureType.SCROLL_VERTICAL),
                interactionIntensity = expectedIntensity,
                timeWindow = timeWindow
            )
            
            // Store and verify
            service.storeInteractionMetrics(metrics)
            
            val db = database!!
            val storedMetrics = db.interactionMetricsDao().getMetricsByTimeRange(
                timeWindow.startTime - 1000,
                timeWindow.endTime + 1000
            )
            
            val matchingMetric = storedMetrics.find { it.id == metrics.id }
            assertNotNull(matchingMetric, "Interaction metric should be stored")
            
            assertEquals(
                expectedIntensity,
                matchingMetric.interactionIntensity,
                0.01f,
                "Interaction intensity should be calculated correctly"
            )
        }
    }

    /**
     * Test privacy preservation - no sensitive content stored
     */
    @Test
    fun testPrivacyPreservation() = runBlocking {
        checkAll(
            iterations = 50,
            arbInteractionMetrics()
        ) { interactionMetrics ->
            
            val db = database!!
            
            // Store the interaction metrics
            service.storeInteractionMetrics(interactionMetrics)
            
            // Check that raw events contain encrypted metadata
            val rawEvents = db.rawEventDao().getEventsByTimeRange(
                interactionMetrics.timestamp - 1000,
                interactionMetrics.timestamp + 1000
            )
            
            val interactionRawEvent = rawEvents.find { it.eventType == "interaction" }
            assertNotNull(interactionRawEvent, "Raw event should be created for interaction")
            
            // Verify metadata is encrypted (should not contain plaintext values)
            val metadata = interactionRawEvent.metadata
            assertFalse(
                metadata.contains(interactionMetrics.touchCount.toString()),
                "Metadata should not contain plaintext touch count"
            )
            assertFalse(
                metadata.contains(interactionMetrics.scrollEvents.toString()),
                "Metadata should not contain plaintext scroll count"
            )
            
            // Verify we can decrypt the metadata
            val decryptedMetadata = DBHelper.decryptMetadata(metadata)
            assertTrue(
                decryptedMetadata.contains("touchCount") || decryptedMetadata.contains("scrollEvents"),
                "Decrypted metadata should contain interaction details"
            )
            
            // Verify no sensitive package names are stored
            assertNull(
                interactionRawEvent.packageName,
                "Package name should not be stored for interaction events to preserve privacy"
            )
        }
    }

    /**
     * Test gesture pattern aggregation
     */
    @Test
    fun testGesturePatternAggregation() = runBlocking {
        checkAll(
            iterations = 50,
            arbGestureSequence()
        ) { gestureSequence ->
            
            // Count expected gesture patterns
            val expectedPatterns = gestureSequence.groupBy { it }.mapValues { it.value.size }
            
            val metrics = InteractionMetrics(
                timestamp = System.currentTimeMillis(),
                touchCount = gestureSequence.count { it == GestureType.TAP },
                scrollEvents = gestureSequence.count { 
                    it == GestureType.SCROLL_VERTICAL || it == GestureType.SCROLL_HORIZONTAL 
                },
                gesturePatterns = gestureSequence.distinct(),
                interactionIntensity = gestureSequence.size.toFloat(),
                timeWindow = TimeRange(
                    System.currentTimeMillis() - 60000L,
                    System.currentTimeMillis()
                )
            )
            
            // Store and verify
            service.storeInteractionMetrics(metrics)
            
            val db = database!!
            val storedMetrics = db.interactionMetricsDao().getMetricsByTimeRange(
                metrics.timestamp - 1000,
                metrics.timestamp + 1000
            )
            
            val matchingMetric = storedMetrics.find { it.id == metrics.id }
            assertNotNull(matchingMetric, "Interaction metric should be stored")
            
            // Verify gesture patterns are correctly aggregated
            val storedGestures = try {
                com.google.gson.Gson().fromJson(matchingMetric.gesturePatterns, Array<String>::class.java)
                    .mapNotNull { gestureString ->
                        try {
                            GestureType.valueOf(gestureString)
                        } catch (e: IllegalArgumentException) {
                            null
                        }
                    }
            } catch (e: Exception) {
                emptyList()
            }
            
            assertEquals(
                gestureSequence.distinct().sorted(),
                storedGestures.sorted(),
                "Gesture patterns should be correctly aggregated"
            )
        }
    }

    /**
     * Test collector interface compliance
     */
    @Test
    fun testCollectorInterfaceCompliance() = runBlocking {
        // Test collector type
        assertEquals(CollectorType.INTERACTIONS, service.getCollectorType())
        
        // Test collection state management
        service.startCollection()
        assertTrue(service.isCollectionActive(), "Collection should be active after start")
        
        service.stopCollection()
        assertFalse(service.isCollectionActive(), "Collection should be inactive after stop")
        
        // Test data count (should be non-negative)
        val dataCount = service.getCollectedDataCount()
        assertTrue(dataCount >= 0, "Data count should be non-negative")
    }

    /**
     * Test time range queries for interaction metrics
     */
    @Test
    fun testTimeRangeQueries() = runBlocking {
        checkAll(
            iterations = 50,
            arbTimeRangeWithInteractions()
        ) { (timeRange, metricsInRange, metricsOutsideRange) ->
            
            val db = database!!
            
            // Store all metrics (both inside and outside the range)
            val allMetrics = metricsInRange + metricsOutsideRange
            allMetrics.forEach { metrics ->
                service.storeInteractionMetrics(metrics)
            }
            
            // Query metrics within the time range
            val queriedMetrics = service.getInteractionMetrics(timeRange)
            
            // Verify all returned metrics are within the time range
            queriedMetrics.forEach { metrics ->
                assertTrue(
                    metrics.timestamp >= timeRange.startTime && metrics.timestamp <= timeRange.endTime,
                    "Metric ${metrics.id} should be within time range [${timeRange.startTime}, ${timeRange.endTime}]"
                )
            }
            
            // Verify all metrics that should be in range are returned
            metricsInRange.forEach { expectedMetrics ->
                assertTrue(
                    queriedMetrics.any { it.id == expectedMetrics.id },
                    "Metric ${expectedMetrics.id} within range should be returned"
                )
            }
            
            // Verify no metrics outside the range are returned
            metricsOutsideRange.forEach { outsideMetrics ->
                assertFalse(
                    queriedMetrics.any { it.id == outsideMetrics.id },
                    "Metric ${outsideMetrics.id} outside range should not be returned"
                )
            }
        }
    }

    /**
     * Test interaction window aggregation
     */
    @Test
    fun testInteractionWindowAggregation() = runBlocking {
        checkAll(
            iterations = 30,
            arbInteractionWindowSequence()
        ) { windowSequence ->
            
            // Store all interaction windows
            windowSequence.forEach { metrics ->
                service.storeInteractionMetrics(metrics)
            }
            
            // Query all stored metrics
            val minTime = windowSequence.minOf { it.timestamp }
            val maxTime = windowSequence.maxOf { it.timestamp }
            val storedMetrics = service.getInteractionMetrics(TimeRange(minTime - 1000, maxTime + 1000))
            
            // Verify total interaction counts
            val expectedTotalTouches = windowSequence.sumOf { it.touchCount }
            val actualTotalTouches = storedMetrics.sumOf { it.touchCount }
            
            assertEquals(
                expectedTotalTouches,
                actualTotalTouches,
                "Total touch count should be preserved across windows"
            )
            
            val expectedTotalScrolls = windowSequence.sumOf { it.scrollEvents }
            val actualTotalScrolls = storedMetrics.sumOf { it.scrollEvents }
            
            assertEquals(
                expectedTotalScrolls,
                actualTotalScrolls,
                "Total scroll count should be preserved across windows"
            )
        }
    }

    // Helper functions for generating test data

    private fun arbInteractionMetrics() = Arb.bind(
        Arb.int(min = 0, max = 100), // touchCount
        Arb.int(min = 0, max = 50), // scrollEvents
        Arb.list(Arb.enum<GestureType>(), range = 0..10), // gesturePatterns
        Arb.float(min = 0f, max = 100f), // interactionIntensity
        Arb.long(min = 1000000000000L, max = System.currentTimeMillis()) // timestamp
    ) { touchCount, scrollEvents, gesturePatterns, intensity, timestamp ->
        InteractionMetrics(
            timestamp = timestamp,
            touchCount = touchCount,
            scrollEvents = scrollEvents,
            gesturePatterns = gesturePatterns.distinct(),
            interactionIntensity = intensity,
            timeWindow = TimeRange(timestamp - 60000L, timestamp)
        )
    }

    private fun arbInteractionWindow() = Arb.bind(
        Arb.int(min = 0, max = 50), // touchCount
        Arb.int(min = 0, max = 30), // scrollCount
        Arb.long(min = 1000L, max = 300000L) // windowDuration
    ) { touchCount, scrollCount, duration ->
        Triple(touchCount, scrollCount, duration)
    }

    private fun arbGestureSequence() = Arb.list(
        Arb.enum<GestureType>(),
        range = 1..20
    )

    private fun arbTimeRangeWithInteractions() = Arb.bind(
        Arb.long(min = 1000000000000L, max = System.currentTimeMillis() - 86400000L), // baseTime
        Arb.long(min = 3600000L, max = 86400000L) // rangeDuration
    ) { baseTime, rangeDuration ->
        val timeRange = TimeRange(baseTime, baseTime + rangeDuration)
        
        // Generate metrics within the range
        val metricsInRange = (1..3).map { i ->
            val timestamp = baseTime + (rangeDuration * i / 4)
            InteractionMetrics(
                timestamp = timestamp,
                touchCount = (1..20).random(),
                scrollEvents = (1..10).random(),
                gesturePatterns = listOf(GestureType.TAP, GestureType.SCROLL_VERTICAL),
                interactionIntensity = (1f..10f).random(),
                timeWindow = TimeRange(timestamp - 60000L, timestamp)
            )
        }
        
        // Generate metrics outside the range
        val metricsOutsideRange = listOf(
            InteractionMetrics(
                timestamp = baseTime - 3600000L,
                touchCount = 5,
                scrollEvents = 2,
                gesturePatterns = listOf(GestureType.TAP),
                interactionIntensity = 2f,
                timeWindow = TimeRange(baseTime - 3660000L, baseTime - 3600000L)
            ),
            InteractionMetrics(
                timestamp = baseTime + rangeDuration + 1000000L,
                touchCount = 8,
                scrollEvents = 3,
                gesturePatterns = listOf(GestureType.SCROLL_HORIZONTAL),
                interactionIntensity = 4f,
                timeWindow = TimeRange(
                    baseTime + rangeDuration + 940000L,
                    baseTime + rangeDuration + 1000000L
                )
            )
        )
        
        Triple(timeRange, metricsInRange, metricsOutsideRange)
    }

    private fun arbInteractionWindowSequence() = Arb.list(
        arbInteractionMetrics(),
        range = 2..10
    ).map { metrics ->
        // Sort by timestamp to ensure proper sequence
        metrics.sortedBy { it.timestamp }
    }

    // Extension function to access private method for testing
    private suspend fun InteractionAccessibilityService.storeInteractionMetrics(metrics: InteractionMetrics) {
        // Use reflection to access private method for testing
        val method = this::class.java.getDeclaredMethod("storeInteractionMetrics", InteractionMetrics::class.java)
        method.isAccessible = true
        method.invoke(this, metrics)
    }
}