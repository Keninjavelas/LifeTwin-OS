package com.lifetwin.mlp.usagestats.test

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.lifetwin.mlp.db.*
import com.lifetwin.mlp.usagestats.UsageStatsCollector
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
 * Property-based tests for UsageStatsCollector
 * Feature: data-collection-intelligence, Property 1: Event Recording Consistency (Usage Events)
 * Feature: data-collection-intelligence, Property 2: Database Persistence Guarantee
 * Feature: data-collection-intelligence, Property 3: Time Range Query Accuracy
 * Validates: Requirements 1.1, 1.2, 1.3, 1.4
 */
@RunWith(AndroidJUnit4::class)
class UsageStatsCollectorPropertyTest {

    private lateinit var context: Context
    private lateinit var collector: UsageStatsCollector
    private var database: AppDatabase? = null

    @Before
    fun setup() = runBlocking {
        context = ApplicationProvider.getApplicationContext()
        collector = UsageStatsCollector(context)
        
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
     * Property 1: Event Recording Consistency (Usage Events)
     * For any behavioral event (app launch, notification, screen change, interaction, sensor reading), 
     * when the event occurs, the corresponding collector should record it with accurate timestamp and metadata
     */
    @Test
    fun testUsageEventRecordingConsistency() = runBlocking {
        checkAll(
            iterations = 100,
            arbUsageEventList()
        ) { usageEvents ->
            
            // Store the events using the collector's internal method
            collector.storeUsageEvents(usageEvents)
            
            // Verify each event was recorded correctly
            usageEvents.forEach { originalEvent ->
                val storedEvents = collector.getStoredUsageEvents(
                    TimeRange(
                        originalEvent.startTime - 1000,
                        originalEvent.endTime + 1000
                    )
                )
                
                val matchingEvent = storedEvents.find { it.id == originalEvent.id }
                assertNotNull(matchingEvent, "Event ${originalEvent.id} should be stored")
                
                // Verify all fields are recorded accurately
                assertEquals(originalEvent.packageName, matchingEvent.packageName)
                assertEquals(originalEvent.startTime, matchingEvent.startTime)
                assertEquals(originalEvent.endTime, matchingEvent.endTime)
                assertEquals(originalEvent.totalTimeInForeground, matchingEvent.totalTimeInForeground)
                assertEquals(originalEvent.lastTimeUsed, matchingEvent.lastTimeUsed)
                assertEquals(originalEvent.eventType, matchingEvent.eventType)
            }
        }
    }

    /**
     * Property 2: Database Persistence Guarantee
     * For any collected event, when it is processed by a data collector, 
     * it should be persisted to the encrypted local database immediately and be retrievable through queries
     */
    @Test
    fun testDatabasePersistenceGuarantee() = runBlocking {
        checkAll(
            iterations = 100,
            arbUsageEvent()
        ) { usageEvent ->
            
            val db = database!!
            
            // Store the event
            collector.storeUsageEvents(listOf(usageEvent))
            
            // Verify it's immediately retrievable from the database
            val storedEvents = db.usageEventDao().getEventsByTimeRange(
                usageEvent.startTime - 1000,
                usageEvent.endTime + 1000
            )
            
            assertTrue(
                storedEvents.any { it.id == usageEvent.id },
                "Event should be immediately retrievable after storage"
            )
            
            // Verify it's also in raw events table
            val rawEvents = db.rawEventDao().getEventsByTimeRange(
                usageEvent.startTime - 1000,
                usageEvent.endTime + 1000
            )
            
            assertTrue(
                rawEvents.any { it.eventType == "usage" && it.packageName == usageEvent.packageName },
                "Raw event should be created for usage event"
            )
        }
    }

    /**
     * Property 3: Time Range Query Accuracy
     * For any time range query, all returned events should have timestamps within the specified range, 
     * and no events within the range should be omitted
     */
    @Test
    fun testTimeRangeQueryAccuracy() = runBlocking {
        checkAll(
            iterations = 50,
            arbTimeRangeWithEvents()
        ) { (timeRange, eventsInRange, eventsOutsideRange) ->
            
            val db = database!!
            
            // Store all events (both inside and outside the range)
            val allEvents = eventsInRange + eventsOutsideRange
            collector.storeUsageEvents(allEvents)
            
            // Query events within the time range
            val queriedEvents = collector.getStoredUsageEvents(timeRange)
            
            // Verify all returned events are within the time range
            queriedEvents.forEach { event ->
                assertTrue(
                    event.startTime >= timeRange.startTime && event.endTime <= timeRange.endTime,
                    "Event ${event.id} should be within time range [${timeRange.startTime}, ${timeRange.endTime}]"
                )
            }
            
            // Verify all events that should be in range are returned
            eventsInRange.forEach { expectedEvent ->
                assertTrue(
                    queriedEvents.any { it.id == expectedEvent.id },
                    "Event ${expectedEvent.id} within range should be returned"
                )
            }
            
            // Verify no events outside the range are returned
            eventsOutsideRange.forEach { outsideEvent ->
                assertFalse(
                    queriedEvents.any { it.id == outsideEvent.id },
                    "Event ${outsideEvent.id} outside range should not be returned"
                )
            }
        }
    }

    /**
     * Test collector lifecycle and state management
     */
    @Test
    fun testCollectorLifecycle() = runBlocking {
        checkAll(
            iterations = 50,
            Arb.boolean()
        ) { shouldStart ->
            
            // Initial state
            assertFalse(collector.isCollectionActive(), "Collector should start inactive")
            assertEquals(CollectorType.USAGE_STATS, collector.getCollectorType())
            
            if (shouldStart) {
                collector.startCollection()
                assertTrue(collector.isCollectionActive(), "Collector should be active after start")
                
                collector.stopCollection()
                assertFalse(collector.isCollectionActive(), "Collector should be inactive after stop")
            }
            
            // Data count should be non-negative
            val dataCount = collector.getCollectedDataCount()
            assertTrue(dataCount >= 0, "Data count should be non-negative")
        }
    }

    /**
     * Test usage event aggregation and session detection
     */
    @Test
    fun testUsageEventAggregation() = runBlocking {
        checkAll(
            iterations = 50,
            arbUsageSessionEvents()
        ) { sessionEvents ->
            
            // Store the session events
            collector.storeUsageEvents(sessionEvents)
            
            // Calculate expected total foreground time
            val expectedTotalTime = sessionEvents
                .filter { it.eventType == UsageEventType.ACTIVITY_PAUSED }
                .sumOf { it.totalTimeInForeground }
            
            // Query all events for this session
            val minTime = sessionEvents.minOf { it.startTime }
            val maxTime = sessionEvents.maxOf { it.endTime }
            val storedEvents = collector.getStoredUsageEvents(TimeRange(minTime - 1000, maxTime + 1000))
            
            // Verify session consistency
            val actualTotalTime = storedEvents
                .filter { it.eventType == UsageEventType.ACTIVITY_PAUSED }
                .sumOf { it.totalTimeInForeground }
            
            assertEquals(
                expectedTotalTime,
                actualTotalTime,
                "Total foreground time should be preserved during storage"
            )
            
            // Verify event ordering is maintained
            val pauseEvents = storedEvents
                .filter { it.eventType == UsageEventType.ACTIVITY_PAUSED }
                .sortedBy { it.startTime }
            
            for (i in 1 until pauseEvents.size) {
                assertTrue(
                    pauseEvents[i].startTime >= pauseEvents[i-1].endTime,
                    "Events should be in chronological order"
                )
            }
        }
    }

    // Helper functions for generating test data

    private fun arbUsageEvent() = Arb.bind(
        Arb.string(minSize = 5, maxSize = 50), // packageName
        Arb.long(min = 1000000000000L, max = System.currentTimeMillis()), // startTime
        Arb.long(min = 1000L, max = 3600000L), // duration
        Arb.enum<UsageEventType>()
    ) { packageName, startTime, duration, eventType ->
        UsageEvent(
            packageName = packageName,
            startTime = startTime,
            endTime = startTime + duration,
            totalTimeInForeground = if (eventType == UsageEventType.ACTIVITY_PAUSED) duration else 0L,
            lastTimeUsed = startTime + duration,
            eventType = eventType
        )
    }

    private fun arbUsageEventList() = Arb.list(arbUsageEvent(), range = 1..20)

    private fun arbTimeRangeWithEvents() = Arb.bind(
        Arb.long(min = 1000000000000L, max = System.currentTimeMillis() - 86400000L), // baseTime
        Arb.long(min = 3600000L, max = 86400000L) // rangeDuration
    ) { baseTime, rangeDuration ->
        val timeRange = TimeRange(baseTime, baseTime + rangeDuration)
        
        // Generate events within the range
        val eventsInRange = (1..5).map { i ->
            val eventStart = baseTime + (rangeDuration * i / 6)
            val eventDuration = rangeDuration / 20
            UsageEvent(
                packageName = "com.test.app$i",
                startTime = eventStart,
                endTime = eventStart + eventDuration,
                totalTimeInForeground = eventDuration,
                lastTimeUsed = eventStart + eventDuration,
                eventType = UsageEventType.ACTIVITY_PAUSED
            )
        }
        
        // Generate events outside the range
        val eventsOutsideRange = listOf(
            UsageEvent(
                packageName = "com.test.before",
                startTime = baseTime - 3600000L,
                endTime = baseTime - 3000000L,
                totalTimeInForeground = 600000L,
                lastTimeUsed = baseTime - 3000000L,
                eventType = UsageEventType.ACTIVITY_PAUSED
            ),
            UsageEvent(
                packageName = "com.test.after",
                startTime = baseTime + rangeDuration + 1000000L,
                endTime = baseTime + rangeDuration + 1600000L,
                totalTimeInForeground = 600000L,
                lastTimeUsed = baseTime + rangeDuration + 1600000L,
                eventType = UsageEventType.ACTIVITY_PAUSED
            )
        )
        
        Triple(timeRange, eventsInRange, eventsOutsideRange)
    }

    private fun arbUsageSessionEvents() = Arb.bind(
        Arb.string(minSize = 5, maxSize = 50), // packageName
        Arb.long(min = 1000000000000L, max = System.currentTimeMillis()), // sessionStart
        Arb.int(min = 2, max = 10) // numberOfEvents
    ) { packageName, sessionStart, numberOfEvents ->
        val events = mutableListOf<UsageEvent>()
        var currentTime = sessionStart
        
        repeat(numberOfEvents / 2) { i ->
            // Activity resumed
            events.add(
                UsageEvent(
                    packageName = packageName,
                    startTime = currentTime,
                    endTime = currentTime,
                    totalTimeInForeground = 0L,
                    lastTimeUsed = currentTime,
                    eventType = UsageEventType.ACTIVITY_RESUMED
                )
            )
            
            currentTime += (60000L..600000L).random() // 1-10 minutes
            
            // Activity paused
            val duration = currentTime - events.last().startTime
            events.add(
                UsageEvent(
                    packageName = packageName,
                    startTime = events.last().startTime,
                    endTime = currentTime,
                    totalTimeInForeground = duration,
                    lastTimeUsed = currentTime,
                    eventType = UsageEventType.ACTIVITY_PAUSED
                )
            )
            
            currentTime += (10000L..60000L).random() // 10 seconds to 1 minute break
        }
        
        events
    }

    // Extension function to access private method for testing
    private suspend fun UsageStatsCollector.storeUsageEvents(events: List<UsageEvent>) {
        // Use reflection to access private method for testing
        val method = this::class.java.getDeclaredMethod("storeUsageEvents", List::class.java)
        method.isAccessible = true
        method.invoke(this, events)
    }
}