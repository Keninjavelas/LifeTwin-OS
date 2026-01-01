package com.lifetwin.mlp.export.test

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.lifetwin.mlp.db.*
import com.lifetwin.mlp.export.DataExporter
import io.kotest.property.Arb
import io.kotest.property.arbitrary.*
import io.kotest.property.checkAll
import kotlinx.coroutines.runBlocking
import org.junit.After
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import kotlin.test.assertTrue
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import java.util.*

@RunWith(AndroidJUnit4::class)
class DataExporterPropertyTest {

    private lateinit var context: Context
    private lateinit var dataExporter: DataExporter
    private lateinit var database: AppDatabase

    @Before
    fun setUp() {
        context = ApplicationProvider.getApplicationContext()
        dataExporter = DataExporter(context)
        database = AppDatabase.getInstance(context)
    }

    @After
    fun tearDown() {
        runBlocking {
            database.clearAllTables()
        }
    }

    /**
     * Property 10: Data Export Completeness
     * Tests that exported data contains all matching records and maintains integrity
     * **Validates: Requirements 9.1, 9.2, 9.3, 9.4, 9.5**
     */
    @Test
    fun `property test - data export completeness for all data types`() = runBlocking {
        checkAll<List<TestDataSet>>(
            iterations = 100,
            Arb.list(
                Arb.bind(
                    Arb.element(CollectorType.values().toList()),
                    Arb.int(1..20),
                    Arb.long(System.currentTimeMillis() - 86400000L..System.currentTimeMillis())
                ) { collectorType, recordCount, baseTimestamp ->
                    TestDataSet(collectorType, recordCount, baseTimestamp)
                },
                range = 1..5
            )
        ) { testDataSets ->
            // Clear database
            database.clearAllTables()
            
            // Insert test data for each collector type
            val expectedCounts = mutableMapOf<CollectorType, Int>()
            
            testDataSets.forEach { testData ->
                val actualCount = insertTestData(testData)
                expectedCounts[testData.collectorType] = actualCount
            }
            
            // Export all data
            val exportedData = dataExporter.exportAllData()
            
            // Validate export is not empty
            assertTrue(exportedData.isNotBlank(), "Exported data should not be empty")
            
            // Validate export data structure
            assertTrue(
                dataExporter.validateExportData(exportedData),
                "Exported data should pass validation"
            )
            
            // Test round-trip: import the exported data into a clean database
            database.clearAllTables()
            val importSuccess = dataExporter.importData(exportedData)
            assertTrue(importSuccess, "Import should succeed for valid export data")
            
            // Verify all data was imported correctly
            expectedCounts.forEach { (collectorType, expectedCount) ->
                val actualCount = getRecordCount(collectorType)
                assertTrue(
                    actualCount >= expectedCount,
                    "Imported data count for $collectorType should be at least $expectedCount, but was $actualCount"
                )
            }
        }
    }

    /**
     * Property test for selective export by collector type
     */
    @Test
    fun `property test - selective export by collector type maintains accuracy`() = runBlocking {
        checkAll<SelectiveExportTestData>(
            iterations = 100,
            Arb.bind(
                Arb.set(Arb.element(CollectorType.values().toList()), range = 1..CollectorType.values().size),
                Arb.list(
                    Arb.bind(
                        Arb.element(CollectorType.values().toList()),
                        Arb.int(1..10)
                    ) { type, count -> type to count },
                    range = 1..CollectorType.values().size
                )
            ) { selectedTypes, dataByType ->
                SelectiveExportTestData(selectedTypes, dataByType.toMap())
            }
        ) { testData ->
            // Clear database
            database.clearAllTables()
            
            // Insert test data for all types
            val insertedCounts = mutableMapOf<CollectorType, Int>()
            testData.dataByType.forEach { (collectorType, count) ->
                val testDataSet = TestDataSet(collectorType, count, System.currentTimeMillis())
                insertedCounts[collectorType] = insertTestData(testDataSet)
            }
            
            // Export only selected types
            val exportedData = dataExporter.exportDataByType(testData.selectedTypes)
            
            // Validate export
            assertTrue(
                dataExporter.validateExportData(exportedData),
                "Selective export should pass validation"
            )
            
            // Import into clean database
            database.clearAllTables()
            val importSuccess = dataExporter.importData(exportedData)
            assertTrue(importSuccess, "Import of selective export should succeed")
            
            // Verify only selected types were imported
            CollectorType.values().forEach { collectorType ->
                val actualCount = getRecordCount(collectorType)
                val expectedCount = if (collectorType in testData.selectedTypes) {
                    insertedCounts[collectorType] ?: 0
                } else {
                    0
                }
                
                assertTrue(
                    actualCount == expectedCount,
                    "Record count for $collectorType should be $expectedCount, but was $actualCount"
                )
            }
        }
    }

    /**
     * Property test for time range export accuracy
     */
    @Test
    fun `property test - time range export includes only records within range`() = runBlocking {
        checkAll<TimeRangeExportTestData>(
            iterations = 100,
            Arb.bind(
                Arb.long(1000000000000L..System.currentTimeMillis() - 86400000L), // Start time
                Arb.long(1..86400000L) // Duration (up to 1 day)
            ) { startTime, duration ->
                TimeRangeExportTestData(
                    timeRange = TimeRange(startTime, startTime + duration),
                    recordsBeforeRange = Arb.int(1..5).next(),
                    recordsInRange = Arb.int(1..10).next(),
                    recordsAfterRange = Arb.int(1..5).next()
                )
            }
        ) { testData ->
            // Clear database
            database.clearAllTables()
            
            val timeRange = testData.timeRange
            
            // Insert records before, during, and after the time range
            insertUsageEventsAtTimes(
                listOf(
                    timeRange.startTime - 3600000L, // 1 hour before
                    timeRange.startTime + (timeRange.endTime - timeRange.startTime) / 2, // Middle
                    timeRange.endTime + 3600000L // 1 hour after
                )
            )
            
            // Export data for the specific time range
            val exportedData = dataExporter.exportDataByTimeRange(timeRange)
            
            // Validate export
            assertTrue(
                dataExporter.validateExportData(exportedData),
                "Time range export should pass validation"
            )
            
            // Import into clean database
            database.clearAllTables()
            val importSuccess = dataExporter.importData(exportedData)
            assertTrue(importSuccess, "Import of time range export should succeed")
            
            // Verify only records within time range were imported
            val importedEvents = database.usageEventDao().getAllEvents()
            importedEvents.forEach { event ->
                assertTrue(
                    event.startTime >= timeRange.startTime && event.startTime <= timeRange.endTime,
                    "Imported event timestamp ${event.startTime} should be within range [${timeRange.startTime}, ${timeRange.endTime}]"
                )
            }
        }
    }

    /**
     * Property test for export validation accuracy
     */
    @Test
    fun `property test - export validation correctly identifies valid and invalid data`() = runBlocking {
        checkAll<ExportValidationTestData>(
            iterations = 100,
            Arb.bind(
                Arb.boolean(), // Is valid export
                Arb.string(1..100), // Random string for invalid cases
                Arb.long() // Random timestamp
            ) { isValid, randomString, timestamp ->
                ExportValidationTestData(isValid, randomString, timestamp)
            }
        ) { testData ->
            if (testData.isValid) {
                // Create valid export data
                database.clearAllTables()
                insertUsageEventsAtTimes(listOf(System.currentTimeMillis() - 3600000L))
                
                val validExport = dataExporter.exportAllData()
                
                // Valid export should pass validation
                assertTrue(
                    dataExporter.validateExportData(validExport),
                    "Valid export data should pass validation"
                )
                
                // Valid export should be importable
                database.clearAllTables()
                val importSuccess = dataExporter.importData(validExport)
                assertTrue(importSuccess, "Valid export should be importable")
                
            } else {
                // Test invalid data
                val invalidExports = listOf(
                    "", // Empty string
                    "{}", // Empty JSON
                    testData.randomString, // Random string
                    """{"invalid": "structure"}""", // Invalid structure
                    """{"exportMetadata": {"exportedAt": -1}}""" // Invalid timestamp
                )
                
                invalidExports.forEach { invalidExport ->
                    // Invalid export should fail validation
                    assertFalse(
                        dataExporter.validateExportData(invalidExport),
                        "Invalid export data should fail validation: $invalidExport"
                    )
                    
                    // Invalid export should not be importable
                    val importSuccess = dataExporter.importData(invalidExport)
                    assertFalse(importSuccess, "Invalid export should not be importable")
                }
            }
        }
    }

    /**
     * Property test for export metadata accuracy
     */
    @Test
    fun `property test - export metadata contains accurate information`() = runBlocking {
        checkAll<Int>(
            iterations = 50,
            Arb.int(0..100)
        ) { recordCount ->
            // Clear database and insert test data
            database.clearAllTables()
            
            if (recordCount > 0) {
                val timestamps = (1..recordCount).map { 
                    System.currentTimeMillis() - (it * 60000L) // 1 minute intervals
                }
                insertUsageEventsAtTimes(timestamps)
            }
            
            val exportStartTime = System.currentTimeMillis()
            val exportedData = dataExporter.exportAllData()
            val exportEndTime = System.currentTimeMillis()
            
            // Validate export contains metadata
            assertTrue(exportedData.contains("exportMetadata"), "Export should contain metadata")
            assertTrue(exportedData.contains("exportedAt"), "Export should contain export timestamp")
            assertTrue(exportedData.contains("exportVersion"), "Export should contain version")
            
            // Validate export is valid
            assertTrue(
                dataExporter.validateExportData(exportedData),
                "Export with $recordCount records should be valid"
            )
            
            // Test round-trip maintains record count
            database.clearAllTables()
            val importSuccess = dataExporter.importData(exportedData)
            assertTrue(importSuccess, "Import should succeed")
            
            val importedCount = database.usageEventDao().getAllEvents().size
            assertTrue(
                importedCount == recordCount,
                "Imported record count should match original: expected $recordCount, got $importedCount"
            )
        }
    }

    // Helper methods

    private suspend fun insertTestData(testData: TestDataSet): Int {
        return when (testData.collectorType) {
            CollectorType.USAGE_STATS -> {
                val events = (1..testData.recordCount).map { index ->
                    UsageEventEntity(
                        id = UUID.randomUUID().toString(),
                        packageName = "com.test.app$index",
                        startTime = testData.baseTimestamp + (index * 60000L),
                        endTime = testData.baseTimestamp + (index * 60000L) + 30000L,
                        totalTimeInForeground = 30000L,
                        lastTimeUsed = testData.baseTimestamp + (index * 60000L),
                        eventType = "ACTIVITY_RESUMED"
                    )
                }
                events.forEach { database.usageEventDao().insert(it) }
                events.size
            }
            CollectorType.NOTIFICATIONS -> {
                val events = (1..testData.recordCount).map { index ->
                    NotificationEventEntity(
                        id = UUID.randomUUID().toString(),
                        packageName = "com.test.app$index",
                        timestamp = testData.baseTimestamp + (index * 60000L),
                        category = "test",
                        priority = 0,
                        hasActions = false,
                        isOngoing = false,
                        interactionType = "posted"
                    )
                }
                events.forEach { database.notificationEventDao().insert(it) }
                events.size
            }
            CollectorType.SCREEN_EVENTS -> {
                val sessions = (1..testData.recordCount).map { index ->
                    ScreenSessionEntity(
                        sessionId = UUID.randomUUID().toString(),
                        startTime = testData.baseTimestamp + (index * 60000L),
                        endTime = testData.baseTimestamp + (index * 60000L) + 30000L,
                        unlockCount = 1,
                        interactionIntensity = 0.5f
                    )
                }
                sessions.forEach { database.screenSessionDao().insert(it) }
                sessions.size
            }
            CollectorType.INTERACTIONS -> {
                val metrics = (1..testData.recordCount).map { index ->
                    InteractionMetricsEntity(
                        id = UUID.randomUUID().toString(),
                        timestamp = testData.baseTimestamp + (index * 60000L),
                        touchCount = 10,
                        scrollEvents = 5,
                        gesturePatterns = """["TAP", "SCROLL_VERTICAL"]""",
                        interactionIntensity = 0.7f,
                        timeWindowStart = testData.baseTimestamp + (index * 60000L),
                        timeWindowEnd = testData.baseTimestamp + (index * 60000L) + 30000L
                    )
                }
                metrics.forEach { database.interactionMetricsDao().insert(it) }
                metrics.size
            }
            CollectorType.SENSORS -> {
                val contexts = (1..testData.recordCount).map { index ->
                    ActivityContextEntity(
                        id = UUID.randomUUID().toString(),
                        activityType = "WALKING",
                        confidence = 0.8f,
                        timestamp = testData.baseTimestamp + (index * 60000L),
                        duration = 30000L,
                        sensorData = """{"accelerometer": [1.0, 2.0, 3.0]}"""
                    )
                }
                contexts.forEach { database.activityContextDao().insert(it) }
                contexts.size
            }
        }
    }

    private suspend fun getRecordCount(collectorType: CollectorType): Int {
        return when (collectorType) {
            CollectorType.USAGE_STATS -> database.usageEventDao().getAllEvents().size
            CollectorType.NOTIFICATIONS -> database.notificationEventDao().getAllEvents().size
            CollectorType.SCREEN_EVENTS -> database.screenSessionDao().getAllSessions().size
            CollectorType.INTERACTIONS -> database.interactionMetricsDao().getAllMetrics().size
            CollectorType.SENSORS -> database.activityContextDao().getAllContexts().size
        }
    }

    private suspend fun insertUsageEventsAtTimes(timestamps: List<Long>) {
        timestamps.forEachIndexed { index, timestamp ->
            val event = UsageEventEntity(
                id = UUID.randomUUID().toString(),
                packageName = "com.test.app$index",
                startTime = timestamp,
                endTime = timestamp + 30000L,
                totalTimeInForeground = 30000L,
                lastTimeUsed = timestamp,
                eventType = "ACTIVITY_RESUMED"
            )
            database.usageEventDao().insert(event)
        }
    }

    // Test data classes

    data class TestDataSet(
        val collectorType: CollectorType,
        val recordCount: Int,
        val baseTimestamp: Long
    )

    data class SelectiveExportTestData(
        val selectedTypes: Set<CollectorType>,
        val dataByType: Map<CollectorType, Int>
    )

    data class TimeRangeExportTestData(
        val timeRange: TimeRange,
        val recordsBeforeRange: Int,
        val recordsInRange: Int,
        val recordsAfterRange: Int
    )

    data class ExportValidationTestData(
        val isValid: Boolean,
        val randomString: String,
        val timestamp: Long
    )
}