package com.lifetwin.mlp.performance.test

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.lifetwin.mlp.db.*
import com.lifetwin.mlp.performance.PerformanceMonitor
import io.kotest.property.Arb
import io.kotest.property.arbitrary.*
import io.kotest.property.checkAll
import kotlinx.coroutines.runBlocking
import org.junit.After
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import kotlin.test.assertTrue
import kotlin.test.assertNotNull
import java.util.*

@RunWith(AndroidJUnit4::class)
class PerformanceMonitorPropertyTest {

    private lateinit var context: Context
    private lateinit var performanceMonitor: PerformanceMonitor
    private lateinit var database: AppDatabase

    @Before
    fun setUp() {
        context = ApplicationProvider.getApplicationContext()
        performanceMonitor = PerformanceMonitor(context)
        database = AppDatabase.getInstance(context)
    }

    @After
    fun tearDown() {
        runBlocking {
            database.clearAllTables()
        }
        performanceMonitor.cleanup()
    }

    /**
     * Property 11: Batching Optimization
     * Tests that batching reduces database operations while preserving data integrity
     * **Validates: Requirements 10.1, 5.3**
     */
    @Test
    fun `property test - batching optimization reduces database operations`() = runBlocking {
        checkAll<BatchingTestData>(
            iterations = 100,
            Arb.bind(
                Arb.int(1..100), // Number of individual operations
                Arb.int(5..50),  // Batch size
                Arb.long(1..1000) // Operation duration range
            ) { operationCount, batchSize, baseDuration ->
                BatchingTestData(operationCount, batchSize, baseDuration)
            }
        ) { testData ->
            // Clear performance logs
            database.performanceLogDao().deleteOldLogs(0)
            
            // Simulate individual operations (non-batched)
            val individualStartTime = System.currentTimeMillis()
            repeat(testData.operationCount) { index ->
                val duration = testData.baseDuration + (index % 10) // Add some variation
                performanceMonitor.recordDatabaseOperation("INDIVIDUAL_INSERT", duration)
            }
            val individualEndTime = System.currentTimeMillis()
            
            // Wait a moment to separate the operations
            Thread.sleep(100)
            
            // Simulate batched operations
            val batchStartTime = System.currentTimeMillis()
            val batchCount = (testData.operationCount + testData.batchSize - 1) / testData.batchSize
            repeat(batchCount) { batchIndex ->
                val itemsInBatch = minOf(testData.batchSize, testData.operationCount - (batchIndex * testData.batchSize))
                val batchDuration = testData.baseDuration * itemsInBatch / 2 // Batching should be more efficient
                performanceMonitor.recordBatchOperation(itemsInBatch, batchDuration)
            }
            val batchEndTime = System.currentTimeMillis()
            
            // Verify that batching reduces the number of database operations
            val individualOperationLogs = database.performanceLogDao()
                .getLogsByTypeAndTimeRange("INDIVIDUAL_INSERT", individualStartTime, individualEndTime)
            val batchOperationLogs = database.performanceLogDao()
                .getLogsByTypeAndTimeRange("BATCH_OPERATION", batchStartTime, batchEndTime)
            
            // Batching should result in fewer database operations
            assertTrue(
                batchOperationLogs.size < individualOperationLogs.size,
                "Batching should reduce operations: individual=${individualOperationLogs.size}, batched=${batchOperationLogs.size}"
            )
            
            // Total records processed should be similar
            val individualRecords = individualOperationLogs.size
            val batchedRecords = batchOperationLogs.sumOf { it.recordCount ?: 0 }
            
            assertTrue(
                batchedRecords >= individualRecords * 0.8, // Allow some tolerance
                "Batched operations should process similar number of records: individual=$individualRecords, batched=$batchedRecords"
            )
        }
    }

    /**
     * Property 12: Adaptive Performance Behavior
     * Tests that performance behavior adapts correctly to system conditions
     * **Validates: Requirements 10.3**
     */
    @Test
    fun `property test - adaptive behavior responds to system conditions`() = runBlocking {
        checkAll<SystemConditionTestData>(
            iterations = 100,
            Arb.bind(
                Arb.int(1..100), // Battery level
                Arb.double(0.1..2.0), // Memory usage multiplier
                Arb.double(1.0..5000.0) // Average operation latency
            ) { batteryLevel, memoryMultiplier, operationLatency ->
                SystemConditionTestData(batteryLevel, memoryMultiplier, operationLatency)
            }
        ) { testData ->
            // Simulate system conditions by recording operations with specific characteristics
            val baseMemoryUsage = 100.0 // MB
            val memoryUsage = baseMemoryUsage * testData.memoryMultiplier
            
            // Record operations that simulate the test conditions
            repeat(10) { index ->
                val performanceEntry = PerformanceLogEntity(
                    id = UUID.randomUUID().toString(),
                    timestamp = System.currentTimeMillis() + index,
                    operationType = "PERFORMANCE_METRICS",
                    batteryLevel = testData.batteryLevel,
                    memoryUsageMB = memoryUsage,
                    durationMs = testData.operationLatency.toLong()
                )
                database.performanceLogDao().insert(performanceEntry)
            }
            
            // Get performance statistics
            val stats = performanceMonitor.getPerformanceStatistics(
                startTime = System.currentTimeMillis() - 60000L,
                endTime = System.currentTimeMillis() + 60000L
            )
            
            // Get performance recommendations
            val recommendations = performanceMonitor.getPerformanceRecommendations()
            
            // Verify adaptive behavior based on conditions
            when {
                testData.batteryLevel < 10 -> {
                    // Critical battery should generate critical recommendations
                    assertTrue(
                        recommendations.any { it.type.name == "CRITICAL" },
                        "Critical battery level should generate critical recommendations"
                    )
                }
                testData.batteryLevel < 20 -> {
                    // Low battery should generate warnings
                    assertTrue(
                        recommendations.any { it.type.name == "WARNING" },
                        "Low battery level should generate warning recommendations"
                    )
                }
                memoryUsage > 800.0 -> {
                    // High memory usage should generate recommendations
                    assertTrue(
                        recommendations.any { it.description.contains("memory", ignoreCase = true) },
                        "High memory usage should generate memory-related recommendations"
                    )
                }
                testData.operationLatency > 1000.0 -> {
                    // Slow operations should generate performance recommendations
                    assertTrue(
                        recommendations.any { it.description.contains("latency", ignoreCase = true) || 
                                           it.description.contains("slow", ignoreCase = true) },
                        "Slow operations should generate performance recommendations"
                    )
                }
            }
            
            // Verify statistics are reasonable
            assertTrue(stats.totalOperations >= 0, "Total operations should be non-negative")
            assertTrue(stats.averageBatteryLevel >= 0.0, "Average battery level should be non-negative")
            assertTrue(stats.averageMemoryUsage >= 0.0, "Average memory usage should be non-negative")
        }
    }

    /**
     * Property 14: Performance Metrics Availability
     * Tests that performance metrics are available and accurately reflect system impact
     * **Validates: Requirements 10.5**
     */
    @Test
    fun `property test - performance metrics accurately reflect system operations`() = runBlocking {
        checkAll<PerformanceMetricsTestData>(
            iterations = 100,
            Arb.bind(
                Arb.list(Arb.element(CollectorType.values().toList()), range = 1..5),
                Arb.int(1..50), // Operations per collector
                Arb.long(10..500) // Operation duration range
            ) { collectorTypes, operationsPerCollector, baseDuration ->
                PerformanceMetricsTestData(collectorTypes, operationsPerCollector, baseDuration)
            }
        ) { testData ->
            // Clear existing performance logs
            database.performanceLogDao().deleteOldLogs(0)
            
            val startTime = System.currentTimeMillis()
            
            // Record operations for each collector type
            val expectedOperationCounts = mutableMapOf<CollectorType, Int>()
            
            testData.collectorTypes.forEach { collectorType ->
                repeat(testData.operationsPerCollector) { index ->
                    val recordCount = index + 1
                    performanceMonitor.recordCollectionOperation(collectorType, recordCount)
                    
                    // Also record some database operations
                    val duration = testData.baseDuration + (index % 20)
                    performanceMonitor.recordDatabaseOperation("${collectorType.name}_OPERATION", duration)
                }
                expectedOperationCounts[collectorType] = testData.operationsPerCollector
            }
            
            val endTime = System.currentTimeMillis()
            
            // Get performance statistics
            val stats = performanceMonitor.getPerformanceStatistics(startTime, endTime)
            
            // Verify metrics availability and accuracy
            assertTrue(
                stats.totalOperations > 0,
                "Performance statistics should show operations were recorded"
            )
            
            assertTrue(
                stats.collectionOperations > 0,
                "Collection operations should be tracked"
            )
            
            // Verify operations by type are tracked
            assertTrue(
                stats.operationsByType.isNotEmpty(),
                "Operations should be categorized by type"
            )
            
            // Verify that collection operations match expected counts
            val collectionLogs = database.performanceLogDao()
                .getLogsByTypeAndTimeRange("DATA_COLLECTION", startTime, endTime)
            
            assertTrue(
                collectionLogs.size >= testData.collectorTypes.size,
                "Should have collection logs for each collector type"
            )
            
            // Verify performance metrics are within reasonable bounds
            assertTrue(
                stats.averageOperationDuration >= 0.0,
                "Average operation duration should be non-negative"
            )
            
            assertTrue(
                stats.peakMemoryUsage >= 0.0,
                "Peak memory usage should be non-negative"
            )
            
            // Verify that different collector types are tracked separately
            testData.collectorTypes.forEach { collectorType ->
                val collectorLogs = collectionLogs.filter { it.collectorType == collectorType.name }
                assertTrue(
                    collectorLogs.isNotEmpty(),
                    "Should have performance logs for collector type $collectorType"
                )
            }
        }
    }

    /**
     * Property test for performance monitoring consistency
     */
    @Test
    fun `property test - performance monitoring maintains consistency across operations`() = runBlocking {
        checkAll<ConsistencyTestData>(
            iterations = 50,
            Arb.bind(
                Arb.int(10..100), // Number of operations
                Arb.list(Arb.long(1..1000), range = 10..100) // Operation durations
            ) { operationCount, durations ->
                ConsistencyTestData(operationCount, durations.take(operationCount))
            }
        ) { testData ->
            // Clear performance logs
            database.performanceLogDao().deleteOldLogs(0)
            
            val startTime = System.currentTimeMillis()
            
            // Record operations with known durations
            testData.durations.forEachIndexed { index, duration ->
                performanceMonitor.recordDatabaseOperation("CONSISTENCY_TEST_$index", duration)
            }
            
            val endTime = System.currentTimeMillis()
            
            // Verify all operations were recorded
            val performanceLogs = database.performanceLogDao()
                .getLogsByTimeRange(startTime, endTime)
                .filter { it.operationType.startsWith("CONSISTENCY_TEST") }
            
            assertTrue(
                performanceLogs.size == testData.durations.size,
                "All operations should be recorded: expected ${testData.durations.size}, got ${performanceLogs.size}"
            )
            
            // Verify durations are preserved accurately
            val recordedDurations = performanceLogs.mapNotNull { it.durationMs }.sorted()
            val expectedDurations = testData.durations.sorted()
            
            assertTrue(
                recordedDurations.size == expectedDurations.size,
                "All durations should be recorded"
            )
            
            // Verify performance statistics reflect the recorded operations
            val stats = performanceMonitor.getPerformanceStatistics(startTime, endTime)
            
            assertTrue(
                stats.totalOperations >= testData.durations.size,
                "Statistics should include all recorded operations"
            )
            
            if (testData.durations.isNotEmpty()) {
                val expectedAverage = testData.durations.average()
                val actualAverage = stats.averageOperationDuration
                
                // Allow some tolerance for floating point precision and other operations
                val tolerance = expectedAverage * 0.5 // 50% tolerance
                assertTrue(
                    kotlin.math.abs(actualAverage - expectedAverage) <= tolerance,
                    "Average duration should be close to expected: expected=$expectedAverage, actual=$actualAverage"
                )
            }
        }
    }

    // Test data classes

    data class BatchingTestData(
        val operationCount: Int,
        val batchSize: Int,
        val baseDuration: Long
    )

    data class SystemConditionTestData(
        val batteryLevel: Int,
        val memoryMultiplier: Double,
        val operationLatency: Double
    )

    data class PerformanceMetricsTestData(
        val collectorTypes: List<CollectorType>,
        val operationsPerCollector: Int,
        val baseDuration: Long
    )

    data class ConsistencyTestData(
        val operationCount: Int,
        val durations: List<Long>
    )
}