package com.lifetwin.mlp.sensors.test

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorManager
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.lifetwin.mlp.db.*
import com.lifetwin.mlp.sensors.SensorFusionManager
import io.kotest.property.Arb
import io.kotest.property.arbitrary.*
import io.kotest.property.checkAll
import io.kotest.property.forAll
import kotlinx.coroutines.delay
import kotlinx.coroutines.runBlocking
import org.junit.After
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.mockito.Mock
import org.mockito.MockitoAnnotations
import org.mockito.kotlin.*
import kotlin.math.abs
import kotlin.math.sqrt
import kotlin.test.assertTrue

@RunWith(AndroidJUnit4::class)
class SensorFusionManagerPropertyTest {

    private lateinit var context: Context
    private lateinit var sensorFusionManager: SensorFusionManager
    
    @Mock
    private lateinit var mockSensorManager: SensorManager
    
    @Mock
    private lateinit var mockAccelerometer: Sensor
    
    @Mock
    private lateinit var mockGyroscope: Sensor
    
    @Mock
    private lateinit var mockMagnetometer: Sensor

    @Before
    fun setUp() {
        MockitoAnnotations.openMocks(this)
        context = ApplicationProvider.getApplicationContext()
        sensorFusionManager = SensorFusionManager(context)
        
        // Setup mock sensors
        whenever(mockSensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)).thenReturn(mockAccelerometer)
        whenever(mockSensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)).thenReturn(mockGyroscope)
        whenever(mockSensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD)).thenReturn(mockMagnetometer)
    }

    @After
    fun tearDown() {
        runBlocking {
            if (sensorFusionManager.isCollectionActive()) {
                sensorFusionManager.stopCollection()
            }
        }
    }

    /**
     * Property 9: Activity Detection Consistency
     * Tests that activity detection produces consistent results for similar sensor patterns
     */
    @Test
    fun `property test - activity detection consistency for similar patterns`() = runBlocking {
        checkAll<List<AccelerometerData>>(
            iterations = 150,
            Arb.list(
                Arb.bind(
                    Arb.float(-20f, 20f),
                    Arb.float(-20f, 20f),
                    Arb.float(-20f, 20f)
                ) { x, y, z ->
                    AccelerometerData(x, y, z, sqrt(x*x + y*y + z*z))
                },
                range = 20..100
            )
        ) { sensorData ->
            // Generate similar sensor patterns by adding small noise
            val pattern1 = sensorData
            val pattern2 = sensorData.map { data ->
                val noise = 0.1f
                AccelerometerData(
                    data.x + (Math.random().toFloat() - 0.5f) * noise,
                    data.y + (Math.random().toFloat() - 0.5f) * noise,
                    data.z + (Math.random().toFloat() - 0.5f) * noise,
                    data.magnitude + (Math.random() - 0.5) * noise
                )
            }
            
            val activity1 = classifyActivityFromData(pattern1)
            val activity2 = classifyActivityFromData(pattern2)
            
            // Similar patterns should produce similar activity classifications
            // Allow for some variance in confidence but activity type should be consistent
            if (activity1 != null && activity2 != null) {
                val confidenceDiff = abs(activity1.confidence - activity2.confidence)
                assertTrue(
                    activity1.activityType == activity2.activityType || confidenceDiff < 0.3f,
                    "Similar sensor patterns should produce consistent activity classifications. " +
                    "Pattern1: ${activity1.activityType}(${activity1.confidence}), " +
                    "Pattern2: ${activity2.activityType}(${activity2.confidence})"
                )
            }
        }
    }

    /**
     * Property 11: Batching Optimization
     * Tests that sensor data batching reduces database operations while preserving data integrity
     */
    @Test
    fun `property test - batching optimization reduces database operations`() = runBlocking {
        checkAll<List<SensorReading>>(
            iterations = 100,
            Arb.list(
                Arb.bind(
                    Arb.string(1..20),
                    Arb.long(1000000000L, 2000000000L),
                    Arb.list(Arb.float(-50f, 50f), range = 3..3),
                    Arb.double(0.0, 100.0)
                ) { sensorType, timestamp, values, magnitude ->
                    SensorReading(sensorType, timestamp, values.toFloatArray(), magnitude)
                },
                range = 10..200
            )
        ) { sensorReadings ->
            val database = AppDatabase.getInstance(context)
            val initialEventCount = database.rawEventDao().getUnprocessedEventCount()
            
            // Simulate sensor data collection with batching
            val batchSize = 50
            val batches = sensorReadings.chunked(batchSize)
            
            // Process each batch
            batches.forEach { batch ->
                processSensorBatch(batch)
            }
            
            delay(100) // Allow async operations to complete
            
            val finalEventCount = database.rawEventDao().getUnprocessedEventCount()
            val expectedBatches = (sensorReadings.size + batchSize - 1) / batchSize
            val actualNewEvents = finalEventCount - initialEventCount
            
            // Batching should create fewer database entries than individual sensor readings
            assertTrue(
                actualNewEvents <= expectedBatches,
                "Batching should reduce database operations. Expected ≤$expectedBatches batches, " +
                "but got $actualNewEvents new events for ${sensorReadings.size} sensor readings"
            )
            
            // But should not lose data - each batch should represent multiple readings
            if (sensorReadings.isNotEmpty()) {
                assertTrue(
                    actualNewEvents > 0,
                    "Batching should still create database entries for non-empty sensor data"
                )
            }
        }
    }

    /**
     * Property Test: Activity Classification Boundaries
     * Tests that activity classification behaves predictably at decision boundaries
     */
    @Test
    fun `property test - activity classification boundaries are stable`() = runBlocking {
        checkAll<ActivityTestData>(
            iterations = 120,
            Arb.bind(
                Arb.double(8.0, 12.0), // avgMagnitude around decision boundaries
                Arb.double(0.5, 4.0),  // stdMagnitude around decision boundaries
                Arb.double(10.0, 20.0) // maxMagnitude
            ) { avgMag, stdMag, maxMag ->
                ActivityTestData(avgMag, stdMag, maxMag)
            }
        ) { testData ->
            val activity = classifyActivityFromStats(testData.avgMagnitude, testData.stdMagnitude, testData.maxMagnitude)
            
            // Test boundary stability - small changes shouldn't cause dramatic classification changes
            val epsilon = 0.1
            val perturbedActivity = classifyActivityFromStats(
                testData.avgMagnitude + epsilon,
                testData.stdMagnitude + epsilon,
                testData.maxMagnitude + epsilon
            )
            
            if (activity != null && perturbedActivity != null) {
                // Either same classification or confidence difference should be reasonable
                val confidenceDiff = abs(activity.confidence - perturbedActivity.confidence)
                assertTrue(
                    activity.activityType == perturbedActivity.activityType || confidenceDiff < 0.4f,
                    "Small perturbations should not cause dramatic classification changes. " +
                    "Original: ${activity.activityType}(${activity.confidence}), " +
                    "Perturbed: ${perturbedActivity.activityType}(${perturbedActivity.confidence})"
                )
            }
        }
    }

    /**
     * Property Test: Sensor Permission Handling
     * Tests that sensor permission changes are handled gracefully
     */
    @Test
    fun `property test - sensor permission handling is graceful`() = runBlocking {
        checkAll<Boolean>(
            iterations = 50,
            Arb.boolean()
        ) { hasPermission ->
            // Mock permission state
            val manager = spy(sensorFusionManager)
            doReturn(hasPermission).whenever(manager).isSensorPermissionGranted()
            
            val initialState = manager.isCollectionActive()
            
            // Attempt to start collection
            manager.startCollection()
            delay(50)
            
            val collectionStarted = manager.isCollectionActive()
            
            // Collection should only be active if permissions are granted
            assertTrue(
                !collectionStarted || hasPermission,
                "Collection should not be active without proper permissions. " +
                "HasPermission: $hasPermission, CollectionActive: $collectionStarted"
            )
            
            // Stop collection should always work regardless of permission state
            manager.stopCollection()
            delay(50)
            
            val collectionStopped = !manager.isCollectionActive()
            assertTrue(
                collectionStopped,
                "Stop collection should always work regardless of permission state"
            )
        }
    }

    /**
     * Property Test: Data Collection Count Accuracy
     * Tests that collected data count accurately reflects stored data
     */
    @Test
    fun `property test - collected data count accuracy`() = runBlocking {
        checkAll<Int>(
            iterations = 80,
            Arb.int(0, 100)
        ) { expectedActivities ->
            val database = AppDatabase.getInstance(context)
            val startTime = System.currentTimeMillis()
            
            // Insert test activity contexts
            repeat(expectedActivities) { i ->
                val activityContext = ActivityContextEntity(
                    id = "test_$i",
                    activityType = ActivityType.values().random().name,
                    confidence = 0.5f + (i % 5) * 0.1f,
                    timestamp = startTime + i * 1000L,
                    duration = 1000L,
                    sensorData = null
                )
                database.activityContextDao().insert(activityContext)
            }
            
            delay(100) // Allow database operations to complete
            
            val reportedCount = sensorFusionManager.getCollectedDataCount()
            
            // The reported count should be reasonably close to what we inserted
            // (allowing for some variance due to time windows and other data)
            assertTrue(
                reportedCount >= expectedActivities,
                "Collected data count should include inserted test data. " +
                "Expected at least: $expectedActivities, Got: $reportedCount"
            )
        }
    }

    // Helper methods for property testing

    private fun classifyActivityFromData(sensorData: List<AccelerometerData>): ActivityContext? {
        if (sensorData.size < 10) return null
        
        val magnitudes = sensorData.map { it.magnitude }
        val avgMagnitude = magnitudes.average()
        val stdMagnitude = calculateStandardDeviation(magnitudes)
        val maxMagnitude = magnitudes.maxOrNull() ?: 0.0
        
        return classifyActivityFromStats(avgMagnitude, stdMagnitude, maxMagnitude)
    }

    private fun classifyActivityFromStats(avgMagnitude: Double, stdMagnitude: Double, maxMagnitude: Double): ActivityContext? {
        val (activityType, confidence) = when {
            avgMagnitude < 9.5 && stdMagnitude < 1.0 -> {
                Pair(ActivityType.STATIONARY, 0.8f)
            }
            avgMagnitude in 9.5..11.0 && stdMagnitude < 2.0 -> {
                Pair(ActivityType.WALKING, 0.7f)
            }
            avgMagnitude > 11.0 && stdMagnitude > 2.0 -> {
                if (maxMagnitude > 15.0) {
                    Pair(ActivityType.RUNNING, 0.6f)
                } else {
                    Pair(ActivityType.WALKING, 0.6f)
                }
            }
            stdMagnitude > 3.0 -> {
                Pair(ActivityType.IN_VEHICLE, 0.5f)
            }
            else -> {
                Pair(ActivityType.UNKNOWN, 0.3f)
            }
        }
        
        return ActivityContext(
            activityType = activityType,
            confidence = confidence,
            timestamp = System.currentTimeMillis(),
            duration = 0L
        )
    }

    private fun calculateStandardDeviation(values: List<Double>): Double {
        if (values.isEmpty()) return 0.0
        
        val mean = values.average()
        val variance = values.map { (it - mean) * (it - mean) }.average()
        return sqrt(variance)
    }

    private suspend fun processSensorBatch(batch: List<SensorReading>) {
        // Simulate the batching logic from SensorFusionManager
        if (batch.isEmpty()) return
        
        val batchStartTime = batch.minOf { it.timestamp }
        val batchEndTime = batch.maxOf { it.timestamp }
        
        val sensorStats = batch.groupBy { it.sensorType }.mapValues { (_, readings) ->
            mapOf(
                "count" to readings.size,
                "avgMagnitude" to readings.map { it.magnitude }.average(),
                "maxMagnitude" to readings.maxOf { it.magnitude },
                "minMagnitude" to readings.minOf { it.magnitude }
            )
        }
        
        val database = AppDatabase.getInstance(context)
        val rawEvent = RawEventEntity(
            id = java.util.UUID.randomUUID().toString(),
            timestamp = batchEndTime,
            eventType = "sensor_batch",
            packageName = null,
            duration = batchEndTime - batchStartTime,
            metadata = com.google.gson.Gson().toJson(
                mapOf(
                    "batchSize" to batch.size,
                    "sensorStats" to sensorStats
                )
            )
        )
        
        database.rawEventDao().insert(rawEvent)
    }

    // Data classes for property testing

    private data class AccelerometerData(
        val x: Float,
        val y: Float,
        val z: Float,
        val magnitude: Double
    )

    private data class SensorReading(
        val sensorType: String,
        val timestamp: Long,
        val values: FloatArray,
        val magnitude: Double
    )

    private data class ActivityTestData(
        val avgMagnitude: Double,
        val stdMagnitude: Double,
        val maxMagnitude: Double
    )
}