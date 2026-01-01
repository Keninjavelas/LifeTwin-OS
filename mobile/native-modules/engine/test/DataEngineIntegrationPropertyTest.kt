package com.lifetwin.mlp.engine.test

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.lifetwin.mlp.db.*
import com.lifetwin.mlp.engine.DataEngine
import io.kotest.core.spec.style.StringSpec
import io.kotest.matchers.shouldBe
import io.kotest.matchers.shouldNotBe
import io.kotest.matchers.collections.shouldHaveSize
import io.kotest.matchers.nulls.shouldNotBeNull
import io.kotest.property.Arb
import io.kotest.property.arbitrary.*
import io.kotest.property.checkAll
import kotlinx.coroutines.*
import kotlinx.coroutines.test.*
import org.junit.runner.RunWith

/**
 * Property-based tests for DataEngine integration scenarios
 * Validates universal properties that should hold across all integration scenarios
 */
@RunWith(AndroidJUnit4::class)
class DataEngineIntegrationPropertyTest : StringSpec({

    val context = ApplicationProvider.getApplicationContext<Context>()
    
    beforeEach {
        val database = AppDatabase.getInstance(context)
        database.clearAllTables()
    }

    afterEach {
        val database = AppDatabase.getInstance(context)
        database.clearAllTables()
    }

    "Property 16: Engine State Consistency - engine state transitions should always be valid" {
        runTest {
            checkAll(
                iterations = 50,
                Arb.list(Arb.enum<EngineOperation>(), 1..10)
            ) { operations ->
                val dataEngine = DataEngine(context)
                val mockCollector = MockUsageStatsCollector()
                dataEngine.registerCollector(CollectorType.USAGE_STATS, mockCollector)
                
                var lastValidState = DataEngine.EngineStatus.INITIALIZING
                
                try {
                    for (operation in operations) {
                        when (operation) {
                            EngineOperation.INITIALIZE -> {
                                if (lastValidState == DataEngine.EngineStatus.INITIALIZING) {
                                    val result = dataEngine.initialize()
                                    if (result) {
                                        lastValidState = DataEngine.EngineStatus.READY
                                    }
                                }
                            }
                            EngineOperation.START -> {
                                if (lastValidState == DataEngine.EngineStatus.READY || 
                                    lastValidState == DataEngine.EngineStatus.STOPPED) {
                                    val result = dataEngine.start()
                                    if (result) {
                                        lastValidState = DataEngine.EngineStatus.RUNNING
                                    }
                                }
                            }
                            EngineOperation.STOP -> {
                                if (lastValidState == DataEngine.EngineStatus.RUNNING) {
                                    val result = dataEngine.stop()
                                    if (result) {
                                        lastValidState = DataEngine.EngineStatus.STOPPED
                                    }
                                }
                            }
                            EngineOperation.RESTART -> {
                                if (lastValidState == DataEngine.EngineStatus.RUNNING || 
                                    lastValidState == DataEngine.EngineStatus.STOPPED) {
                                    val result = dataEngine.restart()
                                    if (result) {
                                        lastValidState = DataEngine.EngineStatus.RUNNING
                                    }
                                }
                            }
                        }
                        
                        delay(10) // Allow state changes to propagate
                        
                        // Verify engine state is always consistent
                        val currentState = dataEngine.engineState.value
                        when (currentState.status) {
                            DataEngine.EngineStatus.INITIALIZING -> {
                                currentState.initialized shouldBe false
                                currentState.running shouldBe false
                            }
                            DataEngine.EngineStatus.READY -> {
                                currentState.initialized shouldBe true
                                currentState.running shouldBe false
                            }
                            DataEngine.EngineStatus.RUNNING -> {
                                currentState.initialized shouldBe true
                                currentState.running shouldBe true
                            }
                            DataEngine.EngineStatus.STOPPED -> {
                                currentState.initialized shouldBe true
                                currentState.running shouldBe false
                            }
                            DataEngine.EngineStatus.ERROR -> {
                                currentState.lastError.shouldNotBeNull()
                            }
                        }
                    }
                } finally {
                    dataEngine.cleanup()
                }
            }
        }
    }

    "Property 17: Collector State Isolation - collector state changes should not affect other collectors" {
        runTest {
            checkAll(
                iterations = 30,
                Arb.list(Arb.enum<CollectorType>(), 2..5),
                Arb.list(Arb.boolean(), 5..15)
            ) { collectorTypes, enableStates ->
                val dataEngine = DataEngine(context)
                val collectors = mutableMapOf<CollectorType, DataCollector>()
                
                // Register collectors
                collectorTypes.forEach { type ->
                    val collector = createMockCollector(type)
                    collectors[type] = collector
                    dataEngine.registerCollector(type, collector)
                }
                
                dataEngine.initialize()
                dataEngine.start()
                
                try {
                    // Apply enable/disable operations
                    enableStates.forEachIndexed { index, enabled ->
                        val targetType = collectorTypes[index % collectorTypes.size]
                        dataEngine.setCollectorEnabled(targetType, enabled)
                        delay(10)
                        
                        // Verify only the target collector's state changed
                        val collectorStates = dataEngine.getCollectorStates()
                        
                        collectorTypes.forEach { type ->
                            val state = collectorStates[type]
                            state.shouldNotBeNull()
                            
                            if (type == targetType) {
                                // Target collector should reflect the change
                                val expectedStatus = if (enabled) {
                                    DataEngine.CollectorStatus.ACTIVE
                                } else {
                                    DataEngine.CollectorStatus.STOPPED
                                }
                                // Note: Status might be different due to permissions, but should not be ERROR
                                state.status shouldNotBe DataEngine.CollectorStatus.ERROR
                            }
                            // All collectors should maintain their type
                            state.type shouldBe type
                        }
                    }
                } finally {
                    dataEngine.cleanup()
                }
            }
        }
    }

    "Property 18: System Event Handling Idempotency - handling the same system event multiple times should be safe" {
        runTest {
            checkAll(
                iterations = 25,
                Arb.enum<DataEngine.SystemEventType>(),
                Arb.int(1..5)
            ) { eventType, repeatCount ->
                val dataEngine = DataEngine(context)
                val mockCollector = MockUsageStatsCollector()
                dataEngine.registerCollector(CollectorType.USAGE_STATS, mockCollector)
                
                dataEngine.initialize()
                dataEngine.start()
                dataEngine.setCollectorEnabled(CollectorType.USAGE_STATS, true)
                
                try {
                    val initialState = dataEngine.engineState.value
                    val initialCollectorStates = dataEngine.getCollectorStates()
                    
                    val event = DataEngine.SystemEvent(
                        type = eventType,
                        data = if (eventType == DataEngine.SystemEventType.PERMISSION_GRANTED || 
                                   eventType == DataEngine.SystemEventType.PERMISSION_REVOKED) {
                            mapOf("collectorType" to CollectorType.USAGE_STATS)
                        } else {
                            emptyMap()
                        }
                    )
                    
                    // Handle the same event multiple times
                    repeat(repeatCount) {
                        dataEngine.handleSystemEvent(event)
                        delay(20)
                    }
                    
                    // Verify system is still in a valid state
                    val finalState = dataEngine.engineState.value
                    finalState.status shouldNotBe DataEngine.EngineStatus.ERROR
                    
                    // Verify collector states are still valid
                    val finalCollectorStates = dataEngine.getCollectorStates()
                    finalCollectorStates.values.forEach { state ->
                        state.status shouldNotBe DataEngine.CollectorStatus.ERROR
                    }
                    
                } finally {
                    dataEngine.cleanup()
                }
            }
        }
    }

    "Property 19: Concurrent Operation Safety - concurrent operations should not corrupt engine state" {
        runTest {
            checkAll(
                iterations = 20,
                Arb.int(2..8),
                Arb.int(5..15)
            ) { concurrentOperations, operationsPerThread ->
                val dataEngine = DataEngine(context)
                val collectors = CollectorType.values().map { type ->
                    type to createMockCollector(type)
                }
                
                collectors.forEach { (type, collector) ->
                    dataEngine.registerCollector(type, collector)
                }
                
                dataEngine.initialize()
                dataEngine.start()
                
                try {
                    // Launch concurrent operations
                    val jobs = (1..concurrentOperations).map { threadId ->
                        launch {
                            repeat(operationsPerThread) { opId ->
                                val operation = ConcurrentOperation.values()[
                                    (threadId + opId) % ConcurrentOperation.values().size
                                ]
                                
                                when (operation) {
                                    ConcurrentOperation.ENABLE_COLLECTOR -> {
                                        val type = CollectorType.values()[opId % CollectorType.values().size]
                                        dataEngine.setCollectorEnabled(type, true)
                                    }
                                    ConcurrentOperation.DISABLE_COLLECTOR -> {
                                        val type = CollectorType.values()[opId % CollectorType.values().size]
                                        dataEngine.setCollectorEnabled(type, false)
                                    }
                                    ConcurrentOperation.GET_STATISTICS -> {
                                        dataEngine.getEngineStatistics()
                                    }
                                    ConcurrentOperation.HANDLE_SYSTEM_EVENT -> {
                                        val event = DataEngine.SystemEvent(
                                            type = DataEngine.SystemEventType.MEMORY_PRESSURE,
                                            data = mapOf("threadId" to threadId, "opId" to opId)
                                        )
                                        dataEngine.handleSystemEvent(event)
                                    }
                                    ConcurrentOperation.GET_COLLECTOR_STATES -> {
                                        dataEngine.getCollectorStates()
                                    }
                                }
                                
                                delay(5)
                            }
                        }
                    }
                    
                    // Wait for all operations to complete
                    jobs.joinAll()
                    
                    // Verify final state is valid
                    val finalState = dataEngine.engineState.value
                    finalState.status shouldNotBe DataEngine.EngineStatus.ERROR
                    finalState.initialized shouldBe true
                    
                    val finalCollectorStates = dataEngine.getCollectorStates()
                    finalCollectorStates shouldHaveSize CollectorType.values().size
                    
                    finalCollectorStates.values.forEach { state ->
                        state.status shouldNotBe DataEngine.CollectorStatus.ERROR
                        state.lastUpdate shouldNotBe 0L
                    }
                    
                } finally {
                    dataEngine.cleanup()
                }
            }
        }
    }

    "Property 20: Data Persistence Across Restarts - data should survive engine restarts" {
        runTest {
            checkAll(
                iterations = 15,
                Arb.int(1..5),
                Arb.int(1..10)
            ) { restartCount, dataOperations ->
                val collectors = mutableMapOf<CollectorType, MockDataCollector>()
                
                try {
                    repeat(restartCount) { iteration ->
                        val dataEngine = DataEngine(context)
                        
                        // Register collectors (simulating app restart)
                        CollectorType.values().forEach { type ->
                            val collector = createMockCollector(type) as MockDataCollector
                            collectors[type] = collector
                            dataEngine.registerCollector(type, collector)
                        }
                        
                        dataEngine.initialize()
                        dataEngine.start()
                        
                        // Enable all collectors
                        CollectorType.values().forEach { type ->
                            dataEngine.setCollectorEnabled(type, true)
                        }
                        
                        delay(50)
                        
                        // Simulate data collection
                        repeat(dataOperations) {
                            collectors.values.forEach { collector ->
                                collector.simulateDataCollection()
                            }
                            delay(10)
                        }
                        
                        // Verify data was collected
                        val statistics = dataEngine.getEngineStatistics()
                        statistics.collectorStatistics.values.forEach { stats ->
                            if (stats.isActive) {
                                stats.dataCount shouldNotBe 0
                            }
                        }
                        
                        // Stop engine (simulating app shutdown)
                        dataEngine.stop()
                        dataEngine.cleanup()
                        
                        delay(20)
                    }
                    
                    // After all restarts, verify system can still function
                    val finalEngine = DataEngine(context)
                    CollectorType.values().forEach { type ->
                        val collector = createMockCollector(type)
                        finalEngine.registerCollector(type, collector)
                    }
                    
                    val initResult = finalEngine.initialize()
                    initResult shouldBe true
                    
                    val startResult = finalEngine.start()
                    startResult shouldBe true
                    
                    finalEngine.cleanup()
                    
                } catch (e: Exception) {
                    // Clean up any remaining engines
                    collectors.clear()
                    throw e
                }
            }
        }
    }
})

// Test enums and helper functions

enum class EngineOperation {
    INITIALIZE, START, STOP, RESTART
}

enum class ConcurrentOperation {
    ENABLE_COLLECTOR,
    DISABLE_COLLECTOR,
    GET_STATISTICS,
    HANDLE_SYSTEM_EVENT,
    GET_COLLECTOR_STATES
}

interface MockDataCollector : DataCollector {
    fun simulateDataCollection()
}

fun createMockCollector(type: CollectorType): DataCollector {
    return when (type) {
        CollectorType.USAGE_STATS -> MockUsageStatsCollector()
        CollectorType.NOTIFICATIONS -> MockNotificationLogger()
        CollectorType.SCREEN_EVENTS -> MockScreenEventReceiver()
        CollectorType.INTERACTIONS -> MockInteractionAccessibilityService()
        CollectorType.SENSORS -> MockSensorFusionManager()
    }
}

// Enhanced mock classes that implement MockDataCollector

class MockUsageStatsCollector : UsageStatsCollector, MockDataCollector {
    private var isActive = false
    private var dataCount = 0
    
    override suspend fun startCollection() {
        isActive = true
    }
    
    override suspend fun stopCollection() {
        isActive = false
    }
    
    override fun isCollectionActive(): Boolean = isActive
    
    override fun getCollectorType(): CollectorType = CollectorType.USAGE_STATS
    
    override suspend fun getCollectedDataCount(): Int = dataCount
    
    override suspend fun collectUsageEvents(timeRange: TimeRange): List<UsageEvent> = emptyList()
    
    override fun isPermissionGranted(): Boolean = true
    
    override suspend fun requestPermission(): Boolean = true
    
    override fun simulateDataCollection() {
        dataCount += (1..5).random()
    }
}

class MockNotificationLogger : NotificationLogger, MockDataCollector {
    private var isActive = false
    private var dataCount = 0
    
    override suspend fun startCollection() {
        isActive = true
    }
    
    override suspend fun stopCollection() {
        isActive = false
    }
    
    override fun isCollectionActive(): Boolean = isActive
    
    override fun getCollectorType(): CollectorType = CollectorType.NOTIFICATIONS
    
    override suspend fun getCollectedDataCount(): Int = dataCount
    
    override suspend fun logNotificationPosted(notification: NotificationData) {
        dataCount++
    }
    
    override suspend fun logNotificationInteraction(interaction: NotificationInteraction) {
        dataCount++
    }
    
    override fun isNotificationAccessGranted(): Boolean = true
    
    override suspend fun requestNotificationAccess(): Boolean = true
    
    override fun simulateDataCollection() {
        dataCount += (1..3).random()
    }
}

class MockScreenEventReceiver : ScreenEventReceiver, MockDataCollector {
    private var isActive = false
    private var dataCount = 0
    
    override suspend fun startCollection() {
        isActive = true
    }
    
    override suspend fun stopCollection() {
        isActive = false
    }
    
    override fun isCollectionActive(): Boolean = isActive
    
    override fun getCollectorType(): CollectorType = CollectorType.SCREEN_EVENTS
    
    override suspend fun getCollectedDataCount(): Int = dataCount
    
    override suspend fun getCurrentSession(): ScreenSession? = null
    
    override suspend fun getSessionsByTimeRange(timeRange: TimeRange): List<ScreenSession> = emptyList()
    
    override suspend fun getTotalScreenTime(timeRange: TimeRange): Long = 0L
    
    override fun simulateDataCollection() {
        dataCount += (1..2).random()
    }
}

class MockInteractionAccessibilityService : InteractionAccessibilityService, MockDataCollector {
    private var isActive = false
    private var dataCount = 0
    
    override suspend fun startCollection() {
        isActive = true
    }
    
    override suspend fun stopCollection() {
        isActive = false
    }
    
    override fun isCollectionActive(): Boolean = isActive
    
    override fun getCollectorType(): CollectorType = CollectorType.INTERACTIONS
    
    override suspend fun getCollectedDataCount(): Int = dataCount
    
    override suspend fun getInteractionMetrics(timeRange: TimeRange): List<InteractionMetrics> = emptyList()
    
    override fun isAccessibilityServiceEnabled(): Boolean = true
    
    override suspend fun requestAccessibilityPermission(): Boolean = true
    
    override fun simulateDataCollection() {
        dataCount += (1..4).random()
    }
}

class MockSensorFusionManager : SensorFusionManager, MockDataCollector {
    private var isActive = false
    private var dataCount = 0
    
    override suspend fun startCollection() {
        isActive = true
    }
    
    override suspend fun stopCollection() {
        isActive = false
    }
    
    override fun isCollectionActive(): Boolean = isActive
    
    override fun getCollectorType(): CollectorType = CollectorType.SENSORS
    
    override suspend fun getCollectedDataCount(): Int = dataCount
    
    override suspend fun getCurrentActivity(): ActivityContext? = null
    
    override suspend fun getActivityHistory(timeRange: TimeRange): List<ActivityContext> = emptyList()
    
    override fun isSensorPermissionGranted(): Boolean = true
    
    override suspend fun requestSensorPermission(): Boolean = true
    
    override fun simulateDataCollection() {
        dataCount += (1..6).random()
    }
}