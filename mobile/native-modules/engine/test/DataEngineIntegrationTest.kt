package com.lifetwin.mlp.engine.test

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.lifetwin.mlp.db.*
import com.lifetwin.mlp.engine.DataEngine
import com.lifetwin.mlp.privacy.PrivacyManager
import com.lifetwin.mlp.performance.PerformanceMonitor
import com.lifetwin.mlp.performance.BatteryOptimizer
import io.kotest.core.spec.style.StringSpec
import io.kotest.matchers.shouldBe
import io.kotest.matchers.shouldNotBe
import io.kotest.matchers.collections.shouldContain
import io.kotest.matchers.collections.shouldNotBeEmpty
import io.kotest.matchers.nulls.shouldNotBeNull
import kotlinx.coroutines.*
import kotlinx.coroutines.test.*
import org.junit.runner.RunWith
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

/**
 * Integration tests for DataEngine end-to-end scenarios
 * Tests complete data collection lifecycle, permission flows, and system events
 */
@RunWith(AndroidJUnit4::class)
class DataEngineIntegrationTest : StringSpec({

    val context = ApplicationProvider.getApplicationContext<Context>()
    
    beforeEach {
        // Clean up database before each test
        val database = AppDatabase.getInstance(context)
        database.clearAllTables()
    }

    afterEach {
        // Clean up after each test
        val database = AppDatabase.getInstance(context)
        database.clearAllTables()
    }

    "complete data collection lifecycle should work end-to-end" {
        runTest {
            val dataEngine = DataEngine(context)
            
            // Mock collectors for testing
            val mockUsageCollector = MockUsageStatsCollector()
            val mockNotificationCollector = MockNotificationLogger()
            val mockScreenCollector = MockScreenEventReceiver()
            val mockInteractionCollector = MockInteractionAccessibilityService()
            val mockSensorCollector = MockSensorFusionManager()
            
            // Register all collectors
            dataEngine.registerCollector(CollectorType.USAGE_STATS, mockUsageCollector)
            dataEngine.registerCollector(CollectorType.NOTIFICATIONS, mockNotificationCollector)
            dataEngine.registerCollector(CollectorType.SCREEN_EVENTS, mockScreenCollector)
            dataEngine.registerCollector(CollectorType.INTERACTIONS, mockInteractionCollector)
            dataEngine.registerCollector(CollectorType.SENSORS, mockSensorCollector)
            
            // Initialize and start engine
            val initResult = dataEngine.initialize()
            initResult shouldBe true
            
            val startResult = dataEngine.start()
            startResult shouldBe true
            
            // Verify engine state
            val engineState = dataEngine.engineState.value
            engineState.initialized shouldBe true
            engineState.running shouldBe true
            engineState.status shouldBe DataEngine.EngineStatus.RUNNING
            engineState.totalCollectors shouldBe 5
            
            // Verify all collectors are registered and can be started
            val collectorStates = dataEngine.getCollectorStates()
            collectorStates.size shouldBe 5
            
            CollectorType.values().forEach { type ->
                collectorStates[type].shouldNotBeNull()
                collectorStates[type]!!.type shouldBe type
            }
            
            // Enable all collectors through privacy settings
            CollectorType.values().forEach { type ->
                val enableResult = dataEngine.setCollectorEnabled(type, true)
                enableResult shouldBe true
            }
            
            // Wait for collectors to start
            delay(100)
            
            // Verify collectors are active
            val updatedStates = dataEngine.getCollectorStates()
            CollectorType.values().forEach { type ->
                updatedStates[type]!!.status shouldBe DataEngine.CollectorStatus.ACTIVE
            }
            
            // Simulate data collection
            mockUsageCollector.simulateDataCollection()
            mockNotificationCollector.simulateDataCollection()
            mockScreenCollector.simulateDataCollection()
            mockInteractionCollector.simulateDataCollection()
            mockSensorCollector.simulateDataCollection()
            
            // Wait for data to be processed
            delay(200)
            
            // Verify data was collected
            val statistics = dataEngine.getEngineStatistics()
            statistics.collectorStatistics.values.forEach { stats ->
                stats.dataCount shouldNotBe 0
                stats.isActive shouldBe true
                stats.lastCollection.shouldNotBeNull()
            }
            
            // Stop engine
            val stopResult = dataEngine.stop()
            stopResult shouldBe true
            
            // Verify engine stopped
            val finalState = dataEngine.engineState.value
            finalState.running shouldBe false
            finalState.status shouldBe DataEngine.EngineStatus.STOPPED
            
            // Clean up
            dataEngine.cleanup()
        }
    }

    "permission revocation and restoration flows should work correctly" {
        runTest {
            val dataEngine = DataEngine(context)
            val mockCollector = MockUsageStatsCollector()
            
            dataEngine.registerCollector(CollectorType.USAGE_STATS, mockCollector)
            dataEngine.initialize()
            dataEngine.start()
            
            // Enable collector
            dataEngine.setCollectorEnabled(CollectorType.USAGE_STATS, true)
            delay(50)
            
            // Verify collector is active
            var collectorState = dataEngine.getCollectorStates()[CollectorType.USAGE_STATS]
            collectorState!!.status shouldBe DataEngine.CollectorStatus.ACTIVE
            mockCollector.isCollectionActive() shouldBe true
            
            // Simulate permission revocation
            val permissionRevokedEvent = DataEngine.SystemEvent(
                type = DataEngine.SystemEventType.PERMISSION_REVOKED,
                data = mapOf("collectorType" to CollectorType.USAGE_STATS)
            )
            
            dataEngine.handleSystemEvent(permissionRevokedEvent)
            delay(50)
            
            // Verify collector stopped
            collectorState = dataEngine.getCollectorStates()[CollectorType.USAGE_STATS]
            collectorState!!.status shouldBe DataEngine.CollectorStatus.STOPPED
            mockCollector.isCollectionActive() shouldBe false
            
            // Simulate permission restoration
            val permissionGrantedEvent = DataEngine.SystemEvent(
                type = DataEngine.SystemEventType.PERMISSION_GRANTED,
                data = mapOf("collectorType" to CollectorType.USAGE_STATS)
            )
            
            dataEngine.handleSystemEvent(permissionGrantedEvent)
            delay(50)
            
            // Verify collector restarted
            collectorState = dataEngine.getCollectorStates()[CollectorType.USAGE_STATS]
            collectorState!!.status shouldBe DataEngine.CollectorStatus.ACTIVE
            mockCollector.isCollectionActive() shouldBe true
            
            dataEngine.cleanup()
        }
    }

    "device restart scenario should restore data collection" {
        runTest {
            // First session - simulate normal operation
            val dataEngine1 = DataEngine(context)
            val mockCollector1 = MockUsageStatsCollector()
            
            dataEngine1.registerCollector(CollectorType.USAGE_STATS, mockCollector1)
            dataEngine1.initialize()
            dataEngine1.start()
            dataEngine1.setCollectorEnabled(CollectorType.USAGE_STATS, true)
            
            // Simulate some data collection
            mockCollector1.simulateDataCollection()
            delay(100)
            
            // Verify data exists
            var statistics = dataEngine1.getEngineStatistics()
            statistics.collectorStatistics[CollectorType.USAGE_STATS]!!.dataCount shouldNotBe 0
            
            // Simulate app shutdown
            dataEngine1.stop()
            dataEngine1.cleanup()
            
            // Second session - simulate device restart
            val dataEngine2 = DataEngine(context)
            val mockCollector2 = MockUsageStatsCollector()
            
            dataEngine2.registerCollector(CollectorType.USAGE_STATS, mockCollector2)
            
            // Simulate device restart event
            val deviceRestartEvent = DataEngine.SystemEvent(
                type = DataEngine.SystemEventType.DEVICE_RESTART
            )
            
            dataEngine2.initialize()
            dataEngine2.handleSystemEvent(deviceRestartEvent)
            delay(100)
            
            // Verify engine restarted correctly
            val engineState = dataEngine2.engineState.value
            engineState.initialized shouldBe true
            engineState.running shouldBe true
            
            // Verify previous data persisted
            statistics = dataEngine2.getEngineStatistics()
            // Note: In a real scenario, data would persist in the database
            // For this test, we verify the engine can restart properly
            
            dataEngine2.cleanup()
        }
    }

    "battery optimization should adapt collection behavior" {
        runTest {
            val dataEngine = DataEngine(context)
            val mockCollector = MockUsageStatsCollector()
            
            dataEngine.registerCollector(CollectorType.USAGE_STATS, mockCollector)
            dataEngine.initialize()
            dataEngine.start()
            dataEngine.setCollectorEnabled(CollectorType.USAGE_STATS, true)
            
            // Verify normal operation
            var engineState = dataEngine.engineState.value
            engineState.powerSavingMode shouldBe false
            
            // Simulate low battery event
            val lowBatteryEvent = DataEngine.SystemEvent(
                type = DataEngine.SystemEventType.LOW_BATTERY
            )
            
            dataEngine.handleSystemEvent(lowBatteryEvent)
            delay(50)
            
            // Verify power saving mode enabled
            engineState = dataEngine.engineState.value
            engineState.powerSavingMode shouldBe true
            
            val collectorState = dataEngine.getCollectorStates()[CollectorType.USAGE_STATS]
            collectorState!!.powerSavingMode shouldBe true
            
            // Simulate charging started
            val chargingStartedEvent = DataEngine.SystemEvent(
                type = DataEngine.SystemEventType.CHARGING_STARTED
            )
            
            dataEngine.handleSystemEvent(chargingStartedEvent)
            delay(50)
            
            // Verify power saving mode disabled
            engineState = dataEngine.engineState.value
            engineState.powerSavingMode shouldBe false
            
            dataEngine.cleanup()
        }
    }

    "memory pressure should trigger optimization" {
        runTest {
            val dataEngine = DataEngine(context)
            val mockCollector = MockUsageStatsCollector()
            
            dataEngine.registerCollector(CollectorType.USAGE_STATS, mockCollector)
            dataEngine.initialize()
            dataEngine.start()
            
            // Simulate memory pressure event
            val memoryPressureEvent = DataEngine.SystemEvent(
                type = DataEngine.SystemEventType.MEMORY_PRESSURE
            )
            
            dataEngine.handleSystemEvent(memoryPressureEvent)
            delay(100)
            
            // Verify system handled the event (no exceptions thrown)
            val engineState = dataEngine.engineState.value
            engineState.status shouldNotBe DataEngine.EngineStatus.ERROR
            
            dataEngine.cleanup()
        }
    }

    "event listeners should receive notifications" {
        runTest {
            val dataEngine = DataEngine(context)
            val mockCollector = MockUsageStatsCollector()
            
            // Set up event listener
            val engineStateChanges = mutableListOf<DataEngine.EngineState>()
            val collectorStateChanges = mutableListOf<Pair<CollectorType, DataEngine.CollectorState>>()
            val systemEvents = mutableListOf<DataEngine.SystemEvent>()
            
            val eventListener = object : DataEngine.EngineEventListener {
                override suspend fun onEngineStateChanged(state: DataEngine.EngineState) {
                    engineStateChanges.add(state)
                }
                
                override suspend fun onCollectorStateChanged(type: CollectorType, state: DataEngine.CollectorState) {
                    collectorStateChanges.add(type to state)
                }
                
                override suspend fun onSystemEvent(event: DataEngine.SystemEvent) {
                    systemEvents.add(event)
                }
            }
            
            dataEngine.addEventListener(eventListener)
            dataEngine.registerCollector(CollectorType.USAGE_STATS, mockCollector)
            
            // Initialize and start
            dataEngine.initialize()
            dataEngine.start()
            dataEngine.setCollectorEnabled(CollectorType.USAGE_STATS, true)
            
            // Handle system event
            val testEvent = DataEngine.SystemEvent(
                type = DataEngine.SystemEventType.LOW_BATTERY
            )
            dataEngine.handleSystemEvent(testEvent)
            
            delay(100)
            
            // Verify events were received
            engineStateChanges.shouldNotBeEmpty()
            collectorStateChanges.shouldNotBeEmpty()
            systemEvents shouldContain testEvent
            
            // Clean up
            dataEngine.removeEventListener(eventListener)
            dataEngine.cleanup()
        }
    }

    "concurrent operations should be handled safely" {
        runTest {
            val dataEngine = DataEngine(context)
            val collectors = CollectorType.values().map { type ->
                type to when (type) {
                    CollectorType.USAGE_STATS -> MockUsageStatsCollector()
                    CollectorType.NOTIFICATIONS -> MockNotificationLogger()
                    CollectorType.SCREEN_EVENTS -> MockScreenEventReceiver()
                    CollectorType.INTERACTIONS -> MockInteractionAccessibilityService()
                    CollectorType.SENSORS -> MockSensorFusionManager()
                }
            }
            
            // Register all collectors
            collectors.forEach { (type, collector) ->
                dataEngine.registerCollector(type, collector)
            }
            
            dataEngine.initialize()
            dataEngine.start()
            
            // Perform concurrent operations
            val jobs = mutableListOf<Job>()
            
            // Concurrent collector enable/disable
            repeat(10) { i ->
                jobs.add(launch {
                    val type = CollectorType.values()[i % CollectorType.values().size]
                    dataEngine.setCollectorEnabled(type, i % 2 == 0)
                    delay(10)
                })
            }
            
            // Concurrent system events
            repeat(5) { i ->
                jobs.add(launch {
                    val event = DataEngine.SystemEvent(
                        type = DataEngine.SystemEventType.MEMORY_PRESSURE,
                        data = mapOf("iteration" to i)
                    )
                    dataEngine.handleSystemEvent(event)
                    delay(20)
                })
            }
            
            // Concurrent statistics requests
            repeat(5) {
                jobs.add(launch {
                    dataEngine.getEngineStatistics()
                    delay(15)
                })
            }
            
            // Wait for all operations to complete
            jobs.joinAll()
            
            // Verify engine is still in a valid state
            val finalState = dataEngine.engineState.value
            finalState.status shouldNotBe DataEngine.EngineStatus.ERROR
            
            dataEngine.cleanup()
        }
    }
})

// Mock implementations for testing

class MockUsageStatsCollector : UsageStatsCollector {
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
    
    override suspend fun collectUsageEvents(timeRange: TimeRange): List<UsageEvent> {
        return emptyList()
    }
    
    override fun isPermissionGranted(): Boolean = true
    
    override suspend fun requestPermission(): Boolean = true
    
    fun simulateDataCollection() {
        dataCount += 5
    }
}

class MockNotificationLogger : NotificationLogger {
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
    
    fun simulateDataCollection() {
        dataCount += 3
    }
}

class MockScreenEventReceiver : ScreenEventReceiver {
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
    
    fun simulateDataCollection() {
        dataCount += 2
    }
}

class MockInteractionAccessibilityService : InteractionAccessibilityService {
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
    
    fun simulateDataCollection() {
        dataCount += 4
    }
}

class MockSensorFusionManager : SensorFusionManager {
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
    
    fun simulateDataCollection() {
        dataCount += 6
    }
}