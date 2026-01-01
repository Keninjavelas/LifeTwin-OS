package com.lifetwin.mlp.summary.test

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.work.testing.TestListenableWorkerBuilder
import com.lifetwin.mlp.db.*
import com.lifetwin.mlp.summary.DailySummaryWorker
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import io.kotest.property.Arb
import io.kotest.property.arbitrary.*
import io.kotest.property.checkAll
import kotlinx.coroutines.runBlocking
import org.junit.After
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import java.text.SimpleDateFormat
import java.util.*
import kotlin.math.abs
import kotlin.test.assertTrue

@RunWith(AndroidJUnit4::class)
class DailySummaryWorkerPropertyTest {

    private lateinit var context: Context
    private lateinit var database: AppDatabase
    private val gson = Gson()
    private val dateFormat = SimpleDateFormat("yyyy-MM-dd", Locale.US)

    @Before
    fun setUp() {
        context = ApplicationProvider.getApplicationContext()
        database = AppDatabase.getInstance(context)
    }

    @After
    fun tearDown() {
        // Clean up test data
        runBlocking {
            database.clearAllTables()
        }
    }

    /**
     * Property 5: Aggregation Accuracy
     * Tests that summary aggregation accurately reflects the underlying raw data
     */
    @Test
    fun `property test - aggregation accuracy for screen time calculation`() = runBlocking {
        checkAll<List<ScreenSessionTestData>>(
            iterations = 100,
            Arb.list(
                Arb.bind(
                    Arb.long(1000L, 10000L), // session duration 1-10 seconds
                    Arb.int(0, 5), // unlock count
                    Arb.float(0f, 1f) // interaction intensity
                ) { duration, unlocks, intensity ->
                    ScreenSessionTestData(duration, unlocks, intensity)
                },
                range = 1..20
            )
        ) { sessionData ->
            val testDate = "2024-01-15"
            val baseTime = dateFormat.parse(testDate)!!.time
            
            // Insert test screen sessions
            val expectedTotalScreenTime = sessionData.sumOf { it.duration }
            val expectedUnlockCount = sessionData.sumOf { it.unlockCount }
            
            sessionData.forEachIndexed { index, data ->
                val startTime = baseTime + (index * 60000L) // 1 minute apart
                val endTime = startTime + data.duration
                
                val session = ScreenSessionEntity(
                    sessionId = "test_session_$index",
                    startTime = startTime,
                    endTime = endTime,
                    unlockCount = data.unlockCount,
                    interactionIntensity = data.interactionIntensity,
                    isActive = false
                )
                
                database.screenSessionDao().insert(session)
            }
            
            // Generate summary
            val worker = TestListenableWorkerBuilder<DailySummaryWorker>(context).build()
            val result = worker.doWork()
            
            // Verify aggregation accuracy
            val summary = database.enhancedDailySummaryDao().getSummaryByDate(testDate)
            
            if (summary != null && sessionData.isNotEmpty()) {
                val accuracyTolerance = 1000L // 1 second tolerance
                assertTrue(
                    abs(summary.totalScreenTime - expectedTotalScreenTime) <= accuracyTolerance,
                    "Screen time aggregation should be accurate. Expected: $expectedTotalScreenTime, " +
                    "Got: ${summary.totalScreenTime}, Difference: ${abs(summary.totalScreenTime - expectedTotalScreenTime)}"
                )
            }
        }
    }

    /**
     * Property 6: Privacy Preservation Through Aggregation
     * Tests that raw events are properly cleaned up after aggregation while preserving summary data
     */
    @Test
    fun `property test - privacy preservation through raw event cleanup`() = runBlocking {
        checkAll<List<RawEventTestData>>(
            iterations = 80,
            Arb.list(
                Arb.bind(
                    Arb.string(5..20),
                    Arb.string(5..50),
                    Arb.long(1000L, 86400000L) // 1 second to 1 day
                ) { eventType, packageName, duration ->
                    RawEventTestData(eventType, packageName, duration)
                },
                range = 5..50
            )
        ) { rawEventData ->
            val testDate = "2024-01-15"
            val baseTime = dateFormat.parse(testDate)!!.time
            
            // Insert test raw events
            val insertedEventIds = mutableListOf<String>()
            
            rawEventData.forEachIndexed { index, data ->
                val eventId = "test_event_$index"
                val timestamp = baseTime + (index * 60000L)
                
                val rawEvent = RawEventEntity(
                    id = eventId,
                    timestamp = timestamp,
                    eventType = data.eventType,
                    packageName = data.packageName,
                    duration = data.duration,
                    metadata = gson.toJson(mapOf("test" to true)),
                    processed = false
                )
                
                database.rawEventDao().insert(rawEvent)
                insertedEventIds.add(eventId)
            }
            
            val initialRawEventCount = database.rawEventDao().getUnprocessedEventCount()
            
            // Generate summary (which should trigger cleanup)
            val worker = TestListenableWorkerBuilder<DailySummaryWorker>(context).build()
            val result = worker.doWork()
            
            // Verify privacy preservation
            val finalRawEventCount = database.rawEventDao().getUnprocessedEventCount()
            val summary = database.enhancedDailySummaryDao().getSummaryByDate(testDate)
            
            if (rawEventData.isNotEmpty()) {
                // Raw events should be marked as processed (privacy preservation)
                assertTrue(
                    finalRawEventCount < initialRawEventCount,
                    "Raw events should be processed/cleaned up after aggregation. " +
                    "Initial: $initialRawEventCount, Final: $finalRawEventCount"
                )
                
                // But summary should still exist (data preservation)
                assertTrue(
                    summary != null,
                    "Summary should be created even after raw event cleanup"
                )
            }
        }
    }

    /**
     * Property 15: Weekly Summary Aggregation
     * Tests that weekly summaries can be accurately generated from daily summaries
     */
    @Test
    fun `property test - weekly summary aggregation from daily summaries`() = runBlocking {
        checkAll<List<DailySummaryTestData>>(
            iterations = 60,
            Arb.list(
                Arb.bind(
                    Arb.long(0L, 86400000L), // 0 to 24 hours screen time
                    Arb.int(0, 200), // notification count
                    Arb.float(0f, 1f), // interaction intensity
                    Arb.int(0, 23) // peak usage hour
                ) { screenTime, notifications, intensity, peakHour ->
                    DailySummaryTestData(screenTime, notifications, intensity, peakHour)
                },
                range = 7..7 // Exactly 7 days for a week
            )
        ) { dailyData ->
            val baseDate = Calendar.getInstance().apply {
                set(2024, Calendar.JANUARY, 15, 0, 0, 0)
                set(Calendar.MILLISECOND, 0)
            }
            
            // Insert daily summaries for a week
            val expectedWeeklyScreenTime = dailyData.sumOf { it.screenTime }
            val expectedWeeklyNotifications = dailyData.sumOf { it.notificationCount }
            val expectedAvgIntensity = dailyData.map { it.interactionIntensity }.average().toFloat()
            
            dailyData.forEachIndexed { dayIndex, data ->
                val currentDate = Calendar.getInstance().apply {
                    timeInMillis = baseDate.timeInMillis
                    add(Calendar.DAY_OF_MONTH, dayIndex)
                }
                
                val dateString = dateFormat.format(currentDate.time)
                
                val dailySummary = EnhancedDailySummaryEntity(
                    date = dateString,
                    totalScreenTime = data.screenTime,
                    appUsageDistribution = DBHelper.encryptMetadata("{}"),
                    notificationCount = data.notificationCount,
                    peakUsageHour = data.peakUsageHour,
                    activityBreakdown = DBHelper.encryptMetadata("{}"),
                    interactionIntensity = data.interactionIntensity
                )
                
                database.enhancedDailySummaryDao().insert(dailySummary)
            }
            
            // Calculate weekly aggregation
            val startDate = dateFormat.format(baseDate.time)
            val endDate = Calendar.getInstance().apply {
                timeInMillis = baseDate.timeInMillis
                add(Calendar.DAY_OF_MONTH, 6)
            }.let { dateFormat.format(it.time) }
            
            val weeklySummaries = database.enhancedDailySummaryDao()
                .getSummariesByDateRange(startDate, endDate)
            
            if (weeklySummaries.size == 7) {
                val actualWeeklyScreenTime = weeklySummaries.sumOf { it.totalScreenTime }
                val actualWeeklyNotifications = weeklySummaries.sumOf { it.notificationCount }
                val actualAvgIntensity = weeklySummaries.map { it.interactionIntensity }.average().toFloat()
                
                // Verify weekly aggregation accuracy
                assertTrue(
                    actualWeeklyScreenTime == expectedWeeklyScreenTime,
                    "Weekly screen time should equal sum of daily screen times. " +
                    "Expected: $expectedWeeklyScreenTime, Got: $actualWeeklyScreenTime"
                )
                
                assertTrue(
                    actualWeeklyNotifications == expectedWeeklyNotifications,
                    "Weekly notification count should equal sum of daily counts. " +
                    "Expected: $expectedWeeklyNotifications, Got: $actualWeeklyNotifications"
                )
                
                val intensityTolerance = 0.1f
                assertTrue(
                    abs(actualAvgIntensity - expectedAvgIntensity) <= intensityTolerance,
                    "Weekly average interaction intensity should be accurate. " +
                    "Expected: $expectedAvgIntensity, Got: $actualAvgIntensity"
                )
            }
        }
    }

    /**
     * Property Test: App Usage Anonymization
     * Tests that app usage data is properly anonymized in summaries
     */
    @Test
    fun `property test - app usage anonymization preserves categories`() = runBlocking {
        checkAll<List<AppUsageTestData>>(
            iterations = 70,
            Arb.list(
                Arb.bind(
                    Arb.element(listOf(
                        "com.android.chrome",
                        "com.facebook.katana", 
                        "com.spotify.music",
                        "com.whatsapp",
                        "com.instagram.android",
                        "com.google.android.gm",
                        "com.random.app.name"
                    )),
                    Arb.long(1000L, 3600000L), // 1 second to 1 hour
                    Arb.int(1, 20) // launch count
                ) { packageName, foregroundTime, launches ->
                    AppUsageTestData(packageName, foregroundTime, launches)
                },
                range = 3..15
            )
        ) { appUsageData ->
            val testDate = "2024-01-15"
            val baseTime = dateFormat.parse(testDate)!!.time
            
            // Insert usage events
            appUsageData.forEachIndexed { index, data ->
                val usageEvent = UsageEventEntity(
                    id = "usage_$index",
                    packageName = data.packageName,
                    startTime = baseTime + (index * 60000L),
                    endTime = baseTime + (index * 60000L) + data.foregroundTime,
                    totalTimeInForeground = data.foregroundTime,
                    lastTimeUsed = baseTime + (index * 60000L),
                    eventType = "ACTIVITY_RESUMED"
                )
                
                database.usageEventDao().insert(usageEvent)
            }
            
            // Generate summary
            val worker = TestListenableWorkerBuilder<DailySummaryWorker>(context).build()
            val result = worker.doWork()
            
            // Verify anonymization
            val summary = database.enhancedDailySummaryDao().getSummaryByDate(testDate)
            
            if (summary != null) {
                val appUsageJson = DBHelper.decryptMetadata(summary.appUsageDistribution)
                val appUsageMap: Map<String, Map<String, Any>> = gson.fromJson(
                    appUsageJson,
                    object : TypeToken<Map<String, Map<String, Any>>>() {}.type
                )
                
                // Verify that package names are anonymized to categories
                val expectedCategories = setOf("browser", "social", "media", "communication", "other")
                val actualCategories = appUsageMap.keys
                
                assertTrue(
                    actualCategories.all { it in expectedCategories },
                    "All app usage should be anonymized to predefined categories. " +
                    "Found categories: $actualCategories, Expected categories: $expectedCategories"
                )
                
                // Verify that usage data is preserved within categories
                val totalOriginalTime = appUsageData.sumOf { it.foregroundTime }
                val totalCategorizedTime = appUsageMap.values.sumOf { 
                    (it["totalForegroundTime"] as? Number)?.toLong() ?: 0L 
                }
                
                assertTrue(
                    totalCategorizedTime == totalOriginalTime,
                    "Total usage time should be preserved during anonymization. " +
                    "Original: $totalOriginalTime, Categorized: $totalCategorizedTime"
                )
            }
        }
    }

    /**
     * Property Test: Summary Generation Idempotency
     * Tests that generating summaries multiple times for the same date produces consistent results
     */
    @Test
    fun `property test - summary generation idempotency`() = runBlocking {
        checkAll<SummaryTestScenario>(
            iterations = 50,
            Arb.bind(
                Arb.long(0L, 86400000L),
                Arb.int(0, 100),
                Arb.float(0f, 1f)
            ) { screenTime, notifications, intensity ->
                SummaryTestScenario(screenTime, notifications, intensity)
            }
        ) { scenario ->
            val testDate = "2024-01-15"
            val baseTime = dateFormat.parse(testDate)!!.time
            
            // Insert consistent test data
            val screenSession = ScreenSessionEntity(
                sessionId = "test_session",
                startTime = baseTime,
                endTime = baseTime + scenario.screenTime,
                unlockCount = 1,
                interactionIntensity = scenario.intensity,
                isActive = false
            )
            database.screenSessionDao().insert(screenSession)
            
            repeat(scenario.notificationCount) { index ->
                val notification = NotificationEventEntity(
                    id = "notification_$index",
                    packageName = "test.app",
                    timestamp = baseTime + (index * 1000L),
                    category = "test",
                    priority = 0,
                    hasActions = false,
                    isOngoing = false,
                    interactionType = "posted"
                )
                database.notificationEventDao().insert(notification)
            }
            
            // Generate summary first time
            val worker1 = TestListenableWorkerBuilder<DailySummaryWorker>(context).build()
            worker1.doWork()
            
            val summary1 = database.enhancedDailySummaryDao().getSummaryByDate(testDate)
            
            // Generate summary second time (should be idempotent)
            val worker2 = TestListenableWorkerBuilder<DailySummaryWorker>(context).build()
            worker2.doWork()
            
            val summary2 = database.enhancedDailySummaryDao().getSummaryByDate(testDate)
            
            // Verify idempotency
            if (summary1 != null && summary2 != null) {
                assertTrue(
                    summary1.totalScreenTime == summary2.totalScreenTime,
                    "Screen time should be consistent across multiple summary generations. " +
                    "First: ${summary1.totalScreenTime}, Second: ${summary2.totalScreenTime}"
                )
                
                assertTrue(
                    summary1.notificationCount == summary2.notificationCount,
                    "Notification count should be consistent across multiple summary generations. " +
                    "First: ${summary1.notificationCount}, Second: ${summary2.notificationCount}"
                )
                
                val intensityTolerance = 0.01f
                assertTrue(
                    abs(summary1.interactionIntensity - summary2.interactionIntensity) <= intensityTolerance,
                    "Interaction intensity should be consistent across multiple summary generations. " +
                    "First: ${summary1.interactionIntensity}, Second: ${summary2.interactionIntensity}"
                )
            }
        }
    }

    // Data classes for property testing

    private data class ScreenSessionTestData(
        val duration: Long,
        val unlockCount: Int,
        val interactionIntensity: Float
    )

    private data class RawEventTestData(
        val eventType: String,
        val packageName: String,
        val duration: Long
    )

    private data class DailySummaryTestData(
        val screenTime: Long,
        val notificationCount: Int,
        val interactionIntensity: Float,
        val peakUsageHour: Int
    )

    private data class AppUsageTestData(
        val packageName: String,
        val foregroundTime: Long,
        val launchCount: Int
    )

    private data class SummaryTestScenario(
        val screenTime: Long,
        val notificationCount: Int,
        val intensity: Float
    )
}