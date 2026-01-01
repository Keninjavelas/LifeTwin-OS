package com.lifetwin.mlp.summary

import android.content.Context
import android.util.Log
import androidx.work.*
import com.lifetwin.mlp.db.*
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.TimeUnit
import kotlin.math.roundToInt

private const val TAG = "DailySummaryWorker"
private const val WORK_NAME = "daily_summary_work"
private const val MAX_RETRY_ATTEMPTS = 3
private const val BASE_BACKOFF_DELAY_MS = 1000L

/**
 * Background worker for generating daily summaries with privacy-preserving aggregation
 * - Processes raw events into aggregated daily summaries
 * - Implements privacy-preserving cleanup of raw events after aggregation
 * - Handles retry logic and error recovery
 * - Optimizes for battery efficiency with intelligent scheduling
 */
class DailySummaryWorker(
    context: Context,
    params: WorkerParameters
) : CoroutineWorker(context, params) {

    private val gson = Gson()
    private val dateFormat = SimpleDateFormat("yyyy-MM-dd", Locale.US)

    override suspend fun doWork(): Result = withContext(Dispatchers.IO) {
        try {
            Log.i(TAG, "Starting daily summary generation")
            
            val database = AppDatabase.getInstance(applicationContext)
            val targetDate = inputData.getString(KEY_TARGET_DATE) 
                ?: dateFormat.format(Date(System.currentTimeMillis() - 24 * 60 * 60 * 1000L))
            
            // Generate summary for the target date
            val summary = generateDailySummary(targetDate, database)
            
            if (summary != null && validateSummaryData(summary)) {
                // Store the summary
                database.enhancedDailySummaryDao().insert(summary)
                
                // Clean up processed raw events (privacy-preserving)
                cleanupProcessedEvents(targetDate, database)
                
                // Log audit event
                logAuditEvent(database, "Daily summary generated for $targetDate")
                
                Log.i(TAG, "Daily summary generated successfully for $targetDate")
                Result.success()
            } else if (summary != null) {
                Log.w(TAG, "Generated summary failed validation for $targetDate")
                
                // Store fallback summary
                val fallbackSummary = createFallbackSummary(targetDate)
                database.enhancedDailySummaryDao().insert(fallbackSummary)
                
                logAuditEvent(database, "Fallback summary created for $targetDate due to validation failure")
                Result.success()
            } else {
                Log.w(TAG, "No data available for daily summary on $targetDate")
                
                // Create minimal summary to maintain consistency
                val fallbackSummary = createFallbackSummary(targetDate)
                database.enhancedDailySummaryDao().insert(fallbackSummary)
                
                logAuditEvent(database, "Fallback summary created for $targetDate due to no data")
                Result.success() // Not a failure, just no data
            }
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to generate daily summary", e)
            
            // Log the error for debugging
            logErrorEvent(e, runAttemptCount)
            
            // Determine if we should retry based on the error type
            when {
                e is OutOfMemoryError -> {
                    Log.e(TAG, "Out of memory during summary generation - attempting cleanup")
                    performEmergencyCleanup()
                    Result.failure()
                }
                e is SecurityException -> {
                    Log.e(TAG, "Security exception during summary generation - permissions issue")
                    Result.failure()
                }
                e is IllegalStateException && e.message?.contains("database") == true -> {
                    Log.w(TAG, "Database state issue - will retry with exponential backoff")
                    if (runAttemptCount < MAX_RETRY_ATTEMPTS) {
                        Result.retry()
                    } else {
                        Result.failure()
                    }
                }
                isTransientError(e) && runAttemptCount < MAX_RETRY_ATTEMPTS -> {
                    val delayMs = calculateBackoffDelay(runAttemptCount)
                    Log.w(TAG, "Transient error - retrying in ${delayMs}ms (attempt ${runAttemptCount + 1})")
                    Result.retry()
                }
                runAttemptCount >= MAX_RETRY_ATTEMPTS -> {
                    Log.e(TAG, "Max retry attempts ($MAX_RETRY_ATTEMPTS) reached for daily summary generation")
                    Result.failure()
                }
                else -> {
                    Log.w(TAG, "Retrying daily summary generation (attempt ${runAttemptCount + 1})")
                    Result.retry()
                }
            }
        }
    }

    /**
     * Generates comprehensive daily summary from raw events
     */
    private suspend fun generateDailySummary(
        targetDate: String,
        database: AppDatabase
    ): EnhancedDailySummaryEntity? {
        
        val calendar = Calendar.getInstance()
        calendar.time = dateFormat.parse(targetDate) ?: return null
        
        val startOfDay = calendar.timeInMillis
        calendar.add(Calendar.DAY_OF_MONTH, 1)
        val endOfDay = calendar.timeInMillis
        
        Log.d(TAG, "Generating summary for $targetDate ($startOfDay to $endOfDay)")
        
        // Collect all raw events for the day
        val rawEvents = database.rawEventDao().getEventsByTimeRange(startOfDay, endOfDay)
        
        if (rawEvents.isEmpty()) {
            Log.d(TAG, "No raw events found for $targetDate")
            return null
        }
        
        // Calculate screen time metrics
        val screenTimeMetrics = calculateScreenTimeMetrics(database, startOfDay, endOfDay)
        
        // Calculate app usage distribution
        val appUsageDistribution = calculateAppUsageDistribution(database, startOfDay, endOfDay)
        
        // Calculate notification metrics
        val notificationMetrics = calculateNotificationMetrics(database, startOfDay, endOfDay)
        
        // Calculate activity breakdown
        val activityBreakdown = calculateActivityBreakdown(database, startOfDay, endOfDay)
        
        // Calculate interaction intensity
        val interactionIntensity = calculateInteractionIntensity(database, startOfDay, endOfDay)
        
        // Find peak usage hour
        val peakUsageHour = findPeakUsageHour(database, startOfDay, endOfDay)
        
        return EnhancedDailySummaryEntity(
            date = targetDate,
            totalScreenTime = screenTimeMetrics.totalScreenTime,
            appUsageDistribution = DBHelper.encryptMetadata(gson.toJson(appUsageDistribution)),
            notificationCount = notificationMetrics.totalCount,
            peakUsageHour = peakUsageHour,
            activityBreakdown = DBHelper.encryptMetadata(gson.toJson(activityBreakdown)),
            interactionIntensity = interactionIntensity,
            createdAt = System.currentTimeMillis(),
            version = 1
        )
    }

    /**
     * Calculates screen time metrics for the day
     */
    private suspend fun calculateScreenTimeMetrics(
        database: AppDatabase,
        startTime: Long,
        endTime: Long
    ): ScreenTimeMetrics {
        
        val screenSessions = database.screenSessionDao().getSessionsByTimeRange(startTime, endTime)
        
        val totalScreenTime = screenSessions
            .filter { it.endTime != null }
            .sumOf { it.endTime!! - it.startTime }
        
        val sessionCount = screenSessions.size
        val averageSessionLength = if (sessionCount > 0) totalScreenTime / sessionCount else 0L
        val totalUnlocks = screenSessions.sumOf { it.unlockCount }
        
        return ScreenTimeMetrics(
            totalScreenTime = totalScreenTime,
            sessionCount = sessionCount,
            averageSessionLength = averageSessionLength,
            totalUnlocks = totalUnlocks
        )
    }

    /**
     * Calculates app usage distribution with privacy preservation
     */
    private suspend fun calculateAppUsageDistribution(
        database: AppDatabase,
        startTime: Long,
        endTime: Long
    ): Map<String, AppUsageStats> {
        
        val usageEvents = database.usageEventDao().getEventsByTimeRange(startTime, endTime)
        
        // Group by package name and calculate usage statistics
        val appUsageMap = mutableMapOf<String, MutableList<UsageEventEntity>>()
        
        usageEvents.forEach { event ->
            val packageName = anonymizePackageName(event.packageName)
            appUsageMap.getOrPut(packageName) { mutableListOf() }.add(event)
        }
        
        return appUsageMap.mapValues { (_, events) ->
            val totalForegroundTime = events.sumOf { it.totalTimeInForeground }
            val launchCount = events.count { it.eventType == "ACTIVITY_RESUMED" }
            val lastUsed = events.maxOfOrNull { it.lastTimeUsed } ?: 0L
            
            AppUsageStats(
                totalForegroundTime = totalForegroundTime,
                launchCount = launchCount,
                lastUsed = lastUsed
            )
        }
    }

    /**
     * Calculates notification metrics for the day
     */
    private suspend fun calculateNotificationMetrics(
        database: AppDatabase,
        startTime: Long,
        endTime: Long
    ): NotificationMetrics {
        
        val notificationEvents = database.notificationEventDao().getEventsByTimeRange(startTime, endTime)
        
        val totalCount = notificationEvents.size
        val interactedCount = notificationEvents.count { 
            it.interactionType in listOf("opened", "dismissed", "action_clicked") 
        }
        val ongoingCount = notificationEvents.count { it.isOngoing }
        
        // Calculate hourly distribution
        val hourlyDistribution = IntArray(24)
        notificationEvents.forEach { event ->
            val hour = Calendar.getInstance().apply { 
                timeInMillis = event.timestamp 
            }.get(Calendar.HOUR_OF_DAY)
            hourlyDistribution[hour]++
        }
        
        return NotificationMetrics(
            totalCount = totalCount,
            interactedCount = interactedCount,
            ongoingCount = ongoingCount,
            hourlyDistribution = hourlyDistribution.toList()
        )
    }

    /**
     * Calculates activity breakdown from sensor data
     */
    private suspend fun calculateActivityBreakdown(
        database: AppDatabase,
        startTime: Long,
        endTime: Long
    ): Map<String, ActivityStats> {
        
        val activityContexts = database.activityContextDao().getContextsByTimeRange(startTime, endTime)
        
        val activityMap = mutableMapOf<String, MutableList<ActivityContextEntity>>()
        
        activityContexts.forEach { context ->
            activityMap.getOrPut(context.activityType) { mutableListOf() }.add(context)
        }
        
        return activityMap.mapValues { (_, contexts) ->
            val totalDuration = contexts.sumOf { it.duration }
            val averageConfidence = contexts.map { it.confidence }.average().toFloat()
            val occurrenceCount = contexts.size
            
            ActivityStats(
                totalDuration = totalDuration,
                averageConfidence = averageConfidence,
                occurrenceCount = occurrenceCount
            )
        }
    }

    /**
     * Calculates interaction intensity for the day
     */
    private suspend fun calculateInteractionIntensity(
        database: AppDatabase,
        startTime: Long,
        endTime: Long
    ): Float {
        
        val interactionMetrics = database.interactionMetricsDao().getMetricsByTimeRange(startTime, endTime)
        
        if (interactionMetrics.isEmpty()) return 0f
        
        val totalTouches = interactionMetrics.sumOf { it.touchCount }
        val totalScrolls = interactionMetrics.sumOf { it.scrollEvents }
        val averageIntensity = interactionMetrics.map { it.interactionIntensity }.average()
        
        // Normalize interaction intensity (0.0 to 1.0)
        val normalizedIntensity = (totalTouches + totalScrolls * 2) / (interactionMetrics.size * 100f)
        
        return minOf(1f, maxOf(0f, normalizedIntensity.toFloat()))
    }

    /**
     * Finds the peak usage hour of the day
     */
    private suspend fun findPeakUsageHour(
        database: AppDatabase,
        startTime: Long,
        endTime: Long
    ): Int {
        
        val usageEvents = database.usageEventDao().getEventsByTimeRange(startTime, endTime)
        val screenSessions = database.screenSessionDao().getSessionsByTimeRange(startTime, endTime)
        
        val hourlyActivity = IntArray(24)
        
        // Count usage events per hour
        usageEvents.forEach { event ->
            val hour = Calendar.getInstance().apply { 
                timeInMillis = event.startTime 
            }.get(Calendar.HOUR_OF_DAY)
            hourlyActivity[hour]++
        }
        
        // Add screen session activity
        screenSessions.forEach { session ->
            val hour = Calendar.getInstance().apply { 
                timeInMillis = session.startTime 
            }.get(Calendar.HOUR_OF_DAY)
            hourlyActivity[hour] += 2 // Weight screen sessions more heavily
        }
        
        return hourlyActivity.indices.maxByOrNull { hourlyActivity[it] } ?: 12
    }

    /**
     * Cleans up processed raw events after aggregation (privacy-preserving)
     */
    private suspend fun cleanupProcessedEvents(
        targetDate: String,
        database: AppDatabase
    ) {
        try {
            val calendar = Calendar.getInstance()
            calendar.time = dateFormat.parse(targetDate) ?: return
            
            val startOfDay = calendar.timeInMillis
            calendar.add(Calendar.DAY_OF_MONTH, 1)
            val endOfDay = calendar.timeInMillis
            
            // Get events to be cleaned up
            val eventsToCleanup = database.rawEventDao().getEventsByTimeRange(startOfDay, endOfDay)
            
            if (eventsToCleanup.isNotEmpty()) {
                // Mark events as processed
                val eventIds = eventsToCleanup.map { it.id }
                database.rawEventDao().markEventsAsProcessed(eventIds)
                
                Log.d(TAG, "Marked ${eventIds.size} raw events as processed for $targetDate")
                
                // Delete old processed events (older than retention period)
                val retentionPeriod = getDataRetentionPeriod(database)
                val cutoffTime = System.currentTimeMillis() - (retentionPeriod * 24 * 60 * 60 * 1000L)
                
                database.rawEventDao().deleteOldProcessedEvents(cutoffTime)
                
                Log.d(TAG, "Cleaned up old processed events older than $retentionPeriod days")
            }
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to cleanup processed events", e)
        }
    }

    /**
     * Gets the data retention period from privacy settings
     */
    private suspend fun getDataRetentionPeriod(database: AppDatabase): Long {
        return try {
            val privacySettings = database.privacySettingsDao().getSettings()
            privacySettings?.dataRetentionDays?.toLong() ?: 7L // Default 7 days
        } catch (e: Exception) {
            Log.w(TAG, "Failed to get data retention period, using default", e)
            7L
        }
    }

    /**
     * Anonymizes package names based on privacy settings
     */
    private fun anonymizePackageName(packageName: String): String {
        // Simple categorization for privacy
        return when {
            packageName.contains("browser") || packageName.contains("chrome") -> "browser"
            packageName.contains("social") || packageName.contains("facebook") || packageName.contains("twitter") -> "social"
            packageName.contains("game") -> "games"
            packageName.contains("music") || packageName.contains("spotify") -> "media"
            packageName.contains("email") || packageName.contains("gmail") -> "communication"
            packageName.contains("camera") || packageName.contains("photo") -> "photography"
            packageName.contains("maps") || packageName.contains("navigation") -> "navigation"
            else -> "other"
        }
    }

    /**
     * Logs error events for debugging and monitoring
     */
    private suspend fun logErrorEvent(error: Exception, attemptCount: Int) {
        try {
            val database = AppDatabase.getInstance(applicationContext)
            val auditEntry = AuditLogEntity(
                id = UUID.randomUUID().toString(),
                timestamp = System.currentTimeMillis(),
                eventType = "SUMMARY_GENERATION_ERROR",
                details = gson.toJson(mapOf(
                    "component" to "DailySummaryWorker",
                    "errorType" to error.javaClass.simpleName,
                    "errorMessage" to (error.message ?: "Unknown error"),
                    "attemptCount" to attemptCount,
                    "stackTrace" to error.stackTrace.take(5).map { it.toString() }
                )),
                userId = null
            )
            
            database.auditLogDao().insert(auditEntry)
        } catch (e: Exception) {
            Log.w(TAG, "Failed to log error event", e)
        }
    }

    /**
     * Determines if an error is transient and worth retrying
     */
    private fun isTransientError(error: Exception): Boolean {
        return when (error) {
            is java.net.SocketTimeoutException,
            is java.net.ConnectException,
            is java.io.IOException,
            is android.database.sqlite.SQLiteException -> true
            else -> error.message?.let { message ->
                message.contains("timeout", ignoreCase = true) ||
                message.contains("connection", ignoreCase = true) ||
                message.contains("network", ignoreCase = true) ||
                message.contains("temporary", ignoreCase = true)
            } ?: false
        }
    }

    /**
     * Calculates exponential backoff delay for retries
     */
    private fun calculateBackoffDelay(attemptCount: Int): Long {
        return BASE_BACKOFF_DELAY_MS * (1L shl attemptCount) // 2^attemptCount
    }

    /**
     * Performs emergency cleanup when memory is low
     */
    private suspend fun performEmergencyCleanup() {
        try {
            Log.i(TAG, "Performing emergency cleanup due to memory pressure")
            
            val database = AppDatabase.getInstance(applicationContext)
            val cutoffTime = System.currentTimeMillis() - (3 * 24 * 60 * 60 * 1000L) // 3 days ago
            
            // Clean up old processed events
            database.rawEventDao().deleteOldProcessedEvents(cutoffTime)
            
            // Clean up old audit logs
            database.auditLogDao().deleteOldLogs(cutoffTime)
            
            // Clean up old interaction metrics
            database.interactionMetricsDao().deleteOldMetrics(cutoffTime)
            
            // Force garbage collection
            System.gc()
            
            Log.i(TAG, "Emergency cleanup completed")
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to perform emergency cleanup", e)
        }
    }

    /**
     * Validates summary data before storage
     */
    private fun validateSummaryData(summary: EnhancedDailySummaryEntity): Boolean {
        return try {
            // Basic validation checks
            when {
                summary.totalScreenTime < 0 -> {
                    Log.w(TAG, "Invalid screen time: ${summary.totalScreenTime}")
                    false
                }
                summary.notificationCount < 0 -> {
                    Log.w(TAG, "Invalid notification count: ${summary.notificationCount}")
                    false
                }
                summary.interactionIntensity < 0f || summary.interactionIntensity > 1f -> {
                    Log.w(TAG, "Invalid interaction intensity: ${summary.interactionIntensity}")
                    false
                }
                summary.peakUsageHour < 0 || summary.peakUsageHour > 23 -> {
                    Log.w(TAG, "Invalid peak usage hour: ${summary.peakUsageHour}")
                    false
                }
                else -> {
                    // Try to decrypt and validate metadata
                    try {
                        DBHelper.decryptMetadata(summary.appUsageDistribution)
                        DBHelper.decryptMetadata(summary.activityBreakdown)
                        true
                    } catch (e: Exception) {
                        Log.w(TAG, "Invalid encrypted metadata", e)
                        false
                    }
                }
            }
        } catch (e: Exception) {
            Log.w(TAG, "Error validating summary data", e)
            false
        }
    }

    /**
     * Creates a fallback summary when data is insufficient
     */
    private fun createFallbackSummary(targetDate: String): EnhancedDailySummaryEntity {
        return EnhancedDailySummaryEntity(
            date = targetDate,
            totalScreenTime = 0L,
            appUsageDistribution = DBHelper.encryptMetadata("{}"),
            notificationCount = 0,
            peakUsageHour = 12, // Default to noon
            activityBreakdown = DBHelper.encryptMetadata("{}"),
            interactionIntensity = 0f,
            createdAt = System.currentTimeMillis(),
            version = 1
        )
    }

    // Data classes for summary calculations

    private data class ScreenTimeMetrics(
        val totalScreenTime: Long,
        val sessionCount: Int,
        val averageSessionLength: Long,
        val totalUnlocks: Int
    )

    private data class AppUsageStats(
        val totalForegroundTime: Long,
        val launchCount: Int,
        val lastUsed: Long
    )

    private data class NotificationMetrics(
        val totalCount: Int,
        val interactedCount: Int,
        val ongoingCount: Int,
        val hourlyDistribution: List<Int>
    )

    private data class ActivityStats(
        val totalDuration: Long,
        val averageConfidence: Float,
        val occurrenceCount: Int
    )

    companion object {
        private const val KEY_TARGET_DATE = "target_date"

        /**
         * Schedules daily summary generation with optimal constraints
         */
        fun scheduleDailySummaryWork(context: Context) {
            val constraints = Constraints.Builder()
                .setRequiredNetworkType(NetworkType.NOT_REQUIRED)
                .setRequiresBatteryNotLow(true)
                .setRequiresCharging(false)
                .setRequiresDeviceIdle(false)
                .build()

            val dailyWorkRequest = PeriodicWorkRequestBuilder<DailySummaryWorker>(
                1, TimeUnit.DAYS
            )
                .setConstraints(constraints)
                .setInitialDelay(1, TimeUnit.HOURS) // Start 1 hour after scheduling
                .setBackoffCriteria(
                    BackoffPolicy.EXPONENTIAL,
                    WorkRequest.MIN_BACKOFF_MILLIS,
                    TimeUnit.MILLISECONDS
                )
                .build()

            WorkManager.getInstance(context).enqueueUniquePeriodicWork(
                WORK_NAME,
                ExistingPeriodicWorkPolicy.KEEP,
                dailyWorkRequest
            )

            Log.i(TAG, "Daily summary work scheduled")
        }

        /**
         * Triggers immediate summary generation for a specific date
         */
        fun generateSummaryForDate(context: Context, targetDate: String) {
            val inputData = Data.Builder()
                .putString(KEY_TARGET_DATE, targetDate)
                .build()

            val oneTimeWorkRequest = OneTimeWorkRequestBuilder<DailySummaryWorker>()
                .setInputData(inputData)
                .setConstraints(
                    Constraints.Builder()
                        .setRequiresBatteryNotLow(true)
                        .build()
                )
                .build()

            WorkManager.getInstance(context).enqueue(oneTimeWorkRequest)

            Log.i(TAG, "One-time summary generation scheduled for $targetDate")
        }

        /**
         * Cancels all scheduled summary work
         */
        fun cancelDailySummaryWork(context: Context) {
            WorkManager.getInstance(context).cancelUniqueWork(WORK_NAME)
            Log.i(TAG, "Daily summary work cancelled")
        }
    }
}

    /**
     * Logs audit event for summary generation
     */
    private suspend fun logAuditEvent(database: AppDatabase, details: String) {
        try {
            val auditEntry = AuditLogEntity(
                id = UUID.randomUUID().toString(),
                timestamp = System.currentTimeMillis(),
                eventType = AuditEventType.DATA_EXPORTED.name, // Summary generation is a form of data processing
                details = gson.toJson(mapOf(
                    "component" to "DailySummaryWorker",
                    "details" to details,
                    "privacyPreserving" to true
                )),
                userId = null
            )
            
            database.auditLogDao().insert(auditEntry)
        } catch (e: Exception) {
            Log.w(TAG, "Failed to log audit event", e)
        }
    }