package com.lifetwin.mlp.usagestats

import android.content.Context
import android.util.Log
import androidx.work.*
import com.lifetwin.mlp.db.TimeRange
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.util.concurrent.TimeUnit

private const val TAG = "UsageStatsWorker"
private const val WORK_NAME = "usage_stats_collection"

/**
 * WorkManager worker for periodic usage stats collection
 */
class UsageStatsWorker(
    context: Context,
    params: WorkerParameters
) : CoroutineWorker(context, params) {

    override suspend fun doWork(): Result = withContext(Dispatchers.IO) {
        try {
            Log.d(TAG, "Starting usage stats collection work")
            
            val collector = UsageStatsCollector(applicationContext)
            
            // Check if we have permission
            if (!collector.isPermissionGranted()) {
                Log.w(TAG, "Usage stats permission not granted, skipping collection")
                return@withContext Result.retry()
            }
            
            // Collect usage events from the last collection period
            val endTime = System.currentTimeMillis()
            val startTime = endTime - getCollectionInterval()
            
            val events = collector.collectUsageEvents(TimeRange(startTime, endTime))
            
            Log.i(TAG, "Collected ${events.size} usage events")
            
            // Schedule next collection
            scheduleNextCollection(applicationContext)
            
            Result.success()
        } catch (e: Exception) {
            Log.e(TAG, "Usage stats collection failed", e)
            Result.retry()
        }
    }

    private fun getCollectionInterval(): Long {
        // Default to 1 hour collection interval
        return inputData.getLong("collection_interval_ms", 60 * 60 * 1000L)
    }

    companion object {
        /**
         * Schedules periodic usage stats collection
         */
        fun schedulePeriodicCollection(context: Context) {
            val constraints = Constraints.Builder()
                .setRequiredNetworkType(NetworkType.NOT_REQUIRED)
                .setRequiresBatteryNotLow(true)
                .setRequiresDeviceIdle(false)
                .build()

            val workRequest = PeriodicWorkRequestBuilder<UsageStatsWorker>(
                1, TimeUnit.HOURS,  // Repeat every hour
                15, TimeUnit.MINUTES // Flex interval
            )
                .setConstraints(constraints)
                .setInputData(
                    Data.Builder()
                        .putLong("collection_interval_ms", 60 * 60 * 1000L) // 1 hour
                        .build()
                )
                .setBackoffCriteria(
                    BackoffPolicy.EXPONENTIAL,
                    WorkRequest.MIN_BACKOFF_DELAY_MILLIS,
                    TimeUnit.MILLISECONDS
                )
                .build()

            WorkManager.getInstance(context)
                .enqueueUniquePeriodicWork(
                    WORK_NAME,
                    ExistingPeriodicWorkPolicy.KEEP,
                    workRequest
                )

            Log.i(TAG, "Scheduled periodic usage stats collection")
        }

        /**
         * Schedules the next one-time collection (for immediate execution)
         */
        private fun scheduleNextCollection(context: Context) {
            val constraints = Constraints.Builder()
                .setRequiredNetworkType(NetworkType.NOT_REQUIRED)
                .setRequiresBatteryNotLow(false)
                .build()

            val workRequest = OneTimeWorkRequestBuilder<UsageStatsWorker>()
                .setConstraints(constraints)
                .setInitialDelay(1, TimeUnit.HOURS)
                .setInputData(
                    Data.Builder()
                        .putLong("collection_interval_ms", 60 * 60 * 1000L)
                        .build()
                )
                .build()

            WorkManager.getInstance(context)
                .enqueue(workRequest)
        }

        /**
         * Cancels all scheduled usage stats collection work
         */
        fun cancelCollection(context: Context) {
            WorkManager.getInstance(context)
                .cancelUniqueWork(WORK_NAME)
            Log.i(TAG, "Cancelled usage stats collection")
        }

        /**
         * Checks if collection is currently scheduled
         */
        suspend fun isCollectionScheduled(context: Context): Boolean {
            val workInfos = WorkManager.getInstance(context)
                .getWorkInfosForUniqueWork(WORK_NAME)
                .await()
            
            return workInfos.any { workInfo ->
                workInfo.state == WorkInfo.State.ENQUEUED || workInfo.state == WorkInfo.State.RUNNING
            }
        }
    }
}