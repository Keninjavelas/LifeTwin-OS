package com.lifetwin.mlp.summaries

import android.content.Context
import androidx.work.CoroutineWorker
import androidx.work.WorkerParameters
import com.lifetwin.mlp.db.AppDatabase
import com.lifetwin.mlp.db.DailySummaryEntity
import com.lifetwin.mlp.db.DBHelper
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.util.Calendar
import java.util.Date

// CoroutineWorker that aggregates raw events from AppDatabase into a daily summary row.
class DailySummaryWorker(appContext: Context, params: WorkerParameters) : CoroutineWorker(appContext, params) {

    override suspend fun doWork(): Result = withContext(Dispatchers.IO) {
        val db = AppDatabase.getInstance(applicationContext)
        val now = Calendar.getInstance()
        // aggregate for the previous day
        now.add(Calendar.DATE, -1)
        now.set(Calendar.HOUR_OF_DAY, 0)
        now.set(Calendar.MINUTE, 0)
        now.set(Calendar.SECOND, 0)
        val since = now.timeInMillis

        val events = try {
            db.appEventDao().getEventsSince(since)
        } catch (e: Exception) {
            // If DB read fails, skip this run and schedule retry
            return@withContext Result.retry()
        }

        // Compute total screen time by pairing screen on/off events within the window.
        val screenEvents = events.filter { it.type == "screen" || it.type == "screen_off" }
            .sortedBy { it.timestamp }

        var totalScreenMillis = 0L
        var lastOn: Long? = null
        for (ev in screenEvents) {
            when (ev.type) {
                "screen" -> lastOn = ev.timestamp
                "screen_off" -> {
                    if (lastOn != null) {
                        val delta = ev.timestamp - lastOn
                        if (delta > 0) totalScreenMillis += delta
                        lastOn = null
                    }
                }
            }
        }
        // If we have an unclosed screen session (device still on), approximate up to end of window
        if (lastOn != null) {
            val endOfWindow = since + 24L * 60L * 60L * 1000L
            val delta = endOfWindow - lastOn
            if (delta > 0) totalScreenMillis += delta
        }

        val totalScreenTimeSeconds = (totalScreenMillis / 1000).toInt()

        // Top apps by event count
        val appCounts = events.groupingBy { it.packageName ?: "unknown" }.eachCount()
        val topApps = appCounts.entries.sortedByDescending { it.value }.map { it.key }.take(3)

        // Most common hour (binned by event timestamps)
        val hourCounts = IntArray(24)
        for (ev in events) {
            val cal = Calendar.getInstance()
            cal.timeInMillis = ev.timestamp
            hourCounts[cal.get(Calendar.HOUR_OF_DAY)]++
        }
        val mostCommonHour = hourCounts.indices.maxByOrNull { hourCounts[it] } ?: 0

        val summary = DailySummaryEntity(
            deviceId = "local-device",
            date = Date(since),
            totalScreenTime = totalScreenTimeSeconds,
            topApps = topApps,
            mostCommonHour = mostCommonHour,
            notificationCount = events.count { it.type == "notification" }
        )

        // Use DBHelper to insert asynchronously; we don't need to block the worker on the insert
        try {
            DBHelper.insertSummaryAsync(applicationContext, summary)
        } catch (e: Exception) {
            // If scheduling the write fails, return retry to avoid losing data
            return@withContext Result.retry()
        }

        // Schedule a background sync to server (non-blocking)
        try {
            SyncManager.enqueueSync(applicationContext)
        } catch (e: Exception) {
            // best-effort: failure to schedule sync should not fail the worker
        }

        // TODO: trigger background sync to backend using native HTTP client or RN bridge

        Result.success()
    }
}
