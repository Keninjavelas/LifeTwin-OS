package com.lifetwin.mlp.usagestats

// Stub for MLP phase. In a full RN Android project this would be
// wired as a NativeModule and use UsageStatsManager to emit app events.

import android.app.usage.UsageStats
import android.app.usage.UsageStatsManager
import android.content.Context
import android.util.Log
import com.lifetwin.mlp.db.AppEventEntity
import com.lifetwin.mlp.db.DBHelper

class UsageStatsCollector(private val context: Context) {

    private val usageStatsManager: UsageStatsManager? =
        context.getSystemService(Context.USAGE_STATS_SERVICE) as? UsageStatsManager

    fun pollRecentEvents() {
        val usm = usageStatsManager ?: run {
            Log.w("UsageStatsCollector", "UsageStatsManager not available on this device")
            return
        }

        // Query the last hour of usage by default
        val end = System.currentTimeMillis()
        val start = end - 60L * 60L * 1000L // 1 hour

        try {
            val stats: List<UsageStats> = usm.queryUsageStats(UsageStatsManager.INTERVAL_DAILY, start, end)
            if (stats.isNullOrEmpty()) {
                // No data available, possibly missing permission
                Log.i("UsageStatsCollector", "No usage stats returned (permission missing or no data)")
                return
            }

            for (s in stats) {
                // Use lastTimeUsed when available; fall back to end
                val ts = if (s.lastTimeUsed > 0) s.lastTimeUsed else end
                val pkg = s.packageName
                val event = AppEventEntity(timestamp = ts, type = "usage", packageName = pkg)
                DBHelper.insertEventAsync(context, event)
            }
        } catch (e: SecurityException) {
            Log.w("UsageStatsCollector", "Missing PACKAGE_USAGE_STATS permission", e)
        } catch (e: Exception) {
            Log.e("UsageStatsCollector", "Failed to query usage stats", e)
        }
    }
}

