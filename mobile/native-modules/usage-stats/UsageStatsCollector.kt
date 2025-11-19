package com.lifetwin.mlp.usagestats

// Stub for MLP phase. In a full RN Android project this would be
// wired as a NativeModule and use UsageStatsManager to emit app events.

import android.app.usage.UsageStatsManager
import android.content.Context

class UsageStatsCollector(private val context: Context) {

    private val usageStatsManager: UsageStatsManager? =
        context.getSystemService(Context.USAGE_STATS_SERVICE) as? UsageStatsManager

    fun pollRecentEvents(): List<AppEvent> {
        // TODO: query UsageStatsManager and map results into AppEvent DTOs
        return emptyList()
    }
}

// Shared DTO for native collectors

data class AppEvent(
    val timestamp: Long,
    val type: String,
    val packageName: String?,
)
