package com.lifetwin.mlp.db

import androidx.room.Entity
import androidx.room.PrimaryKey
import java.util.Date
import androidx.room.ColumnInfo
import androidx.room.TypeConverter
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken


@Entity(tableName = "app_events")
data class AppEventEntity(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val timestamp: Long,
    val type: String,
    val packageName: String? = null
)


@Entity(tableName = "daily_summaries")
data class DailySummaryEntity(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val deviceId: String,
    val date: Date,
    val totalScreenTime: Int = 0,
    val topApps: List<String>? = emptyList(),
    val mostCommonHour: Int = 0,
    val notificationCount: Int = 0
)


@Entity(tableName = "sync_queue")
data class SyncQueueEntity(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val payload: String,
    @ColumnInfo(name = "created_at") val createdAt: Long = System.currentTimeMillis()
)

// New entities for enhanced data collection

@Entity(tableName = "raw_events")
data class RawEventEntity(
    @PrimaryKey val id: String,
    val timestamp: Long,
    val eventType: String, // "usage", "notification", "screen", "interaction", "sensor"
    val packageName: String?,
    val duration: Long?,
    val metadata: String, // Encrypted JSON
    val processed: Boolean = false,
    val createdAt: Long = System.currentTimeMillis()
)

@Entity(tableName = "enhanced_daily_summaries")
data class EnhancedDailySummaryEntity(
    @PrimaryKey val date: String, // YYYY-MM-DD
    val totalScreenTime: Long,
    val appUsageDistribution: String, // Encrypted JSON
    val notificationCount: Int,
    val peakUsageHour: Int,
    val activityBreakdown: String, // Encrypted JSON
    val interactionIntensity: Float,
    val createdAt: Long = System.currentTimeMillis(),
    val version: Int = 1
)

@Entity(tableName = "privacy_settings")
data class PrivacySettingsEntity(
    @PrimaryKey val id: Int = 1, // Single row table
    val enabledCollectors: String, // JSON array of enabled collector types
    val dataRetentionDays: Int = 7,
    val privacyLevel: String = "STANDARD", // MINIMAL, STANDARD, DETAILED
    val anonymizationSettings: String, // JSON object
    val dataSharingSettings: String, // JSON object
    val lastUpdated: Long = System.currentTimeMillis()
)

@Entity(tableName = "usage_events")
data class UsageEventEntity(
    @PrimaryKey val id: String,
    val packageName: String,
    val startTime: Long,
    val endTime: Long,
    val totalTimeInForeground: Long,
    val lastTimeUsed: Long,
    val eventType: String, // "ACTIVITY_RESUMED", "ACTIVITY_PAUSED", etc.
    val createdAt: Long = System.currentTimeMillis()
)

@Entity(tableName = "notification_events")
data class NotificationEventEntity(
    @PrimaryKey val id: String,
    val packageName: String,
    val timestamp: Long,
    val category: String?,
    val priority: Int,
    val hasActions: Boolean,
    val isOngoing: Boolean,
    val interactionType: String?, // "posted", "opened", "dismissed"
    val createdAt: Long = System.currentTimeMillis()
)

@Entity(tableName = "screen_sessions")
data class ScreenSessionEntity(
    @PrimaryKey val sessionId: String,
    val startTime: Long,
    val endTime: Long?,
    val unlockCount: Int = 0,
    val interactionIntensity: Float = 0f,
    val isActive: Boolean = true,
    val createdAt: Long = System.currentTimeMillis()
)

@Entity(tableName = "interaction_metrics")
data class InteractionMetricsEntity(
    @PrimaryKey val id: String,
    val timestamp: Long,
    val touchCount: Int,
    val scrollEvents: Int,
    val gesturePatterns: String, // JSON array of gesture types
    val interactionIntensity: Float,
    val timeWindowStart: Long,
    val timeWindowEnd: Long,
    val createdAt: Long = System.currentTimeMillis()
)

@Entity(tableName = "activity_context")
data class ActivityContextEntity(
    @PrimaryKey val id: String,
    val activityType: String, // STATIONARY, WALKING, IN_VEHICLE, etc.
    val confidence: Float,
    val timestamp: Long,
    val duration: Long,
    val sensorData: String?, // Encrypted JSON of sensor readings
    val createdAt: Long = System.currentTimeMillis()
)

@Entity(tableName = "audit_log")
data class AuditLogEntity(
    @PrimaryKey val id: String,
    val timestamp: Long,
    val eventType: String, // "privacy_setting_changed", "data_exported", etc.
    val details: String, // JSON object with event details
    val userId: String? = null,
    val createdAt: Long = System.currentTimeMillis()
)

@Entity(tableName = "performance_log")
data class PerformanceLogEntity(
    @PrimaryKey val id: String,
    val timestamp: Long,
    val operationType: String, // "DATA_COLLECTION", "BATCH_OPERATION", etc.
    val collectorType: String? = null,
    val recordCount: Int? = null,
    val durationMs: Long? = null,
    val batteryLevel: Int? = null,
    val memoryUsageMB: Double? = null,
    val cpuUsage: Double? = null,
    val createdAt: Long = System.currentTimeMillis()
)

@Entity(tableName = "automation_log")
data class AutomationLogEntity(
    @PrimaryKey val id: String,
    val interventionId: String,
    val timestamp: Long,
    val interventionType: String,
    val trigger: String,
    val reasoning: String,
    val confidence: Float,
    val executed: Boolean,
    val userResponse: String,
    val executionTimeMs: Long,
    val feedbackRating: Int?,
    val feedbackComments: String?,
    val helpful: Boolean?,
    val createdAt: Long = System.currentTimeMillis()
)
