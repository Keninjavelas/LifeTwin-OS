package com.lifetwin.mlp.db

import java.util.UUID

// Core data models for the enhanced data collection system

/**
 * Time range for querying events
 */
data class TimeRange(
    val startTime: Long,
    val endTime: Long
)

/**
 * Usage event data model
 */
data class UsageEvent(
    val id: String = UUID.randomUUID().toString(),
    val packageName: String,
    val startTime: Long,
    val endTime: Long,
    val totalTimeInForeground: Long,
    val lastTimeUsed: Long,
    val eventType: UsageEventType
)

enum class UsageEventType {
    ACTIVITY_RESUMED,
    ACTIVITY_PAUSED,
    CONFIGURATION_CHANGE,
    USER_INTERACTION,
    KEYGUARD_SHOWN,
    KEYGUARD_HIDDEN,
    FOREGROUND_SERVICE_START,
    FOREGROUND_SERVICE_STOP
}

/**
 * Notification data model
 */
data class NotificationData(
    val id: String = UUID.randomUUID().toString(),
    val packageName: String,
    val timestamp: Long,
    val category: String?,
    val priority: Int,
    val hasActions: Boolean,
    val isOngoing: Boolean
)

/**
 * Notification interaction data
 */
data class NotificationInteraction(
    val id: String = UUID.randomUUID().toString(),
    val notificationId: String,
    val interactionType: NotificationInteractionType,
    val timestamp: Long
)

enum class NotificationInteractionType {
    POSTED,
    OPENED,
    DISMISSED,
    ACTION_CLICKED,
    SNOOZED
}

/**
 * Screen session data model
 */
data class ScreenSession(
    val sessionId: String = UUID.randomUUID().toString(),
    val startTime: Long,
    val endTime: Long?,
    val unlockCount: Int = 0,
    val interactionIntensity: Float = 0f
)

/**
 * Interaction metrics data model
 */
data class InteractionMetrics(
    val id: String = UUID.randomUUID().toString(),
    val timestamp: Long,
    val touchCount: Int,
    val scrollEvents: Int,
    val gesturePatterns: List<GestureType>,
    val interactionIntensity: Float,
    val timeWindow: TimeRange
)

enum class GestureType {
    TAP,
    LONG_PRESS,
    SWIPE_UP,
    SWIPE_DOWN,
    SWIPE_LEFT,
    SWIPE_RIGHT,
    PINCH_IN,
    PINCH_OUT,
    SCROLL_VERTICAL,
    SCROLL_HORIZONTAL
}

/**
 * Activity context data model
 */
data class ActivityContext(
    val id: String = UUID.randomUUID().toString(),
    val activityType: ActivityType,
    val confidence: Float,
    val timestamp: Long,
    val duration: Long
)

enum class ActivityType {
    STATIONARY,
    WALKING,
    IN_VEHICLE,
    ON_BICYCLE,
    RUNNING,
    TILTING,
    UNKNOWN
}

/**
 * Privacy settings data model
 */
data class PrivacySettings(
    val enabledCollectors: Set<CollectorType>,
    val dataRetentionDays: Int,
    val privacyLevel: PrivacyLevel,
    val anonymizationSettings: AnonymizationSettings,
    val dataSharingSettings: DataSharingSettings
)

enum class CollectorType {
    USAGE_STATS,
    NOTIFICATIONS,
    SCREEN_EVENTS,
    INTERACTIONS,
    SENSORS
}

enum class PrivacyLevel {
    MINIMAL,    // Only basic screen time and app categories
    STANDARD,   // Includes notifications and interaction patterns
    DETAILED    // Full behavioral analysis with sensor data
}

data class AnonymizationSettings(
    val aggregateAppUsage: Boolean,
    val removePersonalIdentifiers: Boolean,
    val fuzzTimestamps: Boolean,
    val categoryOnlyMode: Boolean,
    val minimumAggregationWindow: Long // Duration in milliseconds
)

data class DataSharingSettings(
    val allowCloudSync: Boolean,
    val allowAnalytics: Boolean,
    val allowResearchParticipation: Boolean,
    val encryptionRequired: Boolean
)

/**
 * Audit log entry
 */
data class AuditLogEntry(
    val id: String = UUID.randomUUID().toString(),
    val timestamp: Long,
    val eventType: AuditEventType,
    val details: Map<String, Any>,
    val userId: String? = null
)

enum class AuditEventType {
    PRIVACY_SETTING_CHANGED,
    DATA_EXPORTED,
    DATA_DELETED,
    DATA_PURGED,
    COLLECTOR_ENABLED,
    COLLECTOR_DISABLED,
    KEY_ROTATED,
    PERMISSION_GRANTED,
    PERMISSION_REVOKED,
    EMERGENCY_MODE_ACTIVATED,
    EMERGENCY_MODE_DEACTIVATED
}

enum class DataType {
    USAGE_EVENTS,
    NOTIFICATION_EVENTS,
    SCREEN_SESSIONS,
    INTERACTION_METRICS,
    SENSOR_DATA,
    DAILY_SUMMARIES,
    RAW_EVENTS
}

// Core interfaces for data collectors

/**
 * Base interface for all data collectors
 */
interface DataCollector {
    suspend fun startCollection()
    suspend fun stopCollection()
    fun isCollectionActive(): Boolean
    fun getCollectorType(): CollectorType
    suspend fun getCollectedDataCount(): Int
}

/**
 * Interface for usage statistics collection
 */
interface UsageStatsCollector : DataCollector {
    suspend fun collectUsageEvents(timeRange: TimeRange): List<UsageEvent>
    fun isPermissionGranted(): Boolean
    suspend fun requestPermission(): Boolean
}

/**
 * Interface for notification logging
 */
interface NotificationLogger : DataCollector {
    suspend fun logNotificationPosted(notification: NotificationData)
    suspend fun logNotificationInteraction(interaction: NotificationInteraction)
    fun isNotificationAccessGranted(): Boolean
    suspend fun requestNotificationAccess(): Boolean
}

/**
 * Interface for screen event monitoring
 */
interface ScreenEventReceiver : DataCollector {
    suspend fun getCurrentSession(): ScreenSession?
    suspend fun getSessionsByTimeRange(timeRange: TimeRange): List<ScreenSession>
    suspend fun getTotalScreenTime(timeRange: TimeRange): Long
}

/**
 * Interface for interaction pattern collection
 */
interface InteractionAccessibilityService : DataCollector {
    suspend fun getInteractionMetrics(timeRange: TimeRange): List<InteractionMetrics>
    fun isAccessibilityServiceEnabled(): Boolean
    suspend fun requestAccessibilityPermission(): Boolean
}

/**
 * Interface for sensor data fusion
 */
interface SensorFusionManager : DataCollector {
    suspend fun getCurrentActivity(): ActivityContext?
    suspend fun getActivityHistory(timeRange: TimeRange): List<ActivityContext>
    fun isSensorPermissionGranted(): Boolean
    suspend fun requestSensorPermission(): Boolean
}

/**
 * Interface for privacy management
 */
interface PrivacyManager {
    suspend fun setCollectorEnabled(collector: CollectorType, enabled: Boolean)
    suspend fun setDataRetentionPeriod(days: Int)
    suspend fun setPrivacyLevel(level: PrivacyLevel)
    suspend fun getPrivacySettings(): PrivacySettings
    suspend fun exportPrivacyReport(): String
    suspend fun setAnonymizationSettings(settings: AnonymizationSettings)
    suspend fun setDataSharingSettings(settings: DataSharingSettings)
    suspend fun enableEmergencyPrivacyMode()
    suspend fun logAuditEvent(event: AuditLogEntry)
}

/**
 * Interface for data export functionality
 */
interface DataExporter {
    suspend fun exportAllData(): String
    suspend fun exportDataByType(types: Set<CollectorType>): String
    suspend fun exportDataByTimeRange(timeRange: TimeRange): String
    suspend fun validateExportData(exportData: String): Boolean
    suspend fun importData(importData: String): Boolean
}