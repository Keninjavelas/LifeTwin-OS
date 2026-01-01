package com.lifetwin.mlp.notifications

import android.content.Context
import android.content.Intent
import android.provider.Settings
import android.util.Log
import com.lifetwin.mlp.db.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

private const val TAG = "NotificationManager"

/**
 * Manager class for coordinating notification collection and handling permission changes
 */
class NotificationManager(private val context: Context) {
    
    /**
     * Initializes notification collection system
     */
    suspend fun initialize() {
        withContext(Dispatchers.IO) {
            try {
                Log.i(TAG, "Initializing notification collection system")
                
                if (!isNotificationAccessGranted()) {
                    Log.w(TAG, "Notification access not granted")
                    logAuditEvent(AuditEventType.PERMISSION_REVOKED, "Notification access not available")
                    return@withContext
                }
                
                logAuditEvent(AuditEventType.COLLECTOR_ENABLED, "Notification collection initialized")
                Log.i(TAG, "Notification collection system initialized successfully")
                
            } catch (e: Exception) {
                Log.e(TAG, "Failed to initialize notification collection", e)
                logAuditEvent(AuditEventType.COLLECTOR_DISABLED, "Failed to initialize: ${e.message}")
            }
        }
    }
    
    /**
     * Handles permission revocation gracefully
     */
    suspend fun handlePermissionRevoked() {
        withContext(Dispatchers.IO) {
            try {
                Log.w(TAG, "Notification access permission revoked")
                
                // Log the permission loss
                logAuditEvent(AuditEventType.PERMISSION_REVOKED, "Notification access permission lost")
                
                // Disable any dependent features gracefully
                disableDependentFeatures()
                
                Log.i(TAG, "Gracefully handled notification permission revocation")
                
            } catch (e: Exception) {
                Log.e(TAG, "Failed to handle permission revocation gracefully", e)
            }
        }
    }
    
    /**
     * Handles permission restoration
     */
    suspend fun handlePermissionGranted() {
        withContext(Dispatchers.IO) {
            try {
                Log.i(TAG, "Notification access permission granted")
                
                // Log the permission grant
                logAuditEvent(AuditEventType.PERMISSION_GRANTED, "Notification access permission granted")
                
                // Re-enable features
                enableDependentFeatures()
                
                Log.i(TAG, "Successfully restored notification collection after permission grant")
                
            } catch (e: Exception) {
                Log.e(TAG, "Failed to handle permission grant", e)
            }
        }
    }
    
    /**
     * Requests notification access permission from user
     */
    suspend fun requestPermission(): Boolean {
        return withContext(Dispatchers.Main) {
            try {
                val intent = Intent(Settings.ACTION_NOTIFICATION_LISTENER_SETTINGS)
                intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                context.startActivity(intent)
                
                logAuditEvent(AuditEventType.PERMISSION_GRANTED, "Notification access permission requested")
                Log.i(TAG, "Opened notification access settings")
                true
            } catch (e: Exception) {
                Log.e(TAG, "Failed to open notification access settings", e)
                logAuditEvent(AuditEventType.PERMISSION_REVOKED, "Failed to request notification access permission")
                false
            }
        }
    }
    
    /**
     * Checks if notification access is currently granted
     */
    fun isNotificationAccessGranted(): Boolean {
        return NotificationLogger.isNotificationAccessGranted(context)
    }
    
    /**
     * Gets collection statistics
     */
    suspend fun getCollectionStats(): NotificationCollectionStats {
        return withContext(Dispatchers.IO) {
            try {
                val database = AppDatabase.getInstance(context)
                val endTime = System.currentTimeMillis()
                val startTime = endTime - (24 * 60 * 60 * 1000L) // Last 24 hours
                
                val totalNotifications = database.notificationEventDao()
                    .getEventCountByTimeRange(startTime, endTime)
                
                val postedCount = database.notificationEventDao()
                    .getEventsByTimeRange(startTime, endTime)
                    .count { it.interactionType == "posted" }
                
                val openedCount = database.notificationEventDao()
                    .getEventsByTimeRange(startTime, endTime)
                    .count { it.interactionType == "opened" }
                
                val dismissedCount = database.notificationEventDao()
                    .getEventsByTimeRange(startTime, endTime)
                    .count { it.interactionType == "dismissed" }
                
                NotificationCollectionStats(
                    totalNotifications = totalNotifications,
                    notificationsPosted = postedCount,
                    notificationsOpened = openedCount,
                    notificationsDismissed = dismissedCount,
                    hasPermission = isNotificationAccessGranted(),
                    lastCollectionTime = getLastCollectionTime()
                )
            } catch (e: Exception) {
                Log.e(TAG, "Failed to get collection stats", e)
                NotificationCollectionStats()
            }
        }
    }
    
    /**
     * Performs cleanup of old notification data based on privacy settings
     */
    suspend fun performDataCleanup() {
        withContext(Dispatchers.IO) {
            try {
                val database = AppDatabase.getInstance(context)
                val privacySettings = database.privacySettingsDao().getSettings()
                val retentionDays = privacySettings?.dataRetentionDays ?: 7
                val cutoffTime = System.currentTimeMillis() - (retentionDays * 24 * 60 * 60 * 1000L)
                
                // Clean up old notification events
                database.notificationEventDao().deleteOldEvents(cutoffTime)
                
                Log.i(TAG, "Notification data cleanup completed")
                logAuditEvent(AuditEventType.DATA_DELETED, "Old notification data cleaned up")
                
            } catch (e: Exception) {
                Log.e(TAG, "Failed to perform data cleanup", e)
            }
        }
    }
    
    /**
     * Exports notification data for user data portability
     */
    suspend fun exportNotificationData(timeRange: TimeRange? = null): String {
        return withContext(Dispatchers.IO) {
            try {
                val database = AppDatabase.getInstance(context)
                
                val events = if (timeRange != null) {
                    database.notificationEventDao().getEventsByTimeRange(
                        timeRange.startTime,
                        timeRange.endTime
                    )
                } else {
                    // Export last 30 days by default
                    val endTime = System.currentTimeMillis()
                    val startTime = endTime - (30 * 24 * 60 * 60 * 1000L)
                    database.notificationEventDao().getEventsByTimeRange(startTime, endTime)
                }
                
                // Convert to JSON format
                val exportData = events.map { event ->
                    mapOf(
                        "id" to event.id,
                        "packageName" to event.packageName,
                        "timestamp" to event.timestamp,
                        "category" to event.category,
                        "priority" to event.priority,
                        "hasActions" to event.hasActions,
                        "isOngoing" to event.isOngoing,
                        "interactionType" to event.interactionType,
                        "createdAt" to event.createdAt
                    )
                }
                
                val exportJson = mapOf(
                    "exportType" to "notification_data",
                    "exportTime" to System.currentTimeMillis(),
                    "eventCount" to events.size,
                    "events" to exportData
                )
                
                val jsonString = com.google.gson.Gson().toJson(exportJson)
                
                logAuditEvent(AuditEventType.DATA_EXPORTED, "Exported ${events.size} notification events")
                Log.i(TAG, "Exported ${events.size} notification events")
                
                jsonString
            } catch (e: Exception) {
                Log.e(TAG, "Failed to export notification data", e)
                "{\"error\":\"Failed to export notification data: ${e.message}\"}"
            }
        }
    }
    
    /**
     * Disables features that depend on notification access
     */
    private suspend fun disableDependentFeatures() {
        try {
            // Update privacy settings to disable notification collection
            val database = AppDatabase.getInstance(context)
            val privacySettings = database.privacySettingsDao().getSettings()
            
            if (privacySettings != null) {
                val enabledCollectors = com.google.gson.Gson()
                    .fromJson(privacySettings.enabledCollectors, Array<String>::class.java)
                    .toMutableList()
                
                if (enabledCollectors.remove("notifications")) {
                    val updatedSettings = privacySettings.copy(
                        enabledCollectors = com.google.gson.Gson().toJson(enabledCollectors),
                        lastUpdated = System.currentTimeMillis()
                    )
                    database.privacySettingsDao().update(updatedSettings)
                    
                    Log.i(TAG, "Disabled notification collection in privacy settings")
                }
            }
        } catch (e: Exception) {
            Log.w(TAG, "Failed to disable dependent features", e)
        }
    }
    
    /**
     * Re-enables features that depend on notification access
     */
    private suspend fun enableDependentFeatures() {
        try {
            // Update privacy settings to re-enable notification collection
            val database = AppDatabase.getInstance(context)
            val privacySettings = database.privacySettingsDao().getSettings()
            
            if (privacySettings != null) {
                val enabledCollectors = com.google.gson.Gson()
                    .fromJson(privacySettings.enabledCollectors, Array<String>::class.java)
                    .toMutableList()
                
                if (!enabledCollectors.contains("notifications")) {
                    enabledCollectors.add("notifications")
                    val updatedSettings = privacySettings.copy(
                        enabledCollectors = com.google.gson.Gson().toJson(enabledCollectors),
                        lastUpdated = System.currentTimeMillis()
                    )
                    database.privacySettingsDao().update(updatedSettings)
                    
                    Log.i(TAG, "Re-enabled notification collection in privacy settings")
                }
            }
        } catch (e: Exception) {
            Log.w(TAG, "Failed to enable dependent features", e)
        }
    }
    
    /**
     * Gets the timestamp of the last successful collection
     */
    private suspend fun getLastCollectionTime(): Long {
        return try {
            val database = AppDatabase.getInstance(context)
            val recentEvents = database.notificationEventDao().getEventsByTimeRange(
                System.currentTimeMillis() - (24 * 60 * 60 * 1000L), // Last 24 hours
                System.currentTimeMillis()
            )
            
            recentEvents.maxOfOrNull { it.createdAt } ?: 0L
        } catch (e: Exception) {
            Log.w(TAG, "Failed to get last collection time", e)
            0L
        }
    }
    
    /**
     * Logs audit events for compliance and debugging
     */
    private suspend fun logAuditEvent(eventType: AuditEventType, details: String) {
        try {
            val database = AppDatabase.getInstance(context)
            val auditEntry = AuditLogEntity(
                timestamp = System.currentTimeMillis(),
                eventType = eventType.name,
                details = """{"component":"NotificationManager","details":"$details"}""",
                userId = null
            )
            
            database.auditLogDao().insert(auditEntry)
        } catch (e: Exception) {
            Log.w(TAG, "Failed to log audit event", e)
        }
    }
}

/**
 * Data class for notification collection statistics
 */
data class NotificationCollectionStats(
    val totalNotifications: Int = 0,
    val notificationsPosted: Int = 0,
    val notificationsOpened: Int = 0,
    val notificationsDismissed: Int = 0,
    val hasPermission: Boolean = false,
    val lastCollectionTime: Long = 0L
)