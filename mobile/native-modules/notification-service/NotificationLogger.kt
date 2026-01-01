package com.lifetwin.mlp.notifications

import android.app.Notification
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.provider.Settings
import android.service.notification.NotificationListenerService
import android.service.notification.StatusBarNotification
import android.text.TextUtils
import android.util.Log
import com.lifetwin.mlp.db.*
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import java.util.UUID

private const val TAG = "NotificationLogger"

class NotificationLogger : NotificationListenerService(), com.lifetwin.mlp.db.NotificationLogger {
    
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    
    @Volatile
    private var isCollecting = false
    
    // System packages to filter out
    private val systemPackages = setOf(
        "android",
        "com.android.systemui",
        "com.android.settings",
        "com.android.vending", // Play Store updates
        "com.google.android.gms",
        "com.google.android.gsf"
    )

    override fun onListenerConnected() {
        super.onListenerConnected()
        Log.i(TAG, "NotificationListener connected")
        isCollecting = true
        
        scope.launch {
            logAuditEvent(AuditEventType.COLLECTOR_ENABLED, "Notification listener connected")
        }
    }

    override fun onListenerDisconnected() {
        super.onListenerDisconnected()
        Log.i(TAG, "NotificationListener disconnected")
        isCollecting = false
        
        scope.launch {
            logAuditEvent(AuditEventType.COLLECTOR_DISABLED, "Notification listener disconnected")
        }
    }

    override fun onNotificationPosted(sbn: StatusBarNotification) {
        scope.launch {
            try {
                val notificationData = createNotificationData(sbn, NotificationInteractionType.POSTED)
                if (shouldLogNotification(notificationData)) {
                    logNotificationPosted(notificationData)
                    Log.d(TAG, "Logged notification posted: ${sbn.packageName}")
                }
            } catch (e: Exception) {
                Log.e(TAG, "Failed to log notification posted", e)
            }
        }
    }

    override fun onNotificationRemoved(sbn: StatusBarNotification, reason: Int) {
        scope.launch {
            try {
                val interactionType = when (reason) {
                    REASON_CLICK -> NotificationInteractionType.OPENED
                    REASON_CANCEL,
                    REASON_CANCEL_ALL -> NotificationInteractionType.DISMISSED
                    else -> NotificationInteractionType.DISMISSED
                }
                
                val interaction = NotificationInteraction(
                    notificationId = createNotificationId(sbn),
                    interactionType = interactionType,
                    timestamp = System.currentTimeMillis()
                )
                
                logNotificationInteraction(interaction)
                Log.d(TAG, "Logged notification removed: ${sbn.packageName}, reason: $reason")
                
            } catch (e: Exception) {
                Log.e(TAG, "Failed to log notification removed", e)
            }
        }
    }

    // Implementation of NotificationLogger interface
    
    override suspend fun logNotificationPosted(notification: NotificationData) {
        try {
            val database = AppDatabase.getInstance(applicationContext)
            
            // Store in notification events table
            val entity = NotificationEventEntity(
                id = notification.id,
                packageName = notification.packageName,
                timestamp = notification.timestamp,
                category = notification.category,
                priority = notification.priority,
                hasActions = notification.hasActions,
                isOngoing = notification.isOngoing,
                interactionType = "posted"
            )
            
            database.notificationEventDao().insert(entity)
            
            // Also create raw event for processing
            val rawEvent = RawEventEntity(
                id = UUID.randomUUID().toString(),
                timestamp = notification.timestamp,
                eventType = "notification",
                packageName = notification.packageName,
                duration = null,
                metadata = DBHelper.encryptMetadata(
                    """{"category":"${notification.category}","priority":${notification.priority},"hasActions":${notification.hasActions},"isOngoing":${notification.isOngoing}}"""
                )
            )
            
            database.rawEventDao().insert(rawEvent)
            
            // Legacy compatibility
            DBHelper.insertEventAsync(
                applicationContext,
                AppEventEntity(
                    timestamp = notification.timestamp,
                    type = "notification",
                    packageName = notification.packageName
                )
            )
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to store notification posted event", e)
        }
    }

    override suspend fun logNotificationInteraction(interaction: NotificationInteraction) {
        try {
            val database = AppDatabase.getInstance(applicationContext)
            
            // Find the original notification to get package name
            val originalNotification = database.notificationEventDao()
                .getEventsByTimeRange(
                    interaction.timestamp - 300000L, // 5 minutes before
                    interaction.timestamp
                )
                .find { it.id == interaction.notificationId }
            
            if (originalNotification != null) {
                // Update the original notification with interaction type
                val updatedEntity = originalNotification.copy(
                    interactionType = interaction.interactionType.name.lowercase()
                )
                
                database.notificationEventDao().insert(updatedEntity)
                
                // Legacy compatibility
                val eventType = when (interaction.interactionType) {
                    NotificationInteractionType.OPENED -> "notification_opened"
                    NotificationInteractionType.DISMISSED -> "notification_removed"
                    else -> "notification_interaction"
                }
                
                DBHelper.insertEventAsync(
                    applicationContext,
                    AppEventEntity(
                        timestamp = interaction.timestamp,
                        type = eventType,
                        packageName = originalNotification.packageName
                    )
                )
            }
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to store notification interaction", e)
        }
    }

    override suspend fun startCollection() {
        // NotificationListenerService is started by the system
        // We just track the state
        isCollecting = true
        Log.i(TAG, "Notification collection started")
    }

    override suspend fun stopCollection() {
        isCollecting = false
        Log.i(TAG, "Notification collection stopped")
    }

    override fun isCollectionActive(): Boolean = isCollecting

    override fun getCollectorType(): CollectorType = CollectorType.NOTIFICATIONS

    override suspend fun getCollectedDataCount(): Int {
        return try {
            val database = AppDatabase.getInstance(applicationContext)
            val endTime = System.currentTimeMillis()
            val startTime = endTime - (24 * 60 * 60 * 1000L) // Last 24 hours
            database.notificationEventDao().getEventCountByTimeRange(startTime, endTime)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to get collected data count", e)
            0
        }
    }

    override fun isNotificationAccessGranted(): Boolean {
        val enabledListeners = Settings.Secure.getString(
            contentResolver,
            "enabled_notification_listeners"
        )
        
        val myComponent = ComponentName(this, NotificationLogger::class.java)
        return enabledListeners?.contains(myComponent.flattenToString()) == true
    }

    override suspend fun requestNotificationAccess(): Boolean {
        return try {
            val intent = Intent(Settings.ACTION_NOTIFICATION_LISTENER_SETTINGS)
            intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            startActivity(intent)
            
            Log.i(TAG, "Opened notification access settings")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to open notification access settings", e)
            false
        }
    }

    // Helper methods

    private fun createNotificationData(
        sbn: StatusBarNotification,
        interactionType: NotificationInteractionType
    ): NotificationData {
        val notification = sbn.notification
        
        return NotificationData(
            id = createNotificationId(sbn),
            packageName = sbn.packageName,
            timestamp = System.currentTimeMillis(),
            category = notification.category,
            priority = notification.priority,
            hasActions = notification.actions?.isNotEmpty() == true,
            isOngoing = notification.flags and Notification.FLAG_ONGOING_EVENT != 0
        )
    }

    private fun createNotificationId(sbn: StatusBarNotification): String {
        return "${sbn.packageName}_${sbn.id}_${sbn.postTime}"
    }

    private fun shouldLogNotification(notificationData: NotificationData): Boolean {
        // Filter out system notifications
        if (systemPackages.contains(notificationData.packageName)) {
            return false
        }
        
        // Filter out very low priority notifications
        if (notificationData.priority < Notification.PRIORITY_LOW) {
            return false
        }
        
        return true
    }

    private suspend fun logAuditEvent(eventType: AuditEventType, details: String) {
        try {
            val database = AppDatabase.getInstance(applicationContext)
            val auditEntry = AuditLogEntity(
                timestamp = System.currentTimeMillis(),
                eventType = eventType.name,
                details = """{"component":"NotificationLogger","details":"$details"}""",
                userId = null
            )
            
            database.auditLogDao().insert(auditEntry)
        } catch (e: Exception) {
            Log.w(TAG, "Failed to log audit event", e)
        }
    }

    companion object {
        /**
         * Checks if notification access is granted for this app
         */
        fun isNotificationAccessGranted(context: Context): Boolean {
            val enabledListeners = Settings.Secure.getString(
                context.contentResolver,
                "enabled_notification_listeners"
            )
            
            val myComponent = ComponentName(context, NotificationLogger::class.java)
            return enabledListeners?.contains(myComponent.flattenToString()) == true
        }

        /**
         * Opens notification access settings
         */
        fun requestNotificationAccess(context: Context): Boolean {
            return try {
                val intent = Intent(Settings.ACTION_NOTIFICATION_LISTENER_SETTINGS)
                intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                context.startActivity(intent)
                true
            } catch (e: Exception) {
                Log.e(TAG, "Failed to open notification access settings", e)
                false
            }
        }
    }
}
