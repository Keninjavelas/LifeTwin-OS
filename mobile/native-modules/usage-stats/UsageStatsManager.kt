package com.lifetwin.mlp.usagestats

import android.content.Context
import android.util.Log
import com.lifetwin.mlp.db.AppDatabase
import com.lifetwin.mlp.db.AuditLogEntity
import com.lifetwin.mlp.db.AuditEventType
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

private const val TAG = "UsageStatsManager"

/**
 * Manager class for coordinating usage stats collection across app lifecycle
 */
class UsageStatsManager(private val context: Context) {
    
    private val collector = UsageStatsCollector(context)
    
    /**
     * Initializes usage stats collection system
     */
    suspend fun initialize() {
        withContext(Dispatchers.IO) {
            try {
                Log.i(TAG, "Initializing usage stats collection system")
                
                // Check if we have permission
                if (!collector.isPermissionGranted()) {
                    Log.w(TAG, "Usage stats permission not granted")
                    logAuditEvent(AuditEventType.PERMISSION_REVOKED, "Usage stats permission not available")
                    return@withContext
                }
                
                // Start the collector
                collector.startCollection()
                
                // Schedule periodic background collection
                UsageStatsWorker.schedulePeriodicCollection(context)
                
                logAuditEvent(AuditEventType.COLLECTOR_ENABLED, "Usage stats collection initialized")
                Log.i(TAG, "Usage stats collection system initialized successfully")
                
            } catch (e: Exception) {
                Log.e(TAG, "Failed to initialize usage stats collection", e)
                logAuditEvent(AuditEventType.COLLECTOR_DISABLED, "Failed to initialize: ${e.message}")
            }
        }
    }
    
    /**
     * Stops usage stats collection
     */
    suspend fun shutdown() {
        withContext(Dispatchers.IO) {
            try {
                Log.i(TAG, "Shutting down usage stats collection")
                
                // Stop the collector
                collector.stopCollection()
                
                // Cancel background work
                UsageStatsWorker.cancelCollection(context)
                
                logAuditEvent(AuditEventType.COLLECTOR_DISABLED, "Usage stats collection stopped")
                Log.i(TAG, "Usage stats collection stopped")
                
            } catch (e: Exception) {
                Log.e(TAG, "Failed to shutdown usage stats collection", e)
            }
        }
    }
    
    /**
     * Checks if collection is currently active
     */
    fun isCollectionActive(): Boolean {
        return collector.isCollectionActive()
    }
    
    /**
     * Gets the current collector instance
     */
    fun getCollector(): UsageStatsCollector {
        return collector
    }
    
    /**
     * Requests usage stats permission from user
     */
    suspend fun requestPermission(): Boolean {
        return withContext(Dispatchers.Main) {
            try {
                val result = collector.requestPermission()
                
                if (result) {
                    logAuditEvent(AuditEventType.PERMISSION_GRANTED, "Usage stats permission requested")
                } else {
                    logAuditEvent(AuditEventType.PERMISSION_REVOKED, "Failed to request usage stats permission")
                }
                
                result
            } catch (e: Exception) {
                Log.e(TAG, "Failed to request permission", e)
                false
            }
        }
    }
    
    /**
     * Performs manual collection for testing or immediate needs
     */
    suspend fun performManualCollection(): Int {
        return withContext(Dispatchers.IO) {
            try {
                if (!collector.isPermissionGranted()) {
                    Log.w(TAG, "Cannot perform manual collection: permission not granted")
                    return@withContext 0
                }
                
                val endTime = System.currentTimeMillis()
                val startTime = endTime - (60 * 60 * 1000L) // Last hour
                
                val events = collector.collectUsageEvents(
                    com.lifetwin.mlp.db.TimeRange(startTime, endTime)
                )
                
                Log.i(TAG, "Manual collection completed: ${events.size} events")
                logAuditEvent(AuditEventType.DATA_EXPORTED, "Manual collection: ${events.size} events")
                
                events.size
            } catch (e: Exception) {
                Log.e(TAG, "Manual collection failed", e)
                0
            }
        }
    }
    
    /**
     * Gets collection statistics
     */
    suspend fun getCollectionStats(): CollectionStats {
        return withContext(Dispatchers.IO) {
            try {
                val totalEvents = collector.getCollectedDataCount()
                val isActive = collector.isCollectionActive()
                val hasPermission = collector.isPermissionGranted()
                val isScheduled = UsageStatsWorker.isCollectionScheduled(context)
                
                CollectionStats(
                    totalEventsCollected = totalEvents,
                    isCollectionActive = isActive,
                    hasPermission = hasPermission,
                    isBackgroundScheduled = isScheduled,
                    lastCollectionTime = getLastCollectionTime()
                )
            } catch (e: Exception) {
                Log.e(TAG, "Failed to get collection stats", e)
                CollectionStats()
            }
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
                details = """{"component":"UsageStatsManager","details":"$details"}""",
                userId = null
            )
            
            database.auditLogDao().insert(auditEntry)
        } catch (e: Exception) {
            Log.w(TAG, "Failed to log audit event", e)
        }
    }
    
    /**
     * Gets the timestamp of the last successful collection
     */
    private suspend fun getLastCollectionTime(): Long {
        return try {
            val database = AppDatabase.getInstance(context)
            val recentEvents = database.usageEventDao().getEventsByTimeRange(
                System.currentTimeMillis() - (24 * 60 * 60 * 1000L), // Last 24 hours
                System.currentTimeMillis()
            )
            
            recentEvents.maxOfOrNull { it.createdAt } ?: 0L
        } catch (e: Exception) {
            Log.w(TAG, "Failed to get last collection time", e)
            0L
        }
    }
}

/**
 * Data class for collection statistics
 */
data class CollectionStats(
    val totalEventsCollected: Int = 0,
    val isCollectionActive: Boolean = false,
    val hasPermission: Boolean = false,
    val isBackgroundScheduled: Boolean = false,
    val lastCollectionTime: Long = 0L
)