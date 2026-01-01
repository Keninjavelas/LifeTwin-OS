package com.lifetwin.mlp.screenevents

import android.content.Context
import android.util.Log
import com.lifetwin.mlp.db.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

private const val TAG = "ScreenEventManager"

/**
 * Manager class for coordinating screen event collection and session management
 */
class ScreenEventManager(private val context: Context) {
    
    private var receiver: ScreenEventReceiver? = null
    
    /**
     * Initializes screen event collection system
     */
    suspend fun initialize() {
        withContext(Dispatchers.IO) {
            try {
                Log.i(TAG, "Initializing screen event collection system")
                
                // Create and register receiver
                val screenReceiver = ScreenEventReceiver()
                ScreenEventReceiver.register(context, screenReceiver)
                receiver = screenReceiver
                
                // Start collection
                screenReceiver.startCollection()
                
                logAuditEvent(AuditEventType.COLLECTOR_ENABLED, "Screen event collection initialized")
                Log.i(TAG, "Screen event collection system initialized successfully")
                
            } catch (e: Exception) {
                Log.e(TAG, "Failed to initialize screen event collection", e)
                logAuditEvent(AuditEventType.COLLECTOR_DISABLED, "Failed to initialize: ${e.message}")
            }
        }
    }
    
    /**
     * Stops screen event collection
     */
    suspend fun shutdown() {
        withContext(Dispatchers.IO) {
            try {
                Log.i(TAG, "Shutting down screen event collection")
                
                receiver?.let { screenReceiver ->
                    screenReceiver.stopCollection()
                    ScreenEventReceiver.unregister(context, screenReceiver)
                }
                receiver = null
                
                logAuditEvent(AuditEventType.COLLECTOR_DISABLED, "Screen event collection stopped")
                Log.i(TAG, "Screen event collection stopped")
                
            } catch (e: Exception) {
                Log.e(TAG, "Failed to shutdown screen event collection", e)
            }
        }
    }
    
    /**
     * Checks if collection is currently active
     */
    fun isCollectionActive(): Boolean {
        return receiver?.isCollectionActive() ?: false
    }
    
    /**
     * Gets the current receiver instance
     */
    fun getReceiver(): ScreenEventReceiver? {
        return receiver
    }
    
    /**
     * Gets the current active screen session
     */
    suspend fun getCurrentSession(): ScreenSession? {
        return receiver?.getCurrentSession()
    }
    
    /**
     * Gets screen sessions for a specific time range
     */
    suspend fun getSessionsByTimeRange(timeRange: TimeRange): List<ScreenSession> {
        return receiver?.getSessionsByTimeRange(timeRange) ?: emptyList()
    }
    
    /**
     * Gets total screen time for a specific time range
     */
    suspend fun getTotalScreenTime(timeRange: TimeRange): Long {
        return receiver?.getTotalScreenTime(timeRange) ?: 0L
    }
    
    /**
     * Gets collection statistics
     */
    suspend fun getCollectionStats(): ScreenEventCollectionStats {
        return withContext(Dispatchers.IO) {
            try {
                val database = AppDatabase.getInstance(context)
                val endTime = System.currentTimeMillis()
                val startTime = endTime - (24 * 60 * 60 * 1000L) // Last 24 hours
                
                val sessions = database.screenSessionDao().getSessionsByTimeRange(startTime, endTime)
                val totalScreenTime = database.screenSessionDao().getTotalScreenTimeByRange(startTime, endTime) ?: 0L
                val averageSessionLength = if (sessions.isNotEmpty()) {
                    sessions.mapNotNull { session ->
                        session.endTime?.let { endTime -> endTime - session.startTime }
                    }.average().toLong()
                } else 0L
                
                val totalUnlocks = sessions.sumOf { it.unlockCount }
                val activeSessions = sessions.count { it.isActive }
                
                ScreenEventCollectionStats(
                    totalSessions = sessions.size,
                    totalScreenTime = totalScreenTime,
                    averageSessionLength = averageSessionLength,
                    totalUnlocks = totalUnlocks,
                    activeSessions = activeSessions,
                    isCollectionActive = isCollectionActive(),
                    lastCollectionTime = getLastCollectionTime()
                )
            } catch (e: Exception) {
                Log.e(TAG, "Failed to get collection stats", e)
                ScreenEventCollectionStats()
            }
        }
    }
    
    /**
     * Performs cleanup of old screen session data based on privacy settings
     */
    suspend fun performDataCleanup() {
        withContext(Dispatchers.IO) {
            try {
                val database = AppDatabase.getInstance(context)
                val privacySettings = database.privacySettingsDao().getSettings()
                val retentionDays = privacySettings?.dataRetentionDays ?: 7
                val cutoffTime = System.currentTimeMillis() - (retentionDays * 24 * 60 * 60 * 1000L)
                
                // Clean up old screen sessions
                database.screenSessionDao().deleteOldSessions(cutoffTime)
                
                Log.i(TAG, "Screen event data cleanup completed")
                logAuditEvent(AuditEventType.DATA_DELETED, "Old screen event data cleaned up")
                
            } catch (e: Exception) {
                Log.e(TAG, "Failed to perform data cleanup", e)
            }
        }
    }
    
    /**
     * Exports screen event data for user data portability
     */
    suspend fun exportScreenEventData(timeRange: TimeRange? = null): String {
        return withContext(Dispatchers.IO) {
            try {
                val database = AppDatabase.getInstance(context)
                
                val sessions = if (timeRange != null) {
                    database.screenSessionDao().getSessionsByTimeRange(
                        timeRange.startTime,
                        timeRange.endTime
                    )
                } else {
                    // Export last 30 days by default
                    val endTime = System.currentTimeMillis()
                    val startTime = endTime - (30 * 24 * 60 * 60 * 1000L)
                    database.screenSessionDao().getSessionsByTimeRange(startTime, endTime)
                }
                
                // Convert to JSON format
                val exportData = sessions.map { session ->
                    mapOf(
                        "sessionId" to session.sessionId,
                        "startTime" to session.startTime,
                        "endTime" to session.endTime,
                        "duration" to (session.endTime?.let { it - session.startTime }),
                        "unlockCount" to session.unlockCount,
                        "interactionIntensity" to session.interactionIntensity,
                        "isActive" to session.isActive,
                        "createdAt" to session.createdAt
                    )
                }
                
                val exportJson = mapOf(
                    "exportType" to "screen_event_data",
                    "exportTime" to System.currentTimeMillis(),
                    "sessionCount" to sessions.size,
                    "totalScreenTime" to sessions.mapNotNull { session ->
                        session.endTime?.let { it - session.startTime }
                    }.sum(),
                    "sessions" to exportData
                )
                
                val jsonString = com.google.gson.Gson().toJson(exportJson)
                
                logAuditEvent(AuditEventType.DATA_EXPORTED, "Exported ${sessions.size} screen sessions")
                Log.i(TAG, "Exported ${sessions.size} screen sessions")
                
                jsonString
            } catch (e: Exception) {
                Log.e(TAG, "Failed to export screen event data", e)
                "{\"error\":\"Failed to export screen event data: ${e.message}\"}"
            }
        }
    }
    
    /**
     * Calculates screen time statistics for a given time range
     */
    suspend fun getScreenTimeStats(timeRange: TimeRange): ScreenTimeStats {
        return withContext(Dispatchers.IO) {
            try {
                val database = AppDatabase.getInstance(context)
                val sessions = database.screenSessionDao().getSessionsByTimeRange(
                    timeRange.startTime,
                    timeRange.endTime
                )
                
                val completedSessions = sessions.filter { it.endTime != null }
                val totalScreenTime = completedSessions.sumOf { it.endTime!! - it.startTime }
                val averageSessionLength = if (completedSessions.isNotEmpty()) {
                    totalScreenTime / completedSessions.size
                } else 0L
                
                val totalUnlocks = sessions.sumOf { it.unlockCount }
                val longestSession = completedSessions.maxOfOrNull { it.endTime!! - it.startTime } ?: 0L
                val shortestSession = completedSessions.minOfOrNull { it.endTime!! - it.startTime } ?: 0L
                
                // Calculate hourly distribution
                val hourlyDistribution = mutableMapOf<Int, Long>()
                sessions.forEach { session ->
                    val startHour = java.util.Calendar.getInstance().apply {
                        timeInMillis = session.startTime
                    }.get(java.util.Calendar.HOUR_OF_DAY)
                    
                    val sessionDuration = session.endTime?.let { it - session.startTime } ?: 0L
                    hourlyDistribution[startHour] = (hourlyDistribution[startHour] ?: 0L) + sessionDuration
                }
                
                val peakUsageHour = hourlyDistribution.maxByOrNull { it.value }?.key ?: 0
                
                ScreenTimeStats(
                    totalScreenTime = totalScreenTime,
                    totalSessions = sessions.size,
                    averageSessionLength = averageSessionLength,
                    totalUnlocks = totalUnlocks,
                    longestSession = longestSession,
                    shortestSession = shortestSession,
                    peakUsageHour = peakUsageHour,
                    hourlyDistribution = hourlyDistribution
                )
            } catch (e: Exception) {
                Log.e(TAG, "Failed to get screen time stats", e)
                ScreenTimeStats()
            }
        }
    }
    
    /**
     * Gets the timestamp of the last successful collection
     */
    private suspend fun getLastCollectionTime(): Long {
        return try {
            val database = AppDatabase.getInstance(context)
            val recentSessions = database.screenSessionDao().getSessionsByTimeRange(
                System.currentTimeMillis() - (24 * 60 * 60 * 1000L), // Last 24 hours
                System.currentTimeMillis()
            )
            
            recentSessions.maxOfOrNull { it.createdAt } ?: 0L
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
                details = """{"component":"ScreenEventManager","details":"$details"}""",
                userId = null
            )
            
            database.auditLogDao().insert(auditEntry)
        } catch (e: Exception) {
            Log.w(TAG, "Failed to log audit event", e)
        }
    }
}

/**
 * Data class for screen event collection statistics
 */
data class ScreenEventCollectionStats(
    val totalSessions: Int = 0,
    val totalScreenTime: Long = 0L,
    val averageSessionLength: Long = 0L,
    val totalUnlocks: Int = 0,
    val activeSessions: Int = 0,
    val isCollectionActive: Boolean = false,
    val lastCollectionTime: Long = 0L
)

/**
 * Data class for detailed screen time statistics
 */
data class ScreenTimeStats(
    val totalScreenTime: Long = 0L,
    val totalSessions: Int = 0,
    val averageSessionLength: Long = 0L,
    val totalUnlocks: Int = 0,
    val longestSession: Long = 0L,
    val shortestSession: Long = 0L,
    val peakUsageHour: Int = 0,
    val hourlyDistribution: Map<Int, Long> = emptyMap()
)