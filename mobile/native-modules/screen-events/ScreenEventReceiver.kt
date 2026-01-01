package com.lifetwin.mlp.screenevents

import android.app.KeyguardManager
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.util.Log
import com.lifetwin.mlp.db.*
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import java.util.UUID

private const val TAG = "ScreenEventReceiver"
private const val SESSION_COALESCE_THRESHOLD = 5000L // 5 seconds

class ScreenEventReceiver : BroadcastReceiver(), com.lifetwin.mlp.db.ScreenEventReceiver {
    
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    
    @Volatile
    private var isCollecting = false
    
    @Volatile
    private var currentSession: ScreenSession? = null
    
    private var lastScreenOffTime: Long = 0L

    override fun onReceive(context: Context, intent: Intent) {
        scope.launch {
            try {
                when (intent.action) {
                    Intent.ACTION_SCREEN_ON -> handleScreenOn(context)
                    Intent.ACTION_SCREEN_OFF -> handleScreenOff(context)
                    Intent.ACTION_USER_PRESENT -> handleUserPresent(context)
                }
            } catch (e: Exception) {
                Log.e(TAG, "Failed to handle screen event: ${intent.action}", e)
            }
        }
    }

    // Implementation of ScreenEventReceiver interface
    
    override suspend fun startCollection() {
        isCollecting = true
        Log.i(TAG, "Screen event collection started")
    }

    override suspend fun stopCollection() {
        isCollecting = false
        // End current session if active
        currentSession?.let { session ->
            if (session.endTime == null) {
                endCurrentSession()
            }
        }
        Log.i(TAG, "Screen event collection stopped")
    }

    override fun isCollectionActive(): Boolean = isCollecting

    override fun getCollectorType(): CollectorType = CollectorType.SCREEN_EVENTS

    override suspend fun getCollectedDataCount(): Int {
        return try {
            val context = getContextFromReceiver() ?: return 0
            val database = AppDatabase.getInstance(context)
            val endTime = System.currentTimeMillis()
            val startTime = endTime - (24 * 60 * 60 * 1000L) // Last 24 hours
            
            val sessions = database.screenSessionDao().getSessionsByTimeRange(startTime, endTime)
            sessions.size
        } catch (e: Exception) {
            Log.e(TAG, "Failed to get collected data count", e)
            0
        }
    }

    override suspend fun getCurrentSession(): ScreenSession? {
        return currentSession
    }

    override suspend fun getSessionsByTimeRange(timeRange: TimeRange): List<ScreenSession> {
        return try {
            val context = getContextFromReceiver() ?: return emptyList()
            val database = AppDatabase.getInstance(context)
            val entities = database.screenSessionDao().getSessionsByTimeRange(
                timeRange.startTime,
                timeRange.endTime
            )
            
            entities.map { entity ->
                ScreenSession(
                    sessionId = entity.sessionId,
                    startTime = entity.startTime,
                    endTime = entity.endTime,
                    unlockCount = entity.unlockCount,
                    interactionIntensity = entity.interactionIntensity
                )
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to get sessions by time range", e)
            emptyList()
        }
    }

    override suspend fun getTotalScreenTime(timeRange: TimeRange): Long {
        return try {
            val context = getContextFromReceiver() ?: return 0L
            val database = AppDatabase.getInstance(context)
            database.screenSessionDao().getTotalScreenTimeByRange(
                timeRange.startTime,
                timeRange.endTime
            ) ?: 0L
        } catch (e: Exception) {
            Log.e(TAG, "Failed to get total screen time", e)
            0L
        }
    }

    // Screen event handlers

    private suspend fun handleScreenOn(context: Context) {
        if (!isCollecting) return
        
        val currentTime = System.currentTimeMillis()
        
        // Check if this is a rapid screen on/off event that should be coalesced
        if (shouldCoalesceWithPreviousSession(currentTime)) {
            Log.d(TAG, "Coalescing rapid screen on/off event")
            // Reactivate the previous session instead of creating a new one
            currentSession?.let { session ->
                val updatedSession = session.copy(endTime = null)
                currentSession = updatedSession
                updateSessionInDatabase(context, updatedSession)
                return
            }
        }
        
        // End any existing session first
        endCurrentSession(context)
        
        // Start new session
        val newSession = ScreenSession(
            sessionId = UUID.randomUUID().toString(),
            startTime = currentTime,
            endTime = null,
            unlockCount = 0,
            interactionIntensity = 0f
        )
        
        currentSession = newSession
        storeScreenSession(context, newSession)
        
        // Store raw event
        storeRawEvent(context, "screen_on", currentTime)
        
        // Legacy compatibility
        DBHelper.insertEventAsync(
            context,
            AppEventEntity(timestamp = currentTime, type = "screen", packageName = null)
        )
        
        Log.d(TAG, "Screen turned on, started session: ${newSession.sessionId}")
    }

    private suspend fun handleScreenOff(context: Context) {
        if (!isCollecting) return
        
        val currentTime = System.currentTimeMillis()
        lastScreenOffTime = currentTime
        
        // End current session
        endCurrentSession(context, currentTime)
        
        // Store raw event
        storeRawEvent(context, "screen_off", currentTime)
        
        // Legacy compatibility
        DBHelper.insertEventAsync(
            context,
            AppEventEntity(timestamp = currentTime, type = "screen_off", packageName = null)
        )
        
        Log.d(TAG, "Screen turned off")
    }

    private suspend fun handleUserPresent(context: Context) {
        if (!isCollecting) return
        
        val currentTime = System.currentTimeMillis()
        
        // Increment unlock count for current session
        currentSession?.let { session ->
            val updatedSession = session.copy(unlockCount = session.unlockCount + 1)
            currentSession = updatedSession
            updateSessionInDatabase(context, updatedSession)
            
            Log.d(TAG, "User unlocked device, unlock count: ${updatedSession.unlockCount}")
        }
        
        // Store raw event
        storeRawEvent(context, "user_present", currentTime)
    }

    private suspend fun endCurrentSession(context: Context? = null, endTime: Long = System.currentTimeMillis()) {
        currentSession?.let { session ->
            if (session.endTime == null) {
                val endedSession = session.copy(endTime = endTime)
                currentSession = null
                
                context?.let { ctx ->
                    updateSessionInDatabase(ctx, endedSession)
                }
                
                Log.d(TAG, "Ended session: ${session.sessionId}, duration: ${endTime - session.startTime}ms")
            }
        }
    }

    private fun shouldCoalesceWithPreviousSession(currentTime: Long): Boolean {
        return (currentTime - lastScreenOffTime) < SESSION_COALESCE_THRESHOLD
    }

    private suspend fun storeScreenSession(context: Context, session: ScreenSession) {
        try {
            val database = AppDatabase.getInstance(context)
            val entity = ScreenSessionEntity(
                sessionId = session.sessionId,
                startTime = session.startTime,
                endTime = session.endTime,
                unlockCount = session.unlockCount,
                interactionIntensity = session.interactionIntensity,
                isActive = session.endTime == null
            )
            
            database.screenSessionDao().insert(entity)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to store screen session", e)
        }
    }

    private suspend fun updateSessionInDatabase(context: Context, session: ScreenSession) {
        try {
            val database = AppDatabase.getInstance(context)
            val entity = ScreenSessionEntity(
                sessionId = session.sessionId,
                startTime = session.startTime,
                endTime = session.endTime,
                unlockCount = session.unlockCount,
                interactionIntensity = session.interactionIntensity,
                isActive = session.endTime == null
            )
            
            database.screenSessionDao().update(entity)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to update screen session", e)
        }
    }

    private suspend fun storeRawEvent(context: Context, eventType: String, timestamp: Long) {
        try {
            val database = AppDatabase.getInstance(context)
            val rawEvent = RawEventEntity(
                id = UUID.randomUUID().toString(),
                timestamp = timestamp,
                eventType = "screen",
                packageName = null,
                duration = null,
                metadata = DBHelper.encryptMetadata(
                    """{"screenEventType":"$eventType","sessionId":"${currentSession?.sessionId}"}"""
                )
            )
            
            database.rawEventDao().insert(rawEvent)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to store raw event", e)
        }
    }

    // Helper method to get context (this is a limitation of BroadcastReceiver)
    private fun getContextFromReceiver(): Context? {
        // This would need to be set by the registering component
        // For now, we'll handle this in the manager class
        return null
    }

    companion object {
        /**
         * Creates an IntentFilter for screen events
         */
        fun createIntentFilter(): IntentFilter {
            return IntentFilter().apply {
                addAction(Intent.ACTION_SCREEN_ON)
                addAction(Intent.ACTION_SCREEN_OFF)
                addAction(Intent.ACTION_USER_PRESENT)
            }
        }

        /**
         * Registers the screen event receiver with the given context
         */
        fun register(context: Context, receiver: ScreenEventReceiver) {
            val filter = createIntentFilter()
            context.registerReceiver(receiver, filter)
            Log.i(TAG, "Screen event receiver registered")
        }

        /**
         * Unregisters the screen event receiver
         */
        fun unregister(context: Context, receiver: ScreenEventReceiver) {
            try {
                context.unregisterReceiver(receiver)
                Log.i(TAG, "Screen event receiver unregistered")
            } catch (e: IllegalArgumentException) {
                Log.w(TAG, "Receiver was not registered", e)
            }
        }
    }
}
