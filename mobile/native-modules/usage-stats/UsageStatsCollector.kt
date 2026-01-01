package com.lifetwin.mlp.usagestats

import android.app.usage.UsageStats
import android.app.usage.UsageStatsManager
import android.app.usage.UsageEvents
import android.content.Context
import android.content.Intent
import android.provider.Settings
import android.util.Log
import com.lifetwin.mlp.db.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.util.UUID

private const val TAG = "UsageStatsCollector"

class UsageStatsCollector(private val context: Context) : com.lifetwin.mlp.db.UsageStatsCollector {
    
    private val usageStatsManager: UsageStatsManager? =
        context.getSystemService(Context.USAGE_STATS_SERVICE) as? UsageStatsManager
    
    @Volatile
    private var isCollecting = false
    
    override suspend fun startCollection() {
        withContext(Dispatchers.IO) {
            if (!isPermissionGranted()) {
                Log.w(TAG, "Cannot start collection: Usage access permission not granted")
                return@withContext
            }
            
            isCollecting = true
            Log.i(TAG, "Usage stats collection started")
        }
    }
    
    override suspend fun stopCollection() {
        withContext(Dispatchers.IO) {
            isCollecting = false
            Log.i(TAG, "Usage stats collection stopped")
        }
    }
    
    override fun isCollectionActive(): Boolean = isCollecting
    
    override fun getCollectorType(): CollectorType = CollectorType.USAGE_STATS
    
    override suspend fun getCollectedDataCount(): Int {
        return withContext(Dispatchers.IO) {
            try {
                val database = AppDatabase.getInstance(context)
                val endTime = System.currentTimeMillis()
                val startTime = endTime - (24 * 60 * 60 * 1000L) // Last 24 hours
                database.usageEventDao().getEventCountByTimeRange(startTime, endTime)
            } catch (e: Exception) {
                Log.e(TAG, "Failed to get collected data count", e)
                0
            }
        }
    }
    
    override suspend fun collectUsageEvents(timeRange: TimeRange): List<UsageEvent> {
        return withContext(Dispatchers.IO) {
            val usm = usageStatsManager ?: run {
                Log.w(TAG, "UsageStatsManager not available on this device")
                return@withContext emptyList()
            }
            
            if (!isPermissionGranted()) {
                Log.w(TAG, "Usage access permission not granted")
                return@withContext emptyList()
            }
            
            try {
                val events = mutableListOf<UsageEvent>()
                
                // Query usage events for the specified time range
                val usageEvents = usm.queryEvents(timeRange.startTime, timeRange.endTime)
                val eventMap = mutableMapOf<String, MutableList<UsageEvents.Event>>()
                
                // Group events by package name
                val event = UsageEvents.Event()
                while (usageEvents.hasNextEvent()) {
                    usageEvents.getNextEvent(event)
                    val packageName = event.packageName ?: continue
                    eventMap.getOrPut(packageName) { mutableListOf() }.add(
                        UsageEvents.Event().apply {
                            packageName = event.packageName
                            className = event.className
                            eventType = event.eventType
                            timeStamp = event.timeStamp
                        }
                    )
                }
                
                // Process events into usage sessions
                eventMap.forEach { (packageName, packageEvents) ->
                    packageEvents.sortBy { it.timeStamp }
                    
                    var sessionStart: Long? = null
                    var totalForegroundTime = 0L
                    
                    packageEvents.forEach { evt ->
                        when (evt.eventType) {
                            UsageEvents.Event.ACTIVITY_RESUMED -> {
                                sessionStart = evt.timeStamp
                            }
                            UsageEvents.Event.ACTIVITY_PAUSED -> {
                                sessionStart?.let { start ->
                                    val duration = evt.timeStamp - start
                                    totalForegroundTime += duration
                                    
                                    events.add(
                                        UsageEvent(
                                            packageName = packageName,
                                            startTime = start,
                                            endTime = evt.timeStamp,
                                            totalTimeInForeground = duration,
                                            lastTimeUsed = evt.timeStamp,
                                            eventType = UsageEventType.ACTIVITY_PAUSED
                                        )
                                    )
                                }
                                sessionStart = null
                            }
                            UsageEvents.Event.USER_INTERACTION -> {
                                events.add(
                                    UsageEvent(
                                        packageName = packageName,
                                        startTime = evt.timeStamp,
                                        endTime = evt.timeStamp,
                                        totalTimeInForeground = 0L,
                                        lastTimeUsed = evt.timeStamp,
                                        eventType = UsageEventType.USER_INTERACTION
                                    )
                                )
                            }
                        }
                    }
                }
                
                // Store events in database
                storeUsageEvents(events)
                
                Log.d(TAG, "Collected ${events.size} usage events for time range")
                events
                
            } catch (e: SecurityException) {
                Log.w(TAG, "Missing PACKAGE_USAGE_STATS permission", e)
                emptyList()
            } catch (e: Exception) {
                Log.e(TAG, "Failed to query usage stats", e)
                emptyList()
            }
        }
    }
    
    override fun isPermissionGranted(): Boolean {
        val usm = usageStatsManager ?: return false
        
        // Check if we have usage access by trying to query recent stats
        val endTime = System.currentTimeMillis()
        val startTime = endTime - 60000L // Last minute
        
        return try {
            val stats = usm.queryUsageStats(UsageStatsManager.INTERVAL_DAILY, startTime, endTime)
            stats.isNotEmpty()
        } catch (e: SecurityException) {
            false
        } catch (e: Exception) {
            Log.w(TAG, "Error checking usage stats permission", e)
            false
        }
    }
    
    override suspend fun requestPermission(): Boolean {
        return withContext(Dispatchers.Main) {
            try {
                val intent = Intent(Settings.ACTION_USAGE_ACCESS_SETTINGS)
                intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                context.startActivity(intent)
                
                // Note: This doesn't wait for the user to grant permission
                // The caller should check isPermissionGranted() later
                Log.i(TAG, "Opened usage access settings")
                true
            } catch (e: Exception) {
                Log.e(TAG, "Failed to open usage access settings", e)
                false
            }
        }
    }
    
    /**
     * Legacy method for backward compatibility
     */
    fun pollRecentEvents() {
        if (!isCollecting) return
        
        val endTime = System.currentTimeMillis()
        val startTime = endTime - 60L * 60L * 1000L // 1 hour
        
        // Use the new collectUsageEvents method
        kotlinx.coroutines.GlobalScope.launch {
            collectUsageEvents(TimeRange(startTime, endTime))
        }
    }
    
    /**
     * Stores usage events in the database
     */
    private suspend fun storeUsageEvents(events: List<UsageEvent>) {
        try {
            val database = AppDatabase.getInstance(context)
            val entities = events.map { event ->
                UsageEventEntity(
                    id = event.id,
                    packageName = event.packageName,
                    startTime = event.startTime,
                    endTime = event.endTime,
                    totalTimeInForeground = event.totalTimeInForeground,
                    lastTimeUsed = event.lastTimeUsed,
                    eventType = event.eventType.name
                )
            }
            
            database.usageEventDao().insertAll(entities)
            
            // Also create raw events for processing
            val rawEvents = events.map { event ->
                RawEventEntity(
                    id = UUID.randomUUID().toString(),
                    timestamp = event.startTime,
                    eventType = "usage",
                    packageName = event.packageName,
                    duration = event.totalTimeInForeground,
                    metadata = DBHelper.encryptMetadata(
                        """{"eventType":"${event.eventType}","endTime":${event.endTime},"lastTimeUsed":${event.lastTimeUsed}}"""
                    )
                )
            }
            
            rawEvents.forEach { rawEvent ->
                database.rawEventDao().insert(rawEvent)
            }
            
            Log.d(TAG, "Stored ${events.size} usage events in database")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to store usage events", e)
        }
    }
    
    /**
     * Gets usage events from database for a time range
     */
    suspend fun getStoredUsageEvents(timeRange: TimeRange): List<UsageEvent> {
        return withContext(Dispatchers.IO) {
            try {
                val database = AppDatabase.getInstance(context)
                val entities = database.usageEventDao().getEventsByTimeRange(
                    timeRange.startTime,
                    timeRange.endTime
                )
                
                entities.map { entity ->
                    UsageEvent(
                        id = entity.id,
                        packageName = entity.packageName,
                        startTime = entity.startTime,
                        endTime = entity.endTime,
                        totalTimeInForeground = entity.totalTimeInForeground,
                        lastTimeUsed = entity.lastTimeUsed,
                        eventType = UsageEventType.valueOf(entity.eventType)
                    )
                }
            } catch (e: Exception) {
                Log.e(TAG, "Failed to get stored usage events", e)
                emptyList()
            }
        }
    }
}

