package com.lifetwin.mlp.accessibility

import android.accessibilityservice.AccessibilityService
import android.accessibilityservice.AccessibilityServiceInfo
import android.content.Intent
import android.provider.Settings
import android.view.accessibility.AccessibilityEvent
import android.util.Log
import com.lifetwin.mlp.db.*
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import java.util.UUID
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicInteger

private const val TAG = "InteractionAccessibilityService"
private const val INTERACTION_WINDOW_MS = 60000L // 1 minute window for aggregation

/**
 * Privacy-compliant AccessibilityService for interaction pattern tracking
 * - Records only interaction patterns and intensity metrics
 * - Does NOT capture text content, coordinates, or sensitive information
 * - Aggregates interactions to preserve privacy
 * - Complies with Play Store policies for accessibility services
 */
class InteractionAccessibilityService : AccessibilityService(), com.lifetwin.mlp.db.InteractionAccessibilityService {
    
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    
    @Volatile
    private var isCollecting = false
    
    // Interaction counters for current window
    private val touchCount = AtomicInteger(0)
    private val scrollCount = AtomicInteger(0)
    private val gesturePatterns = ConcurrentHashMap<GestureType, Int>()
    private var currentWindowStart = System.currentTimeMillis()
    
    // Package filtering for privacy
    private val sensitivePackages = setOf(
        "com.android.inputmethod",
        "com.google.android.inputmethod",
        "com.samsung.android.honeyboard",
        "com.swiftkey.swiftkeyapp"
    )

    override fun onServiceConnected() {
        super.onServiceConnected()
        Log.i(TAG, "Accessibility service connected")
        
        // Configure service info for minimal permissions
        val info = AccessibilityServiceInfo().apply {
            eventTypes = AccessibilityEvent.TYPE_VIEW_CLICKED or
                        AccessibilityEvent.TYPE_VIEW_SCROLLED or
                        AccessibilityEvent.TYPE_GESTURE_DETECTION_START or
                        AccessibilityEvent.TYPE_GESTURE_DETECTION_END or
                        AccessibilityEvent.TYPE_TOUCH_INTERACTION_START or
                        AccessibilityEvent.TYPE_TOUCH_INTERACTION_END
            
            feedbackType = AccessibilityServiceInfo.FEEDBACK_GENERIC
            flags = AccessibilityServiceInfo.FLAG_INCLUDE_NOT_IMPORTANT_VIEWS
            notificationTimeout = 100
        }
        
        serviceInfo = info
        isCollecting = true
        
        scope.launch {
            logAuditEvent(AuditEventType.COLLECTOR_ENABLED, "Accessibility service connected")
        }
    }

    override fun onUnbind(intent: Intent?): Boolean {
        Log.i(TAG, "Accessibility service disconnected")
        isCollecting = false
        
        scope.launch {
            // Flush any remaining interaction data
            flushInteractionWindow()
            logAuditEvent(AuditEventType.COLLECTOR_DISABLED, "Accessibility service disconnected")
        }
        
        return super.onUnbind(intent)
    }

    override fun onAccessibilityEvent(event: AccessibilityEvent?) {
        if (!isCollecting || event == null) return
        
        scope.launch {
            try {
                processAccessibilityEvent(event)
            } catch (e: Exception) {
                Log.w(TAG, "Failed to process accessibility event", e)
            }
        }
    }

    override fun onInterrupt() {
        Log.w(TAG, "Accessibility service interrupted")
    }

    // Implementation of InteractionAccessibilityService interface

    override suspend fun startCollection() {
        isCollecting = true
        Log.i(TAG, "Interaction collection started")
    }

    override suspend fun stopCollection() {
        isCollecting = false
        flushInteractionWindow()
        Log.i(TAG, "Interaction collection stopped")
    }

    override fun isCollectionActive(): Boolean = isCollecting

    override fun getCollectorType(): CollectorType = CollectorType.INTERACTIONS

    override suspend fun getCollectedDataCount(): Int {
        return try {
            val database = AppDatabase.getInstance(applicationContext)
            val endTime = System.currentTimeMillis()
            val startTime = endTime - (24 * 60 * 60 * 1000L) // Last 24 hours
            
            val metrics = database.interactionMetricsDao().getMetricsByTimeRange(startTime, endTime)
            metrics.size
        } catch (e: Exception) {
            Log.e(TAG, "Failed to get collected data count", e)
            0
        }
    }

    override suspend fun getInteractionMetrics(timeRange: TimeRange): List<InteractionMetrics> {
        return try {
            val database = AppDatabase.getInstance(applicationContext)
            val entities = database.interactionMetricsDao().getMetricsByTimeRange(
                timeRange.startTime,
                timeRange.endTime
            )
            
            entities.map { entity ->
                val gestureTypes = try {
                    com.google.gson.Gson().fromJson(entity.gesturePatterns, Array<String>::class.java)
                        .mapNotNull { gestureString ->
                            try {
                                GestureType.valueOf(gestureString)
                            } catch (e: IllegalArgumentException) {
                                null
                            }
                        }
                } catch (e: Exception) {
                    emptyList()
                }
                
                InteractionMetrics(
                    id = entity.id,
                    timestamp = entity.timestamp,
                    touchCount = entity.touchCount,
                    scrollEvents = entity.scrollEvents,
                    gesturePatterns = gestureTypes,
                    interactionIntensity = entity.interactionIntensity,
                    timeWindow = TimeRange(entity.timeWindowStart, entity.timeWindowEnd)
                )
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to get interaction metrics", e)
            emptyList()
        }
    }

    override fun isAccessibilityServiceEnabled(): Boolean {
        val enabledServices = Settings.Secure.getString(
            contentResolver,
            Settings.Secure.ENABLED_ACCESSIBILITY_SERVICES
        )
        
        val serviceName = "${packageName}/${this::class.java.name}"
        return enabledServices?.contains(serviceName) == true
    }

    override suspend fun requestAccessibilityPermission(): Boolean {
        return try {
            val intent = Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS)
            intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            startActivity(intent)
            
            Log.i(TAG, "Opened accessibility settings")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to open accessibility settings", e)
            false
        }
    }

    // Private methods for event processing

    private suspend fun processAccessibilityEvent(event: AccessibilityEvent) {
        val packageName = event.packageName?.toString()
        
        // Filter out sensitive packages (keyboards, etc.)
        if (packageName != null && sensitivePackages.contains(packageName)) {
            return
        }
        
        // Check if we need to flush the current window
        val currentTime = System.currentTimeMillis()
        if (currentTime - currentWindowStart >= INTERACTION_WINDOW_MS) {
            flushInteractionWindow()
            startNewInteractionWindow(currentTime)
        }
        
        // Process the event based on type
        when (event.eventType) {
            AccessibilityEvent.TYPE_VIEW_CLICKED -> {
                touchCount.incrementAndGet()
                incrementGesturePattern(GestureType.TAP)
            }
            
            AccessibilityEvent.TYPE_VIEW_SCROLLED -> {
                scrollCount.incrementAndGet()
                incrementGesturePattern(GestureType.SCROLL_VERTICAL)
            }
            
            AccessibilityEvent.TYPE_GESTURE_DETECTION_START,
            AccessibilityEvent.TYPE_GESTURE_DETECTION_END -> {
                // Detect gesture patterns based on event properties
                detectGesturePattern(event)
            }
            
            AccessibilityEvent.TYPE_TOUCH_INTERACTION_START,
            AccessibilityEvent.TYPE_TOUCH_INTERACTION_END -> {
                touchCount.incrementAndGet()
            }
        }
        
        // Store legacy event for backward compatibility
        val eventType = when (event.eventType) {
            AccessibilityEvent.TYPE_VIEW_CLICKED -> "view_clicked"
            AccessibilityEvent.TYPE_VIEW_SCROLLED -> "view_scrolled"
            else -> "accessibility_event"
        }
        
        DBHelper.insertEventAsync(
            applicationContext,
            AppEventEntity(
                timestamp = currentTime,
                type = eventType,
                packageName = packageName
            )
        )
    }

    private fun detectGesturePattern(event: AccessibilityEvent) {
        // Basic gesture detection based on event properties
        // This is simplified - real implementation would analyze movement patterns
        when {
            event.scrollX != 0 || event.scrollY != 0 -> {
                if (event.scrollX > event.scrollY) {
                    incrementGesturePattern(GestureType.SCROLL_HORIZONTAL)
                } else {
                    incrementGesturePattern(GestureType.SCROLL_VERTICAL)
                }
            }
            else -> {
                incrementGesturePattern(GestureType.TAP)
            }
        }
    }

    private fun incrementGesturePattern(gestureType: GestureType) {
        gesturePatterns.compute(gestureType) { _, count -> (count ?: 0) + 1 }
    }

    private suspend fun flushInteractionWindow() {
        if (touchCount.get() == 0 && scrollCount.get() == 0 && gesturePatterns.isEmpty()) {
            return // Nothing to flush
        }
        
        try {
            val windowEnd = System.currentTimeMillis()
            val windowDuration = windowEnd - currentWindowStart
            
            // Calculate interaction intensity (interactions per minute)
            val totalInteractions = touchCount.get() + scrollCount.get()
            val interactionIntensity = if (windowDuration > 0) {
                (totalInteractions.toFloat() / windowDuration) * 60000f // per minute
            } else 0f
            
            // Create interaction metrics
            val metrics = InteractionMetrics(
                timestamp = windowEnd,
                touchCount = touchCount.get(),
                scrollEvents = scrollCount.get(),
                gesturePatterns = gesturePatterns.keys.toList(),
                interactionIntensity = interactionIntensity,
                timeWindow = TimeRange(currentWindowStart, windowEnd)
            )
            
            // Store in database
            storeInteractionMetrics(metrics)
            
            Log.d(TAG, "Flushed interaction window: ${totalInteractions} interactions, intensity: $interactionIntensity")
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to flush interaction window", e)
        }
    }

    private fun startNewInteractionWindow(startTime: Long) {
        currentWindowStart = startTime
        touchCount.set(0)
        scrollCount.set(0)
        gesturePatterns.clear()
    }

    private suspend fun storeInteractionMetrics(metrics: InteractionMetrics) {
        try {
            val database = AppDatabase.getInstance(applicationContext)
            
            // Store in interaction metrics table
            val entity = InteractionMetricsEntity(
                id = metrics.id,
                timestamp = metrics.timestamp,
                touchCount = metrics.touchCount,
                scrollEvents = metrics.scrollEvents,
                gesturePatterns = com.google.gson.Gson().toJson(
                    metrics.gesturePatterns.map { it.name }
                ),
                interactionIntensity = metrics.interactionIntensity,
                timeWindowStart = metrics.timeWindow.startTime,
                timeWindowEnd = metrics.timeWindow.endTime
            )
            
            database.interactionMetricsDao().insert(entity)
            
            // Also create raw event for processing
            val rawEvent = RawEventEntity(
                id = UUID.randomUUID().toString(),
                timestamp = metrics.timestamp,
                eventType = "interaction",
                packageName = null,
                duration = metrics.timeWindow.endTime - metrics.timeWindow.startTime,
                metadata = DBHelper.encryptMetadata(
                    """{
                        "touchCount": ${metrics.touchCount},
                        "scrollEvents": ${metrics.scrollEvents},
                        "interactionIntensity": ${metrics.interactionIntensity},
                        "gesturePatterns": ${com.google.gson.Gson().toJson(metrics.gesturePatterns.map { it.name })}
                    }"""
                )
            )
            
            database.rawEventDao().insert(rawEvent)
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to store interaction metrics", e)
        }
    }

    private suspend fun logAuditEvent(eventType: AuditEventType, details: String) {
        try {
            val database = AppDatabase.getInstance(applicationContext)
            val auditEntry = AuditLogEntity(
                timestamp = System.currentTimeMillis(),
                eventType = eventType.name,
                details = """{"component":"InteractionAccessibilityService","details":"$details"}""",
                userId = null
            )
            
            database.auditLogDao().insert(auditEntry)
        } catch (e: Exception) {
            Log.w(TAG, "Failed to log audit event", e)
        }
    }

    companion object {
        /**
         * Checks if accessibility service is enabled for this app
         */
        fun isAccessibilityServiceEnabled(context: android.content.Context): Boolean {
            val enabledServices = Settings.Secure.getString(
                context.contentResolver,
                Settings.Secure.ENABLED_ACCESSIBILITY_SERVICES
            )
            
            val serviceName = "${context.packageName}/${InteractionAccessibilityService::class.java.name}"
            return enabledServices?.contains(serviceName) == true
        }

        /**
         * Opens accessibility settings
         */
        fun requestAccessibilityPermission(context: android.content.Context): Boolean {
            return try {
                val intent = Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS)
                intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                context.startActivity(intent)
                true
            } catch (e: Exception) {
                Log.e(TAG, "Failed to open accessibility settings", e)
                false
            }
        }
    }
}
