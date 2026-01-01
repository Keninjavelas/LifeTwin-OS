package com.lifetwin.mlp.accessibility

import android.content.Context
import android.content.Intent
import android.provider.Settings
import android.util.Log
import com.lifetwin.mlp.db.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

private const val TAG = "InteractionManager"

/**
 * Manager class for coordinating interaction tracking and handling permission changes
 */
class InteractionManager(private val context: Context) {
    
    /**
     * Initializes interaction tracking system
     */
    suspend fun initialize() {
        withContext(Dispatchers.IO) {
            try {
                Log.i(TAG, "Initializing interaction tracking system")
                
                if (!isAccessibilityServiceEnabled()) {
                    Log.w(TAG, "Accessibility service not enabled")
                    logAuditEvent(AuditEventType.PERMISSION_REVOKED, "Accessibility service not enabled")
                    return@withContext
                }
                
                logAuditEvent(AuditEventType.COLLECTOR_ENABLED, "Interaction tracking initialized")
                Log.i(TAG, "Interaction tracking system initialized successfully")
                
            } catch (e: Exception) {
                Log.e(TAG, "Failed to initialize interaction tracking", e)
                logAuditEvent(AuditEventType.COLLECTOR_DISABLED, "Failed to initialize: ${e.message}")
            }
        }
    }
    
    /**
     * Handles accessibility permission revocation gracefully
     */
    suspend fun handlePermissionRevoked() {
        withContext(Dispatchers.IO) {
            try {
                Log.w(TAG, "Accessibility permission revoked")
                
                // Log the permission loss
                logAuditEvent(AuditEventType.PERMISSION_REVOKED, "Accessibility permission lost")
                
                // Disable dependent features gracefully
                disableDependentFeatures()
                
                // Switch to reduced functionality mode
                enableReducedFunctionalityMode()
                
                Log.i(TAG, "Gracefully handled accessibility permission revocation")
                
            } catch (e: Exception) {
                Log.e(TAG, "Failed to handle permission revocation gracefully", e)
            }
        }
    }
    
    /**
     * Handles accessibility permission restoration
     */
    suspend fun handlePermissionGranted() {
        withContext(Dispatchers.IO) {
            try {
                Log.i(TAG, "Accessibility permission granted")
                
                // Log the permission grant
                logAuditEvent(AuditEventType.PERMISSION_GRANTED, "Accessibility permission granted")
                
                // Re-enable features
                enableDependentFeatures()
                
                // Disable reduced functionality mode
                disableReducedFunctionalityMode()
                
                Log.i(TAG, "Successfully restored interaction tracking after permission grant")
                
            } catch (e: Exception) {
                Log.e(TAG, "Failed to handle permission grant", e)
            }
        }
    }
    
    /**
     * Requests accessibility permission from user
     */
    suspend fun requestPermission(): Boolean {
        return withContext(Dispatchers.Main) {
            try {
                val intent = Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS)
                intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                context.startActivity(intent)
                
                logAuditEvent(AuditEventType.PERMISSION_GRANTED, "Accessibility permission requested")
                Log.i(TAG, "Opened accessibility settings")
                true
            } catch (e: Exception) {
                Log.e(TAG, "Failed to open accessibility settings", e)
                logAuditEvent(AuditEventType.PERMISSION_REVOKED, "Failed to request accessibility permission")
                false
            }
        }
    }
    
    /**
     * Checks if accessibility service is currently enabled
     */
    fun isAccessibilityServiceEnabled(): Boolean {
        return InteractionAccessibilityService.isAccessibilityServiceEnabled(context)
    }
    
    /**
     * Gets collection statistics
     */
    suspend fun getCollectionStats(): InteractionCollectionStats {
        return withContext(Dispatchers.IO) {
            try {
                val database = AppDatabase.getInstance(context)
                val endTime = System.currentTimeMillis()
                val startTime = endTime - (24 * 60 * 60 * 1000L) // Last 24 hours
                
                val metrics = database.interactionMetricsDao()
                    .getMetricsByTimeRange(startTime, endTime)
                
                val totalTouches = metrics.sumOf { it.touchCount }
                val totalScrolls = metrics.sumOf { it.scrollEvents }
                val averageIntensity = if (metrics.isNotEmpty()) {
                    metrics.map { it.interactionIntensity }.average().toFloat()
                } else 0f
                
                val uniqueGestureTypes = metrics.flatMap { metric ->
                    try {
                        com.google.gson.Gson().fromJson(metric.gesturePatterns, Array<String>::class.java)
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
                }.distinct()
                
                InteractionCollectionStats(
                    totalInteractionWindows = metrics.size,
                    totalTouches = totalTouches,
                    totalScrolls = totalScrolls,
                    averageInteractionIntensity = averageIntensity,
                    uniqueGestureTypes = uniqueGestureTypes.size,
                    hasPermission = isAccessibilityServiceEnabled(),
                    isReducedFunctionalityMode = isReducedFunctionalityModeEnabled(),
                    lastCollectionTime = getLastCollectionTime()
                )
            } catch (e: Exception) {
                Log.e(TAG, "Failed to get collection stats", e)
                InteractionCollectionStats()
            }
        }
    }
    
    /**
     * Performs cleanup of old interaction data based on privacy settings
     */
    suspend fun performDataCleanup() {
        withContext(Dispatchers.IO) {
            try {
                val database = AppDatabase.getInstance(context)
                val privacySettings = database.privacySettingsDao().getSettings()
                val retentionDays = privacySettings?.dataRetentionDays ?: 7
                val cutoffTime = System.currentTimeMillis() - (retentionDays * 24 * 60 * 60 * 1000L)
                
                // Clean up old interaction metrics
                database.interactionMetricsDao().deleteOldMetrics(cutoffTime)
                
                Log.i(TAG, "Interaction data cleanup completed")
                logAuditEvent(AuditEventType.DATA_DELETED, "Old interaction data cleaned up")
                
            } catch (e: Exception) {
                Log.e(TAG, "Failed to perform data cleanup", e)
            }
        }
    }
    
    /**
     * Exports interaction data for user data portability
     */
    suspend fun exportInteractionData(timeRange: TimeRange? = null): String {
        return withContext(Dispatchers.IO) {
            try {
                val database = AppDatabase.getInstance(context)
                
                val metrics = if (timeRange != null) {
                    database.interactionMetricsDao().getMetricsByTimeRange(
                        timeRange.startTime,
                        timeRange.endTime
                    )
                } else {
                    // Export last 30 days by default
                    val endTime = System.currentTimeMillis()
                    val startTime = endTime - (30 * 24 * 60 * 60 * 1000L)
                    database.interactionMetricsDao().getMetricsByTimeRange(startTime, endTime)
                }
                
                // Convert to JSON format (privacy-preserving)
                val exportData = metrics.map { metric ->
                    mapOf(
                        "id" to metric.id,
                        "timestamp" to metric.timestamp,
                        "touchCount" to metric.touchCount,
                        "scrollEvents" to metric.scrollEvents,
                        "interactionIntensity" to metric.interactionIntensity,
                        "timeWindowStart" to metric.timeWindowStart,
                        "timeWindowEnd" to metric.timeWindowEnd,
                        "windowDuration" to (metric.timeWindowEnd - metric.timeWindowStart),
                        "gesturePatterns" to metric.gesturePatterns,
                        "createdAt" to metric.createdAt
                    )
                }
                
                val exportJson = mapOf(
                    "exportType" to "interaction_data",
                    "exportTime" to System.currentTimeMillis(),
                    "metricCount" to metrics.size,
                    "totalTouches" to metrics.sumOf { it.touchCount },
                    "totalScrolls" to metrics.sumOf { it.scrollEvents },
                    "averageIntensity" to if (metrics.isNotEmpty()) {
                        metrics.map { it.interactionIntensity }.average()
                    } else 0.0,
                    "metrics" to exportData
                )
                
                val jsonString = com.google.gson.Gson().toJson(exportJson)
                
                logAuditEvent(AuditEventType.DATA_EXPORTED, "Exported ${metrics.size} interaction metrics")
                Log.i(TAG, "Exported ${metrics.size} interaction metrics")
                
                jsonString
            } catch (e: Exception) {
                Log.e(TAG, "Failed to export interaction data", e)
                "{\"error\":\"Failed to export interaction data: ${e.message}\"}"
            }
        }
    }
    
    /**
     * Enables reduced functionality mode when accessibility permission is not available
     */
    private suspend fun enableReducedFunctionalityMode() {
        try {
            val database = AppDatabase.getInstance(context)
            val privacySettings = database.privacySettingsDao().getSettings()
            
            if (privacySettings != null) {
                // Store the fact that we're in reduced functionality mode
                val updatedSettings = privacySettings.copy(
                    anonymizationSettings = updateAnonymizationSettings(
                        privacySettings.anonymizationSettings,
                        "reducedFunctionalityMode" to true
                    ),
                    lastUpdated = System.currentTimeMillis()
                )
                database.privacySettingsDao().update(updatedSettings)
                
                Log.i(TAG, "Enabled reduced functionality mode")
            }
        } catch (e: Exception) {
            Log.w(TAG, "Failed to enable reduced functionality mode", e)
        }
    }
    
    /**
     * Disables reduced functionality mode when accessibility permission is restored
     */
    private suspend fun disableReducedFunctionalityMode() {
        try {
            val database = AppDatabase.getInstance(context)
            val privacySettings = database.privacySettingsDao().getSettings()
            
            if (privacySettings != null) {
                val updatedSettings = privacySettings.copy(
                    anonymizationSettings = updateAnonymizationSettings(
                        privacySettings.anonymizationSettings,
                        "reducedFunctionalityMode" to false
                    ),
                    lastUpdated = System.currentTimeMillis()
                )
                database.privacySettingsDao().update(updatedSettings)
                
                Log.i(TAG, "Disabled reduced functionality mode")
            }
        } catch (e: Exception) {
            Log.w(TAG, "Failed to disable reduced functionality mode", e)
        }
    }
    
    /**
     * Checks if reduced functionality mode is currently enabled
     */
    private suspend fun isReducedFunctionalityModeEnabled(): Boolean {
        return try {
            val database = AppDatabase.getInstance(context)
            val privacySettings = database.privacySettingsDao().getSettings()
            
            if (privacySettings != null) {
                val anonymizationJson = com.google.gson.Gson().fromJson(
                    privacySettings.anonymizationSettings,
                    Map::class.java
                ) as? Map<String, Any>
                
                anonymizationJson?.get("reducedFunctionalityMode") as? Boolean ?: false
            } else false
        } catch (e: Exception) {
            Log.w(TAG, "Failed to check reduced functionality mode", e)
            false
        }
    }
    
    /**
     * Disables features that depend on accessibility service
     */
    private suspend fun disableDependentFeatures() {
        try {
            // Update privacy settings to disable interaction collection
            val database = AppDatabase.getInstance(context)
            val privacySettings = database.privacySettingsDao().getSettings()
            
            if (privacySettings != null) {
                val enabledCollectors = com.google.gson.Gson()
                    .fromJson(privacySettings.enabledCollectors, Array<String>::class.java)
                    .toMutableList()
                
                if (enabledCollectors.remove("interactions")) {
                    val updatedSettings = privacySettings.copy(
                        enabledCollectors = com.google.gson.Gson().toJson(enabledCollectors),
                        lastUpdated = System.currentTimeMillis()
                    )
                    database.privacySettingsDao().update(updatedSettings)
                    
                    Log.i(TAG, "Disabled interaction collection in privacy settings")
                }
            }
        } catch (e: Exception) {
            Log.w(TAG, "Failed to disable dependent features", e)
        }
    }
    
    /**
     * Re-enables features that depend on accessibility service
     */
    private suspend fun enableDependentFeatures() {
        try {
            // Update privacy settings to re-enable interaction collection
            val database = AppDatabase.getInstance(context)
            val privacySettings = database.privacySettingsDao().getSettings()
            
            if (privacySettings != null) {
                val enabledCollectors = com.google.gson.Gson()
                    .fromJson(privacySettings.enabledCollectors, Array<String>::class.java)
                    .toMutableList()
                
                if (!enabledCollectors.contains("interactions")) {
                    enabledCollectors.add("interactions")
                    val updatedSettings = privacySettings.copy(
                        enabledCollectors = com.google.gson.Gson().toJson(enabledCollectors),
                        lastUpdated = System.currentTimeMillis()
                    )
                    database.privacySettingsDao().update(updatedSettings)
                    
                    Log.i(TAG, "Re-enabled interaction collection in privacy settings")
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
            val recentMetrics = database.interactionMetricsDao().getMetricsByTimeRange(
                System.currentTimeMillis() - (24 * 60 * 60 * 1000L), // Last 24 hours
                System.currentTimeMillis()
            )
            
            recentMetrics.maxOfOrNull { it.createdAt } ?: 0L
        } catch (e: Exception) {
            Log.w(TAG, "Failed to get last collection time", e)
            0L
        }
    }
    
    /**
     * Helper function to update anonymization settings JSON
     */
    private fun updateAnonymizationSettings(currentSettings: String, update: Pair<String, Any>): String {
        return try {
            val settingsMap = com.google.gson.Gson().fromJson(
                currentSettings,
                Map::class.java
            ).toMutableMap()
            
            settingsMap[update.first] = update.second
            com.google.gson.Gson().toJson(settingsMap)
        } catch (e: Exception) {
            Log.w(TAG, "Failed to update anonymization settings", e)
            currentSettings
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
                details = """{"component":"InteractionManager","details":"$details"}""",
                userId = null
            )
            
            database.auditLogDao().insert(auditEntry)
        } catch (e: Exception) {
            Log.w(TAG, "Failed to log audit event", e)
        }
    }
}

/**
 * Data class for interaction collection statistics
 */
data class InteractionCollectionStats(
    val totalInteractionWindows: Int = 0,
    val totalTouches: Int = 0,
    val totalScrolls: Int = 0,
    val averageInteractionIntensity: Float = 0f,
    val uniqueGestureTypes: Int = 0,
    val hasPermission: Boolean = false,
    val isReducedFunctionalityMode: Boolean = false,
    val lastCollectionTime: Long = 0L
)