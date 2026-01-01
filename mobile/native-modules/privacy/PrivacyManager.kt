package com.lifetwin.mlp.privacy

import android.content.Context
import android.util.Log
import com.lifetwin.mlp.db.*
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import java.util.*

private const val TAG = "PrivacyManager"

/**
 * Comprehensive privacy manager with user controls
 * - Manages privacy settings and collector enable/disable functionality
 * - Implements granular privacy level management (Minimal, Standard, Detailed)
 * - Provides data retention and anonymization controls
 * - Handles transparency and audit features
 * - Supports emergency privacy mode
 */
class PrivacyManager(private val context: Context) : com.lifetwin.mlp.db.PrivacyManager {
    
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private val gson = Gson()
    
    @Volatile
    private var currentSettings: PrivacySettings? = null
    
    @Volatile
    private var emergencyModeActive = false

    // Core privacy management interface implementation

    override suspend fun setCollectorEnabled(collector: CollectorType, enabled: Boolean) {
        try {
            val database = AppDatabase.getInstance(context)
            val currentSettings = getPrivacySettings()
            
            val updatedCollectors = currentSettings.enabledCollectors.toMutableSet()
            if (enabled) {
                updatedCollectors.add(collector)
            } else {
                updatedCollectors.remove(collector)
            }
            
            val updatedSettings = currentSettings.copy(enabledCollectors = updatedCollectors)
            savePrivacySettings(updatedSettings)
            
            // Log the change
            logAuditEvent(AuditLogEntry(
                eventType = AuditEventType.PRIVACY_SETTING_CHANGED,
                details = mapOf(
                    "collector" to collector.name,
                    "enabled" to enabled,
                    "changedBy" to "user"
                )
            ))
            
            Log.i(TAG, "Collector ${collector.name} ${if (enabled) "enabled" else "disabled"}")
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to set collector enabled state", e)
            throw e
        }
    }

    override suspend fun setDataRetentionPeriod(days: Int) {
        require(days > 0) { "Data retention period must be positive" }
        require(days <= 365) { "Data retention period cannot exceed 365 days" }
        
        try {
            val currentSettings = getPrivacySettings()
            val updatedSettings = currentSettings.copy(dataRetentionDays = days)
            savePrivacySettings(updatedSettings)
            
            // Trigger cleanup of old data
            cleanupOldData(days)
            
            logAuditEvent(AuditLogEntry(
                eventType = AuditEventType.PRIVACY_SETTING_CHANGED,
                details = mapOf(
                    "setting" to "dataRetentionDays",
                    "oldValue" to currentSettings.dataRetentionDays,
                    "newValue" to days
                )
            ))
            
            Log.i(TAG, "Data retention period set to $days days")
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to set data retention period", e)
            throw e
        }
    }

    /**
     * Cleans up old data based on retention period
     */
    private suspend fun cleanupOldData(retentionDays: Int) {
        try {
            val database = AppDatabase.getInstance(context)
            val cutoffTime = System.currentTimeMillis() - (retentionDays * 24 * 60 * 60 * 1000L)
            
            // Clean up old raw events
            val deletedRawEvents = database.rawEventDao().deleteOldProcessedEvents(cutoffTime)
            
            // Clean up old summaries if retention is very short
            if (retentionDays < 30) {
                val deletedSummaries = database.enhancedDailySummaryDao().deleteOldSummaries(cutoffTime)
                Log.d(TAG, "Deleted $deletedSummaries old summaries")
            }
            
            // Clean up old audit logs
            val deletedAuditLogs = database.auditLogDao().deleteOldLogs(cutoffTime)
            
            Log.i(TAG, "Data cleanup completed: $deletedRawEvents raw events, $deletedAuditLogs audit logs deleted")
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to cleanup old data", e)
        }
    }

    override suspend fun setPrivacyLevel(level: PrivacyLevel) {
        try {
            val currentSettings = getPrivacySettings()
            val updatedSettings = currentSettings.copy(privacyLevel = level)
            
            // Adjust enabled collectors based on privacy level
            val adjustedSettings = adjustCollectorsForPrivacyLevel(updatedSettings, level)
            savePrivacySettings(adjustedSettings)
            
            logAuditEvent(AuditLogEntry(
                eventType = AuditEventType.PRIVACY_SETTING_CHANGED,
                details = mapOf(
                    "setting" to "privacyLevel",
                    "oldValue" to currentSettings.privacyLevel.name,
                    "newValue" to level.name,
                    "adjustedCollectors" to adjustedSettings.enabledCollectors.map { it.name }
                )
            ))
            
            Log.i(TAG, "Privacy level set to ${level.name}")
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to set privacy level", e)
            throw e
        }
    }

    override suspend fun getPrivacySettings(): PrivacySettings {
        if (currentSettings != null && !emergencyModeActive) {
            return currentSettings!!
        }
        
        return try {
            val database = AppDatabase.getInstance(context)
            val settingsEntity = database.privacySettingsDao().getSettings()
            
            if (settingsEntity != null) {
                val settings = convertEntityToSettings(settingsEntity)
                currentSettings = settings
                settings
            } else {
                // Create default settings
                val defaultSettings = createDefaultPrivacySettings()
                savePrivacySettings(defaultSettings)
                defaultSettings
            }
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to get privacy settings", e)
            createDefaultPrivacySettings()
        }
    }

    override suspend fun isCollectorEnabled(collector: CollectorType): Boolean {
        return try {
            if (emergencyModeActive) {
                false // All collectors disabled in emergency mode
            } else {
                val settings = getPrivacySettings()
                settings.enabledCollectors.contains(collector)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to check collector enabled state", e)
            false // Default to disabled on error
        }
    }

    // Privacy level management

    /**
     * Adjusts enabled collectors based on privacy level
     */
    private fun adjustCollectorsForPrivacyLevel(
        settings: PrivacySettings, 
        level: PrivacyLevel
    ): PrivacySettings {
        val enabledCollectors = when (level) {
            PrivacyLevel.MINIMAL -> {
                // Only essential collectors
                setOf(CollectorType.SCREEN_EVENTS)
            }
            PrivacyLevel.STANDARD -> {
                // Standard set of collectors
                setOf(
                    CollectorType.SCREEN_EVENTS,
                    CollectorType.USAGE_STATS,
                    CollectorType.NOTIFICATIONS
                )
            }
            PrivacyLevel.DETAILED -> {
                // All collectors enabled
                CollectorType.values().toSet()
            }
        }
        
        return settings.copy(enabledCollectors = enabledCollectors)
    }

    // Data retention and anonymization controls

    /**
     * Sets anonymization options for collected data
     */
    suspend fun setAnonymizationOptions(options: AnonymizationOptions) {
        try {
            val currentSettings = getPrivacySettings()
            val updatedSettings = currentSettings.copy(anonymizationOptions = options)
            savePrivacySettings(updatedSettings)
            
            logAuditEvent(AuditLogEntry(
                eventType = AuditEventType.PRIVACY_SETTING_CHANGED,
                details = mapOf(
                    "setting" to "anonymizationOptions",
                    "timestampFuzzing" to options.enableTimestampFuzzing,
                    "identifierRemoval" to options.removeIdentifiers,
                    "locationObfuscation" to options.obfuscateLocation
                )
            ))
            
            Log.i(TAG, "Anonymization options updated")
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to set anonymization options", e)
            throw e
        }
    }

    /**
     * Applies anonymization to raw event data based on current settings
     */
    suspend fun anonymizeEventData(eventData: Map<String, Any>): Map<String, Any> {
        val settings = getPrivacySettings()
        val options = settings.anonymizationOptions
        var anonymizedData = eventData.toMutableMap()
        
        try {
            // Apply timestamp fuzzing
            if (options.enableTimestampFuzzing) {
                anonymizedData["timestamp"]?.let { timestamp ->
                    if (timestamp is Long) {
                        val fuzzingRange = options.timestampFuzzingMinutes * 60 * 1000L
                        val fuzzOffset = (Math.random() * fuzzingRange * 2 - fuzzingRange).toLong()
                        anonymizedData["timestamp"] = timestamp + fuzzOffset
                    }
                }
            }
            
            // Remove identifiers
            if (options.removeIdentifiers) {
                anonymizedData.remove("deviceId")
                anonymizedData.remove("userId")
                anonymizedData.remove("sessionId")
                
                // Replace package names with categories if category-only mode is enabled
                if (options.categoryOnlyMode) {
                    anonymizedData["packageName"]?.let { packageName ->
                        if (packageName is String) {
                            anonymizedData["appCategory"] = getAppCategory(packageName)
                            anonymizedData.remove("packageName")
                        }
                    }
                }
            }
            
            // Obfuscate location data
            if (options.obfuscateLocation) {
                anonymizedData.remove("latitude")
                anonymizedData.remove("longitude")
                anonymizedData.remove("location")
                
                // Replace with general area if needed
                anonymizedData["locationArea"] = "general_area"
            }
            
            return anonymizedData
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to anonymize event data", e)
            return eventData // Return original data if anonymization fails
        }
    }
    
    /**
     * Gets app category for anonymization purposes
     */
    private fun getAppCategory(packageName: String): String {
        return when {
            packageName.contains("social") || 
            packageName.contains("facebook") || 
            packageName.contains("twitter") || 
            packageName.contains("instagram") -> "social"
            
            packageName.contains("game") || 
            packageName.contains("play") -> "games"
            
            packageName.contains("news") || 
            packageName.contains("media") -> "news_media"
            
            packageName.contains("work") || 
            packageName.contains("office") || 
            packageName.contains("productivity") -> "productivity"
            
            packageName.contains("shopping") || 
            packageName.contains("commerce") -> "shopping"
            
            packageName.contains("health") || 
            packageName.contains("fitness") -> "health_fitness"
            
            packageName.contains("music") || 
            packageName.contains("video") || 
            packageName.contains("entertainment") -> "entertainment"
            
            else -> "other"
        }
    }

    /**
     * Purges recent data for emergency privacy mode
     */
    suspend fun purgeRecentData(hoursBack: Int = 24) {
        try {
            val database = AppDatabase.getInstance(context)
            val cutoffTime = System.currentTimeMillis() - (hoursBack * 60 * 60 * 1000L)
            
            // Delete recent raw events
            val deletedRawEvents = database.rawEventDao().deleteEventsSince(cutoffTime)
            
            // Delete recent summaries if requested
            val deletedSummaries = database.enhancedDailySummaryDao().deleteSummariesSince(cutoffTime)
            
            logAuditEvent(AuditLogEntry(
                eventType = AuditEventType.DATA_PURGED,
                details = mapOf(
                    "reason" to "emergency_privacy_mode",
                    "hoursBack" to hoursBack,
                    "deletedRawEvents" to deletedRawEvents,
                    "deletedSummaries" to deletedSummaries
                )
            ))
            
            Log.i(TAG, "Purged recent data: $deletedRawEvents raw events, $deletedSummaries summaries")
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to purge recent data", e)
            throw e
        }
    }

    /**
     * Enables emergency privacy mode with optional data purging
     */
    suspend fun enableEmergencyPrivacyMode(purgeRecentData: Boolean = false, hoursBack: Int = 24) {
        try {
            emergencyModeActive = true
            
            // Disable all collectors immediately
            CollectorType.values().forEach { collector ->
                setCollectorEnabled(collector, false)
            }
            
            // Purge recent data if requested
            if (purgeRecentData) {
                purgeRecentData(hoursBack)
            }
            
            // Clear current settings cache to force reload
            currentSettings = null
            
            logAuditEvent(AuditLogEntry(
                eventType = AuditEventType.EMERGENCY_MODE_ACTIVATED,
                details = mapOf(
                    "reason" to "user_requested",
                    "timestamp" to System.currentTimeMillis(),
                    "allCollectorsDisabled" to true,
                    "dataPurged" to purgeRecentData,
                    "hoursBack" to if (purgeRecentData) hoursBack else 0
                )
            ))
            
            Log.w(TAG, "Emergency privacy mode activated - all data collection disabled" + 
                  if (purgeRecentData) ", recent data purged" else "")
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to enable emergency privacy mode", e)
            throw e
        }
    }

    /**
     * Disables emergency privacy mode and restores previous settings
     */
    suspend fun disableEmergencyPrivacyMode() {
        try {
            emergencyModeActive = false
            
            // Restore previous settings
            val settings = getPrivacySettings()
            settings.enabledCollectors.forEach { collector ->
                setCollectorEnabled(collector, true)
            }
            
            logAuditEvent(AuditLogEntry(
                eventType = AuditEventType.EMERGENCY_MODE_DEACTIVATED,
                details = mapOf(
                    "reason" to "user_requested",
                    "timestamp" to System.currentTimeMillis(),
                    "restoredCollectors" to settings.enabledCollectors.map { it.name }
                )
            ))
            
            Log.i(TAG, "Emergency privacy mode deactivated - settings restored")
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to disable emergency privacy mode", e)
            throw e
        }
    }

    /**
     * Sets custom data retention policies for different data types
     */
    suspend fun setCustomRetentionPolicies(policies: Map<DataType, Int>) {
        try {
            val currentSettings = getPrivacySettings()
            val updatedSettings = currentSettings.copy(customRetentionPolicies = policies)
            savePrivacySettings(updatedSettings)
            
            // Apply retention policies immediately
            policies.forEach { (dataType, retentionDays) ->
                cleanupDataByType(dataType, retentionDays)
            }
            
            logAuditEvent(AuditLogEntry(
                eventType = AuditEventType.PRIVACY_SETTING_CHANGED,
                details = mapOf(
                    "setting" to "customRetentionPolicies",
                    "policies" to policies.mapKeys { it.key.name }
                )
            ))
            
            Log.i(TAG, "Custom retention policies set: $policies")
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to set custom retention policies", e)
            throw e
        }
    }

    /**
     * Cleans up data by type based on retention period
     */
    private suspend fun cleanupDataByType(dataType: DataType, retentionDays: Int) {
        try {
            val database = AppDatabase.getInstance(context)
            val cutoffTime = System.currentTimeMillis() - (retentionDays * 24 * 60 * 60 * 1000L)
            
            val deletedCount = when (dataType) {
                DataType.USAGE_EVENTS -> {
                    database.usageEventDao().deleteOldEvents(cutoffTime)
                }
                DataType.NOTIFICATION_EVENTS -> {
                    database.notificationEventDao().deleteOldEvents(cutoffTime)
                }
                DataType.SCREEN_SESSIONS -> {
                    database.screenSessionDao().deleteOldSessions(cutoffTime)
                }
                DataType.INTERACTION_METRICS -> {
                    database.interactionMetricsDao().deleteOldMetrics(cutoffTime)
                }
                DataType.SENSOR_DATA -> {
                    database.activityContextDao().deleteOldContexts(cutoffTime)
                }
                DataType.DAILY_SUMMARIES -> {
                    database.enhancedDailySummaryDao().deleteOldSummaries(cutoffTime)
                }
                DataType.RAW_EVENTS -> {
                    database.rawEventDao().deleteOldProcessedEvents(cutoffTime)
                }
            }
            
            Log.d(TAG, "Cleaned up $deletedCount records of type ${dataType.name}")
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to cleanup data by type $dataType", e)
        }
    }

    // Transparency and audit features

    /**
     * Generates comprehensive privacy transparency report with detailed insights
     */
    suspend fun generateTransparencyReport(): PrivacyTransparencyReport {
        return try {
            val database = AppDatabase.getInstance(context)
            val settings = getPrivacySettings()
            val currentTime = System.currentTimeMillis()
            val last30Days = currentTime - (30 * 24 * 60 * 60 * 1000L)
            
            // Collect data collection statistics
            val dataCollectionStats = mutableMapOf<CollectorType, DataCollectionStats>()
            
            CollectorType.values().forEach { collector ->
                val isEnabled = settings.enabledCollectors.contains(collector)
                val dataCount = getCollectedDataCount(collector, last30Days, currentTime)
                val lastCollection = getLastCollectionTime(collector)
                
                dataCollectionStats[collector] = DataCollectionStats(
                    enabled = isEnabled,
                    dataPointsCollected = dataCount,
                    lastCollectionTime = lastCollection,
                    averageDailyCollection = if (dataCount > 0) dataCount / 30.0 else 0.0,
                    dataStorageSize = getDataStorageSize(collector)
                )
            }
            
            // Get recent privacy setting changes
            val recentChanges = database.auditLogDao()
                .getLogsByTypeAndTimeRange(
                    AuditEventType.PRIVACY_SETTING_CHANGED.name,
                    last30Days,
                    currentTime
                )
            
            // Get data sharing activity
            val dataSharingActivity = getDataSharingActivity(last30Days, currentTime)
            
            // Calculate privacy score
            val privacyScore = calculatePrivacyScore(settings, dataCollectionStats)
            
            PrivacyTransparencyReport(
                generatedAt = currentTime,
                privacyLevel = settings.privacyLevel,
                dataRetentionDays = settings.dataRetentionDays,
                enabledCollectors = settings.enabledCollectors,
                dataCollectionStats = dataCollectionStats,
                recentPrivacyChanges = recentChanges.map { convertAuditLogToEntry(it) },
                emergencyModeActive = emergencyModeActive,
                anonymizationOptions = settings.anonymizationOptions,
                dataSharingActivity = dataSharingActivity,
                privacyScore = privacyScore,
                dataLocations = getDataStorageLocations(),
                thirdPartyAccess = getThirdPartyAccessLog(last30Days, currentTime)
            )
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to generate transparency report", e)
            PrivacyTransparencyReport(
                generatedAt = System.currentTimeMillis(),
                privacyLevel = PrivacyLevel.STANDARD,
                dataRetentionDays = 7,
                enabledCollectors = emptySet(),
                dataCollectionStats = emptyMap(),
                recentPrivacyChanges = emptyList(),
                emergencyModeActive = emergencyModeActive,
                anonymizationOptions = AnonymizationOptions(),
                dataSharingActivity = DataSharingActivity(),
                privacyScore = 0.0,
                dataLocations = emptyList(),
                thirdPartyAccess = emptyList()
            )
        }
    }

    /**
     * Calculates a privacy score based on current settings and data collection
     */
    private fun calculatePrivacyScore(
        settings: PrivacySettings,
        stats: Map<CollectorType, DataCollectionStats>
    ): Double {
        var score = 100.0 // Start with perfect score
        
        // Deduct points for enabled collectors
        val enabledCount = settings.enabledCollectors.size
        val totalCollectors = CollectorType.values().size
        score -= (enabledCount.toDouble() / totalCollectors) * 30.0
        
        // Deduct points for long retention periods
        if (settings.dataRetentionDays > 30) {
            score -= 20.0
        } else if (settings.dataRetentionDays > 7) {
            score -= 10.0
        }
        
        // Add points for anonymization
        if (settings.anonymizationOptions.removeIdentifiers) score += 5.0
        if (settings.anonymizationOptions.enableTimestampFuzzing) score += 5.0
        if (settings.anonymizationOptions.categoryOnlyMode) score += 10.0
        
        // Deduct points for data sharing
        if (settings.dataSharingControls.allowCloudSync) score -= 15.0
        if (settings.dataSharingControls.allowAnalytics) score -= 10.0
        if (settings.dataSharingControls.allowThirdPartySharing) score -= 25.0
        
        return maxOf(0.0, minOf(100.0, score))
    }

    /**
     * Gets data sharing activity for transparency reporting
     */
    private suspend fun getDataSharingActivity(startTime: Long, endTime: Long): DataSharingActivity {
        return try {
            val database = AppDatabase.getInstance(context)
            val exportLogs = database.auditLogDao()
                .getLogsByTypeAndTimeRange(
                    AuditEventType.DATA_EXPORTED.name,
                    startTime,
                    endTime
                )
            
            DataSharingActivity(
                dataExportsCount = exportLogs.size,
                lastExportTime = exportLogs.maxByOrNull { it.timestamp }?.timestamp,
                cloudSyncEnabled = getPrivacySettings().dataSharingControls.allowCloudSync,
                analyticsEnabled = getPrivacySettings().dataSharingControls.allowAnalytics,
                thirdPartyAccessCount = 0 // Would be populated from actual third-party access logs
            )
        } catch (e: Exception) {
            Log.w(TAG, "Failed to get data sharing activity", e)
            DataSharingActivity()
        }
    }

    /**
     * Gets data storage locations for transparency
     */
    private fun getDataStorageLocations(): List<DataStorageLocation> {
        return listOf(
            DataStorageLocation(
                location = "Device Local Storage",
                dataTypes = listOf("Raw Events", "Daily Summaries", "Privacy Settings"),
                encrypted = true,
                description = "All data stored locally on device using SQLCipher encryption"
            ),
            DataStorageLocation(
                location = "Android Keystore",
                dataTypes = listOf("Encryption Keys"),
                encrypted = true,
                description = "Encryption keys stored in secure Android Keystore"
            )
        )
    }

    /**
     * Gets third-party access log for transparency
     */
    private suspend fun getThirdPartyAccessLog(startTime: Long, endTime: Long): List<ThirdPartyAccess> {
        // In a real implementation, this would track actual third-party access
        return emptyList()
    }

    /**
     * Gets data storage size for a specific collector
     */
    private suspend fun getDataStorageSize(collector: CollectorType): Long {
        return try {
            val database = AppDatabase.getInstance(context)
            // This would calculate actual storage size in bytes
            // For now, return estimated size based on record count
            val recordCount = getCollectedDataCount(collector, 0, System.currentTimeMillis())
            recordCount * 1024L // Estimate 1KB per record
        } catch (e: Exception) {
            Log.w(TAG, "Failed to get data storage size for $collector", e)
            0L
        }
    }

    /**
     * Gets comprehensive audit trail for privacy setting changes with filtering options
     */
    suspend fun getPrivacyAuditTrail(
        startTime: Long = System.currentTimeMillis() - (30 * 24 * 60 * 60 * 1000L),
        endTime: Long = System.currentTimeMillis(),
        eventTypes: List<AuditEventType> = listOf(AuditEventType.PRIVACY_SETTING_CHANGED)
    ): List<AuditLogEntry> {
        return try {
            val database = AppDatabase.getInstance(context)
            val auditLogs = mutableListOf<AuditLogEntity>()
            
            eventTypes.forEach { eventType ->
                val logs = database.auditLogDao()
                    .getLogsByTypeAndTimeRange(
                        eventType.name,
                        startTime,
                        endTime
                    )
                auditLogs.addAll(logs)
            }
            
            // Sort by timestamp descending (most recent first)
            auditLogs.sortedByDescending { it.timestamp }
                .map { convertAuditLogToEntry(it) }
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to get privacy audit trail", e)
            emptyList()
        }
    }

    /**
     * Gets detailed data access log for transparency
     */
    suspend fun getDataAccessLog(
        startTime: Long = System.currentTimeMillis() - (7 * 24 * 60 * 60 * 1000L),
        endTime: Long = System.currentTimeMillis()
    ): List<DataAccessEntry> {
        return try {
            val database = AppDatabase.getInstance(context)
            val accessLogs = mutableListOf<DataAccessEntry>()
            
            // Get all audit events that involve data access
            val relevantEventTypes = listOf(
                AuditEventType.DATA_EXPORTED,
                AuditEventType.DATA_DELETED,
                AuditEventType.DATA_PURGED
            )
            
            relevantEventTypes.forEach { eventType ->
                val logs = database.auditLogDao()
                    .getLogsByTypeAndTimeRange(eventType.name, startTime, endTime)
                
                logs.forEach { log ->
                    val details = try {
                        gson.fromJson<Map<String, Any>>(
                            log.details,
                            object : TypeToken<Map<String, Any>>() {}.type
                        ) ?: emptyMap()
                    } catch (e: Exception) {
                        emptyMap()
                    }
                    
                    accessLogs.add(DataAccessEntry(
                        timestamp = log.timestamp,
                        accessType = eventType,
                        dataTypes = extractDataTypesFromDetails(details),
                        purpose = details["reason"]?.toString() ?: "unknown",
                        recordsAffected = details["recordCount"]?.toString()?.toIntOrNull() ?: 0
                    ))
                }
            }
            
            accessLogs.sortedByDescending { it.timestamp }
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to get data access log", e)
            emptyList()
        }
    }

    /**
     * Exports privacy settings and audit trail for user portability
     */
    suspend fun exportPrivacyData(): PrivacyDataExport {
        return try {
            val settings = getPrivacySettings()
            val auditTrail = getPrivacyAuditTrail(
                startTime = 0, // Get all audit logs
                endTime = System.currentTimeMillis(),
                eventTypes = AuditEventType.values().toList()
            )
            val transparencyReport = generateTransparencyReport()
            val dataAccessLog = getDataAccessLog(
                startTime = 0, // Get all access logs
                endTime = System.currentTimeMillis()
            )
            
            PrivacyDataExport(
                exportedAt = System.currentTimeMillis(),
                privacySettings = settings,
                auditTrail = auditTrail,
                transparencyReport = transparencyReport,
                dataAccessLog = dataAccessLog,
                exportVersion = "1.0"
            )
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to export privacy data", e)
            throw e
        }
    }

    /**
     * Validates privacy compliance and returns compliance report
     */
    suspend fun generateComplianceReport(): PrivacyComplianceReport {
        return try {
            val settings = getPrivacySettings()
            val transparencyReport = generateTransparencyReport()
            val issues = mutableListOf<ComplianceIssue>()
            
            // Check for compliance issues
            if (settings.dataRetentionDays > 365) {
                issues.add(ComplianceIssue(
                    severity = ComplianceSeverity.WARNING,
                    issue = "Data retention period exceeds recommended maximum",
                    recommendation = "Consider reducing retention period to 365 days or less"
                ))
            }
            
            if (!settings.anonymizationOptions.removeIdentifiers) {
                issues.add(ComplianceIssue(
                    severity = ComplianceSeverity.HIGH,
                    issue = "Personal identifiers are not being removed",
                    recommendation = "Enable identifier removal for better privacy protection"
                ))
            }
            
            if (settings.dataSharingControls.allowThirdPartySharing && 
                !settings.dataSharingControls.requireExplicitConsent) {
                issues.add(ComplianceIssue(
                    severity = ComplianceSeverity.HIGH,
                    issue = "Third-party sharing enabled without explicit consent requirement",
                    recommendation = "Require explicit consent for third-party data sharing"
                ))
            }
            
            val complianceScore = calculateComplianceScore(issues, transparencyReport.privacyScore)
            
            PrivacyComplianceReport(
                generatedAt = System.currentTimeMillis(),
                complianceScore = complianceScore,
                issues = issues,
                privacyScore = transparencyReport.privacyScore,
                recommendations = generateRecommendations(settings, issues)
            )
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to generate compliance report", e)
            PrivacyComplianceReport(
                generatedAt = System.currentTimeMillis(),
                complianceScore = 0.0,
                issues = listOf(ComplianceIssue(
                    severity = ComplianceSeverity.HIGH,
                    issue = "Failed to generate compliance report",
                    recommendation = "Check system logs for errors"
                )),
                privacyScore = 0.0,
                recommendations = emptyList()
            )
        }
    }

    private fun extractDataTypesFromDetails(details: Map<String, Any>): List<String> {
        return details["dataTypes"]?.let { types ->
            when (types) {
                is List<*> -> types.mapNotNull { it?.toString() }
                is String -> listOf(types)
                else -> emptyList()
            }
        } ?: emptyList()
    }

    private fun calculateComplianceScore(issues: List<ComplianceIssue>, privacyScore: Double): Double {
        var score = privacyScore
        
        issues.forEach { issue ->
            when (issue.severity) {
                ComplianceSeverity.HIGH -> score -= 20.0
                ComplianceSeverity.MEDIUM -> score -= 10.0
                ComplianceSeverity.WARNING -> score -= 5.0
            }
        }
        
        return maxOf(0.0, minOf(100.0, score))
    }

    private fun generateRecommendations(
        settings: PrivacySettings,
        issues: List<ComplianceIssue>
    ): List<String> {
        val recommendations = mutableListOf<String>()
        
        if (settings.privacyLevel == PrivacyLevel.DETAILED) {
            recommendations.add("Consider using Standard privacy level for better privacy protection")
        }
        
        if (settings.dataRetentionDays > 30) {
            recommendations.add("Reduce data retention period to minimize privacy risk")
        }
        
        if (!settings.anonymizationOptions.enableTimestampFuzzing) {
            recommendations.add("Enable timestamp fuzzing to protect temporal privacy")
        }
        
        recommendations.addAll(issues.map { it.recommendation })
        
        return recommendations.distinct()
    }

    /**
     * Sets data sharing controls for cloud sync and analytics
     */
    suspend fun setDataSharingControls(controls: DataSharingControls) {
        try {
            val currentSettings = getPrivacySettings()
            val updatedSettings = currentSettings.copy(dataSharingControls = controls)
            savePrivacySettings(updatedSettings)
            
            logAuditEvent(AuditLogEntry(
                eventType = AuditEventType.PRIVACY_SETTING_CHANGED,
                details = mapOf(
                    "setting" to "dataSharingControls",
                    "allowCloudSync" to controls.allowCloudSync,
                    "allowAnalytics" to controls.allowAnalytics,
                    "allowThirdPartySharing" to controls.allowThirdPartySharing
                )
            ))
            
            Log.i(TAG, "Data sharing controls updated")
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to set data sharing controls", e)
            throw e
        }
    }

    // Helper methods

    private suspend fun savePrivacySettings(settings: PrivacySettings) {
        try {
            val database = AppDatabase.getInstance(context)
            val entity = convertSettingsToEntity(settings)
            database.privacySettingsDao().insertOrUpdate(entity)
            
            // Update cached settings
            currentSettings = settings
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to save privacy settings", e)
            throw e
        }
    }

    private fun createDefaultPrivacySettings(): PrivacySettings {
        return PrivacySettings(
            privacyLevel = PrivacyLevel.STANDARD,
            enabledCollectors = setOf(
                CollectorType.SCREEN_EVENTS,
                CollectorType.USAGE_STATS
            ),
            dataRetentionDays = 7,
            anonymizationOptions = AnonymizationOptions(),
            dataSharingControls = DataSharingControls()
        )
    }

    private fun convertEntityToSettings(entity: PrivacySettingsEntity): PrivacySettings {
        val enabledCollectors = try {
            val collectorNames: List<String> = gson.fromJson(
                entity.enabledCollectors,
                object : TypeToken<List<String>>() {}.type
            )
            collectorNames.mapNotNull { name ->
                try {
                    CollectorType.valueOf(name)
                } catch (e: IllegalArgumentException) {
                    Log.w(TAG, "Unknown collector type: $name")
                    null
                }
            }.toSet()
        } catch (e: Exception) {
            Log.w(TAG, "Failed to parse enabled collectors", e)
            emptySet()
        }
        
        val anonymizationOptions = try {
            gson.fromJson(entity.anonymizationSettings, AnonymizationOptions::class.java)
                ?: AnonymizationOptions()
        } catch (e: Exception) {
            Log.w(TAG, "Failed to parse anonymization options", e)
            AnonymizationOptions()
        }
        
        val dataSharingControls = try {
            gson.fromJson(entity.dataSharingSettings, DataSharingControls::class.java)
                ?: DataSharingControls()
        } catch (e: Exception) {
            Log.w(TAG, "Failed to parse data sharing controls", e)
            DataSharingControls()
        }
        
        return PrivacySettings(
            privacyLevel = PrivacyLevel.valueOf(entity.privacyLevel),
            enabledCollectors = enabledCollectors,
            dataRetentionDays = entity.dataRetentionDays,
            customRetentionPolicies = emptyMap(), // Will be stored separately if needed
            anonymizationOptions = anonymizationOptions,
            dataSharingControls = dataSharingControls
        )
    }

    private fun convertSettingsToEntity(settings: PrivacySettings): PrivacySettingsEntity {
        return PrivacySettingsEntity(
            id = 1, // Single settings record
            privacyLevel = settings.privacyLevel.name,
            enabledCollectors = gson.toJson(settings.enabledCollectors.map { it.name }),
            dataRetentionDays = settings.dataRetentionDays,
            anonymizationSettings = gson.toJson(settings.anonymizationOptions),
            dataSharingSettings = gson.toJson(settings.dataSharingControls),
            lastUpdated = System.currentTimeMillis()
        )
    }

    private suspend fun getCollectedDataCount(
        collector: CollectorType,
        startTime: Long,
        endTime: Long
    ): Int {
        return try {
            val database = AppDatabase.getInstance(context)
            when (collector) {
                CollectorType.USAGE_STATS -> {
                    database.usageEventDao().getEventsByTimeRange(startTime, endTime).size
                }
                CollectorType.NOTIFICATIONS -> {
                    database.notificationEventDao().getEventsByTimeRange(startTime, endTime).size
                }
                CollectorType.SCREEN_EVENTS -> {
                    database.screenSessionDao().getSessionsByTimeRange(startTime, endTime).size
                }
                CollectorType.INTERACTIONS -> {
                    database.interactionMetricsDao().getMetricsByTimeRange(startTime, endTime).size
                }
                CollectorType.SENSORS -> {
                    database.activityContextDao().getContextsByTimeRange(startTime, endTime).size
                }
            }
        } catch (e: Exception) {
            Log.w(TAG, "Failed to get collected data count for $collector", e)
            0
        }
    }

    private suspend fun getLastCollectionTime(collector: CollectorType): Long? {
        return try {
            val database = AppDatabase.getInstance(context)
            when (collector) {
                CollectorType.USAGE_STATS -> {
                    database.usageEventDao().getLatestEvent()?.startTime
                }
                CollectorType.NOTIFICATIONS -> {
                    database.notificationEventDao().getLatestEvent()?.timestamp
                }
                CollectorType.SCREEN_EVENTS -> {
                    database.screenSessionDao().getLatestSession()?.startTime
                }
                CollectorType.INTERACTIONS -> {
                    database.interactionMetricsDao().getLatestMetrics()?.timestamp
                }
                CollectorType.SENSORS -> {
                    database.activityContextDao().getLatestContext()?.timestamp
                }
            }
        } catch (e: Exception) {
            Log.w(TAG, "Failed to get last collection time for $collector", e)
            null
        }
    }

    private suspend fun logAuditEvent(entry: AuditLogEntry) {
        try {
            val database = AppDatabase.getInstance(context)
            val auditEntity = AuditLogEntity(
                id = UUID.randomUUID().toString(),
                timestamp = System.currentTimeMillis(),
                eventType = entry.eventType.name,
                details = gson.toJson(entry.details),
                userId = null
            )
            
            database.auditLogDao().insert(auditEntity)
            
        } catch (e: Exception) {
            Log.w(TAG, "Failed to log audit event", e)
        }
    }

    private fun convertAuditLogToEntry(entity: AuditLogEntity): AuditLogEntry {
        val details = try {
            gson.fromJson<Map<String, Any>>(
                entity.details,
                object : TypeToken<Map<String, Any>>() {}.type
            ) ?: emptyMap()
        } catch (e: Exception) {
            Log.w(TAG, "Failed to parse audit log details", e)
            emptyMap()
        }
        
        return AuditLogEntry(
            eventType = AuditEventType.valueOf(entity.eventType),
            details = details,
            timestamp = entity.timestamp
        )
    }

    // Data classes for privacy management

    data class PrivacySettings(
        val privacyLevel: PrivacyLevel,
        val enabledCollectors: Set<CollectorType>,
        val dataRetentionDays: Int,
        val customRetentionPolicies: Map<DataType, Int> = emptyMap(),
        val anonymizationOptions: AnonymizationOptions,
        val dataSharingControls: DataSharingControls
    )

    data class AnonymizationOptions(
        val enableTimestampFuzzing: Boolean = false,
        val timestampFuzzingMinutes: Int = 15, // Fuzz timestamps by up to 15 minutes
        val removeIdentifiers: Boolean = true,
        val obfuscateLocation: Boolean = true,
        val aggregateOnly: Boolean = false,
        val categoryOnlyMode: Boolean = false, // Store only app categories, not specific apps
        val minimumAggregationWindow: Long = 60 * 60 * 1000L // 1 hour minimum aggregation
    )

    data class DataSharingControls(
        val allowCloudSync: Boolean = false,
        val allowAnalytics: Boolean = false,
        val allowThirdPartySharing: Boolean = false,
        val requireExplicitConsent: Boolean = true
    )

    data class PrivacyTransparencyReport(
        val generatedAt: Long,
        val privacyLevel: PrivacyLevel,
        val dataRetentionDays: Int,
        val enabledCollectors: Set<CollectorType>,
        val dataCollectionStats: Map<CollectorType, DataCollectionStats>,
        val recentPrivacyChanges: List<AuditLogEntry>,
        val emergencyModeActive: Boolean,
        val anonymizationOptions: AnonymizationOptions,
        val dataSharingActivity: DataSharingActivity = DataSharingActivity(),
        val privacyScore: Double = 0.0,
        val dataLocations: List<DataStorageLocation> = emptyList(),
        val thirdPartyAccess: List<ThirdPartyAccess> = emptyList()
    )

    data class DataCollectionStats(
        val enabled: Boolean,
        val dataPointsCollected: Int,
        val lastCollectionTime: Long?,
        val averageDailyCollection: Double = 0.0,
        val dataStorageSize: Long = 0L
    )

    data class DataSharingActivity(
        val dataExportsCount: Int = 0,
        val lastExportTime: Long? = null,
        val cloudSyncEnabled: Boolean = false,
        val analyticsEnabled: Boolean = false,
        val thirdPartyAccessCount: Int = 0
    )

    data class DataStorageLocation(
        val location: String,
        val dataTypes: List<String>,
        val encrypted: Boolean,
        val description: String
    )

    data class ThirdPartyAccess(
        val serviceName: String,
        val accessTime: Long,
        val dataTypesAccessed: List<String>,
        val purpose: String
    )

    data class AuditLogEntry(
        val eventType: AuditEventType,
        val details: Map<String, Any>,
        val timestamp: Long = System.currentTimeMillis()
    )

    data class DataAccessEntry(
        val timestamp: Long,
        val accessType: AuditEventType,
        val dataTypes: List<String>,
        val purpose: String,
        val recordsAffected: Int
    )

    data class PrivacyDataExport(
        val exportedAt: Long,
        val privacySettings: PrivacySettings,
        val auditTrail: List<AuditLogEntry>,
        val transparencyReport: PrivacyTransparencyReport,
        val dataAccessLog: List<DataAccessEntry>,
        val exportVersion: String
    )

    data class PrivacyComplianceReport(
        val generatedAt: Long,
        val complianceScore: Double,
        val issues: List<ComplianceIssue>,
        val privacyScore: Double,
        val recommendations: List<String>
    )

    data class ComplianceIssue(
        val severity: ComplianceSeverity,
        val issue: String,
        val recommendation: String
    )

    enum class ComplianceSeverity {
        HIGH, MEDIUM, WARNING
    }

    enum class PrivacyLevel {
        MINIMAL,    // Only essential data collection
        STANDARD,   // Balanced privacy and functionality
        DETAILED    // Full data collection for maximum insights
    }
}