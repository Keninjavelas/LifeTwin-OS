package com.lifetwin.mlp.export

import android.content.Context
import android.util.Log
import com.lifetwin.mlp.db.*
import com.google.gson.Gson
import com.google.gson.GsonBuilder
import com.google.gson.reflect.TypeToken
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.text.SimpleDateFormat
import java.util.*

private const val TAG = "DataExporter"

/**
 * Comprehensive data export and portability system
 * - Implements JSON export for all user data types
 * - Creates selective export by date range and data type
 * - Implements export validation and integrity checking
 * - Supports data import for portability
 */
class DataExporter(private val context: Context) : com.lifetwin.mlp.db.DataExporter {
    
    private val gson: Gson = GsonBuilder()
        .setPrettyPrinting()
        .setDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSS'Z'")
        .create()
    
    private val dateFormat = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSS'Z'", Locale.US)

    override suspend fun exportAllData(): String = withContext(Dispatchers.IO) {
        try {
            val database = AppDatabase.getInstance(context)
            val exportData = DataExportContainer(
                exportMetadata = ExportMetadata(
                    exportedAt = System.currentTimeMillis(),
                    exportVersion = "1.0",
                    exportType = "FULL_EXPORT",
                    deviceInfo = getDeviceInfo()
                ),
                usageEvents = database.usageEventDao().getAllEvents(),
                notificationEvents = database.notificationEventDao().getAllEvents(),
                screenSessions = database.screenSessionDao().getAllSessions(),
                interactionMetrics = database.interactionMetricsDao().getAllMetrics(),
                activityContexts = database.activityContextDao().getAllContexts(),
                dailySummaries = database.enhancedDailySummaryDao().getAllSummaries(),
                rawEvents = database.rawEventDao().getAllEvents(),
                privacySettings = database.privacySettingsDao().getSettings(),
                auditLogs = database.auditLogDao().getAllLogs()
            )
            
            val jsonData = gson.toJson(exportData)
            
            // Log export event
            logExportEvent("FULL_EXPORT", exportData.calculateRecordCount())
            
            Log.i(TAG, "Successfully exported all data (${exportData.calculateRecordCount()} records)")
            jsonData
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to export all data", e)
            throw DataExportException("Failed to export all data: ${e.message}", e)
        }
    }

    override suspend fun exportDataByType(types: Set<CollectorType>): String = withContext(Dispatchers.IO) {
        try {
            val database = AppDatabase.getInstance(context)
            val exportData = DataExportContainer(
                exportMetadata = ExportMetadata(
                    exportedAt = System.currentTimeMillis(),
                    exportVersion = "1.0",
                    exportType = "SELECTIVE_BY_TYPE",
                    deviceInfo = getDeviceInfo(),
                    selectedTypes = types.map { it.name }
                ),
                usageEvents = if (CollectorType.USAGE_STATS in types) 
                    database.usageEventDao().getAllEvents() else emptyList(),
                notificationEvents = if (CollectorType.NOTIFICATIONS in types) 
                    database.notificationEventDao().getAllEvents() else emptyList(),
                screenSessions = if (CollectorType.SCREEN_EVENTS in types) 
                    database.screenSessionDao().getAllSessions() else emptyList(),
                interactionMetrics = if (CollectorType.INTERACTIONS in types) 
                    database.interactionMetricsDao().getAllMetrics() else emptyList(),
                activityContexts = if (CollectorType.SENSORS in types) 
                    database.activityContextDao().getAllContexts() else emptyList(),
                dailySummaries = database.enhancedDailySummaryDao().getAllSummaries(),
                rawEvents = database.rawEventDao().getAllEvents(),
                privacySettings = database.privacySettingsDao().getSettings(),
                auditLogs = database.auditLogDao().getAllLogs()
            )
            
            val jsonData = gson.toJson(exportData)
            
            // Log export event
            logExportEvent("SELECTIVE_BY_TYPE", exportData.calculateRecordCount(), types.map { it.name })
            
            Log.i(TAG, "Successfully exported data by type (${exportData.calculateRecordCount()} records)")
            jsonData
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to export data by type", e)
            throw DataExportException("Failed to export data by type: ${e.message}", e)
        }
    }

    override suspend fun exportDataByTimeRange(timeRange: TimeRange): String = withContext(Dispatchers.IO) {
        try {
            val database = AppDatabase.getInstance(context)
            val exportData = DataExportContainer(
                exportMetadata = ExportMetadata(
                    exportedAt = System.currentTimeMillis(),
                    exportVersion = "1.0",
                    exportType = "SELECTIVE_BY_TIME_RANGE",
                    deviceInfo = getDeviceInfo(),
                    timeRangeStart = timeRange.startTime,
                    timeRangeEnd = timeRange.endTime
                ),
                usageEvents = database.usageEventDao().getEventsByTimeRange(timeRange.startTime, timeRange.endTime),
                notificationEvents = database.notificationEventDao().getEventsByTimeRange(timeRange.startTime, timeRange.endTime),
                screenSessions = database.screenSessionDao().getSessionsByTimeRange(timeRange.startTime, timeRange.endTime),
                interactionMetrics = database.interactionMetricsDao().getMetricsByTimeRange(timeRange.startTime, timeRange.endTime),
                activityContexts = database.activityContextDao().getContextsByTimeRange(timeRange.startTime, timeRange.endTime),
                dailySummaries = database.enhancedDailySummaryDao().getSummariesByTimeRange(timeRange.startTime, timeRange.endTime),
                rawEvents = database.rawEventDao().getEventsByTimeRange(timeRange.startTime, timeRange.endTime),
                privacySettings = database.privacySettingsDao().getSettings(),
                auditLogs = database.auditLogDao().getLogsByTimeRange(timeRange.startTime, timeRange.endTime)
            )
            
            val jsonData = gson.toJson(exportData)
            
            // Log export event
            logExportEvent("SELECTIVE_BY_TIME_RANGE", exportData.calculateRecordCount())
            
            Log.i(TAG, "Successfully exported data by time range (${exportData.calculateRecordCount()} records)")
            jsonData
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to export data by time range", e)
            throw DataExportException("Failed to export data by time range: ${e.message}", e)
        }
    }

    override suspend fun validateExportData(exportData: String): Boolean = withContext(Dispatchers.IO) {
        try {
            // Parse the JSON to validate structure
            val container = gson.fromJson(exportData, DataExportContainer::class.java)
                ?: return@withContext false
            
            // Validate metadata
            if (container.exportMetadata.exportVersion.isBlank() || 
                container.exportMetadata.exportedAt <= 0) {
                Log.w(TAG, "Invalid export metadata")
                return@withContext false
            }
            
            // Validate data integrity
            val recordCount = container.calculateRecordCount()
            if (recordCount < 0) {
                Log.w(TAG, "Invalid record count: $recordCount")
                return@withContext false
            }
            
            // Validate timestamp consistency
            val exportTime = container.exportMetadata.exportedAt
            val hasInvalidTimestamps = container.usageEvents.any { it.startTime > exportTime } ||
                container.notificationEvents.any { it.timestamp > exportTime } ||
                container.screenSessions.any { it.startTime > exportTime } ||
                container.interactionMetrics.any { it.timestamp > exportTime } ||
                container.activityContexts.any { it.timestamp > exportTime }
            
            if (hasInvalidTimestamps) {
                Log.w(TAG, "Found timestamps after export time")
                return@withContext false
            }
            
            Log.i(TAG, "Export data validation successful ($recordCount records)")
            true
            
        } catch (e: Exception) {
            Log.e(TAG, "Export data validation failed", e)
            false
        }
    }

    override suspend fun importData(importData: String): Boolean = withContext(Dispatchers.IO) {
        try {
            // Validate the import data first
            if (!validateExportData(importData)) {
                Log.e(TAG, "Import data validation failed")
                return@withContext false
            }
            
            val container = gson.fromJson(importData, DataExportContainer::class.java)
            val database = AppDatabase.getInstance(context)
            
            // Import data in transaction for consistency
            database.runInTransaction {
                // Import usage events
                container.usageEvents.forEach { event ->
                    try {
                        database.usageEventDao().insert(event)
                    } catch (e: Exception) {
                        Log.w(TAG, "Failed to import usage event ${event.id}", e)
                    }
                }
                
                // Import notification events
                container.notificationEvents.forEach { event ->
                    try {
                        database.notificationEventDao().insert(event)
                    } catch (e: Exception) {
                        Log.w(TAG, "Failed to import notification event ${event.id}", e)
                    }
                }
                
                // Import screen sessions
                container.screenSessions.forEach { session ->
                    try {
                        database.screenSessionDao().insert(session)
                    } catch (e: Exception) {
                        Log.w(TAG, "Failed to import screen session ${session.sessionId}", e)
                    }
                }
                
                // Import interaction metrics
                container.interactionMetrics.forEach { metrics ->
                    try {
                        database.interactionMetricsDao().insert(metrics)
                    } catch (e: Exception) {
                        Log.w(TAG, "Failed to import interaction metrics ${metrics.id}", e)
                    }
                }
                
                // Import activity contexts
                container.activityContexts.forEach { context ->
                    try {
                        database.activityContextDao().insert(context)
                    } catch (e: Exception) {
                        Log.w(TAG, "Failed to import activity context ${context.id}", e)
                    }
                }
                
                // Import daily summaries
                container.dailySummaries.forEach { summary ->
                    try {
                        database.enhancedDailySummaryDao().insert(summary)
                    } catch (e: Exception) {
                        Log.w(TAG, "Failed to import daily summary ${summary.date}", e)
                    }
                }
                
                // Import raw events
                container.rawEvents.forEach { event ->
                    try {
                        database.rawEventDao().insert(event)
                    } catch (e: Exception) {
                        Log.w(TAG, "Failed to import raw event ${event.id}", e)
                    }
                }
                
                // Import privacy settings (if provided)
                container.privacySettings?.let { settings ->
                    try {
                        database.privacySettingsDao().insertOrUpdate(settings)
                    } catch (e: Exception) {
                        Log.w(TAG, "Failed to import privacy settings", e)
                    }
                }
                
                // Import audit logs
                container.auditLogs.forEach { log ->
                    try {
                        database.auditLogDao().insert(log)
                    } catch (e: Exception) {
                        Log.w(TAG, "Failed to import audit log ${log.id}", e)
                    }
                }
            }
            
            // Log import event
            logImportEvent(container.calculateRecordCount())
            
            Log.i(TAG, "Successfully imported data (${container.calculateRecordCount()} records)")
            true
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to import data", e)
            false
        }
    }

    /**
     * Exports data with custom filters and options
     */
    suspend fun exportDataWithFilters(
        types: Set<CollectorType>? = null,
        timeRange: TimeRange? = null,
        includeRawEvents: Boolean = true,
        includeSummaries: Boolean = true,
        includePrivacySettings: Boolean = true,
        includeAuditLogs: Boolean = true
    ): String = withContext(Dispatchers.IO) {
        try {
            val database = AppDatabase.getInstance(context)
            
            val usageEvents = if (types?.contains(CollectorType.USAGE_STATS) != false) {
                if (timeRange != null) {
                    database.usageEventDao().getEventsByTimeRange(timeRange.startTime, timeRange.endTime)
                } else {
                    database.usageEventDao().getAllEvents()
                }
            } else emptyList()
            
            val notificationEvents = if (types?.contains(CollectorType.NOTIFICATIONS) != false) {
                if (timeRange != null) {
                    database.notificationEventDao().getEventsByTimeRange(timeRange.startTime, timeRange.endTime)
                } else {
                    database.notificationEventDao().getAllEvents()
                }
            } else emptyList()
            
            val screenSessions = if (types?.contains(CollectorType.SCREEN_EVENTS) != false) {
                if (timeRange != null) {
                    database.screenSessionDao().getSessionsByTimeRange(timeRange.startTime, timeRange.endTime)
                } else {
                    database.screenSessionDao().getAllSessions()
                }
            } else emptyList()
            
            val interactionMetrics = if (types?.contains(CollectorType.INTERACTIONS) != false) {
                if (timeRange != null) {
                    database.interactionMetricsDao().getMetricsByTimeRange(timeRange.startTime, timeRange.endTime)
                } else {
                    database.interactionMetricsDao().getAllMetrics()
                }
            } else emptyList()
            
            val activityContexts = if (types?.contains(CollectorType.SENSORS) != false) {
                if (timeRange != null) {
                    database.activityContextDao().getContextsByTimeRange(timeRange.startTime, timeRange.endTime)
                } else {
                    database.activityContextDao().getAllContexts()
                }
            } else emptyList()
            
            val dailySummaries = if (includeSummaries) {
                if (timeRange != null) {
                    database.enhancedDailySummaryDao().getSummariesByTimeRange(timeRange.startTime, timeRange.endTime)
                } else {
                    database.enhancedDailySummaryDao().getAllSummaries()
                }
            } else emptyList()
            
            val rawEvents = if (includeRawEvents) {
                if (timeRange != null) {
                    database.rawEventDao().getEventsByTimeRange(timeRange.startTime, timeRange.endTime)
                } else {
                    database.rawEventDao().getAllEvents()
                }
            } else emptyList()
            
            val privacySettings = if (includePrivacySettings) {
                database.privacySettingsDao().getSettings()
            } else null
            
            val auditLogs = if (includeAuditLogs) {
                if (timeRange != null) {
                    database.auditLogDao().getLogsByTimeRange(timeRange.startTime, timeRange.endTime)
                } else {
                    database.auditLogDao().getAllLogs()
                }
            } else emptyList()
            
            val exportData = DataExportContainer(
                exportMetadata = ExportMetadata(
                    exportedAt = System.currentTimeMillis(),
                    exportVersion = "1.0",
                    exportType = "CUSTOM_FILTERED",
                    deviceInfo = getDeviceInfo(),
                    selectedTypes = types?.map { it.name },
                    timeRangeStart = timeRange?.startTime,
                    timeRangeEnd = timeRange?.endTime,
                    includeRawEvents = includeRawEvents,
                    includeSummaries = includeSummaries,
                    includePrivacySettings = includePrivacySettings,
                    includeAuditLogs = includeAuditLogs
                ),
                usageEvents = usageEvents,
                notificationEvents = notificationEvents,
                screenSessions = screenSessions,
                interactionMetrics = interactionMetrics,
                activityContexts = activityContexts,
                dailySummaries = dailySummaries,
                rawEvents = rawEvents,
                privacySettings = privacySettings,
                auditLogs = auditLogs
            )
            
            val jsonData = gson.toJson(exportData)
            
            // Log export event
            logExportEvent("CUSTOM_FILTERED", exportData.calculateRecordCount())
            
            Log.i(TAG, "Successfully exported filtered data (${exportData.calculateRecordCount()} records)")
            jsonData
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to export filtered data", e)
            throw DataExportException("Failed to export filtered data: ${e.message}", e)
        }
    }

    private fun getDeviceInfo(): DeviceInfo {
        return DeviceInfo(
            manufacturer = android.os.Build.MANUFACTURER,
            model = android.os.Build.MODEL,
            androidVersion = android.os.Build.VERSION.RELEASE,
            apiLevel = android.os.Build.VERSION.SDK_INT,
            appVersion = getAppVersion()
        )
    }

    private fun getAppVersion(): String {
        return try {
            val packageInfo = context.packageManager.getPackageInfo(context.packageName, 0)
            packageInfo.versionName ?: "unknown"
        } catch (e: Exception) {
            "unknown"
        }
    }

    private suspend fun logExportEvent(exportType: String, recordCount: Int, selectedTypes: List<String>? = null) {
        try {
            val database = AppDatabase.getInstance(context)
            val auditEntry = AuditLogEntity(
                id = UUID.randomUUID().toString(),
                timestamp = System.currentTimeMillis(),
                eventType = AuditEventType.DATA_EXPORTED.name,
                details = gson.toJson(mapOf(
                    "exportType" to exportType,
                    "recordCount" to recordCount,
                    "selectedTypes" to selectedTypes,
                    "exportedAt" to System.currentTimeMillis()
                )),
                userId = null
            )
            
            database.auditLogDao().insert(auditEntry)
        } catch (e: Exception) {
            Log.w(TAG, "Failed to log export event", e)
        }
    }

    private suspend fun logImportEvent(recordCount: Int) {
        try {
            val database = AppDatabase.getInstance(context)
            val auditEntry = AuditLogEntity(
                id = UUID.randomUUID().toString(),
                timestamp = System.currentTimeMillis(),
                eventType = "DATA_IMPORTED",
                details = gson.toJson(mapOf(
                    "recordCount" to recordCount,
                    "importedAt" to System.currentTimeMillis()
                )),
                userId = null
            )
            
            database.auditLogDao().insert(auditEntry)
        } catch (e: Exception) {
            Log.w(TAG, "Failed to log import event", e)
        }
    }

    // Data classes for export structure

    data class DataExportContainer(
        val exportMetadata: ExportMetadata,
        val usageEvents: List<UsageEventEntity> = emptyList(),
        val notificationEvents: List<NotificationEventEntity> = emptyList(),
        val screenSessions: List<ScreenSessionEntity> = emptyList(),
        val interactionMetrics: List<InteractionMetricsEntity> = emptyList(),
        val activityContexts: List<ActivityContextEntity> = emptyList(),
        val dailySummaries: List<EnhancedDailySummaryEntity> = emptyList(),
        val rawEvents: List<RawEventEntity> = emptyList(),
        val privacySettings: PrivacySettingsEntity? = null,
        val auditLogs: List<AuditLogEntity> = emptyList()
    ) {
        fun calculateRecordCount(): Int {
            return usageEvents.size + 
                   notificationEvents.size + 
                   screenSessions.size + 
                   interactionMetrics.size + 
                   activityContexts.size + 
                   dailySummaries.size + 
                   rawEvents.size + 
                   (if (privacySettings != null) 1 else 0) + 
                   auditLogs.size
        }
    }

    data class ExportMetadata(
        val exportedAt: Long,
        val exportVersion: String,
        val exportType: String,
        val deviceInfo: DeviceInfo,
        val selectedTypes: List<String>? = null,
        val timeRangeStart: Long? = null,
        val timeRangeEnd: Long? = null,
        val includeRawEvents: Boolean = true,
        val includeSummaries: Boolean = true,
        val includePrivacySettings: Boolean = true,
        val includeAuditLogs: Boolean = true
    )

    data class DeviceInfo(
        val manufacturer: String,
        val model: String,
        val androidVersion: String,
        val apiLevel: Int,
        val appVersion: String
    )

    class DataExportException(message: String, cause: Throwable? = null) : Exception(message, cause)
}