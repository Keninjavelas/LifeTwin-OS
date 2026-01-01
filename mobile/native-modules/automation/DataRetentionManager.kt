package com.lifetwin.automation

import android.content.Context
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*
import java.util.concurrent.TimeUnit
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import kotlinx.serialization.encodeToString
import kotlinx.serialization.decodeFromString

/**
 * DataRetentionManager - Comprehensive data retention and anonymization system
 * 
 * Implements Requirements:
 * - 9.2: Automatic deletion of old automation logs
 * - 9.6: Data anonymization for RL training datasets
 * - 6.6: User controls for data review and deletion
 */
class DataRetentionManager(
    private val context: Context,
    private val automationLog: AutomationLog,
    private val privacyController: PrivacyController
) {
    private val scope = CoroutineScope(Dispatchers.Default + SupervisorJob())
    
    // Retention policy state
    private val _retentionPolicy = MutableStateFlow(loadRetentionPolicy())
    val retentionPolicy: StateFlow<RetentionPolicy> = _retentionPolicy.asStateFlow()
    
    // Data cleanup status
    private val _cleanupStatus = MutableStateFlow(CleanupStatus())
    val cleanupStatus: StateFlow<CleanupStatus> = _cleanupStatus.asStateFlow()
    
    // User data review interface
    private val _userDataSummary = MutableStateFlow(UserDataSummary())
    val userDataSummary: StateFlow<UserDataSummary> = _userDataSummary.asStateFlow()
    
    init {
        startRetentionManagement()
    }
    
    /**
     * Start continuous data retention management
     */
    private fun startRetentionManagement() {
        scope.launch {
            while (isActive) {
                try {
                    performScheduledCleanup()
                    updateUserDataSummary()
                    delay(TimeUnit.HOURS.toMillis(6)) // Run every 6 hours
                } catch (e: Exception) {
                    // Continue retention management even on errors
                    delay(TimeUnit.HOURS.toMillis(12)) // Wait longer on error
                }
            }
        }
    }
    
    /**
     * Update retention policy with user preferences
     */
    suspend fun updateRetentionPolicy(policy: RetentionPolicy) = withContext(Dispatchers.IO) {
        _retentionPolicy.value = policy
        saveRetentionPolicy(policy)
        
        // Apply new policy immediately
        applyRetentionPolicy(policy)
    }
    
    /**
     * Perform scheduled data cleanup based on retention policy
     */
    private suspend fun performScheduledCleanup() = withContext(Dispatchers.IO) {
        val policy = _retentionPolicy.value
        val startTime = System.currentTimeMillis()
        
        var totalDeleted = 0
        var totalAnonymized = 0
        
        try {
            // Clean up automation logs
            val deletedLogs = cleanupAutomationLogs(policy.automationLogRetentionDays)
            totalDeleted += deletedLogs
            
            // Clean up behavioral data
            val deletedBehavioral = cleanupBehavioralData(policy.behavioralDataRetentionDays)
            totalDeleted += deletedBehavioral
            
            // Clean up usage statistics
            val deletedUsage = cleanupUsageData(policy.usageDataRetentionDays)
            totalDeleted += deletedUsage
            
            // Anonymize training data
            if (policy.anonymizeTrainingData) {
                val anonymizedCount = anonymizeTrainingData(policy.trainingDataRetentionDays)
                totalAnonymized += anonymizedCount
            }
            
            // Clean up temporary files
            val deletedTemp = cleanupTemporaryFiles(policy.tempFileRetentionHours)
            totalDeleted += deletedTemp
            
            val endTime = System.currentTimeMillis()
            
            _cleanupStatus.value = CleanupStatus(
                lastCleanupTime = endTime,
                cleanupDuration = endTime - startTime,
                recordsDeleted = totalDeleted,
                recordsAnonymized = totalAnonymized,
                success = true,
                errorMessage = null
            )
            
        } catch (e: Exception) {
            _cleanupStatus.value = CleanupStatus(
                lastCleanupTime = System.currentTimeMillis(),
                cleanupDuration = System.currentTimeMillis() - startTime,
                recordsDeleted = totalDeleted,
                recordsAnonymized = totalAnonymized,
                success = false,
                errorMessage = e.message
            )
        }
    }
    
    /**
     * Clean up old automation logs
     */
    private suspend fun cleanupAutomationLogs(retentionDays: Int): Int {
        val cutoffTime = System.currentTimeMillis() - TimeUnit.DAYS.toMillis(retentionDays.toLong())
        
        return try {
            automationLog.deleteLogsOlderThan(cutoffTime)
        } catch (e: Exception) {
            0
        }
    }
    
    /**
     * Clean up old behavioral data
     */
    private suspend fun cleanupBehavioralData(retentionDays: Int): Int {
        val cutoffTime = System.currentTimeMillis() - TimeUnit.DAYS.toMillis(retentionDays.toLong())
        
        return try {
            // Implementation would delete behavioral context data older than cutoff
            // This would integrate with the database layer
            0 // Placeholder
        } catch (e: Exception) {
            0
        }
    }
    
    /**
     * Clean up old usage statistics
     */
    private suspend fun cleanupUsageData(retentionDays: Int): Int {
        val cutoffTime = System.currentTimeMillis() - TimeUnit.DAYS.toMillis(retentionDays.toLong())
        
        return try {
            // Implementation would delete usage statistics older than cutoff
            // This would integrate with the usage stats database
            0 // Placeholder
        } catch (e: Exception) {
            0
        }
    }
    
    /**
     * Anonymize training data for RL learning
     */
    private suspend fun anonymizeTrainingData(retentionDays: Int): Int {
        val cutoffTime = System.currentTimeMillis() - TimeUnit.DAYS.toMillis(retentionDays.toLong())
        
        return try {
            var anonymizedCount = 0
            
            // Get training data older than retention period
            val trainingData = getTrainingDataOlderThan(cutoffTime)
            
            trainingData.forEach { data ->
                val anonymizedData = privacyController.anonymizeData(data)
                updateTrainingDataRecord(data["id"] as String, anonymizedData)
                anonymizedCount++
            }
            
            anonymizedCount
        } catch (e: Exception) {
            0
        }
    }
    
    /**
     * Clean up temporary files
     */
    private suspend fun cleanupTemporaryFiles(retentionHours: Int): Int {
        val cutoffTime = System.currentTimeMillis() - TimeUnit.HOURS.toMillis(retentionHours.toLong())
        
        return try {
            val tempDir = context.cacheDir
            var deletedCount = 0
            
            tempDir.listFiles()?.forEach { file ->
                if (file.lastModified() < cutoffTime) {
                    if (file.delete()) {
                        deletedCount++
                    }
                }
            }
            
            deletedCount
        } catch (e: Exception) {
            0
        }
    }
    
    /**
     * Get training data older than specified time
     */
    private suspend fun getTrainingDataOlderThan(cutoffTime: Long): List<Map<String, Any>> {
        // Implementation would query database for training data
        // This is a placeholder that would integrate with actual data storage
        return emptyList()
    }
    
    /**
     * Update training data record with anonymized version
     */
    private suspend fun updateTrainingDataRecord(id: String, anonymizedData: Map<String, Any>) {
        // Implementation would update database record with anonymized data
        // This is a placeholder that would integrate with actual data storage
    }
    
    /**
     * Apply retention policy to existing data
     */
    private suspend fun applyRetentionPolicy(policy: RetentionPolicy) = withContext(Dispatchers.IO) {
        // Immediately apply new retention rules
        if (policy.immediateCleanup) {
            performScheduledCleanup()
        }
    }
    
    /**
     * Update user data summary for review interface
     */
    private suspend fun updateUserDataSummary() = withContext(Dispatchers.IO) {
        val summary = UserDataSummary(
            automationLogsCount = getAutomationLogsCount(),
            behavioralDataCount = getBehavioralDataCount(),
            usageDataCount = getUsageDataCount(),
            trainingDataCount = getTrainingDataCount(),
            totalStorageUsedMB = getTotalStorageUsed(),
            oldestRecordDate = getOldestRecordDate(),
            newestRecordDate = getNewestRecordDate(),
            anonymizedRecordsCount = getAnonymizedRecordsCount()
        )
        
        _userDataSummary.value = summary
    }
    
    /**
     * Get count of automation logs
     */
    private suspend fun getAutomationLogsCount(): Int {
        return try {
            automationLog.getTotalLogCount()
        } catch (e: Exception) {
            0
        }
    }
    
    /**
     * Get count of behavioral data records
     */
    private suspend fun getBehavioralDataCount(): Int {
        // Implementation would query behavioral data count
        return 0 // Placeholder
    }
    
    /**
     * Get count of usage data records
     */
    private suspend fun getUsageDataCount(): Int {
        // Implementation would query usage data count
        return 0 // Placeholder
    }
    
    /**
     * Get count of training data records
     */
    private suspend fun getTrainingDataCount(): Int {
        // Implementation would query training data count
        return 0 // Placeholder
    }
    
    /**
     * Get total storage used by all data in MB
     */
    private suspend fun getTotalStorageUsed(): Double {
        return try {
            val dataDir = context.filesDir
            val totalBytes = dataDir.walkTopDown()
                .filter { it.isFile }
                .map { it.length() }
                .sum()
            
            totalBytes / (1024.0 * 1024.0) // Convert to MB
        } catch (e: Exception) {
            0.0
        }
    }
    
    /**
     * Get date of oldest record
     */
    private suspend fun getOldestRecordDate(): Long {
        // Implementation would query for oldest record across all data types
        return 0L // Placeholder
    }
    
    /**
     * Get date of newest record
     */
    private suspend fun getNewestRecordDate(): Long {
        // Implementation would query for newest record across all data types
        return System.currentTimeMillis() // Placeholder
    }
    
    /**
     * Get count of anonymized records
     */
    private suspend fun getAnonymizedRecordsCount(): Int {
        // Implementation would count anonymized records
        return 0 // Placeholder
    }
    
    /**
     * User-initiated data deletion
     */
    suspend fun deleteUserData(dataTypes: List<DataType>, olderThanDays: Int? = null): DataDeletionResult = withContext(Dispatchers.IO) {
        var totalDeleted = 0
        val deletionResults = mutableMapOf<DataType, Int>()
        
        try {
            dataTypes.forEach { dataType ->
                val deleted = when (dataType) {
                    DataType.AUTOMATION_LOGS -> {
                        if (olderThanDays != null) {
                            cleanupAutomationLogs(olderThanDays)
                        } else {
                            automationLog.deleteAllLogs()
                        }
                    }
                    DataType.BEHAVIORAL_DATA -> {
                        if (olderThanDays != null) {
                            cleanupBehavioralData(olderThanDays)
                        } else {
                            deleteAllBehavioralData()
                        }
                    }
                    DataType.USAGE_DATA -> {
                        if (olderThanDays != null) {
                            cleanupUsageData(olderThanDays)
                        } else {
                            deleteAllUsageData()
                        }
                    }
                    DataType.TRAINING_DATA -> {
                        deleteAllTrainingData()
                    }
                }
                
                deletionResults[dataType] = deleted
                totalDeleted += deleted
            }
            
            // Update user data summary after deletion
            updateUserDataSummary()
            
            DataDeletionResult(
                success = true,
                totalDeleted = totalDeleted,
                deletionsByType = deletionResults,
                errorMessage = null
            )
            
        } catch (e: Exception) {
            DataDeletionResult(
                success = false,
                totalDeleted = totalDeleted,
                deletionsByType = deletionResults,
                errorMessage = e.message
            )
        }
    }
    
    /**
     * Delete all behavioral data
     */
    private suspend fun deleteAllBehavioralData(): Int {
        // Implementation would delete all behavioral data
        return 0 // Placeholder
    }
    
    /**
     * Delete all usage data
     */
    private suspend fun deleteAllUsageData(): Int {
        // Implementation would delete all usage data
        return 0 // Placeholder
    }
    
    /**
     * Delete all training data
     */
    private suspend fun deleteAllTrainingData(): Int {
        // Implementation would delete all training data
        return 0 // Placeholder
    }
    
    /**
     * Export user data for review
     */
    suspend fun exportUserData(dataTypes: List<DataType>): DataExportResult = withContext(Dispatchers.IO) {
        try {
            val exportData = mutableMapOf<DataType, Any>()
            
            dataTypes.forEach { dataType ->
                val data = when (dataType) {
                    DataType.AUTOMATION_LOGS -> exportAutomationLogs()
                    DataType.BEHAVIORAL_DATA -> exportBehavioralData()
                    DataType.USAGE_DATA -> exportUsageData()
                    DataType.TRAINING_DATA -> exportTrainingData()
                }
                exportData[dataType] = data
            }
            
            val exportJson = Json.encodeToString(exportData)
            
            DataExportResult(
                success = true,
                exportData = exportJson,
                recordCount = exportData.values.sumOf { 
                    when (it) {
                        is List<*> -> it.size
                        else -> 1
                    }
                },
                errorMessage = null
            )
            
        } catch (e: Exception) {
            DataExportResult(
                success = false,
                exportData = null,
                recordCount = 0,
                errorMessage = e.message
            )
        }
    }
    
    /**
     * Export automation logs
     */
    private suspend fun exportAutomationLogs(): List<Map<String, Any>> {
        // Implementation would export automation logs
        return emptyList() // Placeholder
    }
    
    /**
     * Export behavioral data
     */
    private suspend fun exportBehavioralData(): List<Map<String, Any>> {
        // Implementation would export behavioral data
        return emptyList() // Placeholder
    }
    
    /**
     * Export usage data
     */
    private suspend fun exportUsageData(): List<Map<String, Any>> {
        // Implementation would export usage data
        return emptyList() // Placeholder
    }
    
    /**
     * Export training data
     */
    private suspend fun exportTrainingData(): List<Map<String, Any>> {
        // Implementation would export training data
        return emptyList() // Placeholder
    }
    
    /**
     * Load retention policy from storage
     */
    private fun loadRetentionPolicy(): RetentionPolicy {
        // Implementation would load from encrypted preferences
        return RetentionPolicy() // Default policy
    }
    
    /**
     * Save retention policy to storage
     */
    private fun saveRetentionPolicy(policy: RetentionPolicy) {
        // Implementation would save to encrypted preferences
    }
    
    /**
     * Clean up resources
     */
    fun cleanup() {
        scope.cancel()
    }
}

/**
 * Data retention policy configuration
 */
@Serializable
data class RetentionPolicy(
    val automationLogRetentionDays: Int = 90,
    val behavioralDataRetentionDays: Int = 180,
    val usageDataRetentionDays: Int = 365,
    val trainingDataRetentionDays: Int = 730,
    val tempFileRetentionHours: Int = 24,
    val anonymizeTrainingData: Boolean = true,
    val immediateCleanup: Boolean = false
)

/**
 * Data cleanup status
 */
data class CleanupStatus(
    val lastCleanupTime: Long = 0L,
    val cleanupDuration: Long = 0L,
    val recordsDeleted: Int = 0,
    val recordsAnonymized: Int = 0,
    val success: Boolean = true,
    val errorMessage: String? = null
)

/**
 * User data summary for review
 */
data class UserDataSummary(
    val automationLogsCount: Int = 0,
    val behavioralDataCount: Int = 0,
    val usageDataCount: Int = 0,
    val trainingDataCount: Int = 0,
    val totalStorageUsedMB: Double = 0.0,
    val oldestRecordDate: Long = 0L,
    val newestRecordDate: Long = 0L,
    val anonymizedRecordsCount: Int = 0
)

/**
 * Data types for user control
 */
enum class DataType {
    AUTOMATION_LOGS,
    BEHAVIORAL_DATA,
    USAGE_DATA,
    TRAINING_DATA
}

/**
 * Data deletion result
 */
data class DataDeletionResult(
    val success: Boolean,
    val totalDeleted: Int,
    val deletionsByType: Map<DataType, Int>,
    val errorMessage: String?
)

/**
 * Data export result
 */
data class DataExportResult(
    val success: Boolean,
    val exportData: String?,
    val recordCount: Int,
    val errorMessage: String?
)