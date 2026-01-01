package com.lifetwin.mlp.db

import android.content.Context
import android.util.Log
import com.lifetwin.mlp.security.KeyManager
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch

private const val TAG = "DBHelper"

object DBHelper {
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    
    @Volatile
    private var keyManager: KeyManager? = null
    
    /**
     * Initializes the encrypted database with proper key management
     */
    suspend fun initializeEncrypted(context: Context): AppDatabase? {
        return withContext(Dispatchers.IO) {
            try {
                // Initialize key manager
                val km = KeyManager(context)
                keyManager = km
                
                // Validate key system
                if (!km.validateKeySystem()) {
                    Log.e(TAG, "Key system validation failed")
                    return@withContext null
                }
                
                // Get database passphrase
                val passphrase = km.initializeAndGetDatabasePassphrase()
                if (passphrase == null) {
                    Log.e(TAG, "Failed to get database passphrase")
                    return@withContext null
                }
                
                // Initialize database with encryption
                val database = AppDatabase.getInstance(context, passphrase)
                
                // Initialize default privacy settings if needed
                initializeDefaultPrivacySettings(database)
                
                Log.i(TAG, "Encrypted database initialized successfully")
                database
            } catch (e: Exception) {
                Log.e(TAG, "Failed to initialize encrypted database", e)
                null
            }
        }
    }
    
    /**
     * Legacy method for backward compatibility - initializes with provided passphrase
     */
    fun initializeEncrypted(context: Context, passphrase: String) {
        try {
            AppDatabase.getInstance(context, passphrase)
        } catch (_: Exception) {
            // Best-effort: do not crash the app if initialization fails
        }
    }
    
    /**
     * Gets the current key manager instance
     */
    fun getKeyManager(): KeyManager? = keyManager
    
    /**
     * Initializes default privacy settings if they don't exist
     */
    private suspend fun initializeDefaultPrivacySettings(database: AppDatabase) {
        try {
            val privacyDao = database.privacySettingsDao()
            val existingSettings = privacyDao.getSettings()
            
            if (existingSettings == null) {
                val defaultSettings = PrivacySettingsEntity(
                    enabledCollectors = """["usage", "screen", "notifications"]""",
                    dataRetentionDays = 7,
                    privacyLevel = "STANDARD",
                    anonymizationSettings = """{
                        "aggregateAppUsage": true,
                        "removePersonalIdentifiers": true,
                        "fuzzTimestamps": false,
                        "categoryOnlyMode": false,
                        "minimumAggregationWindow": "PT1H"
                    }""",
                    dataSharingSettings = """{
                        "allowCloudSync": false,
                        "allowAnalytics": false,
                        "allowResearchParticipation": false,
                        "encryptionRequired": true
                    }"""
                )
                
                privacyDao.insert(defaultSettings)
                Log.i(TAG, "Default privacy settings initialized")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize default privacy settings", e)
        }
    }
    
    // Legacy async methods for backward compatibility
    fun insertEventAsync(context: Context, event: AppEventEntity) {
        val db = AppDatabase.getInstance(context)
        scope.launch {
            try {
                db.appEventDao().insert(event)
            } catch (e: Exception) {
                Log.e(TAG, "Failed to insert event", e)
            }
        }
    }

    fun insertSummaryAsync(context: Context, summary: DailySummaryEntity) {
        val db = AppDatabase.getInstance(context)
        scope.launch {
            try {
                db.dailySummaryDao().insert(summary)
            } catch (e: Exception) {
                Log.e(TAG, "Failed to insert summary", e)
            }
        }
    }

    suspend fun enqueueSyncPayload(context: Context, payloadJson: String) {
        val db = AppDatabase.getInstance(context)
        try {
            db.syncQueueDao().insert(SyncQueueEntity(payload = payloadJson))
        } catch (e: Exception) {
            Log.e(TAG, "Failed to enqueue sync payload", e)
        }
    }
    
    /**
     * Encrypts sensitive metadata using the key manager
     */
    fun encryptMetadata(metadata: String): String {
        return keyManager?.encryptData(metadata)?.let { encrypted ->
            """{"ciphertext":"${encrypted.ciphertext}","iv":"${encrypted.iv}"}"""
        } ?: metadata // Fallback to plaintext if encryption fails
    }
    
    /**
     * Decrypts sensitive metadata using the key manager
     */
    fun decryptMetadata(encryptedMetadata: String): String {
        return try {
            if (encryptedMetadata.startsWith("{") && encryptedMetadata.contains("ciphertext")) {
                // Parse encrypted format
                val json = org.json.JSONObject(encryptedMetadata)
                val ciphertext = json.getString("ciphertext")
                val iv = json.getString("iv")
                
                keyManager?.decryptData(
                    com.lifetwin.mlp.security.EncryptedData(ciphertext, iv)
                ) ?: encryptedMetadata
            } else {
                // Assume plaintext for backward compatibility
                encryptedMetadata
            }
        } catch (e: Exception) {
            Log.w(TAG, "Failed to decrypt metadata, returning as-is", e)
            encryptedMetadata
        }
    }
    
    /**
     * Performs database cleanup based on privacy settings
     */
    suspend fun performPrivacyCleanup(database: AppDatabase) {
        try {
            val privacySettings = database.privacySettingsDao().getSettings()
            val retentionDays = privacySettings?.dataRetentionDays ?: 7
            val cutoffTime = System.currentTimeMillis() - (retentionDays * 24 * 60 * 60 * 1000L)
            
            // Clean up old raw events that have been processed
            database.rawEventDao().deleteOldProcessedEvents(cutoffTime)
            
            // Clean up old detailed events based on retention policy
            database.usageEventDao().deleteOldEvents(cutoffTime)
            database.notificationEventDao().deleteOldEvents(cutoffTime)
            database.screenSessionDao().deleteOldSessions(cutoffTime)
            database.interactionMetricsDao().deleteOldMetrics(cutoffTime)
            database.activityContextDao().deleteOldContexts(cutoffTime)
            
            // Clean up old audit logs (keep longer for security)
            val auditCutoffTime = System.currentTimeMillis() - (30 * 24 * 60 * 60 * 1000L) // 30 days
            database.auditLogDao().deleteOldLogs(auditCutoffTime)
            
            Log.i(TAG, "Privacy cleanup completed")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to perform privacy cleanup", e)
        }
    }
}