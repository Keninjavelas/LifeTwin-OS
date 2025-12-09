package com.lifetwin.mlp.db

import android.content.Context
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch

/**
 * Small helper to perform DB writes on a dedicated coroutine scope. This ensures
 * we don't block the calling thread and provides a central place to handle failures.
 */
object DBHelper {
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    /**
     * Optional: initialize an encrypted DB instance if SQLCipher is available.
     * Call this early (e.g. in Application.onCreate) with a user/device passphrase
     * to attempt creating an encrypted database. If SQLCipher isn't present the
     * call will silently fall back to a plain Room DB.
     */
    fun initializeEncrypted(context: Context, passphrase: String) {
        try {
            AppDatabase.getInstance(context, passphrase)
        } catch (_: Exception) {
            // Best-effort: do not crash the app if initialization fails
        }
    }

    fun insertEventAsync(context: Context, event: AppEventEntity) {
        val db = AppDatabase.getInstance(context)
        scope.launch {
            try {
                db.appEventDao().insert(event)
            } catch (e: Exception) {
                // TODO: consider logging to file or telemetry
            }
        }
    }

    fun insertSummaryAsync(context: Context, summary: DailySummaryEntity) {
        val db = AppDatabase.getInstance(context)
        scope.launch {
            try {
                db.dailySummaryDao().insert(summary)
            } catch (e: Exception) {
                // TODO: consider logging to file or telemetry
            }
        }
    }

    suspend fun enqueueSyncPayload(context: Context, payloadJson: String) {
        val db = AppDatabase.getInstance(context)
        try {
            db.syncQueueDao().insert(SyncQueueEntity(payload = payloadJson))
        } catch (e: Exception) {
            // best-effort: if we cannot persist to DB, there's not much we can do here
        }
    }
}
