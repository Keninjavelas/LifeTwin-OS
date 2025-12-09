package com.lifetwin.mlp.summaries

import android.content.Context
import com.google.gson.Gson
import com.lifetwin.mlp.db.AppDatabase
import com.lifetwin.mlp.db.DBHelper
import com.lifetwin.mlp.db.SyncQueueEntity
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody

object SyncManager {
    private val scope = CoroutineScope(Dispatchers.IO)
    private val gson = Gson()
    private val client = OkHttpClient()

    // Configure with your backend endpoint; default to localhost dev server
    var endpoint: String = "http://10.0.2.2:8000/admin/store-encrypted-summary"

    fun enqueueSync(context: Context) {
        scope.launch {
            try {
                val db = AppDatabase.getInstance(context)
                val summaries = db.dailySummaryDao().getSummariesForDevice("local-device")
                val payload = mapOf("device_id" to "local-device", "summaries" to summaries)
                val json = gson.toJson(payload)

                val body = RequestBody.create("application/json; charset=utf-8".toMediaTypeOrNull(), json)
                val req = Request.Builder().url(endpoint).post(body).build()
                client.newCall(req).execute().use { resp ->
                    // For now, just ensure success code; further failure handling can be added
                    if (!resp.isSuccessful) {
                        // Persist payload for retry later
                        DBHelper.enqueueSyncPayload(context, json)
                    }
                }
                // If send succeeded, try to drain the queue
                val queued = db.syncQueueDao().listAll()
                for (q in queued) {
                    val bodyQ = RequestBody.create("application/json; charset=utf-8".toMediaTypeOrNull(), q.payload)
                    val reqQ = Request.Builder().url(endpoint).post(bodyQ).build()
                    client.newCall(reqQ).execute().use { respQ ->
                        if (respQ.isSuccessful) {
                            db.syncQueueDao().deleteById(q.id)
                        }
                    }
                }
            } catch (e: Exception) {
                // Persist payload on exception so it can be retried
                try {
                    // best-effort: store an empty payload indicator if we couldn't construct one above
                    val db = AppDatabase.getInstance(context)
                    DBHelper.enqueueSyncPayload(context, "{}")
                } catch (_: Exception) {
                }
            }
        }
    }
}
