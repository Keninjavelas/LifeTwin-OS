package com.lifetwin.mlp.automation

import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactContextBaseJavaModule
import com.facebook.react.bridge.ReactMethod
import com.facebook.react.bridge.Promise
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject

class AutomationModule(reactContext: ReactApplicationContext) : ReactContextBaseJavaModule(reactContext) {

    companion object {
        // Default to emulator host; update with setServerUrl from JS if needed
        var serverUrl: String = "http://10.0.2.2:8000"
    }

    private val client = OkHttpClient()

    override fun getName(): String {
        return "AutomationModule"
    }

    @ReactMethod
    fun setServerUrl(url: String) {
        serverUrl = url
    }

    /**
     * Trigger the backend export for a device and return the export path.
     * Params:
     *  token: auth token (string)
     *  deviceId: device id used when summaries were uploaded
     */
    @ReactMethod
    fun triggerDailySummaryExport(token: String, deviceId: String, promise: Promise) {
        val url = "$serverUrl/admin/export-summaries"
        val json = JSONObject()
        json.put("token", token)
        json.put("device_id", deviceId)

        val mediaType = "application/json; charset=utf-8".toMediaType()
        val body = json.toString().toRequestBody(mediaType)
        val request = Request.Builder().url(url).post(body).build()

        val executor = java.util.concurrent.Executors.newSingleThreadExecutor()
        executor.execute {
            try {
                client.newCall(request).execute().use { resp ->
                    val code = resp.code
                    val respBody = resp.body?.string()
                    if (!resp.isSuccessful) {
                        promise.reject("http_error", "HTTP $code: $respBody")
                        return@use
                    }
                    if (respBody == null) {
                        promise.reject("empty_response", "Empty response body")
                        return@use
                    }
                    try {
                        val obj = JSONObject(respBody)
                        val exportPath = obj.optString("export_path", "")
                        val count = obj.optInt("count", 0)
                        val result = JSONObject()
                        result.put("export_path", exportPath)
                        result.put("count", count)
                        promise.resolve(result.toString())
                    } catch (je: Exception) {
                        promise.reject("parse_error", je.message)
                    }
                }
            } catch (e: Exception) {
                promise.reject("network_error", e.message)
            }
        }
    }

    @ReactMethod
    fun toggleAutomation(enabled: Boolean, promise: Promise) {
        // TODO: persist toggle and notify AutomationManager
        // For now we just resolve immediately
        promise.resolve(enabled)
    }
}
