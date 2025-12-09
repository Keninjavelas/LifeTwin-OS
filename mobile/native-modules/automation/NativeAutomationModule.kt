package com.lifetwin.mlp.automation

import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactContextBaseJavaModule
import com.facebook.react.bridge.ReactMethod
import com.facebook.react.bridge.Promise
import android.util.Log

private const val TAG = "NativeAutomationModule"

class NativeAutomationModule(reactContext: ReactApplicationContext) : ReactContextBaseJavaModule(reactContext) {
    override fun getName(): String = "NativeAutomation"

    @ReactMethod
    fun startAutomation(promise: Promise) {
        try {
            // Call into AutomationManager (stub)
            val mgr = AutomationManager()
            mgr.start()
            promise.resolve(mapOf("status" to "started"))
        } catch (e: Exception) {
            Log.w(TAG, "startAutomation failed: ${e.message}")
            promise.reject("start_error", e)
        }
    }

    @ReactMethod
    fun stopAutomation(promise: Promise) {
        try {
            val mgr = AutomationManager()
            mgr.stop()
            promise.resolve(mapOf("status" to "stopped"))
        } catch (e: Exception) {
            Log.w(TAG, "stopAutomation failed: ${e.message}")
            promise.reject("stop_error", e)
        }
    }
}
