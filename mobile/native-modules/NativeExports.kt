package com.lifetwin.mlp

import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactContextBaseJavaModule
import com.facebook.react.bridge.ReactMethod
import com.lifetwin.mlp.summaries.SyncManager

class NativeExports(reactContext: ReactApplicationContext) : ReactContextBaseJavaModule(reactContext) {
    override fun getName(): String = "NativeExports"

    @ReactMethod
    fun triggerSummarySync() {
        // Expose a simple trigger to the RN layer; actual paramization and callbacks should be added later
        SyncManager.enqueueSync(reactApplicationContext)
    }
}
