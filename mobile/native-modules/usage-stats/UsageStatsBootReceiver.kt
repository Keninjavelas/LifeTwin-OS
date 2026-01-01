package com.lifetwin.mlp.usagestats

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch

private const val TAG = "UsageStatsBootReceiver"

/**
 * BroadcastReceiver that restarts usage stats collection after device boot
 */
class UsageStatsBootReceiver : BroadcastReceiver() {
    
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    override fun onReceive(context: Context, intent: Intent) {
        when (intent.action) {
            Intent.ACTION_BOOT_COMPLETED,
            Intent.ACTION_MY_PACKAGE_REPLACED,
            Intent.ACTION_PACKAGE_REPLACED -> {
                Log.i(TAG, "Device boot or app update detected, restarting usage stats collection")
                
                scope.launch {
                    try {
                        // Check if collection should be restarted
                        val collector = UsageStatsCollector(context)
                        
                        if (collector.isPermissionGranted()) {
                            // Restart periodic collection
                            UsageStatsWorker.schedulePeriodicCollection(context)
                            
                            // Start the collector
                            collector.startCollection()
                            
                            Log.i(TAG, "Usage stats collection restarted successfully")
                        } else {
                            Log.w(TAG, "Cannot restart collection: permission not granted")
                        }
                    } catch (e: Exception) {
                        Log.e(TAG, "Failed to restart usage stats collection", e)
                    }
                }
            }
        }
    }
}