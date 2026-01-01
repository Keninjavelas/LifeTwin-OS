package com.lifetwin.mlp.automation

import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.Build
import android.os.PowerManager
import android.util.Log
import androidx.work.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.util.concurrent.TimeUnit

private const val TAG = "BackgroundAutomationWorker"

/**
 * WorkManager-based background execution system for automation.
 * Handles reliable background processing with battery optimization integration.
 */
class BackgroundAutomationManager(private val context: Context) {
    
    private val workManager = WorkManager.getInstance(context)
    private val powerManager = context.getSystemService(Context.POWER_SERVICE) as PowerManager
    
    /**
     * Initialize background automation with adaptive scheduling
     */
    fun initialize() {
        Log.i(TAG, "Initializing background automation manager")
        
        // Set up periodic automation evaluation
        schedulePeriodicAutomation()
        
        // Set up battery-aware scheduling
        setupBatteryOptimizedScheduling()
        
        Log.i(TAG, "Background automation manager initialized")
    }
    
    /**
     * Schedule periodic automation evaluation with adaptive frequency
     */
    private fun schedulePeriodicAutomation() {
        val constraints = Constraints.Builder()
            .setRequiredNetworkType(NetworkType.NOT_REQUIRED)
            .setRequiresBatteryNotLow(true)
            .setRequiresDeviceIdle(false)
            .build()
        
        val periodicWork = PeriodicWorkRequestBuilder<AutomationEvaluationWorker>(
            15, TimeUnit.MINUTES, // Repeat every 15 minutes
            5, TimeUnit.MINUTES   // Flex interval of 5 minutes
        )
            .setConstraints(constraints)
            .setBackoffCriteria(
                BackoffPolicy.EXPONENTIAL,
                WorkRequest.MIN_BACKOFF_MILLIS,
                TimeUnit.MILLISECONDS
            )
            .addTag("automation_evaluation")
            .build()
        
        workManager.enqueueUniquePeriodicWork(
            "automation_evaluation",
            ExistingPeriodicWorkPolicy.KEEP,
            periodicWork
        )
        
        Log.d(TAG, "Scheduled periodic automation evaluation")
    }
    
    /**
     * Set up battery-optimized scheduling based on device state
     */
    private fun setupBatteryOptimizedScheduling() {
        // Schedule battery state monitoring
        val batteryMonitorWork = PeriodicWorkRequestBuilder<BatteryStateMonitorWorker>(
            1, TimeUnit.HOURS // Check battery state every hour
        )
            .setConstraints(
                Constraints.Builder()
                    .setRequiredNetworkType(NetworkType.NOT_REQUIRED)
                    .build()
            )
            .addTag("battery_monitor")
            .build()
        
        workManager.enqueueUniquePeriodicWork(
            "battery_monitor",
            ExistingPeriodicWorkPolicy.KEEP,
            batteryMonitorWork
        )
        
        Log.d(TAG, "Set up battery-optimized scheduling")
    }
    
    /**
     * Schedule one-time automation task with device state awareness
     */
    fun scheduleOneTimeAutomation(delayMinutes: Long = 0) {
        val deviceState = getCurrentDeviceState()
        val adjustedDelay = adjustDelayForDeviceState(delayMinutes, deviceState)
        
        val constraints = Constraints.Builder()
            .setRequiredNetworkType(NetworkType.NOT_REQUIRED)
            .setRequiresBatteryNotLow(deviceState.batteryLevel < 0.2f) // Only if battery > 20%
            .setRequiresCharging(false)
            .build()
        
        val oneTimeWork = OneTimeWorkRequestBuilder<AutomationEvaluationWorker>()
            .setInitialDelay(adjustedDelay, TimeUnit.MINUTES)
            .setConstraints(constraints)
            .setBackoffCriteria(
                BackoffPolicy.LINEAR,
                WorkRequest.MIN_BACKOFF_MILLIS,
                TimeUnit.MILLISECONDS
            )
            .addTag("automation_onetime")
            .build()
        
        workManager.enqueue(oneTimeWork)
        Log.d(TAG, "Scheduled one-time automation task with ${adjustedDelay}min delay")
    }
    
    /**
     * Adjust scheduling based on current device state
     */
    private fun adjustDelayForDeviceState(originalDelay: Long, deviceState: DeviceState): Long {
        return when {
            deviceState.batteryLevel < 0.15f -> originalDelay * 2 // Double delay on low battery
            deviceState.isCharging -> maxOf(originalDelay / 2, 1) // Halve delay when charging
            deviceState.isPowerSaveMode -> originalDelay * 3 // Triple delay in power save mode
            else -> originalDelay
        }
    }
    
    /**
     * Get current device state for adaptive scheduling
     */
    private fun getCurrentDeviceState(): DeviceState {
        val batteryIntent = context.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        
        val batteryLevel = batteryIntent?.let { intent ->
            val level = intent.getIntExtra(BatteryManager.EXTRA_LEVEL, -1)
            val scale = intent.getIntExtra(BatteryManager.EXTRA_SCALE, -1)
            if (level >= 0 && scale > 0) level.toFloat() / scale.toFloat() else 0.5f
        } ?: 0.5f
        
        val isCharging = batteryIntent?.let { intent ->
            val status = intent.getIntExtra(BatteryManager.EXTRA_STATUS, -1)
            status == BatteryManager.BATTERY_STATUS_CHARGING || 
            status == BatteryManager.BATTERY_STATUS_FULL
        } ?: false
        
        val isPowerSaveMode = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) {
            powerManager.isPowerSaveMode
        } else {
            false
        }
        
        return DeviceState(
            batteryLevel = batteryLevel,
            isCharging = isCharging,
            isPowerSaveMode = isPowerSaveMode
        )
    }
    
    /**
     * Cancel all background automation work
     */
    fun cancelAllWork() {
        workManager.cancelAllWorkByTag("automation_evaluation")
        workManager.cancelAllWorkByTag("automation_onetime")
        workManager.cancelAllWorkByTag("battery_monitor")
        Log.d(TAG, "Cancelled all background automation work")
    }
    
    /**
     * Get work status for monitoring
     */
    fun getWorkStatus(): BackgroundWorkStatus {
        val periodicWorkInfo = workManager.getWorkInfosForUniqueWork("automation_evaluation")
        val batteryWorkInfo = workManager.getWorkInfosForUniqueWork("battery_monitor")
        
        return BackgroundWorkStatus(
            periodicWorkActive = try {
                periodicWorkInfo.get().any { it.state == WorkInfo.State.ENQUEUED || it.state == WorkInfo.State.RUNNING }
            } catch (e: Exception) {
                false
            },
            batteryMonitorActive = try {
                batteryWorkInfo.get().any { it.state == WorkInfo.State.ENQUEUED || it.state == WorkInfo.State.RUNNING }
            } catch (e: Exception) {
                false
            },
            deviceState = getCurrentDeviceState()
        )
    }
}

/**
 * WorkManager worker for periodic automation evaluation
 */
class AutomationEvaluationWorker(
    context: Context,
    params: WorkerParameters
) : CoroutineWorker(context, params) {
    
    override suspend fun doWork(): Result = withContext(Dispatchers.IO) {
        return@withContext try {
            Log.d(TAG, "Starting automation evaluation work")
            
            // Initialize automation engine
            val automationEngine = AutomationEngine(applicationContext)
            val initSuccess = automationEngine.initialize()
            
            if (!initSuccess) {
                Log.e(TAG, "Failed to initialize automation engine")
                return@withContext Result.retry()
            }
            
            // Evaluate and execute interventions
            val interventions = automationEngine.evaluateInterventions()
            Log.d(TAG, "Evaluated ${interventions.size} potential interventions")
            
            // Execute up to 2 interventions to avoid overwhelming user
            val executedCount = interventions.take(2).count { intervention ->
                try {
                    automationEngine.executeIntervention(intervention)
                    true
                } catch (e: Exception) {
                    Log.e(TAG, "Failed to execute intervention: ${intervention.type}", e)
                    false
                }
            }
            
            Log.d(TAG, "Successfully executed $executedCount interventions")
            
            // Clean up
            automationEngine.cleanup()
            
            Result.success()
            
        } catch (e: Exception) {
            Log.e(TAG, "Automation evaluation work failed", e)
            Result.retry()
        }
    }
}

/**
 * WorkManager worker for monitoring battery state and adjusting scheduling
 */
class BatteryStateMonitorWorker(
    context: Context,
    params: WorkerParameters
) : CoroutineWorker(context, params) {
    
    override suspend fun doWork(): Result = withContext(Dispatchers.IO) {
        return@withContext try {
            Log.d(TAG, "Monitoring battery state for adaptive scheduling")
            
            val backgroundManager = BackgroundAutomationManager(applicationContext)
            val deviceState = backgroundManager.getCurrentDeviceState()
            
            Log.d(TAG, "Device state: battery=${deviceState.batteryLevel}, charging=${deviceState.isCharging}, powerSave=${deviceState.isPowerSaveMode}")
            
            // Adjust automation frequency based on battery state
            when {
                deviceState.batteryLevel < 0.15f -> {
                    // Very low battery - reduce automation frequency
                    Log.i(TAG, "Low battery detected - reducing automation frequency")
                    // Cancel current periodic work and reschedule with longer interval
                    WorkManager.getInstance(applicationContext).cancelUniqueWork("automation_evaluation")
                    
                    val lowBatteryWork = PeriodicWorkRequestBuilder<AutomationEvaluationWorker>(
                        30, TimeUnit.MINUTES // Reduce to every 30 minutes
                    )
                        .setConstraints(
                            Constraints.Builder()
                                .setRequiresBatteryNotLow(false) // Allow on low battery
                                .build()
                        )
                        .addTag("automation_evaluation")
                        .build()
                    
                    WorkManager.getInstance(applicationContext).enqueueUniquePeriodicWork(
                        "automation_evaluation",
                        ExistingPeriodicWorkPolicy.REPLACE,
                        lowBatteryWork
                    )
                }
                
                deviceState.isCharging && deviceState.batteryLevel > 0.8f -> {
                    // High battery and charging - increase automation frequency
                    Log.i(TAG, "High battery and charging - increasing automation frequency")
                    WorkManager.getInstance(applicationContext).cancelUniqueWork("automation_evaluation")
                    
                    val highBatteryWork = PeriodicWorkRequestBuilder<AutomationEvaluationWorker>(
                        10, TimeUnit.MINUTES // Increase to every 10 minutes
                    )
                        .setConstraints(
                            Constraints.Builder()
                                .setRequiresBatteryNotLow(false)
                                .build()
                        )
                        .addTag("automation_evaluation")
                        .build()
                    
                    WorkManager.getInstance(applicationContext).enqueueUniquePeriodicWork(
                        "automation_evaluation",
                        ExistingPeriodicWorkPolicy.REPLACE,
                        highBatteryWork
                    )
                }
            }
            
            Result.success()
            
        } catch (e: Exception) {
            Log.e(TAG, "Battery state monitoring failed", e)
            Result.failure()
        }
    }
}

/**
 * Device state information for adaptive scheduling
 */
data class DeviceState(
    val batteryLevel: Float,
    val isCharging: Boolean,
    val isPowerSaveMode: Boolean
)

/**
 * Background work status information
 */
data class BackgroundWorkStatus(
    val periodicWorkActive: Boolean,
    val batteryMonitorActive: Boolean,
    val deviceState: DeviceState
)