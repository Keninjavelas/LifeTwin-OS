package com.lifetwin.mlp.automation

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.provider.Settings
import android.util.Log
import androidx.core.app.NotificationCompat
import androidx.core.app.NotificationManagerCompat
import androidx.core.content.ContextCompat
import androidx.work.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.time.Duration
import java.util.concurrent.TimeUnit

private const val TAG = "AndroidIntegration"
private const val AUTOMATION_CHANNEL_ID = "lifetwin_automation"
private const val INTERVENTION_NOTIFICATION_ID = 1001

/**
 * Handles all Android system integrations for automation interventions.
 * Provides graceful degradation when permissions are not available.
 */
class AndroidIntegration(private val context: Context) {
    
    private lateinit var notificationManager: NotificationManagerCompat
    private lateinit var systemNotificationManager: NotificationManager
    private val permissionStatus = mutableMapOf<String, Boolean>()
    
    suspend fun initialize(): Boolean {
        return try {
            Log.i(TAG, "Initializing AndroidIntegration...")
            
            // Initialize notification system
            setupNotificationChannels()
            notificationManager = NotificationManagerCompat.from(context)
            systemNotificationManager = context.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            
            // Initialize WorkManager for background tasks
            setupWorkManager()
            
            // Check and cache permission status
            checkPermissions()
            
            Log.i(TAG, "AndroidIntegration initialized successfully")
            true
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize AndroidIntegration", e)
            false
        }
    }
    
    /**
     * Execute a specific intervention through appropriate Android APIs
     */
    suspend fun executeIntervention(intervention: InterventionRecommendation): Boolean {
        return withContext(Dispatchers.Main) {
            try {
                when (intervention.type) {
                    InterventionType.BREAK_SUGGESTION -> {
                        postBreakSuggestionNotification(intervention)
                    }
                    InterventionType.DND_ENABLE -> {
                        enableDoNotDisturb(intervention)
                    }
                    InterventionType.APP_LIMIT_SUGGESTION -> {
                        postAppLimitSuggestion(intervention)
                    }
                    InterventionType.FOCUS_MODE_ENABLE -> {
                        enableFocusMode(intervention)
                    }
                    InterventionType.NOTIFICATION_REDUCTION -> {
                        suggestNotificationReduction(intervention)
                    }
                    InterventionType.ACTIVITY_SUGGESTION -> {
                        postActivitySuggestion(intervention)
                    }
                }
                true
            } catch (e: Exception) {
                Log.e(TAG, "Failed to execute intervention: ${intervention.type}", e)
                false
            }
        }
    }
    
    /**
     * Check and cache permission status for graceful degradation
     */
    private fun checkPermissions() {
        // Check notification permission
        permissionStatus["notifications"] = notificationManager.areNotificationsEnabled()
        
        // Check DND access permission
        permissionStatus["dnd_access"] = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            systemNotificationManager.isNotificationPolicyAccessGranted
        } else {
            true // Not needed on older versions
        }
        
        // Check accessibility service permission (for app blocking)
        permissionStatus["accessibility"] = isAccessibilityServiceEnabled()
        
        // Check usage stats permission
        permissionStatus["usage_stats"] = hasUsageStatsPermission()
        
        Log.d(TAG, "Permission status: $permissionStatus")
    }
    
    /**
     * Get current permission status
     */
    fun getPermissionStatus(): Map<String, Boolean> {
        checkPermissions() // Refresh status
        return permissionStatus.toMap()
    }
    
    /**
     * Request missing permissions with user-friendly explanations
     */
    fun requestMissingPermissions(): List<PermissionRequest> {
        val requests = mutableListOf<PermissionRequest>()
        
        if (permissionStatus["notifications"] == false) {
            requests.add(PermissionRequest(
                type = "notifications",
                title = "Enable Notifications",
                description = "Allow LifeTwin to send helpful suggestions and reminders",
                action = "Open notification settings",
                intent = createNotificationSettingsIntent()
            ))
        }
        
        if (permissionStatus["dnd_access"] == false) {
            requests.add(PermissionRequest(
                type = "dnd_access",
                title = "Do Not Disturb Access",
                description = "Allow LifeTwin to help manage your Do Not Disturb settings for better focus",
                action = "Grant DND access",
                intent = createDNDSettingsIntent()
            ))
        }
        
        if (permissionStatus["accessibility"] == false) {
            requests.add(PermissionRequest(
                type = "accessibility",
                title = "Accessibility Service",
                description = "Enable gentle app usage monitoring and focus assistance (optional)",
                action = "Open accessibility settings",
                intent = createAccessibilitySettingsIntent()
            ))
        }
        
        return requests
    }
    
    /**
     * Enable Do Not Disturb mode with proper permission handling
     */
    private fun enableDoNotDisturb(intervention: InterventionRecommendation) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            if (permissionStatus["dnd_access"] == true) {
                try {
                    // Set DND mode to priority only (allows calls from contacts)
                    systemNotificationManager.setInterruptionFilter(NotificationManager.INTERRUPTION_FILTER_PRIORITY)
                    
                    // Post confirmation notification
                    postDNDConfirmationNotification(intervention)
                    Log.d(TAG, "Successfully enabled Do Not Disturb")
                } catch (e: Exception) {
                    Log.e(TAG, "Failed to enable DND", e)
                    // Fallback to suggestion notification
                    suggestDoNotDisturb(intervention)
                }
            } else {
                // No permission - suggest enabling DND manually
                suggestDoNotDisturb(intervention)
            }
        } else {
            // Older Android versions - suggest manual DND
            suggestDoNotDisturb(intervention)
        }
    }
    
    /**
     * Enable focus mode with app blocking (if accessibility service is available)
     */
    private fun enableFocusMode(intervention: InterventionRecommendation) {
        if (permissionStatus["accessibility"] == true) {
            try {
                // Send intent to accessibility service to enable focus mode
                val intent = Intent("com.lifetwin.mlp.ENABLE_FOCUS_MODE").apply {
                    putExtra("intervention_id", intervention.id)
                    putExtra("reasoning", intervention.reasoning)
                }
                context.sendBroadcast(intent)
                
                postFocusModeConfirmationNotification(intervention)
                Log.d(TAG, "Successfully enabled focus mode")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to enable focus mode", e)
                // Fallback to suggestion notification
                suggestFocusMode(intervention)
            }
        } else {
            // No accessibility service - suggest focus mode manually
            suggestFocusMode(intervention)
        }
    }
    
    /**
     * Check if accessibility service is enabled
     */
    private fun isAccessibilityServiceEnabled(): Boolean {
        return try {
            val accessibilityEnabled = Settings.Secure.getInt(
                context.contentResolver,
                Settings.Secure.ACCESSIBILITY_ENABLED
            )
            
            if (accessibilityEnabled == 1) {
                val services = Settings.Secure.getString(
                    context.contentResolver,
                    Settings.Secure.ENABLED_ACCESSIBILITY_SERVICES
                )
                services?.contains(context.packageName) == true
            } else {
                false
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to check accessibility service status", e)
            false
        }
    }
    
    /**
     * Check if usage stats permission is granted
     */
    private fun hasUsageStatsPermission(): Boolean {
        return try {
            val appOps = context.getSystemService(Context.APP_OPS_SERVICE) as android.app.AppOpsManager
            val mode = appOps.checkOpNoThrow(
                android.app.AppOpsManager.OPSTR_GET_USAGE_STATS,
                android.os.Process.myUid(),
                context.packageName
            )
            mode == android.app.AppOpsManager.MODE_ALLOWED
        } catch (e: Exception) {
            Log.e(TAG, "Failed to check usage stats permission", e)
            false
        }
    }
    
    // Intent creators for permission requests
    
    private fun createNotificationSettingsIntent(): Intent {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            Intent(Settings.ACTION_APP_NOTIFICATION_SETTINGS).apply {
                putExtra(Settings.EXTRA_APP_PACKAGE, context.packageName)
            }
        } else {
            Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS).apply {
                data = android.net.Uri.parse("package:${context.packageName}")
            }
        }
    }
    
    private fun createDNDSettingsIntent(): Intent {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            Intent(Settings.ACTION_NOTIFICATION_POLICY_ACCESS_SETTINGS)
        } else {
            Intent(Settings.ACTION_SOUND_SETTINGS)
        }
    }
    
    private fun createAccessibilitySettingsIntent(): Intent {
        return Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS)
    }
    
    // Enhanced notification methods with fallback handling
    
    private fun postDNDConfirmationNotification(intervention: InterventionRecommendation) {
        if (permissionStatus["notifications"] != true) return
        
        val notification = NotificationCompat.Builder(context, AUTOMATION_CHANNEL_ID)
            .setSmallIcon(android.R.drawable.ic_lock_silent_mode)
            .setContentTitle("🔕 Do Not Disturb Enabled")
            .setContentText("Focus mode is now active. Tap to disable when ready.")
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .setAutoCancel(true)
            .addAction(createDNDAction("disable", "🔔 Disable DND"))
            .addAction(createFeedbackAction(intervention.id, "helpful", "👍 Helpful"))
            .build()
        
        notificationManager.notify(INTERVENTION_NOTIFICATION_ID, notification)
    }
    
    private fun postFocusModeConfirmationNotification(intervention: InterventionRecommendation) {
        if (permissionStatus["notifications"] != true) return
        
        val notification = NotificationCompat.Builder(context, AUTOMATION_CHANNEL_ID)
            .setSmallIcon(android.R.drawable.ic_menu_view)
            .setContentTitle("🎯 Focus Mode Active")
            .setContentText("Distracting apps are now blocked. Stay focused!")
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .setAutoCancel(true)
            .addAction(createFocusModeAction("disable", "🔓 Exit Focus"))
            .addAction(createFeedbackAction(intervention.id, "helpful", "👍 Helpful"))
            .build()
        
        notificationManager.notify(INTERVENTION_NOTIFICATION_ID, notification)
    }
    
    private fun setupNotificationChannels() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                AUTOMATION_CHANNEL_ID,
                "LifeTwin Automation",
                NotificationManager.IMPORTANCE_DEFAULT
            ).apply {
                description = "Automated suggestions for digital wellbeing"
                setShowBadge(false)
                enableVibration(false)
                setSound(null, null)
            }
            
            val systemNotificationManager = context.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            systemNotificationManager.createNotificationChannel(channel)
        }
    }
    
    private fun setupWorkManager() {
        val config = Configuration.Builder()
            .setMinimumLoggingLevel(Log.INFO)
            .build()
        
        WorkManager.initialize(context, config)
    }
    
    private fun postBreakSuggestionNotification(intervention: InterventionRecommendation) {
        if (permissionStatus["notifications"] != true) {
            Log.w(TAG, "Cannot post notification - permission not granted")
            return
        }
        
        val notification = NotificationCompat.Builder(context, AUTOMATION_CHANNEL_ID)
            .setSmallIcon(android.R.drawable.ic_dialog_info)
            .setContentTitle("🌟 Time for a Break!")
            .setContentText(intervention.reasoning)
            .setStyle(NotificationCompat.BigTextStyle().bigText(intervention.reasoning))
            .setPriority(NotificationCompat.PRIORITY_DEFAULT)
            .setAutoCancel(true)
            .addAction(createFeedbackAction(intervention.id, "helpful", "👍 Helpful"))
            .addAction(createFeedbackAction(intervention.id, "dismiss", "👎 Not now"))
            .addAction(createBreakTimerAction())
            .build()
        
        notificationManager.notify(INTERVENTION_NOTIFICATION_ID, notification)
        Log.d(TAG, "Posted break suggestion notification")
    }
    
    private fun suggestDoNotDisturb(intervention: InterventionRecommendation) {
        if (permissionStatus["notifications"] != true) return
        
        val notification = NotificationCompat.Builder(context, AUTOMATION_CHANNEL_ID)
            .setSmallIcon(android.R.drawable.ic_lock_silent_mode)
            .setContentTitle("🌙 Enable Do Not Disturb?")
            .setContentText(intervention.reasoning)
            .setStyle(NotificationCompat.BigTextStyle().bigText(intervention.reasoning))
            .setPriority(NotificationCompat.PRIORITY_DEFAULT)
            .setAutoCancel(true)
            .addAction(createDNDAction("enable", "🔕 Enable DND"))
            .addAction(createFeedbackAction(intervention.id, "dismiss", "Not now"))
            .build()
        
        notificationManager.notify(INTERVENTION_NOTIFICATION_ID, notification)
        Log.d(TAG, "Posted DND suggestion notification")
    }
    
    private fun postAppLimitSuggestion(intervention: InterventionRecommendation) {
        if (permissionStatus["notifications"] != true) return
        
        val notification = NotificationCompat.Builder(context, AUTOMATION_CHANNEL_ID)
            .setSmallIcon(android.R.drawable.ic_dialog_alert)
            .setContentTitle("📱 App Usage Alert")
            .setContentText(intervention.reasoning)
            .setStyle(NotificationCompat.BigTextStyle().bigText(intervention.reasoning))
            .setPriority(NotificationCompat.PRIORITY_DEFAULT)
            .setAutoCancel(true)
            .addAction(createAppLimitAction("set", "⏰ Set Limit"))
            .addAction(createFeedbackAction(intervention.id, "dismiss", "Ignore"))
            .build()
        
        notificationManager.notify(INTERVENTION_NOTIFICATION_ID, notification)
        Log.d(TAG, "Posted app limit suggestion notification")
    }
    
    private fun suggestFocusMode(intervention: InterventionRecommendation) {
        if (permissionStatus["notifications"] != true) return
        
        val notification = NotificationCompat.Builder(context, AUTOMATION_CHANNEL_ID)
            .setSmallIcon(android.R.drawable.ic_menu_view)
            .setContentTitle("🎯 Focus Mode Suggestion")
            .setContentText(intervention.reasoning)
            .setStyle(NotificationCompat.BigTextStyle().bigText(intervention.reasoning))
            .setPriority(NotificationCompat.PRIORITY_DEFAULT)
            .setAutoCancel(true)
            .addAction(createFocusModeAction("enable", "🎯 Enable Focus"))
            .addAction(createFeedbackAction(intervention.id, "dismiss", "Later"))
            .build()
        
        notificationManager.notify(INTERVENTION_NOTIFICATION_ID, notification)
        Log.d(TAG, "Posted focus mode suggestion notification")
    }
    
    private fun suggestNotificationReduction(intervention: InterventionRecommendation) {
        if (permissionStatus["notifications"] != true) return
        
        val notification = NotificationCompat.Builder(context, AUTOMATION_CHANNEL_ID)
            .setSmallIcon(android.R.drawable.ic_dialog_info)
            .setContentTitle("🔔 Reduce Notifications?")
            .setContentText(intervention.reasoning)
            .setStyle(NotificationCompat.BigTextStyle().bigText(intervention.reasoning))
            .setPriority(NotificationCompat.PRIORITY_DEFAULT)
            .setAutoCancel(true)
            .addAction(createNotificationSettingsAction("open", "⚙️ Settings"))
            .addAction(createFeedbackAction(intervention.id, "dismiss", "Keep as is"))
            .build()
        
        notificationManager.notify(INTERVENTION_NOTIFICATION_ID, notification)
        Log.d(TAG, "Posted notification reduction suggestion")
    }
    
    private fun postActivitySuggestion(intervention: InterventionRecommendation) {
        if (permissionStatus["notifications"] != true) return
        
        val activities = listOf("Take a walk", "Do stretches", "Drink water", "Look outside")
        val randomActivity = activities.random()
        
        val notification = NotificationCompat.Builder(context, AUTOMATION_CHANNEL_ID)
            .setSmallIcon(android.R.drawable.ic_menu_compass)
            .setContentTitle("🚶 Activity Suggestion")
            .setContentText("$randomActivity - ${intervention.reasoning}")
            .setStyle(NotificationCompat.BigTextStyle().bigText("$randomActivity - ${intervention.reasoning}"))
            .setPriority(NotificationCompat.PRIORITY_DEFAULT)
            .setAutoCancel(true)
            .addAction(createActivityAction("done", "✅ Done!"))
            .addAction(createFeedbackAction(intervention.id, "dismiss", "Skip"))
            .build()
        
        notificationManager.notify(INTERVENTION_NOTIFICATION_ID, notification)
        Log.d(TAG, "Posted activity suggestion notification")
    }
    
    // Action creators for notifications
    
    private fun createFeedbackAction(interventionId: String, action: String, title: String): NotificationCompat.Action {
        val intent = Intent(context, AutomationActionReceiver::class.java).apply {
            putExtra("intervention_id", interventionId)
            putExtra("action", action)
        }
        
        val pendingIntent = PendingIntent.getBroadcast(
            context,
            interventionId.hashCode() + action.hashCode(),
            intent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
        
        return NotificationCompat.Action.Builder(0, title, pendingIntent).build()
    }
    
    private fun createBreakTimerAction(): NotificationCompat.Action {
        val intent = Intent(context, BreakTimerActivity::class.java)
        val pendingIntent = PendingIntent.getActivity(
            context,
            0,
            intent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
        
        return NotificationCompat.Action.Builder(0, "⏱️ Start Timer", pendingIntent).build()
    }
    
    private fun createDNDAction(action: String, title: String): NotificationCompat.Action {
        val intent = Intent(context, AutomationActionReceiver::class.java).apply {
            putExtra("action", "dnd_$action")
        }
        
        val pendingIntent = PendingIntent.getBroadcast(
            context,
            action.hashCode(),
            intent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
        
        return NotificationCompat.Action.Builder(0, title, pendingIntent).build()
    }
    
    private fun createAppLimitAction(action: String, title: String): NotificationCompat.Action {
        val intent = Intent(context, AppLimitActivity::class.java)
        val pendingIntent = PendingIntent.getActivity(
            context,
            0,
            intent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
        
        return NotificationCompat.Action.Builder(0, title, pendingIntent).build()
    }
    
    private fun createFocusModeAction(action: String, title: String): NotificationCompat.Action {
        val intent = Intent(context, AutomationActionReceiver::class.java).apply {
            putExtra("action", "focus_$action")
        }
        
        val pendingIntent = PendingIntent.getBroadcast(
            context,
            action.hashCode(),
            intent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
        
        return NotificationCompat.Action.Builder(0, title, pendingIntent).build()
    }
    
    private fun createNotificationSettingsAction(action: String, title: String): NotificationCompat.Action {
        val intent = Intent(android.provider.Settings.ACTION_APP_NOTIFICATION_SETTINGS).apply {
            putExtra(android.provider.Settings.EXTRA_APP_PACKAGE, context.packageName)
        }
        
        val pendingIntent = PendingIntent.getActivity(
            context,
            0,
            intent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
        
        return NotificationCompat.Action.Builder(0, title, pendingIntent).build()
    }
    
    private fun createActivityAction(action: String, title: String): NotificationCompat.Action {
        val intent = Intent(context, AutomationActionReceiver::class.java).apply {
            putExtra("action", "activity_$action")
        }
        
        val pendingIntent = PendingIntent.getBroadcast(
            context,
            action.hashCode(),
            intent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
        
        return NotificationCompat.Action.Builder(0, title, pendingIntent).build()
    }
    
    /**
     * Schedule a delayed intervention using WorkManager
     */
    fun scheduleDelayedIntervention(intervention: InterventionRecommendation, delayMinutes: Long) {
        val workRequest = OneTimeWorkRequestBuilder<DelayedInterventionWorker>()
            .setInitialDelay(delayMinutes, TimeUnit.MINUTES)
            .setInputData(workDataOf(
                "intervention_id" to intervention.id,
                "intervention_type" to intervention.type.name,
                "reasoning" to intervention.reasoning
            ))
            .build()
        
        WorkManager.getInstance(context).enqueue(workRequest)
        Log.d(TAG, "Scheduled delayed intervention: ${intervention.type} in $delayMinutes minutes")
    }
    
    /**
     * Cancel all pending notifications
     */
    fun cancelAllNotifications() {
        notificationManager.cancelAll()
        Log.d(TAG, "Cancelled all automation notifications")
    }
}

/**
 * Represents a permission request with user-friendly information
 */
data class PermissionRequest(
    val type: String,
    val title: String,
    val description: String,
    val action: String,
    val intent: Intent
)

/**
 * WorkManager worker for delayed interventions
 */
class DelayedInterventionWorker(
    context: Context,
    params: WorkerParameters
) : CoroutineWorker(context, params) {
    
    override suspend fun doWork(): Result {
        return try {
            val interventionId = inputData.getString("intervention_id") ?: return Result.failure()
            val interventionType = inputData.getString("intervention_type") ?: return Result.failure()
            val reasoning = inputData.getString("reasoning") ?: return Result.failure()
            
            // Recreate intervention and execute
            val intervention = InterventionRecommendation(
                id = interventionId,
                type = InterventionType.valueOf(interventionType),
                trigger = "delayed_execution",
                confidence = 0.8f,
                reasoning = reasoning
            )
            
            val androidIntegration = AndroidIntegration(applicationContext)
            androidIntegration.initialize()
            androidIntegration.executeIntervention(intervention)
            
            Result.success()
        } catch (e: Exception) {
            Log.e(TAG, "Failed to execute delayed intervention", e)
            Result.failure()
        }
    }
}

// Placeholder classes for activities that would be implemented
class BreakTimerActivity : android.app.Activity()
class AppLimitActivity : android.app.Activity()
class AutomationActionReceiver : android.content.BroadcastReceiver() {
    override fun onReceive(context: Context?, intent: Intent?) {
        // Handle automation action responses
        val interventionId = intent?.getStringExtra("intervention_id")
        val action = intent?.getStringExtra("action")
        
        Log.d(TAG, "Received automation action: $action for intervention: $interventionId")
        
        // TODO: Send feedback to AutomationEngine
        when (action) {
            "dnd_enable" -> {
                // Handle DND enable action
                Log.d(TAG, "User requested DND enable")
            }
            "dnd_disable" -> {
                // Handle DND disable action
                if (context != null && Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                    val notificationManager = context.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
                    notificationManager.setInterruptionFilter(NotificationManager.INTERRUPTION_FILTER_ALL)
                }
            }
            "focus_enable" -> {
                // Handle focus mode enable
                Log.d(TAG, "User requested focus mode enable")
            }
            "focus_disable" -> {
                // Handle focus mode disable
                context?.sendBroadcast(Intent("com.lifetwin.mlp.DISABLE_FOCUS_MODE"))
            }
            "activity_done" -> {
                // Handle activity completion
                Log.d(TAG, "User completed activity suggestion")
            }
            "helpful", "dismiss" -> {
                // Handle feedback
                Log.d(TAG, "User provided feedback: $action for intervention: $interventionId")
            }
        }
    }
}