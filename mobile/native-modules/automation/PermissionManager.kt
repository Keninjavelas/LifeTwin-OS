package com.lifetwin.mlp.automation

import android.app.Activity
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.provider.Settings
import android.util.Log
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

private const val TAG = "PermissionManager"

/**
 * Comprehensive permission management with graceful degradation for automation features.
 * Provides user-friendly permission requests and fallback functionality.
 */
class PermissionManager(private val context: Context) {
    
    private val _permissionState = MutableStateFlow(PermissionState())
    val permissionState: StateFlow<PermissionState> = _permissionState.asStateFlow()
    
    private val _fallbackCapabilities = MutableStateFlow(FallbackCapabilities())
    val fallbackCapabilities: StateFlow<FallbackCapabilities> = _fallbackCapabilities.asStateFlow()
    
    /**
     * Initialize permission manager and check current status
     */
    fun initialize() {
        Log.i(TAG, "Initializing permission manager")
        refreshPermissionStatus()
        updateFallbackCapabilities()
        Log.i(TAG, "Permission manager initialized")
    }
    
    /**
     * Refresh all permission statuses
     */
    fun refreshPermissionStatus() {
        val currentState = PermissionState(
            notifications = checkNotificationPermission(),
            dndAccess = checkDNDPermission(),
            accessibility = checkAccessibilityPermission(),
            usageStats = checkUsageStatsPermission(),
            overlay = checkOverlayPermission(),
            deviceAdmin = checkDeviceAdminPermission(),
            batteryOptimization = checkBatteryOptimizationExemption()
        )
        
        _permissionState.value = currentState
        updateFallbackCapabilities()
        
        Log.d(TAG, "Permission status refreshed: $currentState")
    }
    
    /**
     * Get comprehensive permission status with explanations
     */
    fun getDetailedPermissionStatus(): List<PermissionInfo> {
        val state = _permissionState.value
        
        return listOf(
            PermissionInfo(
                type = PermissionType.NOTIFICATIONS,
                granted = state.notifications,
                required = true,
                title = "Notifications",
                description = "Required to send helpful suggestions and reminders",
                impact = if (state.notifications) "Full automation features available" 
                        else "No suggestions will be shown",
                fallbackAvailable = false
            ),
            PermissionInfo(
                type = PermissionType.DND_ACCESS,
                granted = state.dndAccess,
                required = false,
                title = "Do Not Disturb Access",
                description = "Allows automatic Do Not Disturb management for better focus",
                impact = if (state.dndAccess) "Can automatically enable/disable DND" 
                        else "Manual DND suggestions only",
                fallbackAvailable = true
            ),
            PermissionInfo(
                type = PermissionType.ACCESSIBILITY,
                granted = state.accessibility,
                required = false,
                title = "Accessibility Service",
                description = "Enables gentle app usage monitoring and focus assistance",
                impact = if (state.accessibility) "Advanced focus mode with app blocking" 
                        else "Basic focus suggestions only",
                fallbackAvailable = true
            ),
            PermissionInfo(
                type = PermissionType.USAGE_STATS,
                granted = state.usageStats,
                required = true,
                title = "Usage Access",
                description = "Required to understand your app usage patterns",
                impact = if (state.usageStats) "Personalized suggestions based on usage" 
                        else "Generic suggestions only",
                fallbackAvailable = true
            ),
            PermissionInfo(
                type = PermissionType.OVERLAY,
                granted = state.overlay,
                required = false,
                title = "Display Over Other Apps",
                description = "Allows gentle overlay notifications for focus assistance",
                impact = if (state.overlay) "Subtle overlay reminders available" 
                        else "Standard notifications only",
                fallbackAvailable = true
            ),
            PermissionInfo(
                type = PermissionType.BATTERY_OPTIMIZATION,
                granted = state.batteryOptimization,
                required = false,
                title = "Battery Optimization Exemption",
                description = "Ensures reliable background automation",
                impact = if (state.batteryOptimization) "Consistent background operation" 
                        else "May have delayed suggestions",
                fallbackAvailable = true
            )
        )
    }
    
    /**
     * Request specific permission with user-friendly flow
     */
    fun requestPermission(activity: Activity, permissionType: PermissionType, requestCode: Int) {
        when (permissionType) {
            PermissionType.NOTIFICATIONS -> requestNotificationPermission(activity, requestCode)
            PermissionType.DND_ACCESS -> requestDNDPermission(activity)
            PermissionType.ACCESSIBILITY -> requestAccessibilityPermission(activity)
            PermissionType.USAGE_STATS -> requestUsageStatsPermission(activity)
            PermissionType.OVERLAY -> requestOverlayPermission(activity)
            PermissionType.BATTERY_OPTIMIZATION -> requestBatteryOptimizationExemption(activity)
            PermissionType.DEVICE_ADMIN -> requestDeviceAdminPermission(activity)
        }
    }
    
    /**
     * Get fallback functionality when permissions are denied
     */
    private fun updateFallbackCapabilities() {
        val state = _permissionState.value
        
        val capabilities = FallbackCapabilities(
            canShowNotifications = state.notifications,
            canControlDND = state.dndAccess,
            canBlockApps = state.accessibility,
            canAnalyzeUsage = state.usageStats,
            canShowOverlays = state.overlay,
            canRunInBackground = state.batteryOptimization,
            
            // Fallback methods
            fallbackNotifications = !state.notifications && canUseFallbackNotifications(),
            fallbackDND = !state.dndAccess && canUseFallbackDND(),
            fallbackFocus = !state.accessibility && canUseFallbackFocus(),
            fallbackUsageAnalysis = !state.usageStats && canUseFallbackUsageAnalysis(),
            fallbackBackground = !state.batteryOptimization && canUseFallbackBackground()
        )
        
        _fallbackCapabilities.value = capabilities
        Log.d(TAG, "Updated fallback capabilities: $capabilities")
    }
    
    // Permission checking methods
    
    private fun checkNotificationPermission(): Boolean {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            ContextCompat.checkSelfPermission(
                context,
                android.Manifest.permission.POST_NOTIFICATIONS
            ) == PackageManager.PERMISSION_GRANTED
        } else {
            androidx.core.app.NotificationManagerCompat.from(context).areNotificationsEnabled()
        }
    }
    
    private fun checkDNDPermission(): Boolean {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            val notificationManager = context.getSystemService(Context.NOTIFICATION_SERVICE) 
                as android.app.NotificationManager
            notificationManager.isNotificationPolicyAccessGranted
        } else {
            true // Not needed on older versions
        }
    }
    
    private fun checkAccessibilityPermission(): Boolean {
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
            Log.e(TAG, "Failed to check accessibility permission", e)
            false
        }
    }
    
    private fun checkUsageStatsPermission(): Boolean {
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
    
    private fun checkOverlayPermission(): Boolean {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            Settings.canDrawOverlays(context)
        } else {
            true // Not needed on older versions
        }
    }
    
    private fun checkDeviceAdminPermission(): Boolean {
        return try {
            val devicePolicyManager = context.getSystemService(Context.DEVICE_POLICY_SERVICE) 
                as android.app.admin.DevicePolicyManager
            // Check if our app is a device admin (optional feature)
            false // Typically not granted for user apps
        } catch (e: Exception) {
            false
        }
    }
    
    private fun checkBatteryOptimizationExemption(): Boolean {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            val powerManager = context.getSystemService(Context.POWER_SERVICE) as android.os.PowerManager
            powerManager.isIgnoringBatteryOptimizations(context.packageName)
        } else {
            true // Not applicable on older versions
        }
    }
    
    // Permission request methods
    
    private fun requestNotificationPermission(activity: Activity, requestCode: Int) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            ActivityCompat.requestPermissions(
                activity,
                arrayOf(android.Manifest.permission.POST_NOTIFICATIONS),
                requestCode
            )
        } else {
            // Open notification settings for older versions
            val intent = Intent(Settings.ACTION_APP_NOTIFICATION_SETTINGS).apply {
                putExtra(Settings.EXTRA_APP_PACKAGE, context.packageName)
            }
            activity.startActivity(intent)
        }
    }
    
    private fun requestDNDPermission(activity: Activity) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            val intent = Intent(Settings.ACTION_NOTIFICATION_POLICY_ACCESS_SETTINGS)
            activity.startActivity(intent)
        }
    }
    
    private fun requestAccessibilityPermission(activity: Activity) {
        val intent = Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS)
        activity.startActivity(intent)
    }
    
    private fun requestUsageStatsPermission(activity: Activity) {
        val intent = Intent(Settings.ACTION_USAGE_ACCESS_SETTINGS)
        activity.startActivity(intent)
    }
    
    private fun requestOverlayPermission(activity: Activity) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            val intent = Intent(
                Settings.ACTION_MANAGE_OVERLAY_PERMISSION,
                Uri.parse("package:${context.packageName}")
            )
            activity.startActivity(intent)
        }
    }
    
    private fun requestBatteryOptimizationExemption(activity: Activity) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            val intent = Intent(Settings.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS).apply {
                data = Uri.parse("package:${context.packageName}")
            }
            activity.startActivity(intent)
        }
    }
    
    private fun requestDeviceAdminPermission(activity: Activity) {
        // Device admin is typically not requested for user apps
        Log.w(TAG, "Device admin permission not typically granted to user apps")
    }
    
    // Fallback capability checks
    
    private fun canUseFallbackNotifications(): Boolean {
        // Can use system notifications or other methods
        return true
    }
    
    private fun canUseFallbackDND(): Boolean {
        // Can suggest manual DND or use other focus methods
        return true
    }
    
    private fun canUseFallbackFocus(): Boolean {
        // Can use notification-based focus reminders
        return checkNotificationPermission()
    }
    
    private fun canUseFallbackUsageAnalysis(): Boolean {
        // Can use basic app usage tracking or user input
        return true
    }
    
    private fun canUseFallbackBackground(): Boolean {
        // Can use foreground service or user-initiated actions
        return true
    }
    
    /**
     * Get user-friendly permission explanation
     */
    fun getPermissionExplanation(permissionType: PermissionType): PermissionExplanation {
        return when (permissionType) {
            PermissionType.NOTIFICATIONS -> PermissionExplanation(
                title = "Enable Notifications",
                reason = "LifeTwin needs notification access to send you helpful suggestions and reminders for better digital wellbeing.",
                benefits = listOf(
                    "Receive timely break suggestions",
                    "Get focus mode reminders",
                    "See usage insights and tips"
                ),
                consequences = "Without notifications, you won't receive any automation suggestions.",
                isRequired = true
            )
            
            PermissionType.DND_ACCESS -> PermissionExplanation(
                title = "Do Not Disturb Access",
                reason = "This allows LifeTwin to automatically manage your Do Not Disturb settings for better focus and sleep.",
                benefits = listOf(
                    "Automatic DND during focus time",
                    "Sleep-friendly quiet hours",
                    "Reduced interruptions"
                ),
                consequences = "You'll need to manually enable Do Not Disturb when suggested.",
                isRequired = false
            )
            
            PermissionType.ACCESSIBILITY -> PermissionExplanation(
                title = "Accessibility Service",
                reason = "This enables gentle monitoring and focus assistance features.",
                benefits = listOf(
                    "Smart app usage insights",
                    "Focus mode with app blocking",
                    "Gentle usage reminders"
                ),
                consequences = "Focus mode will use notifications instead of app blocking.",
                isRequired = false
            )
            
            PermissionType.USAGE_STATS -> PermissionExplanation(
                title = "Usage Access",
                reason = "LifeTwin needs to understand your app usage patterns to provide personalized suggestions.",
                benefits = listOf(
                    "Personalized usage insights",
                    "Smart break suggestions",
                    "Tailored focus recommendations"
                ),
                consequences = "Suggestions will be generic rather than personalized.",
                isRequired = true
            )
            
            PermissionType.OVERLAY -> PermissionExplanation(
                title = "Display Over Other Apps",
                reason = "This allows subtle overlay reminders that don't interrupt your workflow.",
                benefits = listOf(
                    "Gentle focus reminders",
                    "Non-intrusive break suggestions",
                    "Contextual usage insights"
                ),
                consequences = "All reminders will use standard notifications.",
                isRequired = false
            )
            
            PermissionType.BATTERY_OPTIMIZATION -> PermissionExplanation(
                title = "Battery Optimization Exemption",
                reason = "This ensures LifeTwin can provide consistent automation in the background.",
                benefits = listOf(
                    "Reliable background operation",
                    "Timely suggestions",
                    "Consistent automation"
                ),
                consequences = "Suggestions may be delayed or inconsistent.",
                isRequired = false
            )
            
            PermissionType.DEVICE_ADMIN -> PermissionExplanation(
                title = "Device Administrator",
                reason = "Advanced device management features (rarely needed).",
                benefits = listOf("Advanced security features"),
                consequences = "Standard features will work normally.",
                isRequired = false
            )
        }
    }
}

// Data classes for permission management

enum class PermissionType {
    NOTIFICATIONS,
    DND_ACCESS,
    ACCESSIBILITY,
    USAGE_STATS,
    OVERLAY,
    DEVICE_ADMIN,
    BATTERY_OPTIMIZATION
}

data class PermissionState(
    val notifications: Boolean = false,
    val dndAccess: Boolean = false,
    val accessibility: Boolean = false,
    val usageStats: Boolean = false,
    val overlay: Boolean = false,
    val deviceAdmin: Boolean = false,
    val batteryOptimization: Boolean = false
)

data class PermissionInfo(
    val type: PermissionType,
    val granted: Boolean,
    val required: Boolean,
    val title: String,
    val description: String,
    val impact: String,
    val fallbackAvailable: Boolean
)

data class PermissionExplanation(
    val title: String,
    val reason: String,
    val benefits: List<String>,
    val consequences: String,
    val isRequired: Boolean
)

data class FallbackCapabilities(
    val canShowNotifications: Boolean = false,
    val canControlDND: Boolean = false,
    val canBlockApps: Boolean = false,
    val canAnalyzeUsage: Boolean = false,
    val canShowOverlays: Boolean = false,
    val canRunInBackground: Boolean = false,
    
    // Fallback methods available
    val fallbackNotifications: Boolean = false,
    val fallbackDND: Boolean = false,
    val fallbackFocus: Boolean = false,
    val fallbackUsageAnalysis: Boolean = false,
    val fallbackBackground: Boolean = false
)