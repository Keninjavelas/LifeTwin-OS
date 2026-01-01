package com.lifetwin.mlp.automation

import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import com.lifetwin.mlp.db.*
import kotlinx.serialization.Serializable
import org.json.JSONObject
import java.util.*

/**
 * Comprehensive behavioral context data models for automation decision making.
 * These models capture all relevant user state, usage patterns, and environmental factors.
 */

@Serializable
data class BehavioralContext(
    val currentUsage: UsageSnapshot,
    val recentPatterns: UsagePatterns,
    val timeContext: TimeContext,
    val environmentContext: EnvironmentContext,
    val userState: UserState,
    val contextMetadata: ContextMetadata = ContextMetadata()
) {
    /**
     * Convert to JSON for logging and analysis
     */
    fun toJson(): JSONObject {
        return JSONObject().apply {
            put("current_usage", currentUsage.toJson())
            put("recent_patterns", recentPatterns.toJson())
            put("time_context", timeContext.toJson())
            put("environment_context", environmentContext.toJson())
            put("user_state", userState.toJson())
            put("metadata", contextMetadata.toJson())
        }
    }
    
    /**
     * Check if context indicates high stress/urgency
     */
    fun isHighStressContext(): Boolean {
        return userState.stressLevel > 0.7f || 
               currentUsage.appSwitches > 20 ||
               currentUsage.notificationCount > 15
    }
    
    /**
     * Check if context indicates focus time
     */
    fun isFocusContext(): Boolean {
        return timeContext.isWorkHour && 
               userState.focusLevel > 0.6f &&
               currentUsage.workUsage > currentUsage.socialUsage
    }
    
    /**
     * Check if context indicates relaxation time
     */
    fun isRelaxationContext(): Boolean {
        return !timeContext.isWorkHour &&
               (timeContext.hourOfDay >= 19 || timeContext.isWeekend) &&
               userState.stressLevel < 0.4f
    }
}

@Serializable
data class UsageSnapshot(
    val totalScreenTime: Long,
    val socialUsage: Long,
    val workUsage: Long,
    val entertainmentUsage: Long,
    val communicationUsage: Long,
    val healthUsage: Long,
    val notificationCount: Int,
    val appSwitches: Int,
    val uniqueAppsUsed: Int,
    val longestSessionDuration: Long,
    val averageSessionDuration: Long,
    val pickupCount: Int
) {
    companion object {
        fun fromEvents(events: List<UsageEventEntity>): UsageSnapshot {
            val totalTime = events.sumOf { it.totalTimeInForeground }
            val appCategories = AppCategoryMapping()
            
            val categoryUsage = events.groupBy { appCategories.getCategory(it.packageName) }
                .mapValues { (_, events) -> events.sumOf { it.totalTimeInForeground } }
            
            val sessions = events.map { it.totalTimeInForeground }.filter { it > 0 }
            
            return UsageSnapshot(
                totalScreenTime = totalTime,
                socialUsage = categoryUsage[AppCategory.SOCIAL] ?: 0L,
                workUsage = categoryUsage[AppCategory.PRODUCTIVITY] ?: 0L,
                entertainmentUsage = categoryUsage[AppCategory.ENTERTAINMENT] ?: 0L,
                communicationUsage = categoryUsage[AppCategory.COMMUNICATION] ?: 0L,
                healthUsage = categoryUsage[AppCategory.HEALTH] ?: 0L,
                notificationCount = 0, // Will be filled from notification events
                appSwitches = events.size,
                uniqueAppsUsed = events.map { it.packageName }.distinct().size,
                longestSessionDuration = sessions.maxOrNull() ?: 0L,
                averageSessionDuration = if (sessions.isNotEmpty()) sessions.average().toLong() else 0L,
                pickupCount = events.count { it.totalTimeInForeground > 0 }
            )
        }
    }
    
    fun toJson(): JSONObject {
        return JSONObject().apply {
            put("total_screen_time", totalScreenTime)
            put("social_usage", socialUsage)
            put("work_usage", workUsage)
            put("entertainment_usage", entertainmentUsage)
            put("communication_usage", communicationUsage)
            put("health_usage", healthUsage)
            put("notification_count", notificationCount)
            put("app_switches", appSwitches)
            put("unique_apps_used", uniqueAppsUsed)
            put("longest_session_duration", longestSessionDuration)
            put("average_session_duration", averageSessionDuration)
            put("pickup_count", pickupCount)
        }
    }
    
    /**
     * Calculate usage distribution percentages
     */
    fun getUsageDistribution(): UsageDistribution {
        return if (totalScreenTime > 0) {
            UsageDistribution(
                socialPercentage = socialUsage.toFloat() / totalScreenTime.toFloat(),
                workPercentage = workUsage.toFloat() / totalScreenTime.toFloat(),
                entertainmentPercentage = entertainmentUsage.toFloat() / totalScreenTime.toFloat(),
                communicationPercentage = communicationUsage.toFloat() / totalScreenTime.toFloat(),
                healthPercentage = healthUsage.toFloat() / totalScreenTime.toFloat()
            )
        } else {
            UsageDistribution()
        }
    }
}

@Serializable
data class UsagePatterns(
    val averageDailyUsage: Long,
    val peakUsageHour: Int,
    val weekendVsWeekday: Float,
    val morningUsagePattern: Float,
    val afternoonUsagePattern: Float,
    val eveningUsagePattern: Float,
    val nightUsagePattern: Float,
    val consistencyScore: Float,
    val trendDirection: TrendDirection,
    val seasonalVariation: Float
) {
    companion object {
        fun fromSummary(summary: DailySummaryEntity?): UsagePatterns {
            return UsagePatterns(
                averageDailyUsage = summary?.totalScreenTime ?: 0L,
                peakUsageHour = summary?.mostCommonHour ?: 12,
                weekendVsWeekday = 1.0f, // TODO: Calculate from historical data
                morningUsagePattern = 0.2f, // TODO: Calculate from historical data
                afternoonUsagePattern = 0.4f,
                eveningUsagePattern = 0.3f,
                nightUsagePattern = 0.1f,
                consistencyScore = 0.7f, // TODO: Calculate consistency
                trendDirection = TrendDirection.STABLE,
                seasonalVariation = 0.1f
            )
        }
        
        fun fromHistoricalData(summaries: List<DailySummaryEntity>): UsagePatterns {
            if (summaries.isEmpty()) return fromSummary(null)
            
            val avgUsage = summaries.map { it.totalScreenTime }.average().toLong()
            val peakHour = summaries.mapNotNull { it.mostCommonHour }.groupingBy { it }.eachCount().maxByOrNull { it.value }?.key ?: 12
            
            // Calculate trend direction
            val recentUsage = summaries.takeLast(7).map { it.totalScreenTime }.average()
            val olderUsage = summaries.take(7).map { it.totalScreenTime }.average()
            val trend = when {
                recentUsage > olderUsage * 1.1 -> TrendDirection.INCREASING
                recentUsage < olderUsage * 0.9 -> TrendDirection.DECREASING
                else -> TrendDirection.STABLE
            }
            
            return UsagePatterns(
                averageDailyUsage = avgUsage,
                peakUsageHour = peakHour,
                weekendVsWeekday = calculateWeekendRatio(summaries),
                morningUsagePattern = calculateTimePatternUsage(summaries, 6, 12),
                afternoonUsagePattern = calculateTimePatternUsage(summaries, 12, 18),
                eveningUsagePattern = calculateTimePatternUsage(summaries, 18, 23),
                nightUsagePattern = calculateTimePatternUsage(summaries, 23, 6),
                consistencyScore = calculateConsistencyScore(summaries),
                trendDirection = trend,
                seasonalVariation = calculateSeasonalVariation(summaries)
            )
        }
        
        private fun calculateWeekendRatio(summaries: List<DailySummaryEntity>): Float {
            // TODO: Implement weekend vs weekday calculation
            return 1.2f
        }
        
        private fun calculateTimePatternUsage(summaries: List<DailySummaryEntity>, startHour: Int, endHour: Int): Float {
            // TODO: Implement time pattern calculation
            return 0.25f
        }
        
        private fun calculateConsistencyScore(summaries: List<DailySummaryEntity>): Float {
            if (summaries.size < 2) return 1.0f
            
            val usageTimes = summaries.map { it.totalScreenTime.toDouble() }
            val mean = usageTimes.average()
            val variance = usageTimes.map { (it - mean) * (it - mean) }.average()
            val stdDev = kotlin.math.sqrt(variance)
            
            // Consistency score: lower standard deviation = higher consistency
            return (1.0f - (stdDev / mean).toFloat()).coerceIn(0f, 1f)
        }
        
        private fun calculateSeasonalVariation(summaries: List<DailySummaryEntity>): Float {
            // TODO: Implement seasonal variation calculation
            return 0.1f
        }
    }
    
    fun toJson(): JSONObject {
        return JSONObject().apply {
            put("average_daily_usage", averageDailyUsage)
            put("peak_usage_hour", peakUsageHour)
            put("weekend_vs_weekday", weekendVsWeekday)
            put("morning_usage_pattern", morningUsagePattern)
            put("afternoon_usage_pattern", afternoonUsagePattern)
            put("evening_usage_pattern", eveningUsagePattern)
            put("night_usage_pattern", nightUsagePattern)
            put("consistency_score", consistencyScore)
            put("trend_direction", trendDirection.name)
            put("seasonal_variation", seasonalVariation)
        }
    }
}

@Serializable
data class TimeContext(
    val hourOfDay: Int,
    val dayOfWeek: Int,
    val dayOfMonth: Int,
    val month: Int,
    val year: Int,
    val isWeekend: Boolean,
    val isWorkHour: Boolean,
    val isLunchHour: Boolean,
    val isEveningHour: Boolean,
    val isNightHour: Boolean,
    val timeZone: String,
    val seasonOfYear: Season
) {
    companion object {
        fun fromTimestamp(timestamp: Long): TimeContext {
            val calendar = Calendar.getInstance()
            calendar.timeInMillis = timestamp
            
            val hour = calendar.get(Calendar.HOUR_OF_DAY)
            val dayOfWeek = calendar.get(Calendar.DAY_OF_WEEK)
            val dayOfMonth = calendar.get(Calendar.DAY_OF_MONTH)
            val month = calendar.get(Calendar.MONTH)
            val year = calendar.get(Calendar.YEAR)
            
            val isWeekend = dayOfWeek == Calendar.SATURDAY || dayOfWeek == Calendar.SUNDAY
            val isWorkHour = hour in 9..17 && !isWeekend
            val isLunchHour = hour in 12..13 && !isWeekend
            val isEveningHour = hour in 18..22
            val isNightHour = hour >= 23 || hour <= 6
            
            val season = when (month) {
                Calendar.DECEMBER, Calendar.JANUARY, Calendar.FEBRUARY -> Season.WINTER
                Calendar.MARCH, Calendar.APRIL, Calendar.MAY -> Season.SPRING
                Calendar.JUNE, Calendar.JULY, Calendar.AUGUST -> Season.SUMMER
                else -> Season.FALL
            }
            
            return TimeContext(
                hourOfDay = hour,
                dayOfWeek = dayOfWeek,
                dayOfMonth = dayOfMonth,
                month = month,
                year = year,
                isWeekend = isWeekend,
                isWorkHour = isWorkHour,
                isLunchHour = isLunchHour,
                isEveningHour = isEveningHour,
                isNightHour = isNightHour,
                timeZone = calendar.timeZone.id,
                seasonOfYear = season
            )
        }
    }
    
    fun toJson(): JSONObject {
        return JSONObject().apply {
            put("hour_of_day", hourOfDay)
            put("day_of_week", dayOfWeek)
            put("day_of_month", dayOfMonth)
            put("month", month)
            put("year", year)
            put("is_weekend", isWeekend)
            put("is_work_hour", isWorkHour)
            put("is_lunch_hour", isLunchHour)
            put("is_evening_hour", isEveningHour)
            put("is_night_hour", isNightHour)
            put("time_zone", timeZone)
            put("season_of_year", seasonOfYear.name)
        }
    }
}

@Serializable
data class EnvironmentContext(
    val batteryLevel: Float,
    val isCharging: Boolean,
    val wifiConnected: Boolean,
    val mobileDataConnected: Boolean,
    val locationContext: LocationContext,
    val deviceOrientation: DeviceOrientation,
    val ambientLightLevel: LightLevel,
    val noiseLevel: NoiseLevel,
    val isInMotion: Boolean,
    val proximityToUser: ProximityLevel
) {
    companion object {
        fun getCurrent(context: Context): EnvironmentContext {
            val batteryInfo = getBatteryInfo(context)
            
            return EnvironmentContext(
                batteryLevel = batteryInfo.first,
                isCharging = batteryInfo.second,
                wifiConnected = isWifiConnected(context),
                mobileDataConnected = isMobileDataConnected(context),
                locationContext = LocationContext.UNKNOWN, // TODO: Implement location detection
                deviceOrientation = DeviceOrientation.PORTRAIT, // TODO: Implement orientation detection
                ambientLightLevel = LightLevel.MEDIUM, // TODO: Implement light sensor
                noiseLevel = NoiseLevel.MEDIUM, // TODO: Implement noise detection
                isInMotion = false, // TODO: Implement motion detection
                proximityToUser = ProximityLevel.NEAR // TODO: Implement proximity sensor
            )
        }
        
        private fun getBatteryInfo(context: Context): Pair<Float, Boolean> {
            val batteryIntent = context.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
            
            val level = batteryIntent?.getIntExtra(BatteryManager.EXTRA_LEVEL, -1) ?: -1
            val scale = batteryIntent?.getIntExtra(BatteryManager.EXTRA_SCALE, -1) ?: -1
            val status = batteryIntent?.getIntExtra(BatteryManager.EXTRA_STATUS, -1) ?: -1
            
            val batteryLevel = if (level >= 0 && scale > 0) level.toFloat() / scale.toFloat() else 0.5f
            val isCharging = status == BatteryManager.BATTERY_STATUS_CHARGING || 
                            status == BatteryManager.BATTERY_STATUS_FULL
            
            return Pair(batteryLevel, isCharging)
        }
        
        private fun isWifiConnected(context: Context): Boolean {
            // TODO: Implement WiFi connectivity check
            return true
        }
        
        private fun isMobileDataConnected(context: Context): Boolean {
            // TODO: Implement mobile data connectivity check
            return false
        }
    }
    
    fun toJson(): JSONObject {
        return JSONObject().apply {
            put("battery_level", batteryLevel)
            put("is_charging", isCharging)
            put("wifi_connected", wifiConnected)
            put("mobile_data_connected", mobileDataConnected)
            put("location_context", locationContext.name)
            put("device_orientation", deviceOrientation.name)
            put("ambient_light_level", ambientLightLevel.name)
            put("noise_level", noiseLevel.name)
            put("is_in_motion", isInMotion)
            put("proximity_to_user", proximityToUser.name)
        }
    }
}

@Serializable
data class UserState(
    val currentMood: Float,
    val energyLevel: Float,
    val focusLevel: Float,
    val stressLevel: Float,
    val motivationLevel: Float,
    val socialEngagement: Float,
    val physicalActivity: Float,
    val sleepQuality: Float,
    val cognitiveLoad: Float,
    val emotionalState: EmotionalState
) {
    companion object {
        fun fromPreferences(preferences: Map<String, Any>): UserState {
            return UserState(
                currentMood = preferences["current_mood"] as? Float ?: 0.5f,
                energyLevel = preferences["energy_level"] as? Float ?: 0.5f,
                focusLevel = preferences["focus_level"] as? Float ?: 0.5f,
                stressLevel = preferences["stress_level"] as? Float ?: 0.5f,
                motivationLevel = preferences["motivation_level"] as? Float ?: 0.5f,
                socialEngagement = preferences["social_engagement"] as? Float ?: 0.5f,
                physicalActivity = preferences["physical_activity"] as? Float ?: 0.5f,
                sleepQuality = preferences["sleep_quality"] as? Float ?: 0.5f,
                cognitiveLoad = preferences["cognitive_load"] as? Float ?: 0.5f,
                emotionalState = EmotionalState.NEUTRAL
            )
        }
        
        fun fromBehavioralData(usage: UsageSnapshot, patterns: UsagePatterns): UserState {
            // Infer user state from behavioral patterns
            val stressLevel = when {
                usage.appSwitches > 30 -> 0.8f
                usage.notificationCount > 20 -> 0.7f
                usage.socialUsage > usage.totalScreenTime * 0.6f -> 0.6f
                else -> 0.4f
            }
            
            val focusLevel = when {
                usage.workUsage > usage.totalScreenTime * 0.7f -> 0.8f
                usage.longestSessionDuration > 60 * 60 * 1000 -> 0.7f // 1 hour
                usage.appSwitches < 10 -> 0.6f
                else -> 0.4f
            }
            
            val energyLevel = when (patterns.trendDirection) {
                TrendDirection.INCREASING -> 0.7f
                TrendDirection.DECREASING -> 0.3f
                TrendDirection.STABLE -> 0.5f
            }
            
            return UserState(
                currentMood = 0.5f,
                energyLevel = energyLevel,
                focusLevel = focusLevel,
                stressLevel = stressLevel,
                motivationLevel = 0.5f,
                socialEngagement = (usage.socialUsage.toFloat() / usage.totalScreenTime.toFloat()).coerceIn(0f, 1f),
                physicalActivity = 0.5f, // TODO: Infer from sensor data
                sleepQuality = 0.5f, // TODO: Infer from usage patterns
                cognitiveLoad = stressLevel,
                emotionalState = when {
                    stressLevel > 0.7f -> EmotionalState.STRESSED
                    focusLevel > 0.7f -> EmotionalState.FOCUSED
                    energyLevel > 0.7f -> EmotionalState.ENERGETIC
                    else -> EmotionalState.NEUTRAL
                }
            )
        }
    }
    
    fun toJson(): JSONObject {
        return JSONObject().apply {
            put("current_mood", currentMood)
            put("energy_level", energyLevel)
            put("focus_level", focusLevel)
            put("stress_level", stressLevel)
            put("motivation_level", motivationLevel)
            put("social_engagement", socialEngagement)
            put("physical_activity", physicalActivity)
            put("sleep_quality", sleepQuality)
            put("cognitive_load", cognitiveLoad)
            put("emotional_state", emotionalState.name)
        }
    }
}

@Serializable
data class ContextMetadata(
    val timestamp: Long = System.currentTimeMillis(),
    val version: String = "1.0",
    val dataQuality: DataQuality = DataQuality.HIGH,
    val confidenceScore: Float = 1.0f,
    val dataSourceReliability: Map<String, Float> = emptyMap()
) {
    fun toJson(): JSONObject {
        return JSONObject().apply {
            put("timestamp", timestamp)
            put("version", version)
            put("data_quality", dataQuality.name)
            put("confidence_score", confidenceScore)
            put("data_source_reliability", JSONObject(dataSourceReliability))
        }
    }
}

// Supporting data classes and enums

@Serializable
data class UsageDistribution(
    val socialPercentage: Float = 0f,
    val workPercentage: Float = 0f,
    val entertainmentPercentage: Float = 0f,
    val communicationPercentage: Float = 0f,
    val healthPercentage: Float = 0f
)

enum class TrendDirection {
    INCREASING, DECREASING, STABLE
}

enum class Season {
    SPRING, SUMMER, FALL, WINTER
}

enum class LocationContext {
    HOME, WORK, COMMUTING, OUTDOOR, INDOOR, UNKNOWN
}

enum class DeviceOrientation {
    PORTRAIT, LANDSCAPE, FACE_UP, FACE_DOWN
}

enum class LightLevel {
    DARK, DIM, MEDIUM, BRIGHT, VERY_BRIGHT
}

enum class NoiseLevel {
    SILENT, QUIET, MEDIUM, LOUD, VERY_LOUD
}

enum class ProximityLevel {
    VERY_NEAR, NEAR, MEDIUM, FAR, VERY_FAR
}

enum class EmotionalState {
    HAPPY, SAD, STRESSED, FOCUSED, ENERGETIC, TIRED, ANXIOUS, CALM, NEUTRAL
}

enum class DataQuality {
    LOW, MEDIUM, HIGH, VERY_HIGH
}