package com.lifetwin.mlp.db

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.Query
import androidx.room.Update
import androidx.room.Delete

@Dao
interface AppEventDao {
    @Insert
    suspend fun insert(event: AppEventEntity): Long

    @Query("SELECT * FROM app_events WHERE timestamp >= :since")
    suspend fun getEventsSince(since: Long): List<AppEventEntity>
}


@Dao
interface DailySummaryDao {
    @Insert
    suspend fun insert(summary: DailySummaryEntity): Long

    @Query("SELECT * FROM daily_summaries WHERE deviceId = :deviceId ORDER BY date DESC")
    suspend fun getSummariesForDevice(deviceId: String): List<DailySummaryEntity>
}


@Dao
interface SyncQueueDao {
    @Insert
    suspend fun insert(item: SyncQueueEntity): Long

    @Query("SELECT * FROM sync_queue ORDER BY created_at ASC")
    suspend fun listAll(): List<SyncQueueEntity>

    @Query("DELETE FROM sync_queue WHERE id = :id")
    suspend fun deleteById(id: Long)
}

// New DAOs for enhanced data collection

@Dao
interface RawEventDao {
    @Insert
    suspend fun insert(event: RawEventEntity): Long

    @Query("SELECT * FROM raw_events WHERE processed = 0 ORDER BY timestamp ASC")
    suspend fun getUnprocessedEvents(): List<RawEventEntity>

    @Query("SELECT * FROM raw_events WHERE timestamp BETWEEN :startTime AND :endTime")
    suspend fun getEventsByTimeRange(startTime: Long, endTime: Long): List<RawEventEntity>

    @Query("SELECT * FROM raw_events ORDER BY timestamp DESC")
    suspend fun getAllEvents(): List<RawEventEntity>

    @Query("UPDATE raw_events SET processed = 1 WHERE id IN (:eventIds)")
    suspend fun markEventsAsProcessed(eventIds: List<String>)

    @Query("DELETE FROM raw_events WHERE timestamp < :cutoffTime AND processed = 1")
    suspend fun deleteOldProcessedEvents(cutoffTime: Long): Int

    @Query("DELETE FROM raw_events WHERE timestamp >= :cutoffTime")
    suspend fun deleteEventsSince(cutoffTime: Long): Int

    @Query("SELECT COUNT(*) FROM raw_events WHERE processed = 0")
    suspend fun getUnprocessedEventCount(): Int
}

@Dao
interface EnhancedDailySummaryDao {
    @Insert
    suspend fun insert(summary: EnhancedDailySummaryEntity): Long

    @Query("SELECT * FROM enhanced_daily_summaries ORDER BY date DESC")
    suspend fun getAllSummaries(): List<EnhancedDailySummaryEntity>

    @Query("SELECT * FROM enhanced_daily_summaries WHERE date BETWEEN :startDate AND :endDate ORDER BY date ASC")
    suspend fun getSummariesByDateRange(startDate: String, endDate: String): List<EnhancedDailySummaryEntity>

    @Query("SELECT * FROM enhanced_daily_summaries WHERE createdAt BETWEEN :startTime AND :endTime ORDER BY date ASC")
    suspend fun getSummariesByTimeRange(startTime: Long, endTime: Long): List<EnhancedDailySummaryEntity>

    @Query("SELECT * FROM enhanced_daily_summaries WHERE date = :date")
    suspend fun getSummaryByDate(date: String): EnhancedDailySummaryEntity?

    @Update
    suspend fun update(summary: EnhancedDailySummaryEntity)

    @Query("DELETE FROM enhanced_daily_summaries WHERE date < :cutoffDate")
    suspend fun deleteOldSummaries(cutoffDate: String): Int

    @Query("DELETE FROM enhanced_daily_summaries WHERE createdAt >= :cutoffTime")
    suspend fun deleteSummariesSince(cutoffTime: Long): Int
}

@Dao
interface PrivacySettingsDao {
    @Insert
    suspend fun insert(settings: PrivacySettingsEntity): Long

    @Update
    suspend fun update(settings: PrivacySettingsEntity)

    @Insert
    suspend fun insertOrUpdate(settings: PrivacySettingsEntity) {
        val existing = getSettings()
        if (existing != null) {
            update(settings)
        } else {
            insert(settings)
        }
    }

    @Query("SELECT * FROM privacy_settings WHERE id = 1")
    suspend fun getSettings(): PrivacySettingsEntity?

    @Query("DELETE FROM privacy_settings")
    suspend fun deleteAll()
}

@Dao
interface UsageEventDao {
    @Insert
    suspend fun insert(event: UsageEventEntity): Long

    @Insert
    suspend fun insertAll(events: List<UsageEventEntity>)

    @Query("SELECT * FROM usage_events WHERE startTime BETWEEN :startTime AND :endTime ORDER BY startTime ASC")
    suspend fun getEventsByTimeRange(startTime: Long, endTime: Long): List<UsageEventEntity>

    @Query("SELECT * FROM usage_events ORDER BY startTime DESC")
    suspend fun getAllEvents(): List<UsageEventEntity>

    @Query("SELECT * FROM usage_events WHERE packageName = :packageName AND startTime BETWEEN :startTime AND :endTime")
    suspend fun getEventsForPackage(packageName: String, startTime: Long, endTime: Long): List<UsageEventEntity>

    @Query("SELECT * FROM usage_events ORDER BY startTime DESC LIMIT 1")
    suspend fun getLatestEvent(): UsageEventEntity?

    @Query("DELETE FROM usage_events WHERE startTime < :cutoffTime")
    suspend fun deleteOldEvents(cutoffTime: Long): Int

    @Query("SELECT COUNT(*) FROM usage_events WHERE startTime BETWEEN :startTime AND :endTime")
    suspend fun getEventCountByTimeRange(startTime: Long, endTime: Long): Int

    @Query("SELECT COUNT(*) FROM usage_events")
    suspend fun getEventCount(): Int
}

@Dao
interface NotificationEventDao {
    @Insert
    suspend fun insert(event: NotificationEventEntity): Long

    @Insert
    suspend fun insertAll(events: List<NotificationEventEntity>)

    @Query("SELECT * FROM notification_events WHERE timestamp BETWEEN :startTime AND :endTime ORDER BY timestamp ASC")
    suspend fun getEventsByTimeRange(startTime: Long, endTime: Long): List<NotificationEventEntity>

    @Query("SELECT * FROM notification_events ORDER BY timestamp DESC")
    suspend fun getAllEvents(): List<NotificationEventEntity>

    @Query("SELECT * FROM notification_events WHERE packageName = :packageName AND timestamp BETWEEN :startTime AND :endTime")
    suspend fun getEventsForPackage(packageName: String, startTime: Long, endTime: Long): List<NotificationEventEntity>

    @Query("SELECT * FROM notification_events ORDER BY timestamp DESC LIMIT 1")
    suspend fun getLatestEvent(): NotificationEventEntity?

    @Query("DELETE FROM notification_events WHERE timestamp < :cutoffTime")
    suspend fun deleteOldEvents(cutoffTime: Long): Int

    @Query("SELECT COUNT(*) FROM notification_events WHERE timestamp BETWEEN :startTime AND :endTime")
    suspend fun getEventCountByTimeRange(startTime: Long, endTime: Long): Int
}

@Dao
interface ScreenSessionDao {
    @Insert
    suspend fun insert(session: ScreenSessionEntity): Long

    @Update
    suspend fun update(session: ScreenSessionEntity)

    @Query("SELECT * FROM screen_sessions WHERE isActive = 1 ORDER BY startTime DESC LIMIT 1")
    suspend fun getActiveSession(): ScreenSessionEntity?

    @Query("SELECT * FROM screen_sessions WHERE startTime BETWEEN :startTime AND :endTime ORDER BY startTime ASC")
    suspend fun getSessionsByTimeRange(startTime: Long, endTime: Long): List<ScreenSessionEntity>

    @Query("SELECT * FROM screen_sessions ORDER BY startTime DESC")
    suspend fun getAllSessions(): List<ScreenSessionEntity>

    @Query("SELECT * FROM screen_sessions ORDER BY startTime DESC LIMIT 1")
    suspend fun getLatestSession(): ScreenSessionEntity?

    @Query("DELETE FROM screen_sessions WHERE startTime < :cutoffTime")
    suspend fun deleteOldSessions(cutoffTime: Long): Int

    @Query("SELECT SUM(endTime - startTime) FROM screen_sessions WHERE startTime BETWEEN :startTime AND :endTime AND endTime IS NOT NULL")
    suspend fun getTotalScreenTimeByRange(startTime: Long, endTime: Long): Long?
}

@Dao
interface InteractionMetricsDao {
    @Insert
    suspend fun insert(metrics: InteractionMetricsEntity): Long

    @Insert
    suspend fun insertAll(metrics: List<InteractionMetricsEntity>)

    @Query("SELECT * FROM interaction_metrics WHERE timestamp BETWEEN :startTime AND :endTime ORDER BY timestamp ASC")
    suspend fun getMetricsByTimeRange(startTime: Long, endTime: Long): List<InteractionMetricsEntity>

    @Query("SELECT * FROM interaction_metrics ORDER BY timestamp DESC")
    suspend fun getAllMetrics(): List<InteractionMetricsEntity>

    @Query("SELECT * FROM interaction_metrics ORDER BY timestamp DESC LIMIT 1")
    suspend fun getLatestMetrics(): InteractionMetricsEntity?

    @Query("DELETE FROM interaction_metrics WHERE timestamp < :cutoffTime")
    suspend fun deleteOldMetrics(cutoffTime: Long): Int

    @Query("SELECT AVG(interactionIntensity) FROM interaction_metrics WHERE timestamp BETWEEN :startTime AND :endTime")
    suspend fun getAverageIntensityByRange(startTime: Long, endTime: Long): Float?
}

@Dao
interface ActivityContextDao {
    @Insert
    suspend fun insert(context: ActivityContextEntity): Long

    @Insert
    suspend fun insertAll(contexts: List<ActivityContextEntity>)

    @Query("SELECT * FROM activity_context WHERE timestamp BETWEEN :startTime AND :endTime ORDER BY timestamp ASC")
    suspend fun getContextsByTimeRange(startTime: Long, endTime: Long): List<ActivityContextEntity>

    @Query("SELECT * FROM activity_context ORDER BY timestamp DESC")
    suspend fun getAllContexts(): List<ActivityContextEntity>

    @Query("SELECT * FROM activity_context ORDER BY timestamp DESC LIMIT 1")
    suspend fun getLatestContext(): ActivityContextEntity?

    @Query("DELETE FROM activity_context WHERE timestamp < :cutoffTime")
    suspend fun deleteOldContexts(cutoffTime: Long): Int
}

@Dao
interface AuditLogDao {
    @Insert
    suspend fun insert(log: AuditLogEntity): Long

    @Query("SELECT * FROM audit_log ORDER BY timestamp DESC LIMIT :limit")
    suspend fun getRecentLogs(limit: Int = 100): List<AuditLogEntity>

    @Query("SELECT * FROM audit_log WHERE eventType = :eventType ORDER BY timestamp DESC")
    suspend fun getLogsByType(eventType: String): List<AuditLogEntity>

    @Query("SELECT * FROM audit_log WHERE eventType = :eventType AND timestamp BETWEEN :startTime AND :endTime ORDER BY timestamp DESC")
    suspend fun getLogsByTypeAndTimeRange(eventType: String, startTime: Long, endTime: Long): List<AuditLogEntity>

    @Query("SELECT * FROM audit_log WHERE timestamp BETWEEN :startTime AND :endTime ORDER BY timestamp DESC")
    suspend fun getLogsByTimeRange(startTime: Long, endTime: Long): List<AuditLogEntity>

    @Query("SELECT * FROM audit_log ORDER BY timestamp DESC")
    suspend fun getAllLogs(): List<AuditLogEntity>

    @Query("DELETE FROM audit_log WHERE timestamp < :cutoffTime")
    suspend fun deleteOldLogs(cutoffTime: Long): Int
}

@Dao
interface PerformanceLogDao {
    @Insert
    suspend fun insert(log: PerformanceLogEntity): Long

    @Query("SELECT * FROM performance_log ORDER BY timestamp DESC LIMIT :limit")
    suspend fun getRecentLogs(limit: Int = 100): List<PerformanceLogEntity>

    @Query("SELECT * FROM performance_log WHERE operationType = :operationType ORDER BY timestamp DESC")
    suspend fun getLogsByType(operationType: String): List<PerformanceLogEntity>

    @Query("SELECT * FROM performance_log WHERE timestamp BETWEEN :startTime AND :endTime ORDER BY timestamp DESC")
    suspend fun getLogsByTimeRange(startTime: Long, endTime: Long): List<PerformanceLogEntity>

    @Query("SELECT * FROM performance_log WHERE operationType = :operationType AND timestamp BETWEEN :startTime AND :endTime ORDER BY timestamp DESC")
    suspend fun getLogsByTypeAndTimeRange(operationType: String, startTime: Long, endTime: Long): List<PerformanceLogEntity>

    @Query("SELECT AVG(durationMs) FROM performance_log WHERE operationType = :operationType AND durationMs IS NOT NULL")
    suspend fun getAverageDurationByType(operationType: String): Double?

    @Query("SELECT AVG(batteryLevel) FROM performance_log WHERE batteryLevel IS NOT NULL AND timestamp BETWEEN :startTime AND :endTime")
    suspend fun getAverageBatteryLevel(startTime: Long, endTime: Long): Double?

    @Query("SELECT AVG(memoryUsageMB) FROM performance_log WHERE memoryUsageMB IS NOT NULL AND timestamp BETWEEN :startTime AND :endTime")
    suspend fun getAverageMemoryUsage(startTime: Long, endTime: Long): Double?

    @Query("DELETE FROM performance_log WHERE timestamp < :cutoffTime")
    suspend fun deleteOldLogs(cutoffTime: Long): Int
}

@Dao
interface AutomationLogDao {
    @Insert
    suspend fun insert(log: AutomationLogEntity): Long

    @Insert
    suspend fun insertAll(logs: List<AutomationLogEntity>)

    @Query("SELECT * FROM automation_log ORDER BY timestamp DESC")
    suspend fun getAllLogs(): List<AutomationLogEntity>

    @Query("SELECT * FROM automation_log WHERE timestamp BETWEEN :startTime AND :endTime ORDER BY timestamp DESC")
    suspend fun getLogsByTimeRange(startTime: Long, endTime: Long): List<AutomationLogEntity>

    @Query("SELECT * FROM automation_log WHERE interventionType = :type ORDER BY timestamp DESC")
    suspend fun getLogsByType(type: String): List<AutomationLogEntity>

    @Query("SELECT * FROM automation_log WHERE userResponse = :response ORDER BY timestamp DESC")
    suspend fun getLogsByResponse(response: String): List<AutomationLogEntity>

    @Query("SELECT * FROM automation_log WHERE interventionId = :interventionId")
    suspend fun getLogByInterventionId(interventionId: String): AutomationLogEntity?

    @Query("UPDATE automation_log SET feedbackRating = :rating, feedbackComments = :comments, helpful = :helpful WHERE interventionId = :interventionId")
    suspend fun updateFeedback(interventionId: String, rating: Int, comments: String?, helpful: Boolean)

    @Query("SELECT COUNT(*) FROM automation_log WHERE timestamp BETWEEN :startTime AND :endTime")
    suspend fun getLogCountByTimeRange(startTime: Long, endTime: Long): Int

    @Query("SELECT COUNT(*) FROM automation_log WHERE userResponse = :response AND timestamp BETWEEN :startTime AND :endTime")
    suspend fun getLogCountByResponseAndTimeRange(response: String, startTime: Long, endTime: Long): Int

    @Query("SELECT AVG(confidence) FROM automation_log WHERE timestamp BETWEEN :startTime AND :endTime")
    suspend fun getAverageConfidenceByTimeRange(startTime: Long, endTime: Long): Float?

    @Query("SELECT AVG(feedbackRating) FROM automation_log WHERE feedbackRating IS NOT NULL AND timestamp BETWEEN :startTime AND :endTime")
    suspend fun getAverageRatingByTimeRange(startTime: Long, endTime: Long): Float?

    @Query("DELETE FROM automation_log WHERE timestamp < :cutoffTime")
    suspend fun deleteOldLogs(cutoffTime: Long): Int

    @Query("SELECT * FROM automation_log ORDER BY timestamp DESC LIMIT :limit")
    suspend fun getRecentLogs(limit: Int = 50): List<AutomationLogEntity>
}
