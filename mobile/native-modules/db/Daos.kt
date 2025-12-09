package com.lifetwin.mlp.db

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.Query

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
