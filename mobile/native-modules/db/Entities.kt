package com.lifetwin.mlp.db

import androidx.room.Entity
import androidx.room.PrimaryKey
import java.util.Date
import androidx.room.ColumnInfo


@Entity(tableName = "app_events")
data class AppEventEntity(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val timestamp: Long,
    val type: String,
    val packageName: String? = null
)


@Entity(tableName = "daily_summaries")
data class DailySummaryEntity(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val deviceId: String,
    val date: Date,
    val totalScreenTime: Int = 0,
    val topApps: List<String>? = emptyList(),
    val mostCommonHour: Int = 0,
    val notificationCount: Int = 0
)


@Entity(tableName = "sync_queue")
data class SyncQueueEntity(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val payload: String,
    @ColumnInfo(name = "created_at") val createdAt: Long = System.currentTimeMillis()
)
