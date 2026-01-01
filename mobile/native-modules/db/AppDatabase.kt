package com.lifetwin.mlp.db

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase
import androidx.room.TypeConverters
import android.util.Log
import androidx.room.migration.Migration
import androidx.sqlite.db.SupportSQLiteDatabase

@Database(
    entities = [
        AppEventEntity::class, 
        DailySummaryEntity::class, 
        SyncQueueEntity::class,
        RawEventEntity::class,
        EnhancedDailySummaryEntity::class,
        PrivacySettingsEntity::class,
        UsageEventEntity::class,
        NotificationEventEntity::class,
        ScreenSessionEntity::class,
        InteractionMetricsEntity::class,
        ActivityContextEntity::class,
        AuditLogEntity::class,
        PerformanceLogEntity::class,
        AutomationLogEntity::class
    ], 
    version = 5
)
@TypeConverters(Converters::class)
abstract class AppDatabase : RoomDatabase() {
    // Original DAOs
    abstract fun appEventDao(): AppEventDao
    abstract fun dailySummaryDao(): DailySummaryDao
    abstract fun syncQueueDao(): SyncQueueDao
    
    // New enhanced DAOs
    abstract fun rawEventDao(): RawEventDao
    abstract fun enhancedDailySummaryDao(): EnhancedDailySummaryDao
    abstract fun privacySettingsDao(): PrivacySettingsDao
    abstract fun usageEventDao(): UsageEventDao
    abstract fun notificationEventDao(): NotificationEventDao
    abstract fun screenSessionDao(): ScreenSessionDao
    abstract fun interactionMetricsDao(): InteractionMetricsDao
    abstract fun activityContextDao(): ActivityContextDao
    abstract fun auditLogDao(): AuditLogDao
    abstract fun performanceLogDao(): PerformanceLogDao
    abstract fun automationLogDao(): AutomationLogDao

    companion object {
        @Volatile
        private var INSTANCE: AppDatabase? = null

        // Migration from version 1 -> 2: create the sync_queue table.
        private val MIGRATION_1_2 = object : Migration(1, 2) {
            override fun migrate(database: SupportSQLiteDatabase) {
                database.execSQL(
                    """
                    CREATE TABLE IF NOT EXISTS `sync_queue` (
                        `id` INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
                        `payload` TEXT NOT NULL,
                        `created_at` INTEGER NOT NULL
                    )
                    """.trimIndent()
                )
            }
        }

        // Migration from version 2 -> 3: create all new enhanced data collection tables
        private val MIGRATION_2_3 = object : Migration(2, 3) {
            override fun migrate(database: SupportSQLiteDatabase) {
                // Raw events table
                database.execSQL(
                    """
                    CREATE TABLE IF NOT EXISTS `raw_events` (
                        `id` TEXT PRIMARY KEY NOT NULL,
                        `timestamp` INTEGER NOT NULL,
                        `eventType` TEXT NOT NULL,
                        `packageName` TEXT,
                        `duration` INTEGER,
                        `metadata` TEXT NOT NULL,
                        `processed` INTEGER NOT NULL DEFAULT 0,
                        `createdAt` INTEGER NOT NULL
                    )
                    """.trimIndent()
                )

                // Enhanced daily summaries table
                database.execSQL(
                    """
                    CREATE TABLE IF NOT EXISTS `enhanced_daily_summaries` (
                        `date` TEXT PRIMARY KEY NOT NULL,
                        `totalScreenTime` INTEGER NOT NULL,
                        `appUsageDistribution` TEXT NOT NULL,
                        `notificationCount` INTEGER NOT NULL,
                        `peakUsageHour` INTEGER NOT NULL,
                        `activityBreakdown` TEXT NOT NULL,
                        `interactionIntensity` REAL NOT NULL,
                        `createdAt` INTEGER NOT NULL,
                        `version` INTEGER NOT NULL DEFAULT 1
                    )
                    """.trimIndent()
                )

                // Privacy settings table
                database.execSQL(
                    """
                    CREATE TABLE IF NOT EXISTS `privacy_settings` (
                        `id` INTEGER PRIMARY KEY NOT NULL DEFAULT 1,
                        `enabledCollectors` TEXT NOT NULL,
                        `dataRetentionDays` INTEGER NOT NULL DEFAULT 7,
                        `privacyLevel` TEXT NOT NULL DEFAULT 'STANDARD',
                        `anonymizationSettings` TEXT NOT NULL,
                        `dataSharingSettings` TEXT NOT NULL,
                        `lastUpdated` INTEGER NOT NULL
                    )
                    """.trimIndent()
                )

                // Usage events table
                database.execSQL(
                    """
                    CREATE TABLE IF NOT EXISTS `usage_events` (
                        `id` TEXT PRIMARY KEY NOT NULL,
                        `packageName` TEXT NOT NULL,
                        `startTime` INTEGER NOT NULL,
                        `endTime` INTEGER NOT NULL,
                        `totalTimeInForeground` INTEGER NOT NULL,
                        `lastTimeUsed` INTEGER NOT NULL,
                        `eventType` TEXT NOT NULL,
                        `createdAt` INTEGER NOT NULL
                    )
                    """.trimIndent()
                )

                // Notification events table
                database.execSQL(
                    """
                    CREATE TABLE IF NOT EXISTS `notification_events` (
                        `id` TEXT PRIMARY KEY NOT NULL,
                        `packageName` TEXT NOT NULL,
                        `timestamp` INTEGER NOT NULL,
                        `category` TEXT,
                        `priority` INTEGER NOT NULL,
                        `hasActions` INTEGER NOT NULL,
                        `isOngoing` INTEGER NOT NULL,
                        `interactionType` TEXT,
                        `createdAt` INTEGER NOT NULL
                    )
                    """.trimIndent()
                )

                // Screen sessions table
                database.execSQL(
                    """
                    CREATE TABLE IF NOT EXISTS `screen_sessions` (
                        `sessionId` TEXT PRIMARY KEY NOT NULL,
                        `startTime` INTEGER NOT NULL,
                        `endTime` INTEGER,
                        `unlockCount` INTEGER NOT NULL DEFAULT 0,
                        `interactionIntensity` REAL NOT NULL DEFAULT 0.0,
                        `isActive` INTEGER NOT NULL DEFAULT 1,
                        `createdAt` INTEGER NOT NULL
                    )
                    """.trimIndent()
                )

                // Interaction metrics table
                database.execSQL(
                    """
                    CREATE TABLE IF NOT EXISTS `interaction_metrics` (
                        `id` TEXT PRIMARY KEY NOT NULL,
                        `timestamp` INTEGER NOT NULL,
                        `touchCount` INTEGER NOT NULL,
                        `scrollEvents` INTEGER NOT NULL,
                        `gesturePatterns` TEXT NOT NULL,
                        `interactionIntensity` REAL NOT NULL,
                        `timeWindowStart` INTEGER NOT NULL,
                        `timeWindowEnd` INTEGER NOT NULL,
                        `createdAt` INTEGER NOT NULL
                    )
                    """.trimIndent()
                )

                // Activity context table
                database.execSQL(
                    """
                    CREATE TABLE IF NOT EXISTS `activity_context` (
                        `id` TEXT PRIMARY KEY NOT NULL,
                        `activityType` TEXT NOT NULL,
                        `confidence` REAL NOT NULL,
                        `timestamp` INTEGER NOT NULL,
                        `duration` INTEGER NOT NULL,
                        `sensorData` TEXT,
                        `createdAt` INTEGER NOT NULL
                    )
                    """.trimIndent()
                )

                // Audit log table
                database.execSQL(
                    """
                    CREATE TABLE IF NOT EXISTS `audit_log` (
                        `id` TEXT PRIMARY KEY NOT NULL,
                        `timestamp` INTEGER NOT NULL,
                        `eventType` TEXT NOT NULL,
                        `details` TEXT NOT NULL,
                        `userId` TEXT,
                        `createdAt` INTEGER NOT NULL
                    )
                    """.trimIndent()
                )

                // Create indexes for better query performance
                database.execSQL("CREATE INDEX IF NOT EXISTS `index_raw_events_timestamp` ON `raw_events` (`timestamp`)")
                database.execSQL("CREATE INDEX IF NOT EXISTS `index_raw_events_processed` ON `raw_events` (`processed`)")
                database.execSQL("CREATE INDEX IF NOT EXISTS `index_usage_events_startTime` ON `usage_events` (`startTime`)")
                database.execSQL("CREATE INDEX IF NOT EXISTS `index_notification_events_timestamp` ON `notification_events` (`timestamp`)")
                database.execSQL("CREATE INDEX IF NOT EXISTS `index_screen_sessions_startTime` ON `screen_sessions` (`startTime`)")
                database.execSQL("CREATE INDEX IF NOT EXISTS `index_interaction_metrics_timestamp` ON `interaction_metrics` (`timestamp`)")
                database.execSQL("CREATE INDEX IF NOT EXISTS `index_activity_context_timestamp` ON `activity_context` (`timestamp`)")
                database.execSQL("CREATE INDEX IF NOT EXISTS `index_audit_log_timestamp` ON `audit_log` (`timestamp`)")
            }
        }

        // Migration from version 4 -> 5: add automation log table
        private val MIGRATION_4_5 = object : Migration(4, 5) {
            override fun migrate(database: SupportSQLiteDatabase) {
                // Automation log table
                database.execSQL(
                    """
                    CREATE TABLE IF NOT EXISTS `automation_log` (
                        `id` TEXT PRIMARY KEY NOT NULL,
                        `interventionId` TEXT NOT NULL,
                        `timestamp` INTEGER NOT NULL,
                        `interventionType` TEXT NOT NULL,
                        `trigger` TEXT NOT NULL,
                        `reasoning` TEXT NOT NULL,
                        `confidence` REAL NOT NULL,
                        `executed` INTEGER NOT NULL,
                        `userResponse` TEXT NOT NULL,
                        `executionTimeMs` INTEGER NOT NULL,
                        `feedbackRating` INTEGER,
                        `feedbackComments` TEXT,
                        `helpful` INTEGER,
                        `createdAt` INTEGER NOT NULL
                    )
                    """.trimIndent()
                )

                // Create indexes for better query performance
                database.execSQL("CREATE INDEX IF NOT EXISTS `index_automation_log_timestamp` ON `automation_log` (`timestamp`)")
                database.execSQL("CREATE INDEX IF NOT EXISTS `index_automation_log_interventionId` ON `automation_log` (`interventionId`)")
                database.execSQL("CREATE INDEX IF NOT EXISTS `index_automation_log_interventionType` ON `automation_log` (`interventionType`)")
                database.execSQL("CREATE INDEX IF NOT EXISTS `index_automation_log_userResponse` ON `automation_log` (`userResponse`)")
            }
        }

        fun getInstance(context: Context, passphrase: String? = null): AppDatabase {
            return INSTANCE ?: synchronized(this) {
                val instance = try {
                    // Build Room database. If a non-null passphrase is provided and
                    // SQLCipher's `SupportFactory` is available at runtime, attempt
                    // to open an encrypted database. This reflection-based approach
                    // keeps the dependency optional and non-breaking for builds
                    // that don't include SQLCipher.
                    val builder = Room.databaseBuilder(
                        context.applicationContext,
                        AppDatabase::class.java,
                        "lifetwin_db"
                    ).addMigrations(MIGRATION_1_2, MIGRATION_2_3, MIGRATION_3_4, MIGRATION_4_5)

                    if (!passphrase.isNullOrEmpty()) {
                        try {
                            val clazz = Class.forName("net.sqlcipher.database.SupportFactory")
                            val ctor = clazz.getConstructor(ByteArray::class.java)
                            val factoryInstance = ctor.newInstance(passphrase.toByteArray())
                            if (factoryInstance is androidx.sqlite.db.SupportSQLiteOpenHelper.Factory) {
                                builder.openHelperFactory(factoryInstance)
                                Log.i("AppDatabase", "SQLCipher encryption enabled")
                            }
                        } catch (e: ClassNotFoundException) {
                            Log.w("AppDatabase", "SQLCipher not on classpath; falling back to plain DB")
                        } catch (e: Exception) {
                            Log.w("AppDatabase", "Failed to initialize SQLCipher; falling back to plain DB: ${e.message}")
                        }
                    }

                    builder.build()
                } catch (e: Exception) {
                    // Fall back to an in-memory database if file-based creation fails (device storage/permission issues)
                    Log.w("AppDatabase", "Failed to open file-based DB, falling back to in-memory DB: ${e.message}")
                    Room.inMemoryDatabaseBuilder(context.applicationContext, AppDatabase::class.java).build()
                }
                INSTANCE = instance
                instance
            }
        }

        fun clearInstance() {
            INSTANCE = null
        }
    }
}
