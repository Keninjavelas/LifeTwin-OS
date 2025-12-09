package com.lifetwin.mlp.db

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase
import androidx.room.TypeConverters
import android.util.Log
import androidx.room.migration.Migration
import androidx.sqlite.db.SupportSQLiteDatabase

@Database(entities = [AppEventEntity::class, DailySummaryEntity::class, SyncQueueEntity::class], version = 2)
@TypeConverters(Converters::class)
abstract class AppDatabase : RoomDatabase() {
    abstract fun appEventDao(): AppEventDao
    abstract fun dailySummaryDao(): DailySummaryDao

    companion object {
        @Volatile
        private var INSTANCE: AppDatabase? = null

        // Non-destructive migration from version 1 -> 2: create the sync_queue table.
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
                    ).addMigrations(MIGRATION_1_2)

                    if (!passphrase.isNullOrEmpty()) {
                        try {
                            val clazz = Class.forName("net.sqlcipher.database.SupportFactory")
                            val ctor = clazz.getConstructor(ByteArray::class.java)
                            val factoryInstance = ctor.newInstance(passphrase.toByteArray())
                            if (factoryInstance is androidx.sqlite.db.SupportSQLiteOpenHelper.Factory) {
                                builder.openHelperFactory(factoryInstance)
                            }
                        } catch (e: ClassNotFoundException) {
                            // SQLCipher not on classpath; fall back to plain DB
                        } catch (e: Exception) {
                            // Any reflection/instantiation error -> fall back
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
    }
}
