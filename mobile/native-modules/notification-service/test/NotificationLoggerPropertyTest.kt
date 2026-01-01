package com.lifetwin.mlp.notifications.test

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.lifetwin.mlp.db.*
import com.lifetwin.mlp.notifications.NotificationLogger
import io.kotest.property.Arb
import io.kotest.property.arbitrary.*
import io.kotest.property.checkAll
import kotlinx.coroutines.runBlocking
import org.junit.After
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import kotlin.test.*

/**
 * Property-based tests for NotificationLogger
 * Feature: data-collection-intelligence, Property 1: Event Recording Consistency (Notifications)
 * Feature: data-collection-intelligence, Property 8: Notification Filtering Accuracy
 * Validates: Requirements 2.1, 2.2, 2.3, 2.4
 */
@RunWith(AndroidJUnit4::class)
class NotificationLoggerPropertyTest {

    private lateinit var context: Context
    private lateinit var logger: NotificationLogger
    private var database: AppDatabase? = null

    @Before
    fun setup() = runBlocking {
        context = ApplicationProvider.getApplicationContext()
        logger = NotificationLogger()
        
        // Initialize encrypted database
        database = DBHelper.initializeEncrypted(context)
        assertNotNull(database, "Database should be initialized")
    }

    @After
    fun teardown() {
        database?.close()
        AppDatabase.clearInstance()
    }

    /**
     * Property 1: Event Recording Consistency (Notifications)
     * For any notification event, when it occurs, the notification logger should record it 
     * with accurate timestamp and metadata
     */
    @Test
    fun testNotificationEventRecordingConsistency() = runBlocking {
        checkAll(
            iterations = 100,
            arbNotificationData()
        ) { notificationData ->
            
            // Log the notification
            logger.logNotificationPosted(notificationData)
            
            // Verify it was recorded correctly
            val db = database!!
            val storedEvents = db.notificationEventDao().getEventsByTimeRange(
                notificationData.timestamp - 1000,
                notificationData.timestamp + 1000
            )
            
            val matchingEvent = storedEvents.find { it.id == notificationData.id }
            assertNotNull(matchingEvent, "Notification ${notificationData.id} should be stored")
            
            // Verify all fields are recorded accurately
            assertEquals(notificationData.packageName, matchingEvent.packageName)
            assertEquals(notificationData.timestamp, matchingEvent.timestamp)
            assertEquals(notificationData.category, matchingEvent.category)
            assertEquals(notificationData.priority, matchingEvent.priority)
            assertEquals(notificationData.hasActions, matchingEvent.hasActions)
            assertEquals(notificationData.isOngoing, matchingEvent.isOngoing)
            assertEquals("posted", matchingEvent.interactionType)
        }
    }

    /**
     * Property 8: Notification Filtering Accuracy
     * For any notification event, only user-relevant app notifications should be stored, 
     * while system notifications should be filtered out
     */
    @Test
    fun testNotificationFilteringAccuracy() = runBlocking {
        checkAll(
            iterations = 100,
            arbMixedNotificationData()
        ) { (userNotifications, systemNotifications) ->
            
            val db = database!!
            
            // Log all notifications (both user and system)
            val allNotifications = userNotifications + systemNotifications
            allNotifications.forEach { notification ->
                logger.logNotificationPosted(notification)
            }
            
            // Query all stored notifications
            val minTime = allNotifications.minOf { it.timestamp }
            val maxTime = allNotifications.maxOf { it.timestamp }
            val storedEvents = db.notificationEventDao().getEventsByTimeRange(
                minTime - 1000,
                maxTime + 1000
            )
            
            // Verify all user notifications are stored
            userNotifications.forEach { userNotification ->
                assertTrue(
                    storedEvents.any { it.id == userNotification.id },
                    "User notification ${userNotification.id} from ${userNotification.packageName} should be stored"
                )
            }
            
            // Verify system notifications are filtered out
            // Note: This test assumes the shouldLogNotification method filters system packages
            val systemPackages = setOf(
                "android",
                "com.android.systemui",
                "com.android.settings",
                "com.android.vending",
                "com.google.android.gms",
                "com.google.android.gsf"
            )
            
            systemNotifications.forEach { systemNotification ->
                if (systemPackages.contains(systemNotification.packageName)) {
                    assertFalse(
                        storedEvents.any { it.id == systemNotification.id },
                        "System notification ${systemNotification.id} from ${systemNotification.packageName} should be filtered out"
                    )
                }
            }
        }
    }

    /**
     * Test notification interaction logging
     */
    @Test
    fun testNotificationInteractionLogging() = runBlocking {
        checkAll(
            iterations = 100,
            arbNotificationWithInteraction()
        ) { (notificationData, interaction) ->
            
            val db = database!!
            
            // First log the notification
            logger.logNotificationPosted(notificationData)
            
            // Then log the interaction
            val updatedInteraction = interaction.copy(notificationId = notificationData.id)
            logger.logNotificationInteraction(updatedInteraction)
            
            // Verify the interaction was recorded
            val storedEvents = db.notificationEventDao().getEventsByTimeRange(
                notificationData.timestamp - 1000,
                interaction.timestamp + 1000
            )
            
            // Should find an event with the interaction type
            val interactionEvent = storedEvents.find { 
                it.id == notificationData.id && 
                it.interactionType == interaction.interactionType.name.lowercase()
            }
            
            assertNotNull(
                interactionEvent,
                "Notification interaction should be recorded"
            )
        }
    }

    /**
     * Test notification priority filtering
     */
    @Test
    fun testNotificationPriorityFiltering() = runBlocking {
        checkAll(
            iterations = 50,
            arbNotificationDataWithPriority()
        ) { notifications ->
            
            val db = database!!
            
            // Log all notifications
            notifications.forEach { notification ->
                logger.logNotificationPosted(notification)
            }
            
            // Query stored notifications
            val minTime = notifications.minOf { it.timestamp }
            val maxTime = notifications.maxOf { it.timestamp }
            val storedEvents = db.notificationEventDao().getEventsByTimeRange(
                minTime - 1000,
                maxTime + 1000
            )
            
            // Verify priority filtering (assuming very low priority notifications are filtered)
            notifications.forEach { notification ->
                val shouldBeStored = notification.priority >= -2 // PRIORITY_LOW
                val isStored = storedEvents.any { it.id == notification.id }
                
                if (shouldBeStored) {
                    assertTrue(
                        isStored,
                        "Notification with priority ${notification.priority} should be stored"
                    )
                } else {
                    assertFalse(
                        isStored,
                        "Notification with very low priority ${notification.priority} should be filtered out"
                    )
                }
            }
        }
    }

    /**
     * Test collector interface compliance
     */
    @Test
    fun testCollectorInterfaceCompliance() = runBlocking {
        // Test collector type
        assertEquals(CollectorType.NOTIFICATIONS, logger.getCollectorType())
        
        // Test collection state management
        logger.startCollection()
        assertTrue(logger.isCollectionActive(), "Collection should be active after start")
        
        logger.stopCollection()
        assertFalse(logger.isCollectionActive(), "Collection should be inactive after stop")
        
        // Test data count (should be non-negative)
        val dataCount = logger.getCollectedDataCount()
        assertTrue(dataCount >= 0, "Data count should be non-negative")
    }

    /**
     * Test notification metadata encryption
     */
    @Test
    fun testNotificationMetadataEncryption() = runBlocking {
        checkAll(
            iterations = 50,
            arbNotificationData()
        ) { notificationData ->
            
            val db = database!!
            
            // Log the notification
            logger.logNotificationPosted(notificationData)
            
            // Check that raw events contain encrypted metadata
            val rawEvents = db.rawEventDao().getEventsByTimeRange(
                notificationData.timestamp - 1000,
                notificationData.timestamp + 1000
            )
            
            val notificationRawEvent = rawEvents.find { 
                it.eventType == "notification" && it.packageName == notificationData.packageName 
            }
            
            assertNotNull(notificationRawEvent, "Raw event should be created for notification")
            
            // Verify metadata is encrypted (should not contain plaintext values)
            val metadata = notificationRawEvent.metadata
            assertFalse(
                metadata.contains(notificationData.category ?: ""),
                "Metadata should not contain plaintext category"
            )
            assertFalse(
                metadata.contains(notificationData.priority.toString()),
                "Metadata should not contain plaintext priority"
            )
            
            // Verify we can decrypt the metadata
            val decryptedMetadata = DBHelper.decryptMetadata(metadata)
            assertTrue(
                decryptedMetadata.contains("category") || decryptedMetadata.contains("priority"),
                "Decrypted metadata should contain notification details"
            )
        }
    }

    // Helper functions for generating test data

    private fun arbNotificationData() = Arb.bind(
        Arb.string(minSize = 5, maxSize = 50), // packageName
        Arb.long(min = 1000000000000L, max = System.currentTimeMillis()), // timestamp
        Arb.string(minSize = 1, maxSize = 20).orNull(), // category
        Arb.int(min = -2, max = 2), // priority
        Arb.boolean(), // hasActions
        Arb.boolean() // isOngoing
    ) { packageName, timestamp, category, priority, hasActions, isOngoing ->
        NotificationData(
            packageName = packageName,
            timestamp = timestamp,
            category = category,
            priority = priority,
            hasActions = hasActions,
            isOngoing = isOngoing
        )
    }

    private fun arbMixedNotificationData() = Arb.bind(
        Arb.list(arbUserNotificationData(), range = 1..10),
        Arb.list(arbSystemNotificationData(), range = 1..5)
    ) { userNotifications, systemNotifications ->
        Pair(userNotifications, systemNotifications)
    }

    private fun arbUserNotificationData() = Arb.bind(
        Arb.element(listOf(
            "com.whatsapp",
            "com.facebook.katana",
            "com.twitter.android",
            "com.instagram.android",
            "com.spotify.music",
            "com.netflix.mediaclient"
        )),
        Arb.long(min = 1000000000000L, max = System.currentTimeMillis()),
        Arb.string(minSize = 1, maxSize = 20).orNull(),
        Arb.int(min = -1, max = 2),
        Arb.boolean(),
        Arb.boolean()
    ) { packageName, timestamp, category, priority, hasActions, isOngoing ->
        NotificationData(
            packageName = packageName,
            timestamp = timestamp,
            category = category,
            priority = priority,
            hasActions = hasActions,
            isOngoing = isOngoing
        )
    }

    private fun arbSystemNotificationData() = Arb.bind(
        Arb.element(listOf(
            "android",
            "com.android.systemui",
            "com.android.settings",
            "com.android.vending",
            "com.google.android.gms",
            "com.google.android.gsf"
        )),
        Arb.long(min = 1000000000000L, max = System.currentTimeMillis()),
        Arb.string(minSize = 1, maxSize = 20).orNull(),
        Arb.int(min = -2, max = 1),
        Arb.boolean(),
        Arb.boolean()
    ) { packageName, timestamp, category, priority, hasActions, isOngoing ->
        NotificationData(
            packageName = packageName,
            timestamp = timestamp,
            category = category,
            priority = priority,
            hasActions = hasActions,
            isOngoing = isOngoing
        )
    }

    private fun arbNotificationWithInteraction() = Arb.bind(
        arbNotificationData(),
        Arb.enum<NotificationInteractionType>(),
        Arb.long(min = 1000L, max = 300000L) // delay after notification
    ) { notificationData, interactionType, delay ->
        val interaction = NotificationInteraction(
            notificationId = notificationData.id,
            interactionType = interactionType,
            timestamp = notificationData.timestamp + delay
        )
        Pair(notificationData, interaction)
    }

    private fun arbNotificationDataWithPriority() = Arb.list(
        Arb.bind(
            arbNotificationData(),
            Arb.int(min = -3, max = 2) // Full priority range
        ) { notification, priority ->
            notification.copy(priority = priority)
        },
        range = 5..15
    )
}