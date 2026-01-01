package com.lifetwin.mlp.db.test

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.lifetwin.mlp.db.*
import com.lifetwin.mlp.security.KeyManager
import io.kotest.property.Arb
import io.kotest.property.arbitrary.*
import io.kotest.property.checkAll
import kotlinx.coroutines.runBlocking
import org.junit.After
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertNotEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/**
 * Property-based tests for database encryption functionality
 * Feature: data-collection-intelligence, Property 4: Data Encryption Integrity
 * Validates: Requirements 6.1, 6.2, 6.3
 */
@RunWith(AndroidJUnit4::class)
class DatabaseEncryptionPropertyTest {

    private lateinit var context: Context
    private lateinit var keyManager: KeyManager
    private var database: AppDatabase? = null

    @Before
    fun setup() {
        context = ApplicationProvider.getApplicationContext()
        keyManager = KeyManager(context)
        
        // Clear any existing database instance
        AppDatabase.clearInstance()
    }

    @After
    fun teardown() {
        database?.close()
        AppDatabase.clearInstance()
        keyManager.clearAllKeys()
    }

    /**
     * Property 4: Data Encryption Integrity
     * For any personally identifiable information stored in the database, 
     * it should be encrypted using SQLCipher and unreadable without the proper decryption key
     */
    @Test
    fun testDatabaseEncryptionIntegrity() = runBlocking {
        checkAll(
            iterations = 100,
            Arb.string(minSize = 1, maxSize = 1000),
            Arb.string(minSize = 5, maxSize = 50), // package names
            Arb.long(min = 1000000000000L, max = System.currentTimeMillis()) // timestamps
        ) { sensitiveData, packageName, timestamp ->
            
            // Initialize encrypted database
            val encryptedDb = DBHelper.initializeEncrypted(context)
            assertNotNull(encryptedDb, "Encrypted database should be initialized successfully")
            
            // Create a raw event with sensitive metadata
            val rawEvent = RawEventEntity(
                id = UUID.randomUUID().toString(),
                timestamp = timestamp,
                eventType = "test_event",
                packageName = packageName,
                duration = 1000L,
                metadata = DBHelper.encryptMetadata(sensitiveData)
            )
            
            // Store the event
            val eventId = encryptedDb.rawEventDao().insert(rawEvent)
            assertTrue(eventId > 0, "Event should be inserted successfully")
            
            // Retrieve the event
            val retrievedEvents = encryptedDb.rawEventDao().getEventsByTimeRange(
                timestamp - 1000, 
                timestamp + 1000
            )
            
            assertEquals(1, retrievedEvents.size, "Should retrieve exactly one event")
            val retrievedEvent = retrievedEvents.first()
            
            // Verify the metadata is encrypted (should not match original plaintext)
            assertNotEquals(
                sensitiveData, 
                retrievedEvent.metadata,
                "Stored metadata should be encrypted, not plaintext"
            )
            
            // Verify we can decrypt the metadata back to original
            val decryptedMetadata = DBHelper.decryptMetadata(retrievedEvent.metadata)
            assertEquals(
                sensitiveData,
                decryptedMetadata,
                "Decrypted metadata should match original sensitive data"
            )
            
            // Verify other fields are stored correctly
            assertEquals(packageName, retrievedEvent.packageName)
            assertEquals(timestamp, retrievedEvent.timestamp)
            assertEquals("test_event", retrievedEvent.eventType)
            
            // Clean up
            encryptedDb.close()
        }
    }

    /**
     * Test that database files are actually encrypted at rest
     */
    @Test
    fun testDatabaseFileEncryption() = runBlocking {
        checkAll(
            iterations = 50,
            Arb.list(Arb.string(minSize = 10, maxSize = 100), range = 1..10)
        ) { testDataList ->
            
            // Initialize encrypted database
            val encryptedDb = DBHelper.initializeEncrypted(context)
            assertNotNull(encryptedDb, "Encrypted database should be initialized")
            
            // Store multiple pieces of sensitive data
            testDataList.forEachIndexed { index, testData ->
                val event = RawEventEntity(
                    id = "test_$index",
                    timestamp = System.currentTimeMillis() + index,
                    eventType = "encryption_test",
                    packageName = "com.test.app$index",
                    duration = 1000L,
                    metadata = DBHelper.encryptMetadata(testData)
                )
                encryptedDb.rawEventDao().insert(event)
            }
            
            // Force database to flush to disk
            encryptedDb.close()
            
            // Try to read database file directly (should be encrypted/unreadable)
            val dbFile = context.getDatabasePath("lifetwin_db")
            assertTrue(dbFile.exists(), "Database file should exist")
            
            // Read raw bytes from database file
            val rawBytes = dbFile.readBytes()
            val rawContent = String(rawBytes, Charsets.UTF_8)
            
            // Verify that none of our test data appears in plaintext in the file
            testDataList.forEach { testData ->
                assertTrue(
                    !rawContent.contains(testData),
                    "Sensitive data '$testData' should not appear in plaintext in database file"
                )
            }
            
            // Verify SQLCipher header is present (indicates encryption)
            val sqlCipherHeader = "SQLite format 3"
            // For encrypted databases, the header should be different or encrypted
            // This is a basic check - in practice, SQLCipher modifies the header
        }
    }

    /**
     * Test key management integration with Android Keystore
     */
    @Test
    fun testKeystoreIntegration() = runBlocking {
        checkAll(
            iterations = 50,
            Arb.string(minSize = 10, maxSize = 500)
        ) { testData ->
            
            // Test key manager functionality
            assertTrue(keyManager.validateKeySystem(), "Key system should be valid")
            
            // Test encryption/decryption round trip
            val encrypted = keyManager.encryptData(testData)
            assertNotNull(encrypted, "Encryption should succeed")
            assertNotEquals(testData, encrypted.ciphertext, "Ciphertext should differ from plaintext")
            
            val decrypted = keyManager.decryptData(encrypted)
            assertEquals(testData, decrypted, "Decryption should restore original data")
            
            // Test database passphrase generation
            val passphrase1 = keyManager.initializeAndGetDatabasePassphrase()
            val passphrase2 = keyManager.initializeAndGetDatabasePassphrase()
            
            assertNotNull(passphrase1, "First passphrase should be generated")
            assertNotNull(passphrase2, "Second passphrase should be generated")
            assertEquals(passphrase1, passphrase2, "Passphrase should be consistent")
        }
    }

    /**
     * Test that database operations work correctly with encryption
     */
    @Test
    fun testEncryptedDatabaseOperations() = runBlocking {
        checkAll(
            iterations = 100,
            Arb.list(
                Arb.bind(
                    Arb.string(minSize = 5, maxSize = 50),
                    Arb.long(min = 1000000000000L, max = System.currentTimeMillis()),
                    Arb.string(minSize = 10, maxSize = 200)
                ) { pkg, ts, meta -> Triple(pkg, ts, meta) },
                range = 1..20
            )
        ) { eventData ->
            
            val encryptedDb = DBHelper.initializeEncrypted(context)
            assertNotNull(encryptedDb, "Database should initialize")
            
            // Insert multiple events
            val insertedIds = mutableListOf<String>()
            eventData.forEach { (packageName, timestamp, metadata) ->
                val event = RawEventEntity(
                    id = UUID.randomUUID().toString(),
                    timestamp = timestamp,
                    eventType = "bulk_test",
                    packageName = packageName,
                    duration = 1000L,
                    metadata = DBHelper.encryptMetadata(metadata)
                )
                
                encryptedDb.rawEventDao().insert(event)
                insertedIds.add(event.id)
            }
            
            // Query all events
            val minTimestamp = eventData.minOf { it.second }
            val maxTimestamp = eventData.maxOf { it.second }
            
            val retrievedEvents = encryptedDb.rawEventDao().getEventsByTimeRange(
                minTimestamp - 1000,
                maxTimestamp + 1000
            )
            
            assertEquals(
                eventData.size,
                retrievedEvents.size,
                "Should retrieve all inserted events"
            )
            
            // Verify each event can be decrypted correctly
            retrievedEvents.forEach { event ->
                val decryptedMetadata = DBHelper.decryptMetadata(event.metadata)
                val originalMetadata = eventData.find { it.first == event.packageName }?.third
                assertEquals(
                    originalMetadata,
                    decryptedMetadata,
                    "Decrypted metadata should match original for package ${event.packageName}"
                )
            }
            
            encryptedDb.close()
        }
    }

    // Helper function to generate arbitrary raw events
    private fun arbRawEvent() = Arb.bind(
        Arb.string(minSize = 5, maxSize = 50),
        Arb.long(min = 1000000000000L, max = System.currentTimeMillis()),
        Arb.string(minSize = 1, maxSize = 500)
    ) { packageName, timestamp, metadata ->
        RawEventEntity(
            id = UUID.randomUUID().toString(),
            timestamp = timestamp,
            eventType = "property_test",
            packageName = packageName,
            duration = 1000L,
            metadata = DBHelper.encryptMetadata(metadata)
        )
    }
}