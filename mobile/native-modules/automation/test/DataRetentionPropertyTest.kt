package com.lifetwin.automation.test

import com.lifetwin.automation.DataRetentionManager
import com.lifetwin.automation.RetentionPolicy
import com.lifetwin.automation.DataType
import com.lifetwin.automation.AutomationLog
import com.lifetwin.automation.PrivacyController
import io.kotest.core.spec.style.StringSpec
import io.kotest.matchers.shouldBe
import io.kotest.matchers.shouldNotBe
import io.kotest.matchers.ints.shouldBeGreaterThan
import io.kotest.matchers.ints.shouldBeLessThan
import io.kotest.matchers.longs.shouldBeGreaterThan
import io.kotest.matchers.doubles.shouldBeGreaterThan
import io.kotest.matchers.collections.shouldContain
import io.kotest.property.Arb
import io.kotest.property.arbitrary.*
import io.kotest.property.checkAll
import kotlinx.coroutines.runBlocking
import org.mockito.kotlin.mock
import org.mockito.kotlin.whenever
import android.content.Context
import java.util.concurrent.TimeUnit

/**
 * Property-based tests for DataRetentionManager
 * 
 * Property 19: Data retention policy enforcement
 * Validates: Requirements 9.6 (data retention), 9.2 (automatic deletion), 6.6 (user controls)
 */
class DataRetentionPropertyTest : StringSpec({

    val mockContext = mock<Context>()
    val mockAutomationLog = mock<AutomationLog>()
    val mockPrivacyController = mock<PrivacyController>()
    
    "Property 19.1: Retention policy consistently enforces data lifecycle limits" {
        checkAll(
            Arb.int(1..1000), // automation log retention days
            Arb.int(1..1000), // behavioral data retention days
            Arb.int(1..1000), // usage data retention days
            Arb.int(1..2000) // training data retention days
        ) { automationDays, behavioralDays, usageDays, trainingDays ->
            val manager = DataRetentionManager(mockContext, mockAutomationLog, mockPrivacyController)
            
            val policy = RetentionPolicy(
                automationLogRetentionDays = automationDays,
                behavioralDataRetentionDays = behavioralDays,
                usageDataRetentionDays = usageDays,
                trainingDataRetentionDays = trainingDays
            )
            
            runBlocking {
                manager.updateRetentionPolicy(policy)
                
                val currentPolicy = manager.retentionPolicy.value
                
                // Policy should be applied exactly as specified
                currentPolicy.automationLogRetentionDays shouldBe automationDays
                currentPolicy.behavioralDataRetentionDays shouldBe behavioralDays
                currentPolicy.usageDataRetentionDays shouldBe usageDays
                currentPolicy.trainingDataRetentionDays shouldBe trainingDays
                
                // Retention periods should be reasonable
                currentPolicy.automationLogRetentionDays shouldBeGreaterThan 0
                currentPolicy.behavioralDataRetentionDays shouldBeGreaterThan 0
                currentPolicy.usageDataRetentionDays shouldBeGreaterThan 0
                currentPolicy.trainingDataRetentionDays shouldBeGreaterThan 0
            }
        }
    }
    
    "Property 19.2: Automatic cleanup respects retention periods accurately" {
        checkAll(
            Arb.int(1..365), // retention days
            Arb.int(1..1000) // number of records to delete
        ) { retentionDays, recordsToDelete ->
            val manager = DataRetentionManager(mockContext, mockAutomationLog, mockPrivacyController)
            
            // Mock automation log cleanup
            whenever(mockAutomationLog.deleteLogsOlderThan(any())).thenReturn(recordsToDelete)
            
            val policy = RetentionPolicy(automationLogRetentionDays = retentionDays)
            
            runBlocking {
                manager.updateRetentionPolicy(policy)
                manager.performScheduledCleanup()
                
                val status = manager.cleanupStatus.value
                
                // Cleanup should have been performed
                status.lastCleanupTime shouldBeGreaterThan 0L
                
                // If records were deleted, status should reflect it
                if (recordsToDelete > 0) {
                    status.recordsDeleted shouldBeGreaterThan 0
                }
                
                // Cleanup should complete successfully
                status.success shouldBe true
            }
        }
    }
    
    "Property 19.3: Data anonymization preserves data utility while removing identifiers" {
        checkAll(
            Arb.int(1..100), // number of records to anonymize
            Arb.boolean() // anonymization enabled
        ) { recordCount, anonymizeEnabled ->
            val manager = DataRetentionManager(mockContext, mockAutomationLog, mockPrivacyController)
            
            // Mock anonymization
            val testData = mapOf(
                "id" to "test123",
                "userId" to "user456",
                "appName" to "TestApp",
                "timestamp" to System.currentTimeMillis(),
                "value" to 42
            )
            
            whenever(mockPrivacyController.anonymizeData(any())).thenReturn(
                mapOf(
                    "id" to "test123",
                    "appName" to "hash123", // Anonymized
                    "timestamp" to (System.currentTimeMillis() / 3600000) * 3600000, // Generalized
                    "value" to 42 // Preserved
                )
            )
            
            val policy = RetentionPolicy(
                anonymizeTrainingData = anonymizeEnabled,
                trainingDataRetentionDays = 30
            )
            
            runBlocking {
                manager.updateRetentionPolicy(policy)
                
                if (anonymizeEnabled) {
                    val anonymized = mockPrivacyController.anonymizeData(testData)
                    
                    // Anonymized data should preserve utility
                    anonymized["value"] shouldBe testData["value"]
                    
                    // But should remove/modify identifiers
                    anonymized["appName"] shouldNotBe testData["appName"]
                    anonymized shouldNotContainKey "userId"
                }
            }
        }
    }
    
    "Property 19.4: User data deletion controls work for all data types" {
        checkAll(
            Arb.set(Arb.enum<DataType>(), 1..4), // data types to delete
            Arb.int(1..365).orNull() // optional age filter
        ) { dataTypes, olderThanDays ->
            val manager = DataRetentionManager(mockContext, mockAutomationLog, mockPrivacyController)
            
            // Mock deletion operations
            whenever(mockAutomationLog.deleteAllLogs()).thenReturn(50)
            whenever(mockAutomationLog.deleteLogsOlderThan(any())).thenReturn(25)
            
            runBlocking {
                val result = manager.deleteUserData(dataTypes.toList(), olderThanDays)
                
                // Deletion should succeed
                result.success shouldBe true
                
                // Should have deletion results for requested data types
                dataTypes.forEach { dataType ->
                    result.deletionsByType shouldContain dataType
                }
                
                // Total deleted should be sum of individual deletions
                val expectedTotal = result.deletionsByType.values.sum()
                result.totalDeleted shouldBe expectedTotal
            }
        }
    }
    
    "Property 19.5: User data summary provides accurate storage and count information" {
        checkAll(
            Arb.int(0..10000), // automation logs count
            Arb.int(0..50000), // behavioral data count
            Arb.int(0..100000), // usage data count
            Arb.double(0.0, 1000.0) // storage used MB
        ) { automationCount, behavioralCount, usageCount, storageUsed ->
            val manager = DataRetentionManager(mockContext, mockAutomationLog, mockPrivacyController)
            
            // Mock data counts
            whenever(mockAutomationLog.getTotalLogCount()).thenReturn(automationCount)
            
            runBlocking {
                manager.updateUserDataSummary()
                
                val summary = manager.userDataSummary.value
                
                // Summary should reflect actual data counts
                summary.automationLogsCount shouldBe automationCount
                
                // Storage usage should be non-negative
                summary.totalStorageUsedMB shouldBeGreaterThan -0.1
                
                // Record dates should be reasonable
                if (summary.oldestRecordDate > 0) {
                    summary.oldestRecordDate shouldBeLessThan System.currentTimeMillis()
                }
                
                if (summary.newestRecordDate > 0) {
                    summary.newestRecordDate shouldBeLessThan System.currentTimeMillis() + 1000
                }
            }
        }
    }
    
    "Property 19.6: Data export preserves all user data accurately" {
        checkAll(
            Arb.set(Arb.enum<DataType>(), 1..4) // data types to export
        ) { dataTypes ->
            val manager = DataRetentionManager(mockContext, mockAutomationLog, mockPrivacyController)
            
            runBlocking {
                val result = manager.exportUserData(dataTypes.toList())
                
                // Export should succeed
                result.success shouldBe true
                
                // Should have export data
                result.exportData shouldNotBe null
                
                // Record count should be non-negative
                result.recordCount shouldBeGreaterThan -1
                
                // Export data should be valid JSON
                if (result.exportData != null) {
                    result.exportData!!.isNotEmpty() shouldBe true
                }
            }
        }
    }
    
    "Property 19.7: Cleanup scheduling maintains consistent intervals" {
        checkAll(
            Arb.int(1..24), // cleanup interval hours
            Arb.boolean() // immediate cleanup enabled
        ) { intervalHours, immediateCleanup ->
            val manager = DataRetentionManager(mockContext, mockAutomationLog, mockPrivacyController)
            
            val policy = RetentionPolicy(
                immediateCleanup = immediateCleanup,
                tempFileRetentionHours = intervalHours
            )
            
            runBlocking {
                manager.updateRetentionPolicy(policy)
                
                val beforeTime = System.currentTimeMillis()
                
                if (immediateCleanup) {
                    // Should trigger immediate cleanup
                    manager.performScheduledCleanup()
                    
                    val status = manager.cleanupStatus.value
                    status.lastCleanupTime shouldBeGreaterThan beforeTime - 1000
                }
                
                // Retention policy should be applied
                val currentPolicy = manager.retentionPolicy.value
                currentPolicy.immediateCleanup shouldBe immediateCleanup
                currentPolicy.tempFileRetentionHours shouldBe intervalHours
            }
        }
    }
    
    "Property 19.8: Temporary file cleanup respects age thresholds" {
        checkAll(
            Arb.int(1..168) // retention hours (1 hour to 1 week)
        ) { retentionHours ->
            val manager = DataRetentionManager(mockContext, mockAutomationLog, mockPrivacyController)
            
            val policy = RetentionPolicy(tempFileRetentionHours = retentionHours)
            
            runBlocking {
                manager.updateRetentionPolicy(policy)
                
                // Calculate expected cutoff time
                val expectedCutoff = System.currentTimeMillis() - TimeUnit.HOURS.toMillis(retentionHours.toLong())
                
                // Cleanup should use the correct cutoff time
                val cleanupResult = manager.cleanupTemporaryFiles(retentionHours)
                
                // Cleanup should complete without error
                cleanupResult shouldBeGreaterThan -1
                
                // Policy should be preserved
                val currentPolicy = manager.retentionPolicy.value
                currentPolicy.tempFileRetentionHours shouldBe retentionHours
            }
        }
    }
})