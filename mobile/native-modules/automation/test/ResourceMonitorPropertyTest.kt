package com.lifetwin.automation.test

import com.lifetwin.automation.ResourceMonitor
import com.lifetwin.automation.ResourceUsage
import com.lifetwin.automation.AdaptiveBehavior
import io.kotest.core.spec.style.StringSpec
import io.kotest.matchers.shouldBe
import io.kotest.matchers.shouldNotBe
import io.kotest.matchers.doubles.shouldBeLessThan
import io.kotest.matchers.doubles.shouldBeGreaterThan
import io.kotest.matchers.ints.shouldBeLessThan
import io.kotest.matchers.collections.shouldNotBeEmpty
import io.kotest.property.Arb
import io.kotest.property.arbitrary.*
import io.kotest.property.checkAll
import kotlinx.coroutines.runBlocking
import org.mockito.kotlin.mock
import android.content.Context

/**
 * Property-based tests for ResourceMonitor
 * 
 * Property 17: Resource usage adaptive behavior
 * Validates: Requirements 8.6 (resource monitoring), 8.7 (adaptive behavior)
 */
class ResourceMonitorPropertyTest : StringSpec({

    val mockContext = mock<Context>()
    
    "Property 17.1: Processing frequency adapts inversely to resource pressure" {
        checkAll(
            Arb.int(1..100), // battery level
            Arb.double(0.0, 100.0), // CPU usage
            Arb.double(0.0, 2000.0), // memory usage MB
            Arb.double(20.0, 50.0) // temperature
        ) { batteryLevel, cpuUsage, memoryUsage, temperature ->
            val monitor = ResourceMonitor(mockContext)
            
            val usage = ResourceUsage(
                batteryLevel = batteryLevel,
                cpuUsagePercent = cpuUsage,
                memoryUsageMB = memoryUsage,
                batteryTemperature = temperature
            )
            
            val frequency = runBlocking {
                monitor.calculateProcessingFrequency(usage)
            }
            
            // Lower battery should result in lower frequency
            when {
                batteryLevel <= 10 -> frequency shouldBeLessThan 0.15
                batteryLevel <= 20 -> frequency shouldBeLessThan 0.35
                batteryLevel <= 30 -> frequency shouldBeLessThan 0.55
            }
            
            // High CPU usage should reduce frequency
            if (cpuUsage > 90) {
                frequency shouldBeLessThan 0.5
            }
            
            // High memory usage should reduce frequency
            if (memoryUsage > 1000) {
                frequency shouldBeLessThan 0.5
            }
            
            // High temperature should reduce frequency
            if (temperature > 45) {
                frequency shouldBeLessThan 0.4
            }
            
            // Frequency should never be zero or negative
            frequency shouldBeGreaterThan 0.0
        }
    }
    
    "Property 17.2: Batch size adapts to resource constraints" {
        checkAll(
            Arb.int(1..100), // battery level
            Arb.double(0.0, 100.0), // CPU usage
            Arb.double(0.0, 2000.0) // memory usage MB
        ) { batteryLevel, cpuUsage, memoryUsage ->
            val monitor = ResourceMonitor(mockContext)
            
            val usage = ResourceUsage(
                batteryLevel = batteryLevel,
                cpuUsagePercent = cpuUsage,
                memoryUsageMB = memoryUsage
            )
            
            val batchSize = runBlocking {
                monitor.calculateOptimalBatchSize(usage)
            }
            
            // Resource pressure should reduce batch size
            val hasResourcePressure = batteryLevel <= 20 || cpuUsage > 80 || memoryUsage > 800
            
            if (hasResourcePressure) {
                batchSize shouldBeLessThan 50 // Default batch size
            }
            
            // Batch size should never be too small to be ineffective
            batchSize shouldBeGreaterThan 4
        }
    }
    
    "Property 17.3: Background processing permission correlates with resource availability" {
        checkAll(
            Arb.int(1..100), // battery level
            Arb.double(0.0, 100.0), // CPU usage
            Arb.double(20.0, 50.0) // temperature
        ) { batteryLevel, cpuUsage, temperature ->
            val monitor = ResourceMonitor(mockContext)
            
            val usage = ResourceUsage(
                batteryLevel = batteryLevel,
                cpuUsagePercent = cpuUsage,
                batteryTemperature = temperature
            )
            
            val allowBackground = runBlocking {
                monitor.shouldAllowBackgroundProcessing(usage)
            }
            
            // Critical conditions should disable background processing
            if (batteryLevel <= 15 || cpuUsage >= 85 || temperature >= 42) {
                allowBackground shouldBe false
            }
            
            // Good conditions should allow background processing
            if (batteryLevel > 50 && cpuUsage < 50 && temperature < 35) {
                allowBackground shouldBe true
            }
        }
    }
    
    "Property 17.4: Battery statistics provide accurate drain rate calculations" {
        checkAll(
            Arb.list(Arb.int(1..100), 2..10), // battery levels
            Arb.list(Arb.long(1000..3600000), 2..10) // time intervals
        ) { batteryLevels, timeIntervals ->
            val monitor = ResourceMonitor(mockContext)
            
            // Simulate battery drain over time
            var currentTime = System.currentTimeMillis()
            
            batteryLevels.zip(timeIntervals).forEach { (level, interval) ->
                runBlocking {
                    monitor.updateBatteryStatistics(
                        ResourceUsage(
                            batteryLevel = level,
                            timestamp = currentTime
                        )
                    )
                }
                currentTime += interval
            }
            
            val stats = runBlocking {
                monitor.getBatteryStatistics()
            }
            
            // Statistics should be reasonable
            stats.currentLevel shouldBe batteryLevels.last()
            
            if (stats.hourlyDrainRate > 0) {
                // Drain rate should be reasonable (not more than 100% per hour)
                stats.hourlyDrainRate shouldBeLessThan 100.0
            }
            
            if (stats.dailyDrainRate > 0) {
                // Daily drain rate should be reasonable
                stats.dailyDrainRate shouldBeLessThan 100.0
            }
        }
    }
    
    "Property 17.5: Resource recommendations are contextually appropriate" {
        checkAll(
            Arb.int(1..100), // battery level
            Arb.double(0.0, 100.0), // CPU usage
            Arb.double(0.0, 2000.0), // memory usage MB
            Arb.double(20.0, 50.0), // temperature
            Arb.double(0.0, 1000.0) // available storage MB
        ) { batteryLevel, cpuUsage, memoryUsage, temperature, availableStorage ->
            val monitor = ResourceMonitor(mockContext)
            
            val usage = ResourceUsage(
                batteryLevel = batteryLevel,
                cpuUsagePercent = cpuUsage,
                memoryUsageMB = memoryUsage,
                batteryTemperature = temperature,
                availableStorageMB = availableStorage
            )
            
            val recommendations = runBlocking {
                monitor.generateResourceRecommendations(usage)
            }
            
            // Low battery should trigger battery recommendation
            if (batteryLevel <= 20) {
                recommendations.any { it.contains("battery", ignoreCase = true) } shouldBe true
            }
            
            // High CPU usage should trigger CPU recommendation
            if (cpuUsage > 80) {
                recommendations.any { it.contains("CPU", ignoreCase = true) } shouldBe true
            }
            
            // High memory usage should trigger memory recommendation
            if (memoryUsage > 800) {
                recommendations.any { it.contains("memory", ignoreCase = true) } shouldBe true
            }
            
            // High temperature should trigger temperature recommendation
            if (temperature > 40) {
                recommendations.any { it.contains("heating", ignoreCase = true) || it.contains("temperature", ignoreCase = true) } shouldBe true
            }
            
            // Low storage should trigger storage recommendation
            if (availableStorage < 100) {
                recommendations.any { it.contains("storage", ignoreCase = true) } shouldBe true
            }
        }
    }
    
    "Property 17.6: Cache size adapts to memory pressure" {
        checkAll(
            Arb.double(0.0, 2000.0) // memory usage MB
        ) { memoryUsage ->
            val monitor = ResourceMonitor(mockContext)
            
            val usage = ResourceUsage(memoryUsageMB = memoryUsage)
            
            val cacheSize = runBlocking {
                monitor.calculateOptimalCacheSize(usage)
            }
            
            // High memory usage should reduce cache size
            when {
                memoryUsage > 1000 -> cacheSize shouldBeLessThan 30 // 100 / 4
                memoryUsage > 500 -> cacheSize shouldBeLessThan 60 // 100 / 2
                else -> cacheSize shouldBeLessThan 110 // Allow some margin
            }
            
            // Cache size should never be too small
            cacheSize shouldBeGreaterThan 9
        }
    }
    
    "Property 17.7: Resource summary provides consistent optimization status" {
        checkAll(
            Arb.double(0.0, 100.0), // CPU usage
            Arb.double(0.0, 2000.0), // memory usage
            Arb.int(1..100), // battery level
            Arb.double(0.1, 1.0) // processing frequency
        ) { cpuUsage, memoryUsage, batteryLevel, frequency ->
            val monitor = ResourceMonitor(mockContext)
            
            // Set up resource state
            runBlocking {
                monitor.updateResourceUsage(
                    ResourceUsage(
                        cpuUsagePercent = cpuUsage,
                        memoryUsageMB = memoryUsage,
                        batteryLevel = batteryLevel
                    )
                )
                
                monitor.updateAdaptiveBehavior(
                    AdaptiveBehavior(
                        processingFrequency = frequency,
                        batchSize = if (frequency < 1.0) 25 else 50,
                        backgroundProcessing = frequency >= 0.5
                    )
                )
            }
            
            val summary = runBlocking {
                monitor.getResourceSummary()
            }
            
            // Summary should reflect current state
            summary.cpuUsage shouldBe cpuUsage
            summary.memoryUsage shouldBe memoryUsage
            summary.batteryLevel shouldBe batteryLevel
            summary.adaptiveFrequency shouldBe frequency
            
            // Optimization status should be consistent
            val expectedOptimized = frequency < 1.0 || summary.recommendationsCount > 0
            summary.isOptimized shouldBe expectedOptimized
        }
    }
})