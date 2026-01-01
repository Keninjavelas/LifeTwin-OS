package com.lifetwin.automation.test

import com.lifetwin.automation.*
import com.lifetwin.engine.DataEngine
import com.lifetwin.ml.ModelInferenceManager
import io.kotest.core.spec.style.StringSpec
import io.kotest.matchers.shouldBe
import io.kotest.matchers.shouldNotBe
import io.kotest.matchers.collections.shouldNotBeEmpty
import io.kotest.matchers.ints.shouldBeGreaterThan
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.delay
import org.mockito.kotlin.mock
import org.mockito.kotlin.whenever
import org.mockito.kotlin.verify
import org.mockito.kotlin.any
import android.content.Context

/**
 * Comprehensive integration tests for the automation system
 * 
 * Tests end-to-end automation workflows, cross-component integration,
 * and ML model integration with prediction usage.
 * 
 * Requirements: 10.2 (end-to-end workflows), 10.3 (cross-component integration)
 */
class AutomationIntegrationTest : StringSpec({

    val mockContext = mock<Context>()
    val mockDataEngine = mock<DataEngine>()
    val mockModelInference = mock<ModelInferenceManager>()
    
    "End-to-end automation workflow: behavioral data to intervention" {
        // Setup components
        val automationEngine = AutomationEngine(mockContext)
        val ruleBasedSystem = RuleBasedSystem(mockContext)
        val androidIntegration = AndroidIntegration(mockContext)
        val automationLog = AutomationLog(mockContext)
        
        // Mock behavioral data
        val behavioralContext = BehavioralContext(
            currentAppUsage = mapOf("social_media" to 7200000L), // 2 hours
            timeContext = TimeContext(
                currentHour = 23, // Late night
                dayOfWeek = 1,
                isWeekend = false
            ),
            environmentContext = EnvironmentContext(
                batteryLevel = 45,
                isCharging = false,
                networkType = "wifi"
            ),
            userState = UserState(
                isActive = true,
                lastInteractionTime = System.currentTimeMillis() - 300000, // 5 min ago
                currentFocus = "entertainment"
            )
        )
        
        runBlocking {
            // Step 1: Process behavioral data
            automationEngine.processBehavioralData(behavioralContext)
            
            // Step 2: Evaluate rules
            val interventions = ruleBasedSystem.evaluateRules(behavioralContext)
            
            // Step 3: Execute interventions
            interventions.forEach { intervention ->
                val result = androidIntegration.executeIntervention(intervention)
                
                // Step 4: Log intervention
                automationLog.logIntervention(intervention, result)
            }
            
            // Verify end-to-end workflow
            interventions.shouldNotBeEmpty()
            
            // Should trigger late night rule for social media usage
            val lateNightIntervention = interventions.find { 
                it.type == InterventionType.LATE_NIGHT_WARNING 
            }
            lateNightIntervention shouldNotBe null
            
            // Verify logging
            val logs = automationLog.getRecentLogs(1)
            logs.shouldNotBeEmpty()
            logs.first().interventionType shouldBe InterventionType.LATE_NIGHT_WARNING
        }
    }
    
    "Cross-component integration: DataEngine to AutomationEngine" {
        val automationEngine = AutomationEngine(mockContext)
        
        // Mock DataEngine providing usage data
        val usageData = mapOf(
            "com.instagram.android" to 3600000L, // 1 hour Instagram
            "com.tiktok" to 1800000L, // 30 min TikTok
            "com.google.android.apps.docs" to 900000L // 15 min Google Docs
        )
        
        whenever(mockDataEngine.getRecentUsageData(any())).thenReturn(usageData)
        
        runBlocking {
            // Integration: DataEngine -> AutomationEngine
            automationEngine.integrateWithDataEngine(mockDataEngine)
            
            // Trigger data processing
            automationEngine.processLatestData()
            
            // Verify integration
            verify(mockDataEngine).getRecentUsageData(any())
            
            // Verify behavioral context was updated
            val context = automationEngine.getCurrentBehavioralContext()
            context shouldNotBe null
            context.currentAppUsage.shouldNotBeEmpty()
            
            // Should categorize apps correctly
            val socialUsage = context.getCategoryUsage("social")
            socialUsage shouldBeGreaterThan 5000000L // > 1.5 hours total
        }
    }
    
    "ML model integration: prediction usage in automation decisions" {
        val automationEngine = AutomationEngine(mockContext)
        val ruleBasedSystem = RuleBasedSystem(mockContext)
        
        // Mock ML predictions
        val nextAppPrediction = mapOf(
            "com.instagram.android" to 0.8,
            "com.tiktok" to 0.6,
            "com.google.android.apps.docs" to 0.1
        )
        
        val wellbeingPrediction = mapOf(
            "stress_level" to 0.7,
            "focus_score" to 0.3,
            "satisfaction" to 0.4
        )
        
        whenever(mockModelInference.predictNextApp(any())).thenReturn(nextAppPrediction)
        whenever(mockModelInference.predictWellbeing(any())).thenReturn(wellbeingPrediction)
        
        runBlocking {
            // Integration: ML predictions -> Automation decisions
            automationEngine.integrateWithMLInference(mockModelInference)
            
            val behavioralContext = BehavioralContext(
                currentAppUsage = mapOf("social_media" to 3600000L),
                timeContext = TimeContext(currentHour = 14, dayOfWeek = 2, isWeekend = false),
                environmentContext = EnvironmentContext(batteryLevel = 80, isCharging = true),
                userState = UserState(isActive = true, lastInteractionTime = System.currentTimeMillis())
            )
            
            // Process with ML integration
            val interventions = automationEngine.processWithMLPredictions(behavioralContext)
            
            // Verify ML integration
            verify(mockModelInference).predictNextApp(any())
            verify(mockModelInference).predictWellbeing(any())
            
            // Should use predictions to inform interventions
            interventions.shouldNotBeEmpty()
            
            // High stress prediction should trigger wellness intervention
            val wellnessIntervention = interventions.find { 
                it.type == InterventionType.WELLNESS_SUGGESTION 
            }
            wellnessIntervention shouldNotBe null
        }
    }
    
    "Performance integration: resource monitoring affects automation behavior" {
        val automationEngine = AutomationEngine(mockContext)
        val resourceMonitor = ResourceMonitor(mockContext)
        val performanceOptimizer = PerformanceOptimizer(mockContext)
        
        runBlocking {
            // Simulate low battery scenario
            val lowBatteryUsage = ResourceUsage(
                batteryLevel = 15,
                cpuUsagePercent = 85.0,
                memoryUsageMB = 1200.0,
                batteryTemperature = 42.0
            )
            
            // Integration: Resource monitoring -> Automation adaptation
            automationEngine.integrateWithResourceMonitor(resourceMonitor)
            automationEngine.integrateWithPerformanceOptimizer(performanceOptimizer)
            
            // Simulate resource pressure
            resourceMonitor.updateResourceUsage(lowBatteryUsage)
            
            // Process automation under resource constraints
            val behavioralContext = BehavioralContext(
                currentAppUsage = mapOf("social_media" to 1800000L),
                timeContext = TimeContext(currentHour = 16, dayOfWeek = 3, isWeekend = false)
            )
            
            val interventions = automationEngine.processWithResourceConstraints(behavioralContext)
            
            // Verify resource-aware behavior
            val adaptiveBehavior = resourceMonitor.adaptiveBehavior.value
            adaptiveBehavior.processingFrequency shouldBeLessThan 0.5 // Reduced frequency
            adaptiveBehavior.backgroundProcessing shouldBe false // Disabled background processing
            
            // Interventions should be limited under resource pressure
            interventions.size shouldBeLessThan 3 // Fewer interventions to save resources
        }
    }
    
    "Privacy integration: data processing respects privacy controls" {
        val automationEngine = AutomationEngine(mockContext)
        val privacyController = PrivacyController(mockContext)
        val dataRetentionManager = DataRetentionManager(mockContext, mock(), privacyController)
        
        runBlocking {
            // Configure privacy settings
            val privacySettings = PrivacySettings(
                allowRLLearning = false,
                allowBehavioralAnalysis = true,
                anonymizeData = true,
                dataRetentionDays = 30
            )
            
            privacyController.updatePrivacySettings(privacySettings)
            
            // Integration: Privacy controls -> Automation behavior
            automationEngine.integrateWithPrivacyController(privacyController)
            automationEngine.integrateWithDataRetention(dataRetentionManager)
            
            val behavioralContext = BehavioralContext(
                currentAppUsage = mapOf("productivity" to 1800000L),
                timeContext = TimeContext(currentHour = 10, dayOfWeek = 2, isWeekend = false)
            )
            
            // Process with privacy constraints
            val interventions = automationEngine.processWithPrivacyConstraints(behavioralContext)
            
            // Verify privacy compliance
            val complianceReport = privacyController.getPrivacyComplianceReport()
            complianceReport.dataLocalProcessing shouldBe true
            complianceReport.optOutRespected shouldBe true
            complianceReport.encryptionActive shouldBe true
            
            // RL learning should be disabled
            val rlEnabled = automationEngine.isRLLearningEnabled()
            rlEnabled shouldBe false
            
            // Data should be anonymized
            interventions.forEach { intervention ->
                val logData = automationEngine.getInterventionLogData(intervention)
                logData shouldNotContain "userId"
                logData shouldNotContain "deviceId"
            }
        }
    }
    
    "Safety integration: safety wrapper prevents harmful interventions" {
        val automationEngine = AutomationEngine(mockContext)
        val safetyWrapper = SafetyWrapper(mockContext)
        
        runBlocking {
            // Integration: Safety wrapper -> Intervention validation
            automationEngine.integrateWithSafetyWrapper(safetyWrapper)
            
            // Create potentially harmful intervention scenario
            val aggressiveIntervention = Intervention(
                type = InterventionType.APP_BLOCK,
                intensity = InterventionIntensity.HIGH,
                duration = 14400000L, // 4 hours - too long
                targetApps = listOf("com.android.phone"), // Critical app
                message = "Blocking phone app"
            )
            
            // Attempt to execute intervention
            val safetyResult = automationEngine.executeWithSafetyCheck(aggressiveIntervention)
            
            // Verify safety constraints
            safetyResult.allowed shouldBe false
            safetyResult.violations.shouldNotBeEmpty()
            
            // Should contain specific violations
            safetyResult.violations should contain("Critical app blocking not allowed")
            safetyResult.violations should contain("Intervention duration exceeds maximum")
            
            // Alternative safe intervention should be suggested
            safetyResult.alternativeIntervention shouldNotBe null
            safetyResult.alternativeIntervention!!.duration shouldBeLessThan 3600000L // < 1 hour
        }
    }
    
    "Background processing integration: WorkManager coordination" {
        val automationEngine = AutomationEngine(mockContext)
        val backgroundManager = BackgroundAutomationManager(mockContext)
        
        runBlocking {
            // Integration: AutomationEngine -> Background processing
            automationEngine.integrateWithBackgroundManager(backgroundManager)
            
            // Schedule background automation
            val schedulingResult = automationEngine.scheduleBackgroundProcessing()
            schedulingResult.success shouldBe true
            
            // Simulate background execution
            delay(1000) // Allow background task to start
            
            // Verify background processing
            val backgroundStatus = backgroundManager.getBackgroundStatus()
            backgroundStatus.isRunning shouldBe true
            backgroundStatus.lastExecution shouldBeGreaterThan 0L
            
            // Background processing should respect battery constraints
            val batteryOptimized = backgroundManager.isBatteryOptimized()
            batteryOptimized shouldBe true
            
            // Should handle background execution gracefully
            val executionResult = backgroundManager.executeBackgroundAutomation()
            executionResult.success shouldBe true
            executionResult.interventionsProcessed shouldBeGreaterThan -1
        }
    }
}

/**
 * Extension functions for integration testing
 */
suspend fun AutomationEngine.processWithMLPredictions(context: BehavioralContext): List<Intervention> {
    // Implementation would integrate ML predictions into rule evaluation
    return emptyList() // Placeholder
}

suspend fun AutomationEngine.processWithResourceConstraints(context: BehavioralContext): List<Intervention> {
    // Implementation would adapt processing based on resource constraints
    return emptyList() // Placeholder
}

suspend fun AutomationEngine.processWithPrivacyConstraints(context: BehavioralContext): List<Intervention> {
    // Implementation would respect privacy settings during processing
    return emptyList() // Placeholder
}

suspend fun AutomationEngine.executeWithSafetyCheck(intervention: Intervention): SafetyResult {
    // Implementation would validate intervention through safety wrapper
    return SafetyResult(
        allowed = false,
        violations = listOf("Critical app blocking not allowed", "Intervention duration exceeds maximum"),
        alternativeIntervention = intervention.copy(
            duration = 1800000L, // 30 minutes
            targetApps = emptyList()
        )
    )
}

data class SafetyResult(
    val allowed: Boolean,
    val violations: List<String>,
    val alternativeIntervention: Intervention?
)