package com.lifetwin.automation

import android.content.Context
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import kotlinx.serialization.encodeToString
import kotlinx.serialization.decodeFromString
import kotlin.math.*
import kotlin.random.Random

/**
 * ABTestingFramework - Statistical testing for rule-based vs RL effectiveness
 * 
 * Implements Requirements:
 * - 4.5: Statistical testing for rule-based vs RL effectiveness
 * - User satisfaction measurement and comparison
 * - Long-term behavioral change outcome tracking
 */
class ABTestingFramework(
    private val context: Context,
    private val automationEngine: AutomationEngine,
    private val ruleBasedSystem: RuleBasedSystem
) {
    private val scope = CoroutineScope(Dispatchers.Default + SupervisorJob())
    
    // A/B test state management
    private val _activeTests = MutableStateFlow<Map<String, ABTest>>(emptyMap())
    val activeTests: StateFlow<Map<String, ABTest>> = _activeTests.asStateFlow()
    
    // User assignment tracking
    private val _userAssignments = MutableStateFlow<Map<String, TestAssignment>>(emptyMap())
    val userAssignments: StateFlow<Map<String, TestAssignment>> = _userAssignments.asStateFlow()
    
    // Results tracking
    private val _testResults = MutableStateFlow<Map<String, TestResults>>(emptyMap())
    val testResults: StateFlow<Map<String, TestResults>> = _testResults.asStateFlow()
    
    init {
        loadActiveTests()
        startResultsCollection()
    }
    
    /**
     * Create and start a new A/B test
     */
    suspend fun createABTest(
        testName: String,
        description: String,
        controlPolicy: PolicyType,
        treatmentPolicy: PolicyType,
        trafficSplit: Double = 0.5,
        durationDays: Int = 30,
        minimumSampleSize: Int = 100,
        metrics: List<TestMetric>
    ): ABTest = withContext(Dispatchers.IO) {
        
        val test = ABTest(
            id = generateTestId(),
            name = testName,
            description = description,
            controlPolicy = controlPolicy,
            treatmentPolicy = treatmentPolicy,
            trafficSplit = trafficSplit,
            startTime = System.currentTimeMillis(),
            endTime = System.currentTimeMillis() + (durationDays * 24 * 3600 * 1000L),
            minimumSampleSize = minimumSampleSize,
            metrics = metrics,
            status = TestStatus.ACTIVE
        )
        
        val currentTests = _activeTests.value.toMutableMap()
        currentTests[test.id] = test
        _activeTests.value = currentTests
        
        saveActiveTests()
        
        test
    }
    
    /**
     * Assign user to test group (control or treatment)
     */
    suspend fun assignUserToTest(userId: String, testId: String): TestAssignment = withContext(Dispatchers.IO) {
        val test = _activeTests.value[testId] 
            ?: throw IllegalArgumentException("Test $testId not found")
        
        // Use consistent hashing for stable assignment
        val hash = (userId + testId).hashCode()
        val normalizedHash = (hash.toDouble() / Int.MAX_VALUE + 1.0) / 2.0 // Normalize to 0-1
        
        val group = if (normalizedHash < test.trafficSplit) {
            TestGroup.TREATMENT
        } else {
            TestGroup.CONTROL
        }
        
        val assignment = TestAssignment(
            userId = userId,
            testId = testId,
            group = group,
            assignmentTime = System.currentTimeMillis(),
            policy = if (group == TestGroup.CONTROL) test.controlPolicy else test.treatmentPolicy
        )
        
        val currentAssignments = _userAssignments.value.toMutableMap()
        currentAssignments[userId] = assignment
        _userAssignments.value = currentAssignments
        
        saveUserAssignments()
        
        assignment
    }
    
    /**
     * Get user's current test assignment
     */
    fun getUserAssignment(userId: String): TestAssignment? {
        return _userAssignments.value[userId]
    }
    
    /**
     * Record intervention outcome for A/B testing
     */
    suspend fun recordInterventionOutcome(
        userId: String,
        intervention: Intervention,
        outcome: InterventionOutcome
    ) = withContext(Dispatchers.IO) {
        
        val assignment = getUserAssignment(userId) ?: return@withContext
        val test = _activeTests.value[assignment.testId] ?: return@withContext
        
        if (test.status != TestStatus.ACTIVE) return@withContext
        
        val outcomeRecord = OutcomeRecord(
            userId = userId,
            testId = assignment.testId,
            group = assignment.group,
            intervention = intervention,
            outcome = outcome,
            timestamp = System.currentTimeMillis()
        )
        
        // Update test results
        val currentResults = _testResults.value[assignment.testId] ?: TestResults(
            testId = assignment.testId,
            controlOutcomes = mutableListOf(),
            treatmentOutcomes = mutableListOf(),
            controlMetrics = mutableMapOf(),
            treatmentMetrics = mutableMapOf()
        )
        
        when (assignment.group) {
            TestGroup.CONTROL -> currentResults.controlOutcomes.add(outcomeRecord)
            TestGroup.TREATMENT -> currentResults.treatmentOutcomes.add(outcomeRecord)
        }
        
        // Update metrics
        updateTestMetrics(currentResults, outcomeRecord)
        
        val updatedResults = _testResults.value.toMutableMap()
        updatedResults[assignment.testId] = currentResults
        _testResults.value = updatedResults
        
        saveTestResults()
        
        // Check if test should be concluded
        checkTestCompletion(assignment.testId)
    }
    
    /**
     * Record user satisfaction rating
     */
    suspend fun recordUserSatisfaction(
        userId: String,
        satisfactionRating: Double, // 0.0 to 1.0
        feedbackText: String? = null
    ) = withContext(Dispatchers.IO) {
        
        val assignment = getUserAssignment(userId) ?: return@withContext
        
        val satisfactionRecord = SatisfactionRecord(
            userId = userId,
            testId = assignment.testId,
            group = assignment.group,
            rating = satisfactionRating,
            feedback = feedbackText,
            timestamp = System.currentTimeMillis()
        )
        
        val currentResults = _testResults.value[assignment.testId] ?: return@withContext
        
        when (assignment.group) {
            TestGroup.CONTROL -> currentResults.controlSatisfaction.add(satisfactionRecord)
            TestGroup.TREATMENT -> currentResults.treatmentSatisfaction.add(satisfactionRecord)
        }
        
        val updatedResults = _testResults.value.toMutableMap()
        updatedResults[assignment.testId] = currentResults
        _testResults.value = updatedResults
        
        saveTestResults()
    }
    
    /**
     * Record long-term behavioral change metrics
     */
    suspend fun recordBehavioralChange(
        userId: String,
        changeMetrics: Map<String, Double>
    ) = withContext(Dispatchers.IO) {
        
        val assignment = getUserAssignment(userId) ?: return@withContext
        
        val changeRecord = BehavioralChangeRecord(
            userId = userId,
            testId = assignment.testId,
            group = assignment.group,
            metrics = changeMetrics,
            timestamp = System.currentTimeMillis()
        )
        
        val currentResults = _testResults.value[assignment.testId] ?: return@withContext
        
        when (assignment.group) {
            TestGroup.CONTROL -> currentResults.controlBehavioralChanges.add(changeRecord)
            TestGroup.TREATMENT -> currentResults.treatmentBehavioralChanges.add(changeRecord)
        }
        
        val updatedResults = _testResults.value.toMutableMap()
        updatedResults[assignment.testId] = currentResults
        _testResults.value = updatedResults
        
        saveTestResults()
    }
    
    /**
     * Analyze test results and determine statistical significance
     */
    suspend fun analyzeTestResults(testId: String): TestAnalysis = withContext(Dispatchers.IO) {
        val test = _activeTests.value[testId] 
            ?: throw IllegalArgumentException("Test $testId not found")
        
        val results = _testResults.value[testId] 
            ?: throw IllegalArgumentException("No results found for test $testId")
        
        val analysis = TestAnalysis(
            testId = testId,
            testName = test.name,
            analysisTime = System.currentTimeMillis(),
            sampleSizes = SampleSizes(
                control = results.controlOutcomes.size,
                treatment = results.treatmentOutcomes.size
            ),
            metricAnalysis = analyzeMetrics(results, test.metrics),
            satisfactionAnalysis = analyzeSatisfaction(results),
            behavioralChangeAnalysis = analyzeBehavioralChanges(results),
            overallSignificance = false, // Will be calculated
            recommendation = TestRecommendation.CONTINUE // Will be determined
        )
        
        // Calculate overall significance
        val significantMetrics = analysis.metricAnalysis.values.count { it.isSignificant }
        analysis.overallSignificance = significantMetrics > 0 && 
                                     analysis.sampleSizes.control >= test.minimumSampleSize &&
                                     analysis.sampleSizes.treatment >= test.minimumSampleSize
        
        // Determine recommendation
        analysis.recommendation = determineRecommendation(analysis, test)
        
        analysis
    }
    
    /**
     * Update test metrics based on outcome record
     */
    private fun updateTestMetrics(results: TestResults, outcome: OutcomeRecord) {
        val groupMetrics = when (outcome.group) {
            TestGroup.CONTROL -> results.controlMetrics
            TestGroup.TREATMENT -> results.treatmentMetrics
        }
        
        // Update intervention effectiveness
        val effectivenessKey = "intervention_effectiveness"
        val currentEffectiveness = groupMetrics[effectivenessKey] ?: 0.0
        val newEffectiveness = when (outcome.outcome.effectiveness) {
            EffectivenessRating.VERY_EFFECTIVE -> 1.0
            EffectivenessRating.EFFECTIVE -> 0.75
            EffectivenessRating.SOMEWHAT_EFFECTIVE -> 0.5
            EffectivenessRating.NOT_EFFECTIVE -> 0.0
        }
        
        val outcomeCount = when (outcome.group) {
            TestGroup.CONTROL -> results.controlOutcomes.size
            TestGroup.TREATMENT -> results.treatmentOutcomes.size
        }
        
        // Running average
        groupMetrics[effectivenessKey] = (currentEffectiveness * (outcomeCount - 1) + newEffectiveness) / outcomeCount
        
        // Update engagement metrics
        val engagementKey = "user_engagement"
        val engagement = if (outcome.outcome.userEngaged) 1.0 else 0.0
        val currentEngagement = groupMetrics[engagementKey] ?: 0.0
        groupMetrics[engagementKey] = (currentEngagement * (outcomeCount - 1) + engagement) / outcomeCount
        
        // Update dismissal rate
        val dismissalKey = "dismissal_rate"
        val dismissed = if (outcome.outcome.dismissed) 1.0 else 0.0
        val currentDismissal = groupMetrics[dismissalKey] ?: 0.0
        groupMetrics[dismissalKey] = (currentDismissal * (outcomeCount - 1) + dismissed) / outcomeCount
    }
    
    /**
     * Analyze metrics for statistical significance
     */
    private fun analyzeMetrics(results: TestResults, testMetrics: List<TestMetric>): Map<String, MetricAnalysis> {
        val analysis = mutableMapOf<String, MetricAnalysis>()
        
        testMetrics.forEach { metric ->
            val controlValues = results.controlMetrics[metric.name] ?: 0.0
            val treatmentValues = results.treatmentMetrics[metric.name] ?: 0.0
            
            val controlSample = results.controlOutcomes.size
            val treatmentSample = results.treatmentOutcomes.size
            
            // Perform t-test (simplified)
            val pooledStdDev = calculatePooledStandardDeviation(
                controlValues, treatmentValues, controlSample, treatmentSample
            )
            
            val standardError = pooledStdDev * sqrt(1.0/controlSample + 1.0/treatmentSample)
            val tStatistic = abs(treatmentValues - controlValues) / standardError
            val pValue = calculatePValue(tStatistic, controlSample + treatmentSample - 2)
            
            analysis[metric.name] = MetricAnalysis(
                metricName = metric.name,
                controlMean = controlValues,
                treatmentMean = treatmentValues,
                difference = treatmentValues - controlValues,
                percentChange = if (controlValues != 0.0) {
                    ((treatmentValues - controlValues) / controlValues) * 100.0
                } else 0.0,
                pValue = pValue,
                isSignificant = pValue < 0.05,
                confidenceInterval = calculateConfidenceInterval(
                    treatmentValues - controlValues, standardError
                )
            )
        }
        
        return analysis
    }
    
    /**
     * Analyze user satisfaction differences
     */
    private fun analyzeSatisfaction(results: TestResults): SatisfactionAnalysis {
        val controlSatisfaction = results.controlSatisfaction.map { it.rating }
        val treatmentSatisfaction = results.treatmentSatisfaction.map { it.rating }
        
        val controlMean = controlSatisfaction.average().takeIf { !it.isNaN() } ?: 0.0
        val treatmentMean = treatmentSatisfaction.average().takeIf { !it.isNaN() } ?: 0.0
        
        return SatisfactionAnalysis(
            controlMean = controlMean,
            treatmentMean = treatmentMean,
            difference = treatmentMean - controlMean,
            sampleSizes = SampleSizes(
                control = controlSatisfaction.size,
                treatment = treatmentSatisfaction.size
            ),
            isSignificant = abs(treatmentMean - controlMean) > 0.1 && // Practical significance
                           controlSatisfaction.size >= 30 && treatmentSatisfaction.size >= 30
        )
    }
    
    /**
     * Analyze behavioral change differences
     */
    private fun analyzeBehavioralChanges(results: TestResults): BehavioralChangeAnalysis {
        val controlChanges = results.controlBehavioralChanges
        val treatmentChanges = results.treatmentBehavioralChanges
        
        val metricAnalysis = mutableMapOf<String, Double>()
        
        // Analyze each behavioral metric
        val allMetrics = (controlChanges.flatMap { it.metrics.keys } + 
                         treatmentChanges.flatMap { it.metrics.keys }).toSet()
        
        allMetrics.forEach { metric ->
            val controlValues = controlChanges.mapNotNull { it.metrics[metric] }
            val treatmentValues = treatmentChanges.mapNotNull { it.metrics[metric] }
            
            if (controlValues.isNotEmpty() && treatmentValues.isNotEmpty()) {
                val controlMean = controlValues.average()
                val treatmentMean = treatmentValues.average()
                metricAnalysis[metric] = treatmentMean - controlMean
            }
        }
        
        return BehavioralChangeAnalysis(
            metricDifferences = metricAnalysis,
            sampleSizes = SampleSizes(
                control = controlChanges.size,
                treatment = treatmentChanges.size
            ),
            significantChanges = metricAnalysis.filter { abs(it.value) > 0.1 }.keys.toList()
        )
    }
    
    /**
     * Determine test recommendation based on analysis
     */
    private fun determineRecommendation(analysis: TestAnalysis, test: ABTest): TestRecommendation {
        val hasSignificantResults = analysis.overallSignificance
        val hasSufficientSample = analysis.sampleSizes.control >= test.minimumSampleSize &&
                                 analysis.sampleSizes.treatment >= test.minimumSampleSize
        
        val treatmentBetter = analysis.metricAnalysis.values.count { 
            it.isSignificant && it.difference > 0 
        } > analysis.metricAnalysis.values.count { 
            it.isSignificant && it.difference < 0 
        }
        
        return when {
            !hasSufficientSample -> TestRecommendation.CONTINUE
            hasSignificantResults && treatmentBetter -> TestRecommendation.ADOPT_TREATMENT
            hasSignificantResults && !treatmentBetter -> TestRecommendation.KEEP_CONTROL
            else -> TestRecommendation.NO_DIFFERENCE
        }
    }
    
    /**
     * Check if test should be concluded
     */
    private suspend fun checkTestCompletion(testId: String) {
        val test = _activeTests.value[testId] ?: return
        val results = _testResults.value[testId] ?: return
        
        val now = System.currentTimeMillis()
        val hasEnoughSamples = results.controlOutcomes.size >= test.minimumSampleSize &&
                              results.treatmentOutcomes.size >= test.minimumSampleSize
        val timeExpired = now >= test.endTime
        
        if (hasEnoughSamples && timeExpired) {
            concludeTest(testId)
        }
    }
    
    /**
     * Conclude a test and generate final analysis
     */
    private suspend fun concludeTest(testId: String) = withContext(Dispatchers.IO) {
        val currentTests = _activeTests.value.toMutableMap()
        val test = currentTests[testId] ?: return@withContext
        
        currentTests[testId] = test.copy(
            status = TestStatus.COMPLETED,
            endTime = System.currentTimeMillis()
        )
        _activeTests.value = currentTests
        
        // Generate final analysis
        val finalAnalysis = analyzeTestResults(testId)
        
        // Apply recommendation if auto-apply is enabled
        if (test.autoApply && finalAnalysis.recommendation == TestRecommendation.ADOPT_TREATMENT) {
            applyTreatmentPolicy(test.treatmentPolicy)
        }
        
        saveActiveTests()
    }
    
    /**
     * Apply treatment policy system-wide
     */
    private suspend fun applyTreatmentPolicy(policy: PolicyType) {
        when (policy) {
            PolicyType.RULE_BASED_ENHANCED -> {
                // Apply enhanced rule-based system
                automationEngine.setPrimaryPolicy(policy)
            }
            PolicyType.RL_POLICY -> {
                // Apply RL policy
                automationEngine.setPrimaryPolicy(policy)
            }
            PolicyType.HYBRID -> {
                // Apply hybrid approach
                automationEngine.setPrimaryPolicy(policy)
            }
        }
    }
    
    /**
     * Start continuous results collection
     */
    private fun startResultsCollection() {
        scope.launch {
            while (isActive) {
                try {
                    collectOngoingResults()
                    delay(3600000) // Collect every hour
                } catch (e: Exception) {
                    delay(7200000) // Wait longer on error
                }
            }
        }
    }
    
    /**
     * Collect ongoing results from active tests
     */
    private suspend fun collectOngoingResults() = withContext(Dispatchers.IO) {
        _activeTests.value.values.filter { it.status == TestStatus.ACTIVE }.forEach { test ->
            checkTestCompletion(test.id)
        }
    }
    
    /**
     * Utility functions for statistical calculations
     */
    private fun calculatePooledStandardDeviation(
        mean1: Double, mean2: Double, n1: Int, n2: Int
    ): Double {
        // Simplified calculation - in practice would use actual sample data
        return sqrt(((n1 - 1) * 0.25 + (n2 - 1) * 0.25) / (n1 + n2 - 2))
    }
    
    private fun calculatePValue(tStatistic: Double, degreesOfFreedom: Int): Double {
        // Simplified p-value calculation - in practice would use proper statistical library
        return if (abs(tStatistic) > 1.96) 0.04 else 0.06
    }
    
    private fun calculateConfidenceInterval(difference: Double, standardError: Double): ConfidenceInterval {
        val margin = 1.96 * standardError // 95% confidence interval
        return ConfidenceInterval(
            lower = difference - margin,
            upper = difference + margin
        )
    }
    
    private fun generateTestId(): String {
        return "test_${System.currentTimeMillis()}_${Random.nextInt(1000)}"
    }
    
    /**
     * Persistence functions
     */
    private fun loadActiveTests() {
        // Implementation would load from encrypted storage
    }
    
    private fun saveActiveTests() {
        // Implementation would save to encrypted storage
    }
    
    private fun saveUserAssignments() {
        // Implementation would save to encrypted storage
    }
    
    private fun saveTestResults() {
        // Implementation would save to encrypted storage
    }
    
    /**
     * Clean up resources
     */
    fun cleanup() {
        scope.cancel()
    }
}

/**
 * Data classes for A/B testing
 */
@Serializable
data class ABTest(
    val id: String,
    val name: String,
    val description: String,
    val controlPolicy: PolicyType,
    val treatmentPolicy: PolicyType,
    val trafficSplit: Double,
    val startTime: Long,
    val endTime: Long,
    val minimumSampleSize: Int,
    val metrics: List<TestMetric>,
    val status: TestStatus,
    val autoApply: Boolean = false
)

@Serializable
data class TestAssignment(
    val userId: String,
    val testId: String,
    val group: TestGroup,
    val assignmentTime: Long,
    val policy: PolicyType
)

@Serializable
data class TestMetric(
    val name: String,
    val description: String,
    val type: MetricType
)

@Serializable
data class OutcomeRecord(
    val userId: String,
    val testId: String,
    val group: TestGroup,
    val intervention: Intervention,
    val outcome: InterventionOutcome,
    val timestamp: Long
)

@Serializable
data class SatisfactionRecord(
    val userId: String,
    val testId: String,
    val group: TestGroup,
    val rating: Double,
    val feedback: String?,
    val timestamp: Long
)

@Serializable
data class BehavioralChangeRecord(
    val userId: String,
    val testId: String,
    val group: TestGroup,
    val metrics: Map<String, Double>,
    val timestamp: Long
)

data class TestResults(
    val testId: String,
    val controlOutcomes: MutableList<OutcomeRecord>,
    val treatmentOutcomes: MutableList<OutcomeRecord>,
    val controlMetrics: MutableMap<String, Double>,
    val treatmentMetrics: MutableMap<String, Double>,
    val controlSatisfaction: MutableList<SatisfactionRecord> = mutableListOf(),
    val treatmentSatisfaction: MutableList<SatisfactionRecord> = mutableListOf(),
    val controlBehavioralChanges: MutableList<BehavioralChangeRecord> = mutableListOf(),
    val treatmentBehavioralChanges: MutableList<BehavioralChangeRecord> = mutableListOf()
)

data class TestAnalysis(
    val testId: String,
    val testName: String,
    val analysisTime: Long,
    val sampleSizes: SampleSizes,
    val metricAnalysis: Map<String, MetricAnalysis>,
    val satisfactionAnalysis: SatisfactionAnalysis,
    val behavioralChangeAnalysis: BehavioralChangeAnalysis,
    var overallSignificance: Boolean,
    var recommendation: TestRecommendation
)

data class MetricAnalysis(
    val metricName: String,
    val controlMean: Double,
    val treatmentMean: Double,
    val difference: Double,
    val percentChange: Double,
    val pValue: Double,
    val isSignificant: Boolean,
    val confidenceInterval: ConfidenceInterval
)

data class SatisfactionAnalysis(
    val controlMean: Double,
    val treatmentMean: Double,
    val difference: Double,
    val sampleSizes: SampleSizes,
    val isSignificant: Boolean
)

data class BehavioralChangeAnalysis(
    val metricDifferences: Map<String, Double>,
    val sampleSizes: SampleSizes,
    val significantChanges: List<String>
)

data class SampleSizes(
    val control: Int,
    val treatment: Int
)

data class ConfidenceInterval(
    val lower: Double,
    val upper: Double
)

/**
 * Enums for A/B testing
 */
enum class PolicyType {
    RULE_BASED,
    RULE_BASED_ENHANCED,
    RL_POLICY,
    HYBRID
}

enum class TestGroup {
    CONTROL,
    TREATMENT
}

enum class TestStatus {
    ACTIVE,
    PAUSED,
    COMPLETED,
    CANCELLED
}

enum class TestRecommendation {
    CONTINUE,
    ADOPT_TREATMENT,
    KEEP_CONTROL,
    NO_DIFFERENCE
}

enum class MetricType {
    EFFECTIVENESS,
    ENGAGEMENT,
    SATISFACTION,
    BEHAVIORAL_CHANGE
}