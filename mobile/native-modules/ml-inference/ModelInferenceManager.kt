package com.lifetwin.mlp.ml

import android.content.Context
import android.util.Log
import com.lifetwin.mlp.db.*
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*
import org.json.JSONObject
import org.json.JSONArray
import java.io.File
import java.io.FileInputStream
import java.io.InputStream
import java.util.concurrent.ConcurrentHashMap

private const val TAG = "ModelInferenceManager"

/**
 * Manages on-device ML model inference for next-app prediction and time-series forecasting.
 * Integrates trained models with Android data collection system.
 */
class ModelInferenceManager(private val context: Context) {
    
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    
    // Model instances
    private var nextAppModel: NextAppPredictor? = null
    private var timeSeriesModel: TimeSeriesForecaster? = null
    
    // Model metadata
    private val modelMetadata = ConcurrentHashMap<String, ModelMetadata>()
    
    // Inference cache
    private val predictionCache = ConcurrentHashMap<String, CachedPrediction>()
    
    // State flows for predictions
    private val _nextAppPredictions = MutableStateFlow<List<AppPrediction>>(emptyList())
    val nextAppPredictions: StateFlow<List<AppPrediction>> = _nextAppPredictions.asStateFlow()
    
    private val _timeSeriesPredictions = MutableStateFlow<TimeSeriesPrediction?>(null)
    val timeSeriesPredictions: StateFlow<TimeSeriesPrediction?> = _timeSeriesPredictions.asStateFlow()
    
    /**
     * Initialize the model inference manager
     */
    suspend fun initialize(): Boolean {
        return try {
            Log.i(TAG, "Initializing ModelInferenceManager...")
            
            // Load available models
            loadAvailableModels()
            
            // Initialize models
            initializeModels()
            
            Log.i(TAG, "ModelInferenceManager initialized successfully")
            true
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize ModelInferenceManager", e)
            false
        }
    }
    
    /**
     * Load metadata for available models
     */
    private suspend fun loadAvailableModels() {
        val modelsDir = File(context.filesDir, "ml_models")
        if (!modelsDir.exists()) {
            Log.w(TAG, "Models directory not found: ${modelsDir.absolutePath}")
            return
        }
        
        // Load next-app model metadata
        val nextAppMetaFile = File(modelsDir, "next_app_model.json")
        if (nextAppMetaFile.exists()) {
            try {
                val metadata = loadModelMetadata(nextAppMetaFile)
                modelMetadata["next_app"] = metadata
                Log.i(TAG, "Loaded next-app model metadata: ${metadata.modelName}")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to load next-app model metadata", e)
            }
        }
        
        // Load time-series model metadata
        val timeSeriesMetaFile = File(modelsDir, "time_series_twin.json")
        if (timeSeriesMetaFile.exists()) {
            try {
                val metadata = loadModelMetadata(timeSeriesMetaFile)
                modelMetadata["time_series"] = metadata
                Log.i(TAG, "Loaded time-series model metadata: ${metadata.modelName}")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to load time-series model metadata", e)
            }
        }
    }
    
    /**
     * Initialize model instances
     */
    private suspend fun initializeModels() {
        // Initialize next-app predictor
        modelMetadata["next_app"]?.let { metadata ->
            try {
                nextAppModel = NextAppPredictor(context, metadata)
                nextAppModel?.initialize()
                Log.i(TAG, "Next-app predictor initialized")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to initialize next-app predictor", e)
            }
        }
        
        // Initialize time-series forecaster
        modelMetadata["time_series"]?.let { metadata ->
            try {
                timeSeriesModel = TimeSeriesForecaster(context, metadata)
                timeSeriesModel?.initialize()
                Log.i(TAG, "Time-series forecaster initialized")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to initialize time-series forecaster", e)
            }
        }
    }
    
    /**
     * Predict next app based on current usage sequence
     */
    suspend fun predictNextApp(currentSequence: List<String>, topK: Int = 5): List<AppPrediction> {
        return try {
            val cacheKey = "next_app_${currentSequence.hashCode()}_$topK"
            
            // Check cache
            predictionCache[cacheKey]?.let { cached ->
                if (System.currentTimeMillis() - cached.timestamp < 60000) { // 1 minute cache
                    return cached.predictions as List<AppPrediction>
                }
            }
            
            val predictions = nextAppModel?.predict(currentSequence, topK) ?: emptyList()
            
            // Cache predictions
            predictionCache[cacheKey] = CachedPrediction(
                predictions = predictions,
                timestamp = System.currentTimeMillis()
            )
            
            // Update state flow
            _nextAppPredictions.value = predictions
            
            Log.d(TAG, "Generated ${predictions.size} next-app predictions")
            predictions
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to predict next app", e)
            emptyList()
        }
    }
    
    /**
     * Predict time-series metrics (screen time, energy, focus, mood)
     */
    suspend fun predictTimeSeries(historicalData: List<DailySummaryEntity>, 
                                 forecastDays: Int = 7): TimeSeriesPrediction? {
        return try {
            val cacheKey = "time_series_${historicalData.hashCode()}_$forecastDays"
            
            // Check cache
            predictionCache[cacheKey]?.let { cached ->
                if (System.currentTimeMillis() - cached.timestamp < 3600000) { // 1 hour cache
                    return cached.predictions as TimeSeriesPrediction
                }
            }
            
            val prediction = timeSeriesModel?.predict(historicalData, forecastDays)
            
            // Cache prediction
            if (prediction != null) {
                predictionCache[cacheKey] = CachedPrediction(
                    predictions = prediction,
                    timestamp = System.currentTimeMillis()
                )
                
                // Update state flow
                _timeSeriesPredictions.value = prediction
            }
            
            Log.d(TAG, "Generated time-series prediction for $forecastDays days")
            prediction
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to predict time series", e)
            null
        }
    }
    
    /**
     * Get current app usage sequence from database
     */
    suspend fun getCurrentAppSequence(maxLength: Int = 10): List<String> {
        return try {
            val database = AppDatabase.getInstance(context)
            val currentTime = System.currentTimeMillis()
            val lookbackTime = currentTime - (24 * 60 * 60 * 1000L) // Last 24 hours
            
            val recentEvents = database.usageEventDao().getEventsByTimeRange(lookbackTime, currentTime)
            
            // Create sequence from recent events
            val sequence = recentEvents
                .filter { it.totalTimeInForeground > 5000 } // At least 5 seconds
                .sortedBy { it.startTime }
                .map { it.packageName }
                .distinct()
                .takeLast(maxLength)
            
            Log.d(TAG, "Current app sequence: ${sequence.size} apps")
            sequence
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to get current app sequence", e)
            emptyList()
        }
    }
    
    /**
     * Get historical data for time-series prediction
     */
    suspend fun getHistoricalData(days: Int = 30): List<DailySummaryEntity> {
        return try {
            val database = AppDatabase.getInstance(context)
            val endTime = System.currentTimeMillis()
            val startTime = endTime - (days * 24 * 60 * 60 * 1000L)
            
            val summaries = database.dailySummaryDao().getSummariesByDateRange(startTime, endTime)
            
            Log.d(TAG, "Retrieved ${summaries.size} days of historical data")
            summaries
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to get historical data", e)
            emptyList()
        }
    }
    
    /**
     * Run periodic predictions
     */
    fun startPeriodicPredictions() {
        scope.launch {
            while (true) {
                try {
                    // Update next-app predictions every 5 minutes
                    val currentSequence = getCurrentAppSequence()
                    if (currentSequence.isNotEmpty()) {
                        predictNextApp(currentSequence)
                    }
                    
                    // Update time-series predictions every hour
                    val currentMinute = System.currentTimeMillis() / (60 * 1000) % 60
                    if (currentMinute == 0L) { // Top of the hour
                        val historicalData = getHistoricalData()
                        if (historicalData.isNotEmpty()) {
                            predictTimeSeries(historicalData)
                        }
                    }
                    
                    delay(5 * 60 * 1000) // 5 minutes
                    
                } catch (e: Exception) {
                    Log.e(TAG, "Error in periodic predictions", e)
                    delay(60 * 1000) // Wait 1 minute before retrying
                }
            }
        }
    }
    
    /**
     * Get model performance metrics
     */
    fun getModelMetrics(): Map<String, Any> {
        val metrics = mutableMapOf<String, Any>()
        
        modelMetadata["next_app"]?.let { metadata ->
            metrics["next_app_model"] = mapOf(
                "accuracy" to metadata.metrics.getOrDefault("test_accuracy", 0.0),
                "vocab_size" to metadata.vocabSize,
                "framework" to metadata.framework
            )
        }
        
        modelMetadata["time_series"]?.let { metadata ->
            metrics["time_series_model"] = mapOf(
                "mae" to metadata.metrics.getOrDefault("overall_mae", 0.0),
                "r2" to metadata.metrics.getOrDefault("overall_r2", 0.0),
                "framework" to metadata.framework
            )
        }
        
        metrics["cache_size"] = predictionCache.size
        metrics["models_loaded"] = modelMetadata.size
        
        return metrics
    }
    
    /**
     * Clear prediction cache
     */
    fun clearCache() {
        predictionCache.clear()
        Log.i(TAG, "Prediction cache cleared")
    }
    
    /**
     * Cleanup resources
     */
    fun cleanup() {
        scope.cancel()
        nextAppModel?.cleanup()
        timeSeriesModel?.cleanup()
        predictionCache.clear()
        Log.i(TAG, "ModelInferenceManager cleaned up")
    }
    
    // Helper methods
    
    private fun loadModelMetadata(file: File): ModelMetadata {
        val json = JSONObject(file.readText())
        
        return ModelMetadata(
            modelName = json.getString("model_name"),
            modelType = json.getString("model_type"),
            framework = json.getString("framework"),
            vocabSize = json.optInt("vocab_size", 0),
            maxSeqLength = json.optInt("max_seq_length", 0),
            targetMetrics = json.optJSONArray("target_metrics")?.let { array ->
                (0 until array.length()).map { array.getString(it) }
            } ?: emptyList(),
            metrics = json.optJSONObject("final_metrics")?.let { metricsObj ->
                metricsObj.keys().asSequence().associateWith { key ->
                    metricsObj.getDouble(key)
                }
            } ?: emptyMap(),
            savedAt = json.getString("saved_at")
        )
    }
    
    // Data classes
    
    data class ModelMetadata(
        val modelName: String,
        val modelType: String,
        val framework: String,
        val vocabSize: Int,
        val maxSeqLength: Int,
        val targetMetrics: List<String>,
        val metrics: Map<String, Double>,
        val savedAt: String
    )
    
    data class AppPrediction(
        val packageName: String,
        val probability: Float,
        val rank: Int
    )
    
    data class TimeSeriesPrediction(
        val forecastDays: Int,
        val predictions: Map<String, List<Float>>, // metric -> daily predictions
        val confidence: Map<String, List<Float>>, // metric -> confidence intervals
        val generatedAt: Long = System.currentTimeMillis()
    )
    
    private data class CachedPrediction(
        val predictions: Any,
        val timestamp: Long
    )
}

/**
 * Next-app prediction model wrapper
 */
class NextAppPredictor(
    private val context: Context,
    private val metadata: ModelInferenceManager.ModelMetadata
) {
    
    private var vocabulary: Map<String, Int> = emptyMap()
    private var reverseVocab: Map<Int, String> = emptyMap()
    
    suspend fun initialize(): Boolean {
        return try {
            // Load vocabulary
            val vocabFile = File(context.filesDir, "ml_models/vocab.json")
            if (vocabFile.exists()) {
                val vocabJson = JSONObject(vocabFile.readText())
                vocabulary = vocabJson.keys().asSequence().associateWith { vocabJson.getInt(it) }
                reverseVocab = vocabulary.entries.associate { it.value to it.key }
                Log.i(TAG, "Loaded vocabulary with ${vocabulary.size} entries")
            }
            
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize NextAppPredictor", e)
            false
        }
    }
    
    suspend fun predict(sequence: List<String>, topK: Int): List<ModelInferenceManager.AppPrediction> {
        return try {
            // Convert sequence to IDs
            val sequenceIds = sequence.mapNotNull { app ->
                vocabulary[app] ?: vocabulary["<UNK>"]
            }
            
            if (sequenceIds.isEmpty()) {
                return emptyList()
            }
            
            // For now, return mock predictions based on vocabulary frequency
            // In a real implementation, this would use ONNX Runtime or TensorFlow Lite
            val predictions = generateMockPredictions(sequenceIds, topK)
            
            predictions
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to generate next-app predictions", e)
            emptyList()
        }
    }
    
    private fun generateMockPredictions(sequenceIds: List<Int>, topK: Int): List<ModelInferenceManager.AppPrediction> {
        // Mock implementation - in reality would use trained model
        val commonApps = listOf(
            "com.android.chrome",
            "com.whatsapp",
            "com.instagram.android",
            "com.spotify.music",
            "com.google.android.youtube"
        )
        
        return commonApps.take(topK).mapIndexed { index, app ->
            ModelInferenceManager.AppPrediction(
                packageName = app,
                probability = (1.0f - index * 0.15f).coerceAtLeast(0.1f),
                rank = index + 1
            )
        }
    }
    
    fun cleanup() {
        // Cleanup model resources
    }
}

/**
 * Time-series forecasting model wrapper
 */
class TimeSeriesForecaster(
    private val context: Context,
    private val metadata: ModelInferenceManager.ModelMetadata
) {
    
    suspend fun initialize(): Boolean {
        return try {
            Log.i(TAG, "TimeSeriesForecaster initialized")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize TimeSeriesForecaster", e)
            false
        }
    }
    
    suspend fun predict(historicalData: List<DailySummaryEntity>, 
                       forecastDays: Int): ModelInferenceManager.TimeSeriesPrediction? {
        return try {
            if (historicalData.isEmpty()) {
                return null
            }
            
            // Extract time series data
            val screenTimes = historicalData.map { it.totalScreenTime.toFloat() }
            val energyLevels = historicalData.map { it.energyLevel }
            val focusLevels = historicalData.map { it.focusLevel }
            val moodLevels = historicalData.map { it.moodLevel }
            
            // Generate mock predictions - in reality would use trained model
            val predictions = mapOf(
                "screen_time" to generateTrendPrediction(screenTimes, forecastDays),
                "energy_level" to generateTrendPrediction(energyLevels, forecastDays),
                "focus_level" to generateTrendPrediction(focusLevels, forecastDays),
                "mood_level" to generateTrendPrediction(moodLevels, forecastDays)
            )
            
            val confidence = predictions.mapValues { (_, values) ->
                values.map { 0.8f } // Mock confidence intervals
            }
            
            ModelInferenceManager.TimeSeriesPrediction(
                forecastDays = forecastDays,
                predictions = predictions,
                confidence = confidence
            )
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to generate time-series predictions", e)
            null
        }
    }
    
    private fun generateTrendPrediction(historicalValues: List<Float>, forecastDays: Int): List<Float> {
        if (historicalValues.isEmpty()) return List(forecastDays) { 0.5f }
        
        // Simple trend-based prediction
        val recentValues = historicalValues.takeLast(7)
        val mean = recentValues.average().toFloat()
        val trend = if (recentValues.size >= 2) {
            (recentValues.last() - recentValues.first()) / recentValues.size
        } else {
            0f
        }
        
        return (1..forecastDays).map { day ->
            (mean + trend * day).coerceIn(0f, 1f)
        }
    }
    
    fun cleanup() {
        // Cleanup model resources
    }
}