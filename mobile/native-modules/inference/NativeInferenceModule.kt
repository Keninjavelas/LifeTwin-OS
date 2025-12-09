package com.lifetwin.mlp.inference

import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactContextBaseJavaModule
import com.facebook.react.bridge.ReactMethod
import com.facebook.react.bridge.ReadableArray
import com.facebook.react.bridge.Promise

/**
 * Minimal React Native module stub for next-app inference.
 *
 * This is a placeholder implementation that returns `null` for now.
 * Replace the placeholder logic with a real model inference call (ONNX/TorchMobile) later.
 */
class NativeInferenceModule(reactContext: ReactApplicationContext) : ReactContextBaseJavaModule(reactContext) {
    override fun getName(): String = "NativeInference"

    @ReactMethod
    fun reloadModel(promise: Promise) {
        try {
            val ctx = reactApplicationContext
            val loaded = try { NativeModelLoader.loadModel(ctx) } catch (_: Exception) { false }
            val status = mapOf(
                "modelPresent" to NativeModelLoader.modelExists(ctx),
                "onnxRuntimeAvailable" to NativeModelLoader.onnxRuntimeAvailable,
                "modelSessionLoaded" to NativeModelLoader.modelSessionLoaded,
                "loaderReturned" to loaded
            )
            promise.resolve(com.facebook.react.bridge.Arguments.makeNativeMap(status))
        } catch (e: Exception) {
            promise.reject("reload_error", e)
        }
    }

    @ReactMethod
    fun unloadModel(promise: Promise) {
        try {
            // Best-effort unload: mark session unloaded and release references if present.
            // If a real runtime is integrated, call the native session close APIs here.
            NativeModelLoader.modelSessionLoaded = false
            promise.resolve(mapOf("status" to "ok", "modelSessionLoaded" to false))
        } catch (e: Exception) {
            promise.reject("unload_error", e)
        }
    }

    @ReactMethod
    fun getSessionInfo(promise: Promise) {
        try {
            val ctx = reactApplicationContext
            val info = mapOf(
                "modelPresent" to NativeModelLoader.modelExists(ctx),
                "onnxRuntimeAvailable" to NativeModelLoader.onnxRuntimeAvailable,
                "modelSessionLoaded" to NativeModelLoader.modelSessionLoaded
            )
            promise.resolve(com.facebook.react.bridge.Arguments.makeNativeMap(info))
        } catch (e: Exception) {
            promise.reject("session_info_error", e)
        }
    }

    @ReactMethod
    fun runInference(history: ReadableArray, promise: Promise) {
        try {
            // This exposes a simple inference entrypoint. If a native runtime is integrated,
            // call it here and return model outputs. For the stub we call the existing heuristic.
            val seq = mutableListOf<String>()
            for (i in 0 until history.size()) {
                val v = history.getString(i)
                seq.add(v ?: "unknown")
            }

            val ctx = reactApplicationContext
            val loaded = try { NativeModelLoader.loadModel(ctx) } catch (_: Exception) { false }

            if (loaded && NativeModelLoader.modelSessionLoaded) {
                // Runtime present but actual inference not implemented.
                promise.resolve(null)
                return
            }

            // Fallback heuristic (same as predictNextApp)
            val vocabMap = try { NativeModelLoader.loadVocabMap(ctx) } catch (_: Exception) { null }
            val counts = HashMap<String, Int>()
            for (s in seq) counts[s] = (counts[s] ?: 0) + 1
            var candidate: String? = null
            if (vocabMap != null && vocabMap.isNotEmpty()) {
                candidate = counts.entries.filter { vocabMap.containsKey(it.key) }.maxByOrNull { it.value }?.key
            }
            val best = candidate ?: counts.maxByOrNull { it.value }?.key ?: seq.lastOrNull()
            promise.resolve(best)
        } catch (e: Exception) {
            promise.reject("run_inference_error", e)
        }
    }

    @ReactMethod
    fun getModelStatus(promise: Promise) {
        try {
            val ctx = reactApplicationContext
            val status = mapOf(
                "modelPresent" to NativeModelLoader.modelExists(ctx),
                "onnxRuntimeAvailable" to NativeModelLoader.onnxRuntimeAvailable,
                "modelSessionLoaded" to NativeModelLoader.modelSessionLoaded
            )
            promise.resolve(com.facebook.react.bridge.Arguments.makeNativeMap(status))
        } catch (e: Exception) {
            promise.reject("status_error", e)
        }
    }

    @ReactMethod
    fun predictNextApp(history: ReadableArray, promise: Promise) {
        try {
            // Convert ReadableArray -> List<String>
            val seq = mutableListOf<String>()
            for (i in 0 until history.size()) {
                val v = history.getString(i)
                seq.add(v ?: "unknown")
            }

            // If a model loader is present and can load, we'd call into the model runtime here.
            // For now: try to load model (checks for file) and if not present, use a simple heuristic:
            val ctx = reactApplicationContext
            val loaded = try { NativeModelLoader.loadModel(ctx) } catch (_: Exception) { false }

            if (loaded) {
                // Model runtime not implemented in this stub; return null to indicate no prediction
                promise.resolve(null)
                return
            }

            // Heuristic fallback: prefer packages present in vocab (if available),
            // else return the most frequent package or the last seen.
            val vocabSet = try { NativeModelLoader.loadVocabSet(ctx) } catch (_: Exception) { null }
            val vocabMap = try { NativeModelLoader.loadVocabMap(ctx) } catch (_: Exception) { null }

            val counts = HashMap<String, Int>()
            for (s in seq) counts[s] = (counts[s] ?: 0) + 1

            var candidate: String? = null
            if (vocabMap != null && vocabMap.isNotEmpty()) {
                // choose most frequent token that exists in vocab map
                candidate = counts.entries.filter { vocabMap.containsKey(it.key) }.maxByOrNull { it.value }?.key
            } else if (vocabSet != null && vocabSet.isNotEmpty()) {
                candidate = counts.entries.filter { vocabSet.contains(it.key) }.maxByOrNull { it.value }?.key
            }
            val best = candidate ?: counts.maxByOrNull { it.value }?.key ?: seq.lastOrNull()
            promise.resolve(best)
        } catch (e: Exception) {
            promise.reject("inference_error", e)
        }
    }
}
