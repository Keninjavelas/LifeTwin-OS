package com.lifetwin.mlp.inference

import android.content.Context
import android.util.Log

/**
 * Lightweight model loader stub.
 *
 * Looks for `next_app_model.onnx` in the app's files directory and reports
 * whether a model file exists. Replace the placeholder loader with actual
 * ONNX/ORT or TorchMobile loading code when integrating a runtime.
 */
object NativeModelLoader {
    private const val TAG = "NativeModelLoader"
    // Flags set when attempting to load runtime/model. Kept simple for now.
    var onnxRuntimeAvailable: Boolean = false
        private set
    var modelSessionLoaded: Boolean = false
        private set
    // Hold references to environment/session when created via reflection
    private var ortEnvironment: Any? = null
    private var ortSession: Any? = null

    fun modelExists(context: Context): Boolean {
        try {
            val f = context.filesDir.resolve("next_app_model.onnx")
            return f.exists()
        } catch (e: Exception) {
            Log.w(TAG, "modelExists check failed: ${e.message}")
            return false
        }
    }

    private fun copyAssetIfPresent(context: Context): Boolean {
        try {
            val assetName = "next_app_model.onnx"
            val assetManager = context.assets
            val assetList = try { assetManager.list("")?.toList() ?: emptyList() } catch (_: Exception) { emptyList<String>() }
            if (!assetList.contains(assetName)) return false

            val outFile = context.filesDir.resolve(assetName)
            if (outFile.exists()) return true

            assetManager.open(assetName).use { input ->
                outFile.outputStream().use { output ->
                    input.copyTo(output)
                }
            }
            Log.i(TAG, "Copied asset $assetName to ${outFile.absolutePath}")
            return true
        } catch (e: Exception) {
            Log.w(TAG, "copyAssetIfPresent failed: ${e.message}")
            return false
        }
    }

    fun loadModel(context: Context): Boolean {
        var exists = modelExists(context)
        if (!exists) {
            // Try to copy from packaged assets (useful for debug builds where a sample
            // model is bundled in assets). If copy succeeds, update exists.
            val copied = copyAssetIfPresent(context)
            if (copied) {
                exists = modelExists(context)
            }
        }

        if (!exists) {
            Log.i(TAG, "No model file found in filesDir or assets; skipping load")
            return false
        }

        // Quick non-breaking detection of ONNX Runtime (ai.onnxruntime.OrtEnvironment)
        try {
            Class.forName("ai.onnxruntime.OrtEnvironment")
            onnxRuntimeAvailable = true
            Log.i(TAG, "ONNX Runtime classes detected on classpath")
        } catch (e: ClassNotFoundException) {
            onnxRuntimeAvailable = false
            Log.i(TAG, "ONNX Runtime not found on classpath; loader remains a stub")
        }

        // If the ONNX runtime is present, attempt to create an OrtEnvironment and session via reflection.
        if (onnxRuntimeAvailable) {
            try {
                val envCls = Class.forName("ai.onnxruntime.OrtEnvironment")

                // Try several common static entrypoints to obtain an environment instance
                val env: Any? = try {
                    // common: OrtEnvironment.getEnvironment()
                    envCls.getMethod("getEnvironment").invoke(null)
                } catch (_: Exception) {
                    try {
                        // fallback: OrtEnvironment.getInstance()
                        envCls.getMethod("getInstance").invoke(null)
                    } catch (_: Exception) {
                        try {
                            // fallback: OrtEnvironment.createEnvironment()
                            envCls.getMethod("createEnvironment").invoke(null)
                        } catch (ex: Exception) {
                            Log.w(TAG, "Could not obtain OrtEnvironment via known static methods: ${ex.message}")
                            null
                        }
                    }
                }

                if (env != null) {
                    val modelFilePath = context.filesDir.resolve("next_app_model.onnx").absolutePath

                    // Search for a createSession method on the env instance with common signatures
                    val methods = env::class.java.methods.filter { it.name == "createSession" }
                    val candidate = methods.firstOrNull { m ->
                        val params = m.parameterTypes
                        // accept createSession(String), createSession(String, SessionOptions), createSession(File)
                        params.size >= 1 && (params[0] == java.lang.String::class.java || params[0].name.contains("File") || params[0].name.contains("String"))
                    }

                    if (candidate != null) {
                        try {
                            val session = when (candidate.parameterTypes.size) {
                                1 -> candidate.invoke(env, modelFilePath)
                                2 -> {
                                    // try to construct SessionOptions if available, else call single-arg
                                    val sessionOptionsCls = try { Class.forName("ai.onnxruntime.OrtSession\$SessionOptions") } catch (_: Exception) { null }
                                    val opts = sessionOptionsCls?.getConstructor()?.newInstance()
                                    if (opts != null) candidate.invoke(env, modelFilePath, opts) else candidate.invoke(env, modelFilePath)
                                }
                                else -> candidate.invoke(env, modelFilePath)
                            }
                            if (session != null) {
                                // store environment and session references for later use
                                ortEnvironment = env
                                ortSession = session
                                modelSessionLoaded = true
                                Log.i(TAG, "ONNX Runtime session created successfully via reflection")
                                return true
                            }
                        } catch (e: Exception) {
                            Log.w(TAG, "ONNX session creation via reflection failed: ${e.message}")
                        }
                    } else {
                        Log.w(TAG, "No suitable createSession method found on OrtEnvironment via reflection")
                    }
                }
            } catch (e: Exception) {
                Log.w(TAG, "ONNX reflective load failed: ${e.message}")
            }
        }

        // Fallback: mark model present but no runtime session
        modelSessionLoaded = false
        return true
    }

    fun unloadSession(): Boolean {
        try {
            ortSession?.let { s ->
                // try to call close() or release() if present
                try {
                    val closeMethod = s::class.java.methods.firstOrNull { it.name == "close" || it.name == "release" }
                    closeMethod?.invoke(s)
                } catch (_: Exception) {
                }
            }
            ortSession = null
            ortEnvironment = null
            modelSessionLoaded = false
            Log.i(TAG, "Unloaded ONNX runtime session")
            return true
        } catch (e: Exception) {
            Log.w(TAG, "unloadSession failed: ${e.message}")
            return false
        }
    }

    fun runInference(context: Context, inputs: List<String>): String? {
        try {
            // If a reflective ORT session exists, attempt a best-effort run
            val s = ortSession ?: return null

            // Try to find a run method that accepts a Map
            val runMethod = s::class.java.methods.firstOrNull { it.name == "run" && it.parameterTypes.any { p -> p.name.contains("Map") || p.name.contains("java.util.Map") } }
            if (runMethod != null) {
                try {
                    val inputMap = java.util.HashMap<String, Any>()
                    // Best-effort: no typed tensor creation (requires OnnxTensor); try empty map to get outputs
                    val out = runMethod.invoke(s, inputMap)
                    if (out != null) return out.toString()
                } catch (e: Exception) {
                    Log.w(TAG, "runInference via reflection failed: ${e.message}")
                }
            } else {
                Log.w(TAG, "No suitable run(Map) method found on OrtSession via reflection")
            }
        } catch (e: Exception) {
            Log.w(TAG, "runInference failed: ${e.message}")
        }
        return null
    }

    fun loadVocabSet(context: Context): Set<String>? {
        val map = loadVocabMap(context) ?: return null
        return map.keys
    }

    fun loadVocabMap(context: Context): Map<String, Int>? {
        try {
            val f = context.filesDir.resolve("vocab.json")
            if (!f.exists()) return null
            val text = f.readText()
            val obj = org.json.JSONObject(text)
            val keys = obj.keys()
            val map = HashMap<String, Int>()
            while (keys.hasNext()) {
                val k = keys.next()
                val v = try { obj.getInt(k) } catch (_: Exception) { -1 }
                map[k] = v
            }
            return map
        } catch (e: Exception) {
            Log.w(TAG, "loadVocabMap failed: ${e.message}")
            return null
        }
    }
}
