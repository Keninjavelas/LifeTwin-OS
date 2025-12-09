package com.lifetwin.mlp.security

import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactContextBaseJavaModule
import com.facebook.react.bridge.ReactMethod
import com.facebook.react.bridge.Promise
import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyProperties
import java.security.KeyPairGenerator
import android.util.Log

private const val TAG = "KeystoreModule"

class KeystoreModule(reactContext: ReactApplicationContext) : ReactContextBaseJavaModule(reactContext) {
    override fun getName(): String = "KeystoreModule"

    @ReactMethod
    fun generateKeyPair(alias: String, promise: Promise) {
        try {
            val keyPairGenerator = KeyPairGenerator.getInstance(
                KeyProperties.KEY_ALGORITHM_RSA, "AndroidKeyStore"
            )
            val spec = KeyGenParameterSpec.Builder(
                alias,
                KeyProperties.PURPOSE_SIGN or KeyProperties.PURPOSE_VERIFY
            )
                .setDigests(KeyProperties.DIGEST_SHA256)
                .setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_RSA_PKCS1)
                .build()
            keyPairGenerator.initialize(spec)
            keyPairGenerator.generateKeyPair()
            promise.resolve(mapOf("status" to "ok", "alias" to alias))
        } catch (e: Exception) {
            Log.w(TAG, "generateKeyPair failed: ${e.message}")
            promise.reject("keystore_error", e)
        }
    }

    @ReactMethod
    fun generateWrappedDataKey(alias: String, promise: Promise) {
        try {
            val keyStore = java.security.KeyStore.getInstance("AndroidKeyStore")
            keyStore.load(null)

            val entry = keyStore.getEntry(alias, null) as? java.security.KeyStore.PrivateKeyEntry
            if (entry == null) {
                promise.reject("keystore_error", "Key pair not found for alias: $alias")
                return
            }

            val publicKey = entry.certificate.publicKey

            // Generate random 32-byte DEK
            val dek = ByteArray(32)
            java.security.SecureRandom().nextBytes(dek)

            // Encrypt (wrap) with RSA public key using PKCS1 padding
            val cipher = javax.crypto.Cipher.getInstance("RSA/ECB/PKCS1Padding")
            cipher.init(javax.crypto.Cipher.ENCRYPT_MODE, publicKey)
            val wrapped = cipher.doFinal(dek)

            val wrappedB64 = android.util.Base64.encodeToString(wrapped, android.util.Base64.NO_WRAP)

            // Return wrapped DEK (base64). The raw DEK is not returned for security.
            promise.resolve(mapOf("wrapped" to wrappedB64))
        } catch (e: Exception) {
            Log.w(TAG, "generateWrappedDataKey failed: ${e.message}")
            promise.reject("keystore_error", e)
        }
    }

    @ReactMethod
    fun unwrapDataKey(alias: String, wrappedB64: String, promise: Promise) {
        try {
            val keyStore = java.security.KeyStore.getInstance("AndroidKeyStore")
            keyStore.load(null)

            val entry = keyStore.getEntry(alias, null) as? java.security.KeyStore.PrivateKeyEntry
            if (entry == null) {
                promise.reject("keystore_error", "Key pair not found for alias: $alias")
                return
            }

            val privateKey = entry.privateKey
            val wrapped = android.util.Base64.decode(wrappedB64, android.util.Base64.NO_WRAP)
            val cipher = javax.crypto.Cipher.getInstance("RSA/ECB/PKCS1Padding")
            cipher.init(javax.crypto.Cipher.DECRYPT_MODE, privateKey)
            val dek = cipher.doFinal(wrapped)

            val dekB64 = android.util.Base64.encodeToString(dek, android.util.Base64.NO_WRAP)
            promise.resolve(mapOf("dek" to dekB64))
        } catch (e: Exception) {
            Log.w(TAG, "unwrapDataKey failed: ${e.message}")
            promise.reject("keystore_error", e)
        }
    }
}
