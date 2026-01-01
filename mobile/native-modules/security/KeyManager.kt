package com.lifetwin.mlp.security

import android.content.Context
import android.content.SharedPreferences
import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyProperties
import android.util.Log
import java.security.KeyPairGenerator
import java.security.KeyStore
import java.security.SecureRandom
import javax.crypto.Cipher
import javax.crypto.KeyGenerator
import javax.crypto.SecretKey
import javax.crypto.spec.GCMParameterSpec
import javax.crypto.spec.SecretKeySpec
import android.util.Base64

private const val TAG = "KeyManager"
private const val KEYSTORE_PROVIDER = "AndroidKeyStore"
private const val DB_KEY_ALIAS = "lifetwin_db_key"
private const val ENCRYPTION_KEY_ALIAS = "lifetwin_encryption_key"
private const val PREFS_NAME = "lifetwin_key_prefs"
private const val WRAPPED_DB_KEY_PREF = "wrapped_db_key"
private const val DB_KEY_IV_PREF = "db_key_iv"

class KeyManager(private val context: Context) {
    
    private val sharedPrefs: SharedPreferences = 
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    /**
     * Initializes the key management system and returns the database passphrase
     * Creates keys if they don't exist, or retrieves existing ones
     */
    suspend fun initializeAndGetDatabasePassphrase(): String? {
        return try {
            // Ensure we have a master key in Android Keystore
            ensureMasterKeyExists()
            
            // Get or create the database encryption key
            val dbKey = getOrCreateDatabaseKey()
            
            // Convert to string for SQLCipher
            Base64.encodeToString(dbKey, Base64.NO_WRAP)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize key management", e)
            null
        }
    }

    /**
     * Ensures the master RSA key pair exists in Android Keystore
     */
    private fun ensureMasterKeyExists() {
        val keyStore = KeyStore.getInstance(KEYSTORE_PROVIDER)
        keyStore.load(null)
        
        if (!keyStore.containsAlias(ENCRYPTION_KEY_ALIAS)) {
            Log.i(TAG, "Creating new master key pair in Android Keystore")
            createMasterKeyPair()
        } else {
            Log.d(TAG, "Master key pair already exists")
        }
    }

    /**
     * Creates a new RSA key pair in Android Keystore for wrapping database keys
     */
    private fun createMasterKeyPair() {
        val keyPairGenerator = KeyPairGenerator.getInstance(
            KeyProperties.KEY_ALGORITHM_RSA, 
            KEYSTORE_PROVIDER
        )
        
        val spec = KeyGenParameterSpec.Builder(
            ENCRYPTION_KEY_ALIAS,
            KeyProperties.PURPOSE_ENCRYPT or KeyProperties.PURPOSE_DECRYPT
        )
            .setDigests(KeyProperties.DIGEST_SHA256, KeyProperties.DIGEST_SHA512)
            .setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_RSA_OAEP)
            .setKeySize(2048)
            .setUserAuthenticationRequired(false) // Allow background access
            .build()
            
        keyPairGenerator.initialize(spec)
        keyPairGenerator.generateKeyPair()
    }

    /**
     * Gets existing database key or creates a new one
     * The key is wrapped with the master key and stored in SharedPreferences
     */
    private fun getOrCreateDatabaseKey(): ByteArray {
        val wrappedKeyB64 = sharedPrefs.getString(WRAPPED_DB_KEY_PREF, null)
        val ivB64 = sharedPrefs.getString(DB_KEY_IV_PREF, null)
        
        return if (wrappedKeyB64 != null && ivB64 != null) {
            // Unwrap existing key
            Log.d(TAG, "Unwrapping existing database key")
            unwrapDatabaseKey(wrappedKeyB64, ivB64)
        } else {
            // Create new key
            Log.i(TAG, "Creating new database key")
            createAndWrapDatabaseKey()
        }
    }

    /**
     * Creates a new AES database key and wraps it with the master key
     */
    private fun createAndWrapDatabaseKey(): ByteArray {
        // Generate random 256-bit AES key
        val dbKey = ByteArray(32)
        SecureRandom().nextBytes(dbKey)
        
        // Wrap the key with the master key
        val keyStore = KeyStore.getInstance(KEYSTORE_PROVIDER)
        keyStore.load(null)
        
        val entry = keyStore.getEntry(ENCRYPTION_KEY_ALIAS, null) as KeyStore.PrivateKeyEntry
        val publicKey = entry.certificate.publicKey
        
        val cipher = Cipher.getInstance("RSA/ECB/OAEPWithSHA-256AndMGF1Padding")
        cipher.init(Cipher.ENCRYPT_MODE, publicKey)
        val wrappedKey = cipher.doFinal(dbKey)
        
        // Store wrapped key in SharedPreferences
        val wrappedKeyB64 = Base64.encodeToString(wrappedKey, Base64.NO_WRAP)
        sharedPrefs.edit()
            .putString(WRAPPED_DB_KEY_PREF, wrappedKeyB64)
            .apply()
            
        Log.d(TAG, "Database key created and wrapped successfully")
        return dbKey
    }

    /**
     * Unwraps the database key using the master private key
     */
    private fun unwrapDatabaseKey(wrappedKeyB64: String, ivB64: String): ByteArray {
        val keyStore = KeyStore.getInstance(KEYSTORE_PROVIDER)
        keyStore.load(null)
        
        val entry = keyStore.getEntry(ENCRYPTION_KEY_ALIAS, null) as KeyStore.PrivateKeyEntry
        val privateKey = entry.privateKey
        
        val wrappedKey = Base64.decode(wrappedKeyB64, Base64.NO_WRAP)
        
        val cipher = Cipher.getInstance("RSA/ECB/OAEPWithSHA-256AndMGF1Padding")
        cipher.init(Cipher.DECRYPT_MODE, privateKey)
        
        return cipher.doFinal(wrappedKey)
    }

    /**
     * Encrypts sensitive data using AES-GCM
     */
    fun encryptData(plaintext: String): EncryptedData? {
        return try {
            val dbKey = getOrCreateDatabaseKey()
            val secretKey = SecretKeySpec(dbKey, "AES")
            
            val cipher = Cipher.getInstance("AES/GCM/NoPadding")
            cipher.init(Cipher.ENCRYPT_MODE, secretKey)
            
            val iv = cipher.iv
            val ciphertext = cipher.doFinal(plaintext.toByteArray(Charsets.UTF_8))
            
            EncryptedData(
                ciphertext = Base64.encodeToString(ciphertext, Base64.NO_WRAP),
                iv = Base64.encodeToString(iv, Base64.NO_WRAP)
            )
        } catch (e: Exception) {
            Log.e(TAG, "Failed to encrypt data", e)
            null
        }
    }

    /**
     * Decrypts sensitive data using AES-GCM
     */
    fun decryptData(encryptedData: EncryptedData): String? {
        return try {
            val dbKey = getOrCreateDatabaseKey()
            val secretKey = SecretKeySpec(dbKey, "AES")
            
            val cipher = Cipher.getInstance("AES/GCM/NoPadding")
            val iv = Base64.decode(encryptedData.iv, Base64.NO_WRAP)
            val spec = GCMParameterSpec(128, iv)
            cipher.init(Cipher.DECRYPT_MODE, secretKey, spec)
            
            val ciphertext = Base64.decode(encryptedData.ciphertext, Base64.NO_WRAP)
            val plaintext = cipher.doFinal(ciphertext)
            
            String(plaintext, Charsets.UTF_8)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to decrypt data", e)
            null
        }
    }

    /**
     * Rotates the database key (creates new key and re-encrypts data)
     * This should be called periodically for security
     */
    suspend fun rotateDatabaseKey(): Boolean {
        return try {
            Log.i(TAG, "Starting database key rotation")
            
            // Clear existing wrapped key
            sharedPrefs.edit()
                .remove(WRAPPED_DB_KEY_PREF)
                .remove(DB_KEY_IV_PREF)
                .apply()
            
            // Create new key
            createAndWrapDatabaseKey()
            
            Log.i(TAG, "Database key rotation completed successfully")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to rotate database key", e)
            false
        }
    }

    /**
     * Clears all stored keys (for app uninstall or reset)
     */
    fun clearAllKeys() {
        try {
            // Clear SharedPreferences
            sharedPrefs.edit().clear().apply()
            
            // Remove keys from Android Keystore
            val keyStore = KeyStore.getInstance(KEYSTORE_PROVIDER)
            keyStore.load(null)
            
            if (keyStore.containsAlias(ENCRYPTION_KEY_ALIAS)) {
                keyStore.deleteEntry(ENCRYPTION_KEY_ALIAS)
            }
            
            Log.i(TAG, "All keys cleared successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to clear keys", e)
        }
    }

    /**
     * Validates that the key management system is working correctly
     */
    fun validateKeySystem(): Boolean {
        return try {
            val testData = "test_encryption_data_${System.currentTimeMillis()}"
            val encrypted = encryptData(testData)
            
            if (encrypted == null) {
                Log.e(TAG, "Key system validation failed: encryption returned null")
                return false
            }
            
            val decrypted = decryptData(encrypted)
            val isValid = decrypted == testData
            
            if (isValid) {
                Log.d(TAG, "Key system validation passed")
            } else {
                Log.e(TAG, "Key system validation failed: decryption mismatch")
            }
            
            isValid
        } catch (e: Exception) {
            Log.e(TAG, "Key system validation failed with exception", e)
            false
        }
    }
}

/**
 * Data class for encrypted data with IV
 */
data class EncryptedData(
    val ciphertext: String,
    val iv: String
)