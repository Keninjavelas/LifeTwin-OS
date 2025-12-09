package com.lifetwin.mlp.db

import android.app.Application

// Example Application showing how to initialize optional encrypted DB.
// Add this pattern to your real Application.onCreate and wire secure passphrase retrieval
// (Android Keystore or user-provided secret). Do not hardcode secrets in production.
class ExampleApplication : Application() {
    override fun onCreate() {
        super.onCreate()
        // Example: derive or fetch a secure passphrase here.
        val passphrase = "change-me-in-production"
        DBHelper.initializeEncrypted(this, passphrase)
    }
}
