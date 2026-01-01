package com.lifetwin.mlp.validation

import android.content.Context
import android.util.Log
import kotlinx.coroutines.*

private const val TAG = "ValidationRunner"

/**
 * Simple runner for executing system validation
 * Can be called from tests or as part of deployment verification
 */
object ValidationRunner {
    
    /**
     * Runs complete system validation and returns the report
     */
    suspend fun runValidation(context: Context): SystemValidation.ValidationReport {
        Log.i(TAG, "Starting Data Collection & Intelligence system validation...")
        
        val validation = SystemValidation(context)
        val report = validation.runCompleteValidation()
        
        // Log summary
        Log.i(TAG, "Validation completed:")
        Log.i(TAG, "Total tests: ${report.totalTests}")
        Log.i(TAG, "Passed: ${report.passedTests}")
        Log.i(TAG, "Failed: ${report.failedTests}")
        Log.i(TAG, "Success rate: ${String.format("%.1f", report.successRate)}%")
        
        if (report.overallPassed) {
            Log.i(TAG, "✅ All validation tests passed - system ready for deployment")
        } else {
            Log.w(TAG, "❌ Some validation tests failed - review required")
            
            // Log failed tests
            report.results.filter { !it.passed }.forEach { result ->
                Log.w(TAG, "Failed: ${result.testName} - ${result.message}")
            }
        }
        
        return report
    }
    
    /**
     * Runs validation and prints detailed report to console
     */
    suspend fun runValidationWithDetailedOutput(context: Context) {
        val report = runValidation(context)
        
        println("\n" + "=".repeat(80))
        println(report.summary)
        println("=".repeat(80))
        
        if (!report.overallPassed) {
            println("\nDetailed failure information:")
            println("-".repeat(40))
            
            report.results.filter { !it.passed }.forEach { result ->
                println("\n❌ ${result.testName}")
                println("   Message: ${result.message}")
                if (result.details.isNotEmpty()) {
                    println("   Details:")
                    result.details.forEach { (key, value) ->
                        println("     $key: $value")
                    }
                }
            }
        }
        
        println("\n" + "=".repeat(80))
    }
}

/**
 * Example usage in a test or main function
 */
suspend fun main() {
    // This would typically be called from an Android test or application context
    // ValidationRunner.runValidationWithDetailedOutput(context)
}