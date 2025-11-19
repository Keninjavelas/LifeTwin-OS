package com.lifetwin.mlp.summaries

// Stub WorkManager worker that will aggregate raw events into daily/weekly summaries.

import android.content.Context
import androidx.work.Worker
import androidx.work.WorkerParameters

class DailySummaryWorker(appContext: Context, params: WorkerParameters) : Worker(appContext, params) {

    override fun doWork(): Result {
        // TODO: read raw events from local DB, aggregate into daily_summary rows, and
        // optionally trigger sync with backend.
        return Result.success()
    }
}
