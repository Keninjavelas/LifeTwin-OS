package com.lifetwin.mlp.screenevents

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import com.lifetwin.mlp.db.AppDatabase
import com.lifetwin.mlp.db.AppEventEntity
import com.lifetwin.mlp.db.DBHelper

class ScreenEventReceiver : BroadcastReceiver() {

    override fun onReceive(context: Context, intent: Intent) {
        val db = AppDatabase.getInstance(context)
        val executor = java.util.concurrent.Executors.newSingleThreadExecutor()
        when (intent.action) {
            Intent.ACTION_SCREEN_ON -> {
                DBHelper.insertEventAsync(context, AppEventEntity(timestamp = System.currentTimeMillis(), type = "screen", packageName = null))
            }
            Intent.ACTION_SCREEN_OFF -> {
                DBHelper.insertEventAsync(context, AppEventEntity(timestamp = System.currentTimeMillis(), type = "screen_off", packageName = null))
            }
        }
    }
}
