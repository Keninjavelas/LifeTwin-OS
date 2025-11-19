package com.lifetwin.mlp.screenevents

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent

class ScreenEventReceiver : BroadcastReceiver() {

    override fun onReceive(context: Context, intent: Intent) {
        when (intent.action) {
            Intent.ACTION_SCREEN_ON -> {
                // TODO: log session start into local DB / RN bridge
            }
            Intent.ACTION_SCREEN_OFF -> {
                // TODO: log session end and compute session duration
            }
        }
    }
}
