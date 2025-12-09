package com.lifetwin.mlp.accessibility

import android.accessibilityservice.AccessibilityService
import android.view.accessibility.AccessibilityEvent
import android.util.Log
import com.lifetwin.mlp.db.AppEventEntity
import com.lifetwin.mlp.db.DBHelper

/**
 * Conservative AccessibilityService stub.
 * - Records only that an interaction occurred (package name + timestamp).
 * - Does NOT capture event text to avoid sensitive data collection.
 */
class InteractionAccessibilityService : AccessibilityService() {
    override fun onAccessibilityEvent(event: AccessibilityEvent?) {
        try {
            val pkg = event?.packageName?.toString()
            val type = when (event?.eventType) {
                AccessibilityEvent.TYPE_VIEW_CLICKED -> "view_clicked"
                AccessibilityEvent.TYPE_VIEW_SCROLLED -> "view_scrolled"
                AccessibilityEvent.TYPE_VIEW_TEXT_CHANGED -> "text_changed"
                else -> "accessibility_event"
            }

            // Only emit a generic event; do not persist text or sensitive content
            val e = AppEventEntity(timestamp = System.currentTimeMillis(), type = type, packageName = pkg)
            DBHelper.insertEventAsync(applicationContext, e)
        } catch (ex: Exception) {
            Log.w("InteractionAccessibility", "failed to record event", ex)
        }
    }

    override fun onInterrupt() {
        // no-op
    }
}
