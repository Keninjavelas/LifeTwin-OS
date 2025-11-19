package com.lifetwin.mlp.accessibility

// Stub AccessibilityService. Later will capture tap/scroll/gesture patterns
// and map them into high-level interaction events for ML.

import android.accessibilityservice.AccessibilityService
import android.view.accessibility.AccessibilityEvent

class InteractionAccessibilityService : AccessibilityService() {

    override fun onAccessibilityEvent(event: AccessibilityEvent?) {
        // TODO: inspect event types (TYPE_VIEW_CLICKED, TYPE_VIEW_SCROLLED, etc.)
        // and forward summarized interaction events to local storage / RN bridge.
    }

    override fun onInterrupt() {
        // no-op for now
    }
}
