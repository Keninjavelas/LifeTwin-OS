package com.lifetwin.mlp.notifications

// Stub NotificationListener for MLP documentation purposes.
// Later, this will record notification actions (posted, dismissed, opened)
// into the local event store and/or forward to the RN bridge.

import android.service.notification.NotificationListenerService
import android.service.notification.StatusBarNotification

class NotificationLogger : NotificationListenerService() {

    override fun onNotificationPosted(sbn: StatusBarNotification) {
        // TODO: persist or forward notification metadata (title, package, category)
    }

    override fun onNotificationRemoved(sbn: StatusBarNotification) {
        // TODO: record dismissal events for behavior modeling
    }
}
