package com.lifetwin.mlp.notifications

// Stub NotificationListener for MLP documentation purposes.
// Later, this will record notification actions (posted, dismissed, opened)
// into the local event store and/or forward to the RN bridge.

import android.service.notification.NotificationListenerService
import android.service.notification.StatusBarNotification
import com.lifetwin.mlp.db.AppDatabase
import com.lifetwin.mlp.db.AppEventEntity
import com.lifetwin.mlp.db.DBHelper

class NotificationLogger : NotificationListenerService() {

    override fun onNotificationPosted(sbn: StatusBarNotification) {
        DBHelper.insertEventAsync(
            applicationContext,
            AppEventEntity(
                timestamp = System.currentTimeMillis(),
                type = "notification",
                packageName = sbn.packageName
            )
        )
    }

    override fun onNotificationRemoved(sbn: StatusBarNotification) {
        DBHelper.insertEventAsync(
            applicationContext,
            AppEventEntity(
                timestamp = System.currentTimeMillis(),
                type = "notification_removed",
                packageName = sbn.packageName
            )
        )
    }
}
