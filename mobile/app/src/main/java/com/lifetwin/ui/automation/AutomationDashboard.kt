package com.lifetwin.ui.automation

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.lifetwin.automation.*
import kotlinx.coroutines.flow.StateFlow
import java.text.SimpleDateFormat
import java.util.*

/**
 * AutomationDashboard - Main dashboard showing automation activity and metrics
 * 
 * Implements Requirements:
 * - 7.1: Dashboard showing recent interventions and outcomes
 * - 7.5: Intervention effectiveness metrics and trend visualization
 * - Automation activity timeline and summary views
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun AutomationDashboard(
    viewModel: AutomationDashboardViewModel = viewModel()
) {
    val dashboardState by viewModel.dashboardState.collectAsState()
    val recentInterventions by viewModel.recentInterventions.collectAsState()
    val effectivenessMetrics by viewModel.effectivenessMetrics.collectAsState()
    
    LazyColumn(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        // Dashboard Header
        item {
            DashboardHeader(dashboardState)
        }
        
        // Quick Stats Cards
        item {
            QuickStatsRow(effectivenessMetrics)
        }
        
        // Recent Interventions Section
        item {
            Text(
                text = "Recent Interventions",
                style = MaterialTheme.typography.headlineSmall,
                fontWeight = FontWeight.Bold,
                modifier = Modifier.padding(vertical = 8.dp)
            )
        }
        
        items(recentInterventions) { intervention ->
            InterventionCard(intervention)
        }
        
        // Effectiveness Trends
        item {
            EffectivenessTrendsCard(effectivenessMetrics)
        }
        
        // Activity Timeline
        item {
            ActivityTimelineCard(dashboardState.activityTimeline)
        }
        
        // Weekly Summary
        item {
            WeeklySummaryCard(dashboardState.weeklySummary)
        }
    }
}

@Composable
fun DashboardHeader(dashboardState: DashboardState) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.primaryContainer
        )
    ) {
        Column(
            modifier = Modifier.padding(20.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Column {
                    Text(
                        text = "Automation Status",
                        style = MaterialTheme.typography.headlineMedium,
                        fontWeight = FontWeight.Bold
                    )
                    Text(
                        text = "Last updated: ${formatTime(dashboardState.lastUpdateTime)}",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onPrimaryContainer.copy(alpha = 0.7f)
                    )
                }
                
                StatusIndicator(dashboardState.automationEnabled)
            }
            
            Spacer(modifier = Modifier.height(16.dp))
            
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceEvenly
            ) {
                StatusMetric(
                    label = "Today's Interventions",
                    value = dashboardState.todayInterventions.toString(),
                    icon = Icons.Default.Notifications
                )
                
                StatusMetric(
                    label = "Effectiveness",
                    value = "${(dashboardState.overallEffectiveness * 100).toInt()}%",
                    icon = Icons.Default.TrendingUp
                )
                
                StatusMetric(
                    label = "Battery Impact",
                    value = "${dashboardState.batteryImpact}%",
                    icon = Icons.Default.Battery6Bar
                )
            }
        }
    }
}

@Composable
fun StatusIndicator(enabled: Boolean) {
    Row(
        verticalAlignment = Alignment.CenterVertically
    ) {
        Icon(
            imageVector = if (enabled) Icons.Default.CheckCircle else Icons.Default.Cancel,
            contentDescription = if (enabled) "Active" else "Inactive",
            tint = if (enabled) Color.Green else Color.Red,
            modifier = Modifier.size(24.dp)
        )
        Spacer(modifier = Modifier.width(8.dp))
        Text(
            text = if (enabled) "Active" else "Inactive",
            fontWeight = FontWeight.Medium,
            color = if (enabled) Color.Green else Color.Red
        )
    }
}

@Composable
fun StatusMetric(
    label: String,
    value: String,
    icon: androidx.compose.ui.graphics.vector.ImageVector
) {
    Column(
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Icon(
            imageVector = icon,
            contentDescription = label,
            modifier = Modifier.size(32.dp),
            tint = MaterialTheme.colorScheme.primary
        )
        Spacer(modifier = Modifier.height(4.dp))
        Text(
            text = value,
            style = MaterialTheme.typography.headlineSmall,
            fontWeight = FontWeight.Bold
        )
        Text(
            text = label,
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f)
        )
    }
}

@Composable
fun QuickStatsRow(metrics: EffectivenessMetrics) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        QuickStatCard(
            title = "Success Rate",
            value = "${(metrics.successRate * 100).toInt()}%",
            trend = metrics.successRateTrend,
            modifier = Modifier.weight(1f)
        )
        
        QuickStatCard(
            title = "User Satisfaction",
            value = "${(metrics.userSatisfaction * 100).toInt()}%",
            trend = metrics.satisfactionTrend,
            modifier = Modifier.weight(1f)
        )
        
        QuickStatCard(
            title = "Response Time",
            value = "${metrics.avgResponseTimeMs}ms",
            trend = metrics.responseTimeTrend,
            modifier = Modifier.weight(1f)
        )
    }
}

@Composable
fun QuickStatCard(
    title: String,
    value: String,
    trend: Double,
    modifier: Modifier = Modifier
) {
    Card(
        modifier = modifier,
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surface
        )
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text(
                text = title,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f)
            )
            Spacer(modifier = Modifier.height(4.dp))
            Text(
                text = value,
                style = MaterialTheme.typography.headlineSmall,
                fontWeight = FontWeight.Bold
            )
            
            // Trend indicator
            Row(
                verticalAlignment = Alignment.CenterVertically
            ) {
                Icon(
                    imageVector = if (trend >= 0) Icons.Default.TrendingUp else Icons.Default.TrendingDown,
                    contentDescription = "Trend",
                    tint = if (trend >= 0) Color.Green else Color.Red,
                    modifier = Modifier.size(16.dp)
                )
                Text(
                    text = "${if (trend >= 0) "+" else ""}${(trend * 100).toInt()}%",
                    style = MaterialTheme.typography.bodySmall,
                    color = if (trend >= 0) Color.Green else Color.Red
                )
            }
        }
    }
}

@Composable
fun InterventionCard(intervention: InterventionSummary) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceVariant
        )
    ) {
        Column(
            modifier = Modifier.padding(16.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Row(
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Icon(
                        imageVector = getInterventionIcon(intervention.type),
                        contentDescription = intervention.type.name,
                        tint = getInterventionColor(intervention.type),
                        modifier = Modifier.size(24.dp)
                    )
                    Spacer(modifier = Modifier.width(12.dp))
                    Column {
                        Text(
                            text = intervention.title,
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.Medium
                        )
                        Text(
                            text = formatTime(intervention.timestamp),
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f)
                        )
                    }
                }
                
                EffectivenessChip(intervention.effectiveness)
            }
            
            if (intervention.description.isNotEmpty()) {
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = intervention.description,
                    style = MaterialTheme.typography.bodyMedium
                )
            }
            
            if (intervention.userFeedback != null) {
                Spacer(modifier = Modifier.height(8.dp))
                UserFeedbackRow(intervention.userFeedback)
            }
        }
    }
}

@Composable
fun EffectivenessChip(effectiveness: Double) {
    val color = when {
        effectiveness >= 0.8 -> Color.Green
        effectiveness >= 0.6 -> Color(0xFFFF9800) // Orange
        else -> Color.Red
    }
    
    Surface(
        color = color.copy(alpha = 0.1f),
        shape = RoundedCornerShape(12.dp)
    ) {
        Text(
            text = "${(effectiveness * 100).toInt()}% effective",
            modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp),
            style = MaterialTheme.typography.bodySmall,
            color = color,
            fontWeight = FontWeight.Medium
        )
    }
}

@Composable
fun UserFeedbackRow(feedback: UserFeedback) {
    Row(
        verticalAlignment = Alignment.CenterVertically
    ) {
        Icon(
            imageVector = when (feedback.rating) {
                in 4..5 -> Icons.Default.ThumbUp
                3 -> Icons.Default.ThumbsUpDown
                else -> Icons.Default.ThumbDown
            },
            contentDescription = "Feedback",
            tint = when (feedback.rating) {
                in 4..5 -> Color.Green
                3 -> Color(0xFFFF9800)
                else -> Color.Red
            },
            modifier = Modifier.size(16.dp)
        )
        Spacer(modifier = Modifier.width(8.dp))
        Text(
            text = "User rated ${feedback.rating}/5",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f)
        )
        if (feedback.comment.isNotEmpty()) {
            Text(
                text = " • ${feedback.comment}",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f)
            )
        }
    }
}

@Composable
fun EffectivenessTrendsCard(metrics: EffectivenessMetrics) {
    Card(
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(
            modifier = Modifier.padding(16.dp)
        ) {
            Text(
                text = "Effectiveness Trends",
                style = MaterialTheme.typography.titleLarge,
                fontWeight = FontWeight.Bold
            )
            Spacer(modifier = Modifier.height(16.dp))
            
            // Simple trend visualization
            TrendVisualization(metrics.weeklyTrends)
        }
    }
}

@Composable
fun TrendVisualization(trends: List<Double>) {
    // Simplified trend visualization - in a real app, you'd use a charting library
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceEvenly
    ) {
        trends.forEachIndexed { index, value ->
            Column(
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Box(
                    modifier = Modifier
                        .width(8.dp)
                        .height((value * 100).dp)
                        .background(
                            MaterialTheme.colorScheme.primary,
                            RoundedCornerShape(4.dp)
                        )
                )
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = "D${index + 1}",
                    style = MaterialTheme.typography.bodySmall
                )
            }
        }
    }
}

@Composable
fun ActivityTimelineCard(timeline: List<ActivityEvent>) {
    Card(
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(
            modifier = Modifier.padding(16.dp)
        ) {
            Text(
                text = "Activity Timeline",
                style = MaterialTheme.typography.titleLarge,
                fontWeight = FontWeight.Bold
            )
            Spacer(modifier = Modifier.height(16.dp))
            
            timeline.take(5).forEach { event ->
                TimelineItem(event)
                Spacer(modifier = Modifier.height(8.dp))
            }
        }
    }
}

@Composable
fun TimelineItem(event: ActivityEvent) {
    Row(
        verticalAlignment = Alignment.CenterVertically
    ) {
        Icon(
            imageVector = getActivityIcon(event.type),
            contentDescription = event.type,
            tint = MaterialTheme.colorScheme.primary,
            modifier = Modifier.size(20.dp)
        )
        Spacer(modifier = Modifier.width(12.dp))
        Column(
            modifier = Modifier.weight(1f)
        ) {
            Text(
                text = event.description,
                style = MaterialTheme.typography.bodyMedium
            )
            Text(
                text = formatTime(event.timestamp),
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f)
            )
        }
    }
}

@Composable
fun WeeklySummaryCard(summary: WeeklySummary) {
    Card(
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(
            modifier = Modifier.padding(16.dp)
        ) {
            Text(
                text = "Weekly Summary",
                style = MaterialTheme.typography.titleLarge,
                fontWeight = FontWeight.Bold
            )
            Spacer(modifier = Modifier.height(16.dp))
            
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceEvenly
            ) {
                SummaryMetric(
                    label = "Interventions",
                    value = summary.totalInterventions.toString()
                )
                SummaryMetric(
                    label = "Success Rate",
                    value = "${(summary.successRate * 100).toInt()}%"
                )
                SummaryMetric(
                    label = "Time Saved",
                    value = "${summary.timeSavedMinutes}min"
                )
            }
        }
    }
}

@Composable
fun SummaryMetric(label: String, value: String) {
    Column(
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            text = value,
            style = MaterialTheme.typography.headlineMedium,
            fontWeight = FontWeight.Bold,
            color = MaterialTheme.colorScheme.primary
        )
        Text(
            text = label,
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f)
        )
    }
}

// Helper functions
fun getInterventionIcon(type: InterventionType): androidx.compose.ui.graphics.vector.ImageVector {
    return when (type) {
        InterventionType.NOTIFICATION_LIMIT -> Icons.Default.NotificationsOff
        InterventionType.BREAK_REMINDER -> Icons.Default.AccessTime
        InterventionType.FOCUS_MODE -> Icons.Default.Visibility
        InterventionType.BEDTIME_REMINDER -> Icons.Default.Bedtime
        InterventionType.APP_LIMIT -> Icons.Default.Block
        else -> Icons.Default.Info
    }
}

fun getInterventionColor(type: InterventionType): Color {
    return when (type) {
        InterventionType.NOTIFICATION_LIMIT -> Color(0xFF2196F3)
        InterventionType.BREAK_REMINDER -> Color(0xFF4CAF50)
        InterventionType.FOCUS_MODE -> Color(0xFF9C27B0)
        InterventionType.BEDTIME_REMINDER -> Color(0xFF3F51B5)
        InterventionType.APP_LIMIT -> Color(0xFFFF5722)
        else -> Color.Gray
    }
}

fun getActivityIcon(type: String): androidx.compose.ui.graphics.vector.ImageVector {
    return when (type) {
        "intervention" -> Icons.Default.Notifications
        "feedback" -> Icons.Default.Feedback
        "setting_change" -> Icons.Default.Settings
        "system_event" -> Icons.Default.Info
        else -> Icons.Default.Circle
    }
}

fun formatTime(timestamp: Long): String {
    val formatter = SimpleDateFormat("MMM dd, HH:mm", Locale.getDefault())
    return formatter.format(Date(timestamp))
}

// Data classes for UI state
data class DashboardState(
    val automationEnabled: Boolean = true,
    val lastUpdateTime: Long = System.currentTimeMillis(),
    val todayInterventions: Int = 0,
    val overallEffectiveness: Double = 0.0,
    val batteryImpact: Double = 0.0,
    val activityTimeline: List<ActivityEvent> = emptyList(),
    val weeklySummary: WeeklySummary = WeeklySummary()
)

data class EffectivenessMetrics(
    val successRate: Double = 0.0,
    val successRateTrend: Double = 0.0,
    val userSatisfaction: Double = 0.0,
    val satisfactionTrend: Double = 0.0,
    val avgResponseTimeMs: Long = 0L,
    val responseTimeTrend: Double = 0.0,
    val weeklyTrends: List<Double> = emptyList()
)

data class InterventionSummary(
    val id: String,
    val type: InterventionType,
    val title: String,
    val description: String,
    val timestamp: Long,
    val effectiveness: Double,
    val userFeedback: UserFeedback? = null
)

data class UserFeedback(
    val rating: Int, // 1-5
    val comment: String = ""
)

data class ActivityEvent(
    val type: String,
    val description: String,
    val timestamp: Long
)

data class WeeklySummary(
    val totalInterventions: Int = 0,
    val successRate: Double = 0.0,
    val timeSavedMinutes: Int = 0
)

enum class InterventionType {
    NOTIFICATION_LIMIT,
    BREAK_REMINDER,
    FOCUS_MODE,
    BEDTIME_REMINDER,
    APP_LIMIT,
    CUSTOM
}