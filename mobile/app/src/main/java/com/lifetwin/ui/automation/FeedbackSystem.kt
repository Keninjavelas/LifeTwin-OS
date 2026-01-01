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
import androidx.lifecycle.viewmodel.compose.viewModel
import com.lifetwin.automation.*

/**
 * FeedbackSystem - UI for collecting and displaying user feedback on interventions
 * 
 * Implements Requirements:
 * - 7.4: Intervention feedback collection UI
 * - Rating mechanisms for intervention helpfulness
 * - Feedback history and pattern analysis display
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun FeedbackSystem(
    viewModel: FeedbackSystemViewModel = viewModel()
) {
    val feedbackState by viewModel.feedbackState.collectAsState()
    val pendingFeedback by viewModel.pendingFeedback.collectAsState()
    val feedbackHistory by viewModel.feedbackHistory.collectAsState()
    val feedbackAnalytics by viewModel.feedbackAnalytics.collectAsState()
    
    LazyColumn(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        // Pending Feedback Section
        if (pendingFeedback.isNotEmpty()) {
            item {
                Text(
                    text = "Rate Recent Interventions",
                    style = MaterialTheme.typography.headlineSmall,
                    fontWeight = FontWeight.Bold
                )
            }
            
            items(pendingFeedback) { intervention ->
                PendingFeedbackCard(
                    intervention = intervention,
                    onFeedbackSubmitted = { rating, comment ->
                        viewModel.submitFeedback(intervention.id, rating, comment)
                    },
                    onDismiss = { viewModel.dismissFeedback(intervention.id) }
                )
            }
        }
        
        // Feedback Analytics
        item {
            FeedbackAnalyticsCard(feedbackAnalytics)
        }
        
        // Feedback History
        item {
            Text(
                text = "Feedback History",
                style = MaterialTheme.typography.headlineSmall,
                fontWeight = FontWeight.Bold
            )
        }
        
        items(feedbackHistory) { feedback ->
            FeedbackHistoryCard(feedback)
        }
        
        // Feedback Insights
        item {
            FeedbackInsightsCard(feedbackAnalytics)
        }
    }
}

@Composable
fun PendingFeedbackCard(
    intervention: InterventionSummary,
    onFeedbackSubmitted: (Int, String) -> Unit,
    onDismiss: () -> Unit
) {
    var selectedRating by remember { mutableStateOf(0) }
    var comment by remember { mutableStateOf("") }
    var showCommentField by remember { mutableStateOf(false) }
    
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
                Column(
                    modifier = Modifier.weight(1f)
                ) {
                    Text(
                        text = intervention.title,
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.Bold
                    )
                    Text(
                        text = intervention.description,
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onPrimaryContainer.copy(alpha = 0.8f)
                    )
                    Text(
                        text = formatTime(intervention.timestamp),
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onPrimaryContainer.copy(alpha = 0.6f)
                    )
                }
                
                IconButton(onClick = onDismiss) {
                    Icon(
                        imageVector = Icons.Default.Close,
                        contentDescription = "Dismiss",
                        tint = MaterialTheme.colorScheme.onPrimaryContainer
                    )
                }
            }
            
            Spacer(modifier = Modifier.height(16.dp))
            
            Text(
                text = "How helpful was this intervention?",
                style = MaterialTheme.typography.bodyMedium,
                fontWeight = FontWeight.Medium
            )
            
            Spacer(modifier = Modifier.height(12.dp))
            
            // Star Rating
            StarRating(
                rating = selectedRating,
                onRatingChanged = { selectedRating = it }
            )
            
            Spacer(modifier = Modifier.height(16.dp))
            
            // Quick Feedback Buttons
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                QuickFeedbackChip(
                    text = "Very Helpful",
                    selected = selectedRating == 5,
                    onClick = { selectedRating = 5 }
                )
                QuickFeedbackChip(
                    text = "Somewhat Helpful",
                    selected = selectedRating == 3,
                    onClick = { selectedRating = 3 }
                )
                QuickFeedbackChip(
                    text = "Not Helpful",
                    selected = selectedRating == 1,
                    onClick = { selectedRating = 1 }
                )
            }
            
            Spacer(modifier = Modifier.height(16.dp))
            
            // Comment Section
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                TextButton(
                    onClick = { showCommentField = !showCommentField }
                ) {
                    Icon(
                        imageVector = if (showCommentField) Icons.Default.ExpandLess else Icons.Default.ExpandMore,
                        contentDescription = "Toggle comment"
                    )
                    Spacer(modifier = Modifier.width(4.dp))
                    Text("Add comment")
                }
                
                Button(
                    onClick = { 
                        if (selectedRating > 0) {
                            onFeedbackSubmitted(selectedRating, comment)
                        }
                    },
                    enabled = selectedRating > 0
                ) {
                    Text("Submit")
                }
            }
            
            if (showCommentField) {
                Spacer(modifier = Modifier.height(12.dp))
                
                OutlinedTextField(
                    value = comment,
                    onValueChange = { comment = it },
                    label = { Text("Additional feedback (optional)") },
                    placeholder = { Text("Tell us more about your experience...") },
                    modifier = Modifier.fillMaxWidth(),
                    maxLines = 3
                )
            }
        }
    }
}

@Composable
fun StarRating(
    rating: Int,
    onRatingChanged: (Int) -> Unit,
    maxRating: Int = 5
) {
    Row(
        horizontalArrangement = Arrangement.spacedBy(4.dp)
    ) {
        for (i in 1..maxRating) {
            IconButton(
                onClick = { onRatingChanged(i) },
                modifier = Modifier.size(40.dp)
            ) {
                Icon(
                    imageVector = if (i <= rating) Icons.Default.Star else Icons.Default.StarBorder,
                    contentDescription = "Star $i",
                    tint = if (i <= rating) Color(0xFFFFD700) else MaterialTheme.colorScheme.onSurface.copy(alpha = 0.4f),
                    modifier = Modifier.size(32.dp)
                )
            }
        }
    }
}

@Composable
fun QuickFeedbackChip(
    text: String,
    selected: Boolean,
    onClick: () -> Unit
) {
    FilterChip(
        onClick = onClick,
        label = { Text(text) },
        selected = selected,
        colors = FilterChipDefaults.filterChipColors(
            selectedContainerColor = MaterialTheme.colorScheme.primary,
            selectedLabelColor = MaterialTheme.colorScheme.onPrimary
        )
    )
}

@Composable
fun FeedbackAnalyticsCard(analytics: FeedbackAnalytics) {
    Card(
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(
            modifier = Modifier.padding(16.dp)
        ) {
            Text(
                text = "Feedback Overview",
                style = MaterialTheme.typography.titleLarge,
                fontWeight = FontWeight.Bold
            )
            
            Spacer(modifier = Modifier.height(16.dp))
            
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceEvenly
            ) {
                AnalyticsMetric(
                    label = "Average Rating",
                    value = String.format("%.1f", analytics.averageRating),
                    icon = Icons.Default.Star,
                    color = Color(0xFFFFD700)
                )
                
                AnalyticsMetric(
                    label = "Response Rate",
                    value = "${(analytics.responseRate * 100).toInt()}%",
                    icon = Icons.Default.TrendingUp,
                    color = MaterialTheme.colorScheme.primary
                )
                
                AnalyticsMetric(
                    label = "Total Feedback",
                    value = analytics.totalFeedback.toString(),
                    icon = Icons.Default.Feedback,
                    color = MaterialTheme.colorScheme.secondary
                )
            }
            
            Spacer(modifier = Modifier.height(16.dp))
            
            // Rating Distribution
            Text(
                text = "Rating Distribution",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Medium
            )
            
            Spacer(modifier = Modifier.height(8.dp))
            
            analytics.ratingDistribution.forEachIndexed { index, count ->
                RatingDistributionBar(
                    stars = index + 1,
                    count = count,
                    total = analytics.totalFeedback
                )
            }
        }
    }
}

@Composable
fun AnalyticsMetric(
    label: String,
    value: String,
    icon: androidx.compose.ui.graphics.vector.ImageVector,
    color: Color
) {
    Column(
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Icon(
            imageVector = icon,
            contentDescription = label,
            tint = color,
            modifier = Modifier.size(32.dp)
        )
        Spacer(modifier = Modifier.height(4.dp))
        Text(
            text = value,
            style = MaterialTheme.typography.headlineSmall,
            fontWeight = FontWeight.Bold,
            color = color
        )
        Text(
            text = label,
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f)
        )
    }
}

@Composable
fun RatingDistributionBar(
    stars: Int,
    count: Int,
    total: Int
) {
    val percentage = if (total > 0) count.toFloat() / total else 0f
    
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 2.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Row(
            verticalAlignment = Alignment.CenterVertically,
            modifier = Modifier.width(60.dp)
        ) {
            Text(
                text = "$stars",
                style = MaterialTheme.typography.bodySmall
            )
            Icon(
                imageVector = Icons.Default.Star,
                contentDescription = "star",
                tint = Color(0xFFFFD700),
                modifier = Modifier.size(16.dp)
            )
        }
        
        Box(
            modifier = Modifier
                .weight(1f)
                .height(8.dp)
                .background(
                    MaterialTheme.colorScheme.surfaceVariant,
                    RoundedCornerShape(4.dp)
                )
        ) {
            Box(
                modifier = Modifier
                    .fillMaxWidth(percentage)
                    .fillMaxHeight()
                    .background(
                        MaterialTheme.colorScheme.primary,
                        RoundedCornerShape(4.dp)
                    )
            )
        }
        
        Text(
            text = count.toString(),
            style = MaterialTheme.typography.bodySmall,
            modifier = Modifier.width(40.dp)
        )
    }
}

@Composable
fun FeedbackHistoryCard(feedback: FeedbackEntry) {
    Card(
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(
            modifier = Modifier.padding(16.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Column(
                    modifier = Modifier.weight(1f)
                ) {
                    Text(
                        text = feedback.interventionTitle,
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.Medium
                    )
                    Text(
                        text = formatTime(feedback.timestamp),
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f)
                    )
                }
                
                Row(
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    repeat(feedback.rating) {
                        Icon(
                            imageVector = Icons.Default.Star,
                            contentDescription = "star",
                            tint = Color(0xFFFFD700),
                            modifier = Modifier.size(16.dp)
                        )
                    }
                    repeat(5 - feedback.rating) {
                        Icon(
                            imageVector = Icons.Default.StarBorder,
                            contentDescription = "empty star",
                            tint = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.3f),
                            modifier = Modifier.size(16.dp)
                        )
                    }
                }
            }
            
            if (feedback.comment.isNotEmpty()) {
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = feedback.comment,
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.8f)
                )
            }
        }
    }
}

@Composable
fun FeedbackInsightsCard(analytics: FeedbackAnalytics) {
    Card(
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(
            modifier = Modifier.padding(16.dp)
        ) {
            Text(
                text = "Insights & Patterns",
                style = MaterialTheme.typography.titleLarge,
                fontWeight = FontWeight.Bold
            )
            
            Spacer(modifier = Modifier.height(16.dp))
            
            analytics.insights.forEach { insight ->
                InsightItem(insight)
                Spacer(modifier = Modifier.height(8.dp))
            }
            
            if (analytics.recommendations.isNotEmpty()) {
                Spacer(modifier = Modifier.height(8.dp))
                
                Text(
                    text = "Recommendations",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Medium
                )
                
                Spacer(modifier = Modifier.height(8.dp))
                
                analytics.recommendations.forEach { recommendation ->
                    RecommendationItem(recommendation)
                    Spacer(modifier = Modifier.height(4.dp))
                }
            }
        }
    }
}

@Composable
fun InsightItem(insight: FeedbackInsight) {
    Row(
        verticalAlignment = Alignment.CenterVertically
    ) {
        Icon(
            imageVector = getInsightIcon(insight.type),
            contentDescription = insight.type,
            tint = getInsightColor(insight.type),
            modifier = Modifier.size(20.dp)
        )
        Spacer(modifier = Modifier.width(12.dp))
        Text(
            text = insight.message,
            style = MaterialTheme.typography.bodyMedium
        )
    }
}

@Composable
fun RecommendationItem(recommendation: String) {
    Row(
        verticalAlignment = Alignment.CenterVertically
    ) {
        Icon(
            imageVector = Icons.Default.Lightbulb,
            contentDescription = "recommendation",
            tint = MaterialTheme.colorScheme.primary,
            modifier = Modifier.size(16.dp)
        )
        Spacer(modifier = Modifier.width(8.dp))
        Text(
            text = recommendation,
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.8f)
        )
    }
}

// Helper functions
fun getInsightIcon(type: String): androidx.compose.ui.graphics.vector.ImageVector {
    return when (type) {
        "trend_positive" -> Icons.Default.TrendingUp
        "trend_negative" -> Icons.Default.TrendingDown
        "pattern" -> Icons.Default.Pattern
        "improvement" -> Icons.Default.Upgrade
        else -> Icons.Default.Info
    }
}

fun getInsightColor(type: String): Color {
    return when (type) {
        "trend_positive" -> Color.Green
        "trend_negative" -> Color.Red
        "pattern" -> Color(0xFFFF9800)
        "improvement" -> Color(0xFF2196F3)
        else -> Color.Gray
    }
}

// Data classes for feedback system
data class FeedbackSystemState(
    val isEnabled: Boolean = true,
    val feedbackPromptDelay: Long = 300000, // 5 minutes
    val showQuickFeedback: Boolean = true
)

data class FeedbackEntry(
    val id: String,
    val interventionId: String,
    val interventionTitle: String,
    val rating: Int,
    val comment: String,
    val timestamp: Long
)

data class FeedbackAnalytics(
    val averageRating: Double = 0.0,
    val responseRate: Double = 0.0,
    val totalFeedback: Int = 0,
    val ratingDistribution: List<Int> = listOf(0, 0, 0, 0, 0), // 1-5 stars
    val insights: List<FeedbackInsight> = emptyList(),
    val recommendations: List<String> = emptyList()
)

data class FeedbackInsight(
    val type: String,
    val message: String,
    val confidence: Double
)