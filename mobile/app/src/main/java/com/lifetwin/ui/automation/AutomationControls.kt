package com.lifetwin.ui.automation

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.selection.selectable
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
 * AutomationControls - UI for managing automation settings and preferences
 * 
 * Implements Requirements:
 * - 7.2: Toggle controls for each intervention type with descriptions
 * - 7.3: Threshold customization UI for all rule-based triggers
 * - 7.6: Preset automation profiles (Focus, Wellness, Minimal modes)
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun AutomationControls(
    viewModel: AutomationControlsViewModel = viewModel()
) {
    val controlsState by viewModel.controlsState.collectAsState()
    val automationProfiles by viewModel.automationProfiles.collectAsState()
    val thresholdSettings by viewModel.thresholdSettings.collectAsState()
    
    LazyColumn(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        // Master Control
        item {
            MasterControlCard(
                enabled = controlsState.masterEnabled,
                onToggle = { viewModel.toggleMasterControl() }
            )
        }
        
        // Automation Profiles
        item {
            AutomationProfilesCard(
                profiles = automationProfiles,
                selectedProfile = controlsState.selectedProfile,
                onProfileSelected = { viewModel.selectProfile(it) }
            )
        }
        
        // Intervention Type Controls
        item {
            Text(
                text = "Intervention Controls",
                style = MaterialTheme.typography.headlineSmall,
                fontWeight = FontWeight.Bold,
                modifier = Modifier.padding(vertical = 8.dp)
            )
        }
        
        items(controlsState.interventionControls) { control ->
            InterventionControlCard(
                control = control,
                onToggle = { viewModel.toggleIntervention(control.type) },
                onThresholdChange = { threshold -> 
                    viewModel.updateThreshold(control.type, threshold) 
                }
            )
        }
        
        // Advanced Threshold Settings
        item {
            AdvancedThresholdCard(
                settings = thresholdSettings,
                onSettingChange = { key, value -> 
                    viewModel.updateAdvancedSetting(key, value) 
                }
            )
        }
        
        // Quick Actions
        item {
            QuickActionsCard(
                onResetToDefaults = { viewModel.resetToDefaults() },
                onExportSettings = { viewModel.exportSettings() },
                onImportSettings = { viewModel.importSettings() }
            )
        }
    }
}

@Composable
fun MasterControlCard(
    enabled: Boolean,
    onToggle: () -> Unit
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = if (enabled) 
                MaterialTheme.colorScheme.primaryContainer 
            else 
                MaterialTheme.colorScheme.surfaceVariant
        )
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(20.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column(
                modifier = Modifier.weight(1f)
            ) {
                Text(
                    text = "Automation System",
                    style = MaterialTheme.typography.titleLarge,
                    fontWeight = FontWeight.Bold
                )
                Text(
                    text = if (enabled) 
                        "All automation features are active and monitoring your digital wellbeing" 
                    else 
                        "Automation is disabled. Enable to start receiving intelligent interventions",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f)
                )
            }
            
            Switch(
                checked = enabled,
                onCheckedChange = { onToggle() },
                modifier = Modifier.size(48.dp)
            )
        }
    }
}

@Composable
fun AutomationProfilesCard(
    profiles: List<AutomationProfile>,
    selectedProfile: String,
    onProfileSelected: (String) -> Unit
) {
    Card(
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(
            modifier = Modifier.padding(16.dp)
        ) {
            Text(
                text = "Automation Profiles",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold
            )
            Text(
                text = "Choose a preset that matches your digital wellbeing goals",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f)
            )
            
            Spacer(modifier = Modifier.height(16.dp))
            
            profiles.forEach { profile ->
                ProfileOption(
                    profile = profile,
                    selected = selectedProfile == profile.id,
                    onSelected = { onProfileSelected(profile.id) }
                )
                Spacer(modifier = Modifier.height(8.dp))
            }
        }
    }
}

@Composable
fun ProfileOption(
    profile: AutomationProfile,
    selected: Boolean,
    onSelected: () -> Unit
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .selectable(
                selected = selected,
                onClick = onSelected
            ),
        colors = CardDefaults.cardColors(
            containerColor = if (selected) 
                MaterialTheme.colorScheme.primaryContainer 
            else 
                MaterialTheme.colorScheme.surface
        ),
        border = if (selected) 
            androidx.compose.foundation.BorderStroke(2.dp, MaterialTheme.colorScheme.primary) 
        else null
    ) {
        Row(
            modifier = Modifier.padding(16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Icon(
                imageVector = getProfileIcon(profile.type),
                contentDescription = profile.name,
                tint = if (selected) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurface,
                modifier = Modifier.size(32.dp)
            )
            
            Spacer(modifier = Modifier.width(16.dp))
            
            Column(
                modifier = Modifier.weight(1f)
            ) {
                Text(
                    text = profile.name,
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Medium,
                    color = if (selected) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurface
                )
                Text(
                    text = profile.description,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f)
                )
            }
            
            if (selected) {
                Icon(
                    imageVector = Icons.Default.CheckCircle,
                    contentDescription = "Selected",
                    tint = MaterialTheme.colorScheme.primary,
                    modifier = Modifier.size(24.dp)
                )
            }
        }
    }
}

@Composable
fun InterventionControlCard(
    control: InterventionControl,
    onToggle: () -> Unit,
    onThresholdChange: (Double) -> Unit
) {
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
                Row(
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Icon(
                        imageVector = getInterventionIcon(control.type),
                        contentDescription = control.name,
                        tint = if (control.enabled) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurface.copy(alpha = 0.5f),
                        modifier = Modifier.size(24.dp)
                    )
                    
                    Spacer(modifier = Modifier.width(12.dp))
                    
                    Column {
                        Text(
                            text = control.name,
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.Medium
                        )
                        Text(
                            text = control.description,
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f)
                        )
                    }
                }
                
                Switch(
                    checked = control.enabled,
                    onCheckedChange = { onToggle() }
                )
            }
            
            if (control.enabled && control.hasThreshold) {
                Spacer(modifier = Modifier.height(16.dp))
                
                ThresholdSlider(
                    label = control.thresholdLabel,
                    value = control.threshold,
                    range = control.thresholdRange,
                    unit = control.thresholdUnit,
                    onValueChange = onThresholdChange
                )
            }
            
            if (control.enabled && control.examples.isNotEmpty()) {
                Spacer(modifier = Modifier.height(12.dp))
                
                Text(
                    text = "Examples:",
                    style = MaterialTheme.typography.bodySmall,
                    fontWeight = FontWeight.Medium,
                    color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.8f)
                )
                
                control.examples.forEach { example ->
                    Text(
                        text = "• $example",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f),
                        modifier = Modifier.padding(start = 8.dp)
                    )
                }
            }
        }
    }
}

@Composable
fun ThresholdSlider(
    label: String,
    value: Double,
    range: ClosedFloatingPointRange<Double>,
    unit: String,
    onValueChange: (Double) -> Unit
) {
    Column {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Text(
                text = label,
                style = MaterialTheme.typography.bodyMedium,
                fontWeight = FontWeight.Medium
            )
            Text(
                text = "${value.toInt()} $unit",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.primary,
                fontWeight = FontWeight.Bold
            )
        }
        
        Spacer(modifier = Modifier.height(8.dp))
        
        Slider(
            value = value.toFloat(),
            onValueChange = { onValueChange(it.toDouble()) },
            valueRange = range.start.toFloat()..range.endInclusive.toFloat(),
            steps = ((range.endInclusive - range.start) / 5).toInt().coerceAtLeast(0)
        )
        
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Text(
                text = "${range.start.toInt()} $unit",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f)
            )
            Text(
                text = "${range.endInclusive.toInt()} $unit",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f)
            )
        }
    }
}

@Composable
fun AdvancedThresholdCard(
    settings: Map<String, Double>,
    onSettingChange: (String, Double) -> Unit
) {
    Card(
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(
            modifier = Modifier.padding(16.dp)
        ) {
            Text(
                text = "Advanced Settings",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold
            )
            Text(
                text = "Fine-tune automation behavior for your specific needs",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f)
            )
            
            Spacer(modifier = Modifier.height(16.dp))
            
            settings.forEach { (key, value) ->
                AdvancedSettingRow(
                    key = key,
                    value = value,
                    onValueChange = { onSettingChange(key, it) }
                )
                Spacer(modifier = Modifier.height(12.dp))
            }
        }
    }
}

@Composable
fun AdvancedSettingRow(
    key: String,
    value: Double,
    onValueChange: (Double) -> Unit
) {
    val (label, range, unit) = getAdvancedSettingInfo(key)
    
    ThresholdSlider(
        label = label,
        value = value,
        range = range,
        unit = unit,
        onValueChange = onValueChange
    )
}

@Composable
fun QuickActionsCard(
    onResetToDefaults: () -> Unit,
    onExportSettings: () -> Unit,
    onImportSettings: () -> Unit
) {
    Card(
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(
            modifier = Modifier.padding(16.dp)
        ) {
            Text(
                text = "Quick Actions",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold
            )
            
            Spacer(modifier = Modifier.height(16.dp))
            
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                OutlinedButton(
                    onClick = onResetToDefaults,
                    modifier = Modifier.weight(1f)
                ) {
                    Icon(
                        imageVector = Icons.Default.Refresh,
                        contentDescription = "Reset",
                        modifier = Modifier.size(18.dp)
                    )
                    Spacer(modifier = Modifier.width(4.dp))
                    Text("Reset")
                }
                
                OutlinedButton(
                    onClick = onExportSettings,
                    modifier = Modifier.weight(1f)
                ) {
                    Icon(
                        imageVector = Icons.Default.FileUpload,
                        contentDescription = "Export",
                        modifier = Modifier.size(18.dp)
                    )
                    Spacer(modifier = Modifier.width(4.dp))
                    Text("Export")
                }
                
                OutlinedButton(
                    onClick = onImportSettings,
                    modifier = Modifier.weight(1f)
                ) {
                    Icon(
                        imageVector = Icons.Default.FileDownload,
                        contentDescription = "Import",
                        modifier = Modifier.size(18.dp)
                    )
                    Spacer(modifier = Modifier.width(4.dp))
                    Text("Import")
                }
            }
        }
    }
}

// Helper functions
fun getProfileIcon(type: String): androidx.compose.ui.graphics.vector.ImageVector {
    return when (type) {
        "focus" -> Icons.Default.Visibility
        "wellness" -> Icons.Default.Favorite
        "minimal" -> Icons.Default.MinimizeOutlined
        "custom" -> Icons.Default.Settings
        else -> Icons.Default.AutoMode
    }
}

fun getAdvancedSettingInfo(key: String): Triple<String, ClosedFloatingPointRange<Double>, String> {
    return when (key) {
        "intervention_frequency" -> Triple("Intervention Frequency", 0.1..2.0, "x")
        "sensitivity" -> Triple("Detection Sensitivity", 0.5..2.0, "x")
        "cooldown_period" -> Triple("Cooldown Period", 5.0..60.0, "min")
        "learning_rate" -> Triple("Learning Rate", 0.01..0.5, "")
        "confidence_threshold" -> Triple("Confidence Threshold", 0.5..0.95, "")
        else -> Triple(key.replace("_", " ").capitalize(), 0.0..1.0, "")
    }
}

// Data classes for controls state
data class AutomationControlsState(
    val masterEnabled: Boolean = true,
    val selectedProfile: String = "wellness",
    val interventionControls: List<InterventionControl> = emptyList()
)

data class InterventionControl(
    val type: InterventionType,
    val name: String,
    val description: String,
    val enabled: Boolean,
    val hasThreshold: Boolean,
    val threshold: Double,
    val thresholdRange: ClosedFloatingPointRange<Double>,
    val thresholdLabel: String,
    val thresholdUnit: String,
    val examples: List<String>
)

data class AutomationProfile(
    val id: String,
    val name: String,
    val description: String,
    val type: String,
    val settings: Map<String, Any>
)