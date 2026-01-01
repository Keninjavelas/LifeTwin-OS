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

/**
 * EducationalContent - Educational content and help system for automation features
 * 
 * Implements Requirements:
 * - 7.7: Educational content about digital wellbeing and automation
 * - 9.4: Contextual help and explanation systems
 * - Onboarding flow for automation features
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun EducationalContent(
    viewModel: EducationalContentViewModel = viewModel()
) {
    val contentState by viewModel.contentState.collectAsState()
    val educationalModules by viewModel.educationalModules.collectAsState()
    val helpTopics by viewModel.helpTopics.collectAsState()
    
    LazyColumn(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        // Welcome Section
        item {
            WelcomeCard(contentState.isFirstTime)
        }
        
        // Quick Start Guide
        if (contentState.showQuickStart) {
            item {
                QuickStartCard(
                    onStartOnboarding = { viewModel.startOnboarding() },
                    onDismiss = { viewModel.dismissQuickStart() }
                )
            }
        }
        
        // Educational Modules
        item {
            Text(
                text = "Learn About Digital Wellbeing"