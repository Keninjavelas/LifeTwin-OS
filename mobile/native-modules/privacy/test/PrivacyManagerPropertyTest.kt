package com.lifetwin.mlp.privacy.test

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.lifetwin.mlp.db.*
import com.lifetwin.mlp.privacy.PrivacyManager
import io.kotest.property.Arb
import io.kotest.property.arbitrary.*
import io.kotest.property.checkAll
import kotlinx.coroutines.runBlocking
import org.junit.After
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import kotlin.test.assertTrue

@RunWith(AndroidJUnit4::class)
class PrivacyManagerPropertyTest {

    private lateinit var context: Context
    private lateinit var privacyManager: PrivacyManager
    private lateinit var database: AppDatabase

    @Before
    fun setUp() {
        context = ApplicationProvider.getApplicationContext()
        privacyManager = PrivacyManager(context)
        database = AppDatabase.getInstance(context)
    }

    @After
    fun tearDown() {
        runBlocking {
            database.clearAllTables()
        }
    }

    /**
     * Property 13: Component Independence Control
     * Tests that enabling/disabling individual collectors works independently
     * **Validates: Requirements 8.2, 8.6, 8.7**
     */
    @Test
    fun `property test - component independence control`() = runBlocking {
        checkAll<List<CollectorControlAction>>(
            iterations = 100,
            Arb.list(
                Arb.bind(
                    Arb.element(CollectorType.values().toList()),
                    Arb.boolean()
                ) { collector, enabled ->
                    CollectorControlAction(collector, enabled)
                },
                range = 1..10
            )
        ) { actions ->
            // Apply all collector control actions
            val expectedStates = mutableMapOf<CollectorType, Boolean>()
            
            actions.forEach { action ->
                privacyManager.setCollectorEnabled(action.collectorType, action.enabled)
                expectedStates[action.collectorType] = action.enabled
            }
            
            // Verify each collector state independently
            val currentSettings = privacyManager.getPrivacySettings()
            
            expectedStates.forEach { (collectorType, expectedEnabled) ->
                val actualEnabled = currentSettings.enabledCollectors.contains(collectorType)
                assertTrue(
                    actualEnabled == expectedEnabled,
                    "Collector $collectorType should be ${if (expectedEnabled) "enabled" else "disabled"} but was ${if (actualEnabled) "enabled" else "disabled"}"
                )
            }
            
            // Verify that changing one collector doesn't affect others
            val unchangedCollectors = CollectorType.values().filter { it !in expectedStates.keys }
            unchangedCollectors.forEach { collectorType ->
                // These should maintain their default state (enabled by default)
                val actualEnabled = currentSettings.enabledCollectors.contains(collectorType)
                assertTrue(
                    actualEnabled,
                    "Unchanged collector $collectorType should remain enabled"
                )
            }
        }
    }

    data class CollectorControlAction(
        val collectorType: CollectorType,
        val enabled: Boolean
    )