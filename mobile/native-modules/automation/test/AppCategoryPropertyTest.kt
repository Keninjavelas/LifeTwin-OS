package com.lifetwin.mlp.automation.test

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.lifetwin.mlp.automation.*
import com.lifetwin.mlp.db.*
import io.kotest.property.Arb
import io.kotest.property.arbitrary.*
import io.kotest.property.checkAll
import kotlinx.coroutines.runBlocking
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import kotlin.test.assertTrue
import kotlin.test.assertEquals
import kotlin.test.assertNotNull

@RunWith(AndroidJUnit4::class)
class AppCategoryPropertyTest {

    private lateinit var context: Context
    private lateinit var appCategoryMapping: AppCategoryMapping

    @Before
    fun setup() {
        context = ApplicationProvider.getApplicationContext()
        appCategoryMapping = AppCategoryMapping()
        appCategoryMapping.initialize()
    }

    /**
     * Property 5: App category usage computation accuracy
     * Validates that category usage calculations are accurate and consistent
     */
    @Test
    fun `property test - category usage computation accuracy`() = runBlocking {
        checkAll(
            iterations = 30,
            Arb.usageEventList()
        ) { events ->
            val categoryUsage = appCategoryMapping.computeCategoryUsage(events)
            
            // Total usage should equal sum of all category usage
            val totalFromCategories = categoryUsage.values.sum()
            val totalFromEvents = events.sumOf { it.totalTimeInForeground }
            
            assertEquals(
                totalFromEvents,
                totalFromCategories,
                "Total usage from categories should equal total from events"
            )
            
            // All categories should be represented (even with 0 usage)
            AppCategory.values().forEach { category ->
                assertTrue(
                    categoryUsage.containsKey(category),
                    "Category $category should be present in usage map"
                )
                assertTrue(
                    categoryUsage[category]!! >= 0L,
                    "Category usage should be non-negative for $category"
                )
            }
            
            // Verify specific app categorizations
            events.forEach { event ->
                val expectedCategory = appCategoryMapping.getCategory(event.packageName)
                val actualUsage = categoryUsage[expectedCategory] ?: 0L
                
                assertTrue(
                    actualUsage > 0L,
                    "Category $expectedCategory should have usage > 0 for package ${event.packageName}"
                )
            }
        }
    }

    /**
     * Property 6: Category distribution consistency
     * Validates that category distributions sum to 1.0 and are non-negative
     */
    @Test
    fun `property test - category distribution consistency`() = runBlocking {
        checkAll(
            iterations = 30,
            Arb.usageEventList()
        ) { events ->
            val categoryUsage = appCategoryMapping.computeCategoryUsage(events)
            val distribution = appCategoryMapping.getCategoryDistribution(categoryUsage)
            
            // All percentages should be non-negative
            distribution.values.forEach { percentage ->
                assertTrue(
                    percentage >= 0.0f,
                    "Category percentage should be non-negative, got $percentage"
                )
                assertTrue(
                    percentage <= 1.0f,
                    "Category percentage should be <= 1.0, got $percentage"
                )
            }
            
            // Total should sum to 1.0 (or 0.0 if no usage)
            val totalPercentage = distribution.values.sum()
            val totalUsage = categoryUsage.values.sum()
            
            if (totalUsage > 0L) {
                assertEquals(
                    1.0f,
                    totalPercentage,
                    0.001f,
                    "Category percentages should sum to 1.0 when there's usage"
                )
            } else {
                assertEquals(
                    0.0f,
                    totalPercentage,
                    0.001f,
                    "Category percentages should sum to 0.0 when there's no usage"
                )
            }
        }
    }

    /**
     * Property 7: Category trigger generation
     * Validates that category triggers are generated correctly based on thresholds
     */
    @Test
    fun `property test - category trigger generation`() = runBlocking {
        checkAll(
            iterations = 20,
            Arb.usageEventList(),
            Arb.categoryThresholds()
        ) { events, thresholds ->
            val categoryUsage = appCategoryMapping.computeCategoryUsage(events)
            val triggers = appCategoryMapping.getCategoryTriggers(categoryUsage, thresholds)
            
            // Verify triggers are only generated for categories exceeding thresholds
            triggers.forEach { trigger ->
                val usage = categoryUsage[trigger.category] ?: 0L
                val threshold = thresholds[trigger.category] ?: Long.MAX_VALUE
                
                assertTrue(
                    usage > threshold,
                    "Trigger should only be generated when usage ($usage) exceeds threshold ($threshold) for ${trigger.category}"
                )
                
                assertTrue(
                    trigger.severity > 0.0f,
                    "Trigger severity should be positive, got ${trigger.severity}"
                )
                
                assertNotNull(
                    trigger.recommendedAction,
                    "Trigger should have a recommended action"
                )
            }
            
            // Verify no triggers for categories below threshold
            for ((category, threshold) in thresholds) {
                val usage = categoryUsage[category] ?: 0L
                val hasTrigger = triggers.any { it.category == category }
                
                if (usage <= threshold) {
                    assertTrue(
                        !hasTrigger,
                        "No trigger should be generated when usage ($usage) is below threshold ($threshold) for $category"
                    )
                }
            }
            
            // Triggers should be sorted by severity (descending)
            if (triggers.size > 1) {
                for (i in 0 until triggers.size - 1) {
                    assertTrue(
                        triggers[i].severity >= triggers[i + 1].severity,
                        "Triggers should be sorted by severity: ${triggers[i].severity} >= ${triggers[i + 1].severity}"
                    )
                }
            }
        }
    }

    /**
     * Property 8: Top apps by category accuracy
     * Validates that top apps are correctly identified and sorted
     */
    @Test
    fun `property test - top apps by category accuracy`() = runBlocking {
        checkAll(
            iterations = 20,
            Arb.usageEventListWithKnownCategories(),
            Arb.enum<AppCategory>(),
            Arb.int(1, 10) // limit
        ) { events, category, limit ->
            val topApps = appCategoryMapping.getTopAppsByCategory(events, category, limit)
            
            // Should not exceed the requested limit
            assertTrue(
                topApps.size <= limit,
                "Top apps list should not exceed limit of $limit, got ${topApps.size}"
            )
            
            // All returned apps should belong to the requested category
            topApps.forEach { appInfo ->
                assertEquals(
                    category,
                    appInfo.category,
                    "App ${appInfo.packageName} should belong to category $category"
                )
                
                assertTrue(
                    appInfo.totalUsage >= 0L,
                    "App usage should be non-negative for ${appInfo.packageName}"
                )
            }
            
            // Apps should be sorted by usage (descending)
            if (topApps.size > 1) {
                for (i in 0 until topApps.size - 1) {
                    assertTrue(
                        topApps[i].totalUsage >= topApps[i + 1].totalUsage,
                        "Apps should be sorted by usage: ${topApps[i].totalUsage} >= ${topApps[i + 1].totalUsage}"
                    )
                }
            }
            
            // Verify usage calculations are correct
            val eventsByPackage = events
                .filter { appCategoryMapping.getCategory(it.packageName) == category }
                .groupBy { it.packageName }
                .mapValues { (_, events) -> events.sumOf { it.totalTimeInForeground } }
            
            topApps.forEach { appInfo ->
                val expectedUsage = eventsByPackage[appInfo.packageName] ?: 0L
                assertEquals(
                    expectedUsage,
                    appInfo.totalUsage,
                    "Usage calculation should be correct for ${appInfo.packageName}"
                )
            }
        }
    }

    /**
     * Property 9: Custom category mapping persistence
     * Validates that custom category mappings work correctly
     */
    @Test
    fun `property test - custom category mapping persistence`() = runBlocking {
        checkAll(
            iterations = 20,
            Arb.string(5, 20), // package name
            Arb.enum<AppCategory>()
        ) { packageName, customCategory ->
            // Set custom category
            appCategoryMapping.setCustomCategory(packageName, customCategory)
            
            // Verify the mapping is applied
            val retrievedCategory = appCategoryMapping.getCategory(packageName)
            assertEquals(
                customCategory,
                retrievedCategory,
                "Custom category mapping should be applied for $packageName"
            )
            
            // Test with usage events
            val testEvents = listOf(
                UsageEventEntity(
                    id = "test_${System.currentTimeMillis()}",
                    packageName = packageName,
                    startTime = System.currentTimeMillis() - 60000,
                    endTime = System.currentTimeMillis(),
                    totalTimeInForeground = 30000L,
                    lastTimeUsed = System.currentTimeMillis(),
                    eventType = "ACTIVITY_RESUMED"
                )
            )
            
            val categoryUsage = appCategoryMapping.computeCategoryUsage(testEvents)
            val customCategoryUsage = categoryUsage[customCategory] ?: 0L
            
            assertTrue(
                customCategoryUsage > 0L,
                "Custom category $customCategory should have usage > 0 for custom mapping"
            )
        }
    }

    /**
     * Property 10: Category mapping determinism
     * Validates that category mappings are deterministic and consistent
     */
    @Test
    fun `property test - category mapping determinism`() = runBlocking {
        checkAll(
            iterations = 30,
            Arb.string(5, 30) // package name
        ) { packageName ->
            // Get category multiple times
            val category1 = appCategoryMapping.getCategory(packageName)
            val category2 = appCategoryMapping.getCategory(packageName)
            val category3 = appCategoryMapping.getCategory(packageName)
            
            // Should always return the same category
            assertEquals(
                category1,
                category2,
                "Category mapping should be deterministic for $packageName"
            )
            assertEquals(
                category2,
                category3,
                "Category mapping should be deterministic for $packageName"
            )
            
            // Category should be valid
            assertTrue(
                AppCategory.values().contains(category1),
                "Returned category should be valid: $category1"
            )
        }
    }
}

// Arbitrary generators for property-based testing

fun Arb.Companion.usageEventList(): Arb<List<UsageEventEntity>> = arbitrary { rs ->
    val count = int(1, 20).bind(rs)
    (1..count).map { i ->
        val packageName = packageName().bind(rs)
        val duration = long(1000L, 3600000L).bind(rs) // 1 second to 1 hour
        val startTime = System.currentTimeMillis() - duration - (i * 60000L)
        
        UsageEventEntity(
            id = "test_$i",
            packageName = packageName,
            startTime = startTime,
            endTime = startTime + duration,
            totalTimeInForeground = duration,
            lastTimeUsed = startTime + duration,
            eventType = "ACTIVITY_RESUMED"
        )
    }
}

fun Arb.Companion.usageEventListWithKnownCategories(): Arb<List<UsageEventEntity>> = arbitrary { rs ->
    val count = int(1, 15).bind(rs)
    (1..count).map { i ->
        val packageName = knownPackageName().bind(rs)
        val duration = long(1000L, 3600000L).bind(rs)
        val startTime = System.currentTimeMillis() - duration - (i * 60000L)
        
        UsageEventEntity(
            id = "test_$i",
            packageName = packageName,
            startTime = startTime,
            endTime = startTime + duration,
            totalTimeInForeground = duration,
            lastTimeUsed = startTime + duration,
            eventType = "ACTIVITY_RESUMED"
        )
    }
}

fun Arb.Companion.packageName(): Arb<String> = arbitrary { rs ->
    val domains = listOf("com", "org", "net")
    val companies = listOf("google", "facebook", "microsoft", "amazon", "apple", "netflix", "spotify")
    val apps = listOf("android", "app", "mobile", "client", "main", "ui")
    
    val domain = domains.random()
    val company = companies.random()
    val app = apps.random()
    
    "$domain.$company.$app"
}

fun Arb.Companion.knownPackageName(): Arb<String> = arbitrary { rs ->
    val knownPackages = listOf(
        "com.facebook.katana",
        "com.instagram.android",
        "com.google.android.youtube",
        "com.netflix.mediaclient",
        "com.spotify.music",
        "com.microsoft.office.outlook",
        "com.google.android.gm",
        "com.slack",
        "com.google.android.apps.fitness",
        "com.headspace.android",
        "com.whatsapp",
        "com.telegram.messenger"
    )
    
    knownPackages.random()
}

fun Arb.Companion.categoryThresholds(): Arb<Map<AppCategory, Long>> = arbitrary { rs ->
    AppCategory.values().associateWith { category ->
        when (category) {
            AppCategory.SOCIAL -> long(30 * 60 * 1000L, 2 * 60 * 60 * 1000L).bind(rs) // 30 min to 2 hours
            AppCategory.ENTERTAINMENT -> long(60 * 60 * 1000L, 4 * 60 * 60 * 1000L).bind(rs) // 1 to 4 hours
            AppCategory.PRODUCTIVITY -> long(2 * 60 * 60 * 1000L, 8 * 60 * 60 * 1000L).bind(rs) // 2 to 8 hours
            else -> long(15 * 60 * 1000L, 60 * 60 * 1000L).bind(rs) // 15 min to 1 hour
        }
    }
}