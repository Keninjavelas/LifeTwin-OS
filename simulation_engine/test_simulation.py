#!/usr/bin/env python3
"""
Test script for the enhanced simulation engine.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from simulation_engine.engine import run_comprehensive_scenario
import json


def test_basic_simulation():
    """Test basic simulation functionality."""
    print("Testing basic simulation...")
    
    # Sample base history
    base_history = {
        "total_screen_time": 360,
        "social_screen_time": 120,
        "work_screen_time": 240,
        "notification_count": 45,
        "energy_level": 0.7,
        "focus_level": 0.6,
        "mood_level": 0.75
    }
    
    # Test scenario: Digital detox
    scenario_params = {
        "social_usage_delta_min": -60,
        "screen_break_frequency": 3,
        "notification_delta": -20
    }
    
    result = run_comprehensive_scenario(base_history, scenario_params)
    
    print("✅ Basic simulation completed")
    print(f"Model available: {result['model_info']['model_available']}")
    print(f"Prediction confidence: {result['model_info']['prediction_confidence']:.2f}")
    
    if 'impact_analysis' in result:
        overall = result['impact_analysis']['overall']
        print(f"Overall impact: {overall['interpretation']}")
        print(f"Recommendation: {overall['recommendation']}")
    
    return result


def test_preset_scenarios():
    """Test various preset scenarios."""
    print("\nTesting preset scenarios...")
    
    base_history = {
        "total_screen_time": 300,
        "social_screen_time": 90,
        "work_screen_time": 180,
        "notification_count": 35,
        "energy_level": 0.6,
        "focus_level": 0.5,
        "mood_level": 0.65
    }
    
    presets = [
        {
            "name": "Digital Detox",
            "params": {
                "social_usage_delta_min": -60,
                "screen_break_frequency": 3,
                "notification_delta": -20
            }
        },
        {
            "name": "Productivity Boost",
            "params": {
                "bedtime_shift_hours": -1,
                "work_app_delta_min": 30,
                "social_usage_delta_min": -30,
                "screen_break_frequency": 2
            }
        },
        {
            "name": "Wellness Focus",
            "params": {
                "exercise_delta_min": 30,
                "bedtime_shift_hours": -1,
                "sleep_quality_modifier": 0.2,
                "social_usage_delta_min": -45
            }
        }
    ]
    
    for preset in presets:
        print(f"\n--- Testing {preset['name']} ---")
        result = run_comprehensive_scenario(base_history, preset['params'])
        
        if 'impact_analysis' in result:
            overall = result['impact_analysis']['overall']
            print(f"Impact: {overall['interpretation']}")
            
            # Show metric changes
            for metric in ['energy', 'focus', 'mood']:
                if metric in result['impact_analysis']:
                    change = result['impact_analysis'][metric]['average_change']
                    print(f"{metric.capitalize()}: {change:+.3f}")
    
    print("✅ Preset scenarios tested")


def test_extreme_scenarios():
    """Test edge cases and extreme scenarios."""
    print("\nTesting extreme scenarios...")
    
    base_history = {
        "total_screen_time": 480,  # 8 hours
        "social_screen_time": 180,  # 3 hours
        "work_screen_time": 300,   # 5 hours
        "notification_count": 100,
        "energy_level": 0.3,  # Low energy
        "focus_level": 0.2,   # Low focus
        "mood_level": 0.4     # Low mood
    }
    
    # Extreme digital detox
    extreme_params = {
        "social_usage_delta_min": -150,  # Cut social media drastically
        "bedtime_shift_hours": -2,       # Much earlier bedtime
        "exercise_delta_min": 60,        # Add 1 hour exercise
        "screen_break_frequency": 6,     # Break every 10 minutes
        "notification_delta": -80,       # Drastically reduce notifications
        "sleep_quality_modifier": 0.5    # Improve sleep quality
    }
    
    result = run_comprehensive_scenario(base_history, extreme_params)
    
    print("--- Extreme Digital Detox ---")
    if 'impact_analysis' in result:
        overall = result['impact_analysis']['overall']
        print(f"Impact: {overall['interpretation']}")
        print(f"Recommendation: {overall['recommendation']}")
        
        for metric in ['energy', 'focus', 'mood']:
            if metric in result['impact_analysis']:
                impact = result['impact_analysis'][metric]
                change = impact['average_change']
                percent = impact['percent_change']
                print(f"{metric.capitalize()}: {change:+.3f} ({percent:+.1f}%)")
    
    print("✅ Extreme scenarios tested")


def save_test_results():
    """Save test results for debugging."""
    print("\nSaving test results...")
    
    base_history = {
        "total_screen_time": 360,
        "social_screen_time": 120,
        "work_screen_time": 240,
        "notification_count": 45,
        "energy_level": 0.7,
        "focus_level": 0.6,
        "mood_level": 0.75
    }
    
    scenario_params = {
        "social_usage_delta_min": -30,
        "exercise_delta_min": 20,
        "screen_break_frequency": 2
    }
    
    result = run_comprehensive_scenario(base_history, scenario_params)
    
    # Save to file with custom serialization
    output_path = project_root / "simulation_engine" / "test_results.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"✅ Test results saved to {output_path}")


def main():
    """Run all tests."""
    print("🚀 Starting simulation engine tests...\n")
    
    try:
        test_basic_simulation()
        test_preset_scenarios()
        test_extreme_scenarios()
        save_test_results()
        
        print("\n🎉 All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())