"""Enhanced simulation engine for LifeTwin OS.

Supports comprehensive behavioral scenarios using trained time-series forecasting models.
Integrates with the advanced ML models to provide realistic predictions.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path
import json
from datetime import datetime, timedelta

from ml.time_series_models.forecast_twin import forecast_next_hours
from simulation_engine.model_loader import load_time_series_model

logger = logging.getLogger(__name__)


class SimulationEngine:
    """Enhanced simulation engine with comprehensive scenario support."""
    
    def __init__(self):
        self.model_obj = None
        self.model_metadata = None
        self._load_model()
    
    def _load_model(self):
        """Load the trained time-series model and its metadata."""
        try:
            self.model_obj = load_time_series_model()
            
            # Load model metadata for better feature engineering
            project_root = Path(__file__).resolve().parents[1]
            metadata_path = project_root / 'ml' / 'models' / 'time_series_twin.json'
            
            if metadata_path.exists():
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    self.model_metadata = json.load(f)
                logger.info(f"Loaded model metadata: {self.model_metadata.get('model_name', 'unknown')}")
            
        except Exception as e:
            logger.warning(f"Failed to load model: {e}")
            self.model_obj = None
            self.model_metadata = None
    
    def run_comprehensive_scenario(self, 
                                 base_history: Dict,
                                 scenario_params: Dict) -> Dict:
        """Run a comprehensive behavioral scenario with multiple parameters.
        
        Args:
            base_history: Historical behavioral data
            scenario_params: Dictionary of scenario modifications including:
                - bedtime_shift_hours: Change in bedtime (positive = later)
                - social_usage_delta_min: Change in social media usage (minutes)
                - work_app_delta_min: Change in productivity app usage
                - exercise_delta_min: Change in fitness/health app usage
                - notification_delta: Change in notification frequency
                - screen_break_frequency: Frequency of screen breaks (per hour)
                - sleep_quality_modifier: Sleep quality adjustment (-1 to 1)
        
        Returns:
            Dictionary with baseline and simulated predictions
        """
        
        # Create modified scenario data
        modified_history = self._apply_scenario_modifications(base_history, scenario_params)
        
        # Generate predictions for both baseline and modified scenarios
        baseline_forecast = self._generate_forecast(base_history, "baseline")
        simulated_forecast = self._generate_forecast(modified_history, "simulated")
        
        # Calculate impact metrics
        impact_analysis = self._calculate_impact_metrics(baseline_forecast, simulated_forecast)
        
        return {
            "baseline": baseline_forecast,
            "simulated": simulated_forecast,
            "impact_analysis": impact_analysis,
            "scenario_params": scenario_params,
            "model_info": {
                "model_available": self.model_obj is not None,
                "model_type": self.model_metadata.get('model_type', 'unknown') if self.model_metadata else 'unknown',
                "prediction_confidence": self._get_prediction_confidence()
            }
        }
    
    def _apply_scenario_modifications(self, base_history: Dict, scenario_params: Dict) -> Dict:
        """Apply scenario modifications to base history data."""
        modified = dict(base_history)
        
        # Apply bedtime shift
        bedtime_shift = scenario_params.get('bedtime_shift_hours', 0)
        if bedtime_shift != 0:
            modified['bedtime_shift_hours'] = bedtime_shift
            # Bedtime affects sleep quality and next-day energy
            sleep_impact = max(-0.3, min(0.2, -bedtime_shift * 0.1))
            modified['sleep_quality_modifier'] = modified.get('sleep_quality_modifier', 0) + sleep_impact
        
        # Apply social media usage changes
        social_delta = scenario_params.get('social_usage_delta_min', 0)
        if social_delta != 0:
            modified['social_usage_delta_min'] = social_delta
            current_social = modified.get('social_screen_time', 120)  # Default 2 hours
            new_social = max(0, current_social + social_delta)
            modified['social_screen_time'] = new_social
            
            # Social media affects focus and mood
            focus_impact = -social_delta * 0.001  # More social = less focus
            mood_impact = social_delta * 0.0005 if social_delta > 0 else social_delta * 0.002  # Complex relationship
            modified['focus_modifier'] = modified.get('focus_modifier', 0) + focus_impact
            modified['mood_modifier'] = modified.get('mood_modifier', 0) + mood_impact
        
        # Apply work/productivity app changes
        work_delta = scenario_params.get('work_app_delta_min', 0)
        if work_delta != 0:
            modified['work_app_delta_min'] = work_delta
            current_work = modified.get('work_screen_time', 240)  # Default 4 hours
            new_work = max(0, current_work + work_delta)
            modified['work_screen_time'] = new_work
            
            # Work apps affect focus and energy differently
            if work_delta > 0:
                focus_impact = work_delta * 0.0008  # More work can improve focus initially
                energy_impact = -work_delta * 0.001  # But drains energy
            else:
                focus_impact = work_delta * 0.0005  # Less work = less focus practice
                energy_impact = -work_delta * 0.0008  # But more energy
            
            modified['focus_modifier'] = modified.get('focus_modifier', 0) + focus_impact
            modified['energy_modifier'] = modified.get('energy_modifier', 0) + energy_impact
        
        # Apply exercise/fitness changes
        exercise_delta = scenario_params.get('exercise_delta_min', 0)
        if exercise_delta != 0:
            modified['exercise_delta_min'] = exercise_delta
            # Exercise positively affects energy, mood, and sleep
            energy_impact = exercise_delta * 0.002
            mood_impact = exercise_delta * 0.0015
            sleep_impact = exercise_delta * 0.001
            
            modified['energy_modifier'] = modified.get('energy_modifier', 0) + energy_impact
            modified['mood_modifier'] = modified.get('mood_modifier', 0) + mood_impact
            modified['sleep_quality_modifier'] = modified.get('sleep_quality_modifier', 0) + sleep_impact
        
        # Apply notification frequency changes
        notification_delta = scenario_params.get('notification_delta', 0)
        if notification_delta != 0:
            modified['notification_delta'] = notification_delta
            current_notifications = modified.get('notification_count', 50)
            new_notifications = max(0, current_notifications + notification_delta)
            modified['notification_count'] = new_notifications
            
            # More notifications = less focus, more stress
            focus_impact = -notification_delta * 0.01
            mood_impact = -notification_delta * 0.005
            
            modified['focus_modifier'] = modified.get('focus_modifier', 0) + focus_impact
            modified['mood_modifier'] = modified.get('mood_modifier', 0) + mood_impact
        
        # Apply screen break frequency
        screen_breaks = scenario_params.get('screen_break_frequency', 0)
        if screen_breaks > 0:
            modified['screen_break_frequency'] = screen_breaks
            # Regular breaks improve focus and energy
            focus_impact = screen_breaks * 0.05
            energy_impact = screen_breaks * 0.03
            
            modified['focus_modifier'] = modified.get('focus_modifier', 0) + focus_impact
            modified['energy_modifier'] = modified.get('energy_modifier', 0) + energy_impact
        
        # Apply direct sleep quality modifier
        sleep_modifier = scenario_params.get('sleep_quality_modifier', 0)
        if sleep_modifier != 0:
            modified['sleep_quality_modifier'] = modified.get('sleep_quality_modifier', 0) + sleep_modifier
        
        return modified
    
    def _generate_forecast(self, history_data: Dict, scenario_type: str) -> Dict:
        """Generate forecast using the trained model or fallback logic."""
        try:
            # Use the enhanced forecasting function
            forecast = forecast_next_hours(history_data, model_obj=self.model_obj)
            
            # Apply scenario modifiers if present
            if scenario_type == "simulated":
                forecast = self._apply_modifiers_to_forecast(forecast, history_data)
            
            # Add additional metrics
            forecast['scenario_type'] = scenario_type
            forecast['generated_at'] = datetime.utcnow().isoformat() + 'Z'
            
            return forecast
            
        except Exception as e:
            logger.error(f"Failed to generate forecast: {e}")
            return self._generate_fallback_forecast(scenario_type)
    
    def _apply_modifiers_to_forecast(self, forecast: Dict, modified_history: Dict) -> Dict:
        """Apply scenario modifiers to the forecast results."""
        modified_forecast = dict(forecast)
        
        # Apply energy modifier
        energy_mod = modified_history.get('energy_modifier', 0)
        if energy_mod != 0 and 'energy' in modified_forecast:
            modified_forecast['energy'] = [
                max(0.0, min(1.0, val + energy_mod)) for val in modified_forecast['energy']
            ]
        
        # Apply focus modifier
        focus_mod = modified_history.get('focus_modifier', 0)
        if focus_mod != 0 and 'focus' in modified_forecast:
            modified_forecast['focus'] = [
                max(0.0, min(1.0, val + focus_mod)) for val in modified_forecast['focus']
            ]
        
        # Apply mood modifier
        mood_mod = modified_history.get('mood_modifier', 0)
        if mood_mod != 0 and 'mood' in modified_forecast:
            modified_forecast['mood'] = [
                max(0.0, min(1.0, val + mood_mod)) for val in modified_forecast['mood']
            ]
        
        return modified_forecast
    
    def _calculate_impact_metrics(self, baseline: Dict, simulated: Dict) -> Dict:
        """Calculate impact metrics comparing baseline vs simulated scenarios."""
        impact = {}
        
        metrics = ['energy', 'focus', 'mood']
        
        for metric in metrics:
            if metric in baseline and metric in simulated:
                baseline_vals = baseline[metric]
                simulated_vals = simulated[metric]
                
                if len(baseline_vals) == len(simulated_vals):
                    # Calculate average change
                    avg_baseline = np.mean(baseline_vals)
                    avg_simulated = np.mean(simulated_vals)
                    avg_change = avg_simulated - avg_baseline
                    percent_change = (avg_change / max(avg_baseline, 0.01)) * 100
                    
                    # Calculate trend change (slope difference)
                    baseline_trend = self._calculate_trend(baseline_vals)
                    simulated_trend = self._calculate_trend(simulated_vals)
                    trend_change = simulated_trend - baseline_trend
                    
                    impact[metric] = {
                        'average_change': round(avg_change, 3),
                        'percent_change': round(percent_change, 1),
                        'trend_change': round(trend_change, 3),
                        'improvement': avg_change > 0.02,  # Threshold for meaningful improvement
                        'degradation': avg_change < -0.02  # Threshold for meaningful degradation
                    }
        
        # Overall impact score
        overall_changes = [impact[m]['average_change'] for m in impact.keys()]
        overall_score = np.mean(overall_changes) if overall_changes else 0
        
        impact['overall'] = {
            'score': round(overall_score, 3),
            'interpretation': self._interpret_overall_score(overall_score),
            'recommendation': self._generate_recommendation(impact)
        }
        
        return impact
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate the trend (slope) of a series of values."""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        y = np.array(values)
        
        # Simple linear regression slope
        slope = np.polyfit(x, y, 1)[0]
        return float(slope)
    
    def _interpret_overall_score(self, score: float) -> str:
        """Interpret the overall impact score."""
        if score > 0.1:
            return "Significantly positive impact"
        elif score > 0.05:
            return "Moderately positive impact"
        elif score > 0.02:
            return "Slightly positive impact"
        elif score > -0.02:
            return "Minimal impact"
        elif score > -0.05:
            return "Slightly negative impact"
        elif score > -0.1:
            return "Moderately negative impact"
        else:
            return "Significantly negative impact"
    
    def _generate_recommendation(self, impact: Dict) -> str:
        """Generate a recommendation based on impact analysis."""
        positive_metrics = [m for m in ['energy', 'focus', 'mood'] 
                          if m in impact and impact[m].get('improvement', False)]
        negative_metrics = [m for m in ['energy', 'focus', 'mood'] 
                          if m in impact and impact[m].get('degradation', False)]
        
        if len(positive_metrics) > len(negative_metrics):
            return f"Recommended: This scenario improves {', '.join(positive_metrics)}"
        elif len(negative_metrics) > len(positive_metrics):
            return f"Not recommended: This scenario negatively affects {', '.join(negative_metrics)}"
        else:
            return "Neutral: This scenario has mixed effects. Consider your priorities."
    
    def _get_prediction_confidence(self) -> float:
        """Get prediction confidence based on model availability and quality."""
        if not self.model_obj:
            return 0.3  # Low confidence with fallback logic
        
        if self.model_metadata:
            # Use model performance metrics to determine confidence
            overall_r2 = self.model_metadata.get('metrics', {}).get('overall_r2', 0.5)
            return min(0.95, max(0.5, overall_r2))
        
        return 0.7  # Default confidence when model is available
    
    def _generate_fallback_forecast(self, scenario_type: str) -> Dict:
        """Generate a fallback forecast when model is not available."""
        return {
            "hours_ahead": [1, 2, 3, 4, 6, 8, 12, 24],
            "energy": [0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.6],
            "focus": [0.7, 0.68, 0.65, 0.6, 0.55, 0.5, 0.45, 0.5],
            "mood": [0.75, 0.73, 0.7, 0.68, 0.65, 0.6, 0.55, 0.6],
            "scenario_type": scenario_type,
            "generated_at": datetime.utcnow().isoformat() + 'Z',
            "fallback_used": True
        }


# Global simulation engine instance
_simulation_engine = SimulationEngine()


def run_scenario(base_history: Dict, bedtime_shift_hours: int = 0, social_usage_delta_min: int = 0) -> Dict:
    """Legacy function for backward compatibility."""
    scenario_params = {
        'bedtime_shift_hours': bedtime_shift_hours,
        'social_usage_delta_min': social_usage_delta_min
    }
    return _simulation_engine.run_comprehensive_scenario(base_history, scenario_params)


def run_comprehensive_scenario(base_history: Dict, scenario_params: Dict) -> Dict:
    """Run a comprehensive behavioral scenario with multiple parameters."""
    return _simulation_engine.run_comprehensive_scenario(base_history, scenario_params)
