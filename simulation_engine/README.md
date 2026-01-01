# LifeTwin OS Simulation Engine

The simulation engine provides comprehensive behavioral modeling and prediction capabilities for the LifeTwin OS platform. It allows users to explore "what-if" scenarios and understand how different lifestyle changes might affect their digital wellbeing.

## Features

### 🎯 Comprehensive Behavioral Modeling
- **Bedtime adjustments**: Model the impact of earlier/later bedtimes on energy and mood
- **App usage modifications**: Simulate changes in social media, work apps, and exercise apps
- **Notification management**: Explore the effects of reducing notification frequency
- **Screen break patterns**: Model the benefits of regular screen breaks
- **Sleep quality factors**: Incorporate sleep quality improvements into predictions

### 📊 Advanced Predictions
- **Multi-metric forecasting**: Predicts energy, focus, and mood levels over time
- **Trend analysis**: Calculates behavioral trends and their changes
- **Impact scoring**: Quantifies the overall impact of lifestyle changes
- **Confidence intervals**: Provides prediction confidence based on model quality

### 🚀 API Integration
- **RESTful API**: Complete FastAPI integration with comprehensive endpoints
- **Preset scenarios**: Pre-configured scenarios for common use cases
- **Real-time processing**: Fast scenario execution with caching
- **Health monitoring**: API health checks and model status reporting

### 🎨 Interactive Dashboard
- **Visual controls**: Intuitive sliders and controls for all parameters
- **Real-time charts**: Side-by-side comparison of baseline vs simulated predictions
- **Impact visualization**: Color-coded impact analysis with recommendations
- **Responsive design**: Works across desktop and mobile devices

## Quick Start

### Running a Basic Simulation

```python
from simulation_engine.engine import run_comprehensive_scenario

# Define baseline behavioral data
base_history = {
    "total_screen_time": 360,  # 6 hours
    "social_screen_time": 120,  # 2 hours
    "work_screen_time": 240,   # 4 hours
    "notification_count": 45,
    "energy_level": 0.7,
    "focus_level": 0.6,
    "mood_level": 0.75
}

# Define scenario modifications
scenario_params = {
    "social_usage_delta_min": -60,  # Reduce social media by 1 hour
    "screen_break_frequency": 3,    # Take breaks every 20 minutes
    "notification_delta": -20       # Reduce notifications by 20
}

# Run simulation
result = run_comprehensive_scenario(base_history, scenario_params)

# Access results
print(f"Overall impact: {result['impact_analysis']['overall']['interpretation']}")
print(f"Recommendation: {result['impact_analysis']['overall']['recommendation']}")
```

### Using the API

```bash
# Run a comprehensive simulation
curl -X POST "http://localhost:8000/simulate/comprehensive" \
  -H "Content-Type: application/json" \
  -d '{
    "base_history": {
      "total_screen_time": 360,
      "energy_level": 0.7,
      "focus_level": 0.6,
      "mood_level": 0.75
    },
    "social_usage_delta_min": -30,
    "exercise_delta_min": 20,
    "screen_break_frequency": 2
  }'

# Get available presets
curl "http://localhost:8000/simulate/presets"

# Run a preset scenario
curl -X POST "http://localhost:8000/simulate/preset/digital_detox" \
  -H "Content-Type: application/json" \
  -d '{"total_screen_time": 360, "energy_level": 0.7}'
```

## Available Presets

### 🧘 Digital Detox
- Reduces social media usage by 60 minutes
- Increases screen breaks to every 20 minutes
- Reduces notifications by 20

### 💼 Productivity Boost
- Shifts bedtime 1 hour earlier
- Increases work app usage by 30 minutes
- Reduces social media by 30 minutes
- Adds regular screen breaks

### 🏃 Wellness Focus
- Adds 30 minutes of exercise
- Improves sleep quality
- Reduces social media usage
- Earlier bedtime

### 🎯 Mindful Usage
- Balanced approach with moderate changes
- Regular screen breaks
- Reduced notifications
- Slight reduction in all screen time

### 🌅 Early Bird
- 2-hour earlier bedtime
- Increased morning productivity
- Added exercise routine
- Improved sleep quality

## Architecture

### Core Components

1. **SimulationEngine**: Main engine class handling scenario processing
2. **Model Integration**: Seamless integration with trained ML models
3. **Impact Analysis**: Sophisticated impact calculation and interpretation
4. **API Layer**: FastAPI endpoints with comprehensive validation
5. **Dashboard UI**: React-based interactive interface

### Data Flow

```
User Input → Scenario Parameters → Model Processing → Impact Analysis → Visualization
     ↓              ↓                    ↓               ↓              ↓
Dashboard → API Validation → Engine Processing → Results → Charts/Metrics
```

## Testing

Run the comprehensive test suite:

```bash
python simulation_engine/test_simulation.py
```

The test suite includes:
- Basic functionality tests
- Preset scenario validation
- Extreme scenario edge cases
- JSON serialization verification
- Performance benchmarking

## Model Integration

The simulation engine automatically detects and uses trained time-series models:

- **Model Detection**: Automatically loads models from `ml/models/`
- **Fallback Logic**: Graceful degradation when models aren't available
- **Confidence Scoring**: Adjusts confidence based on model performance
- **Metadata Integration**: Uses model metadata for enhanced predictions

## Performance

- **Fast Execution**: Typical simulation completes in <100ms
- **Caching**: Intelligent caching of predictions and model objects
- **Scalability**: Designed for concurrent API requests
- **Memory Efficient**: Minimal memory footprint with lazy loading

## Future Enhancements

- **Real-time Learning**: Continuous model updates based on user feedback
- **Personalization**: User-specific model fine-tuning
- **Advanced Scenarios**: More complex multi-day scenario modeling
- **Social Features**: Collaborative scenario sharing and comparison
- **Mobile Integration**: Native mobile app integration with Android ML inference

---

The simulation engine represents a significant advancement in digital wellbeing technology, providing users with actionable insights into how their behavioral choices affect their overall wellbeing and productivity.