import React, { useState, useEffect } from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

interface SimulationParams {
  bedtime_shift_hours: number;
  social_usage_delta_min: number;
  work_app_delta_min: number;
  exercise_delta_min: number;
  notification_delta: number;
  screen_break_frequency: number;
  sleep_quality_modifier: number;
}

interface SimulationResult {
  baseline: {
    hours_ahead: number[];
    energy: number[];
    focus: number[];
    mood: number[];
  };
  simulated: {
    hours_ahead: number[];
    energy: number[];
    focus: number[];
    mood: number[];
  };
  impact_analysis: {
    [key: string]: {
      average_change: number;
      percent_change: number;
      improvement: boolean;
      degradation: boolean;
    };
    overall: {
      score: number;
      interpretation: string;
      recommendation: string;
    };
  };
  model_info: {
    model_available: boolean;
    prediction_confidence: number;
  };
}

interface Preset {
  name: string;
  description: string;
  parameters: Partial<SimulationParams>;
}

const SimulationDashboard: React.FC = () => {
  const [simulationParams, setSimulationParams] = useState<SimulationParams>({
    bedtime_shift_hours: 0,
    social_usage_delta_min: 0,
    work_app_delta_min: 0,
    exercise_delta_min: 0,
    notification_delta: 0,
    screen_break_frequency: 0,
    sleep_quality_modifier: 0,
  });

  const [baseHistory] = useState({
    total_screen_time: 360,
    social_screen_time: 120,
    work_screen_time: 240,
    notification_count: 45,
    energy_level: 0.7,
    focus_level: 0.6,
    mood_level: 0.75,
  });

  const [simulationResult, setSimulationResult] = useState<SimulationResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [presets, setPresets] = useState<Preset[]>([]);
  const [selectedPreset, setSelectedPreset] = useState<string>('');

  // Load presets on component mount
  useEffect(() => {
    fetchPresets();
  }, []);

  const fetchPresets = async () => {
    try {
      const response = await fetch('/api/simulate/presets');
      const data = await response.json();
      setPresets(data);
    } catch (error) {
      console.error('Failed to fetch presets:', error);
    }
  };

  const runSimulation = async () => {
    setLoading(true);
    try {
      const response = await fetch('/api/simulate/comprehensive', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          base_history: baseHistory,
          ...simulationParams,
        }),
      });

      if (!response.ok) {
        throw new Error('Simulation failed');
      }

      const result = await response.json();
      setSimulationResult(result);
    } catch (error) {
      console.error('Simulation error:', error);
      alert('Simulation failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const applyPreset = (presetName: string) => {
    const preset = presets.find(p => p.name === presetName);
    if (preset) {
      setSimulationParams(prev => ({
        ...prev,
        ...preset.parameters,
      }));
      setSelectedPreset(presetName);
    }
  };

  const resetParams = () => {
    setSimulationParams({
      bedtime_shift_hours: 0,
      social_usage_delta_min: 0,
      work_app_delta_min: 0,
      exercise_delta_min: 0,
      notification_delta: 0,
      screen_break_frequency: 0,
      sleep_quality_modifier: 0,
    });
    setSelectedPreset('');
  };

  const updateParam = (key: keyof SimulationParams, value: number) => {
    setSimulationParams(prev => ({
      ...prev,
      [key]: value,
    }));
    setSelectedPreset(''); // Clear preset selection when manually adjusting
  };

  const getChartData = () => {
    if (!simulationResult) return null;

    const { baseline, simulated } = simulationResult;
    const hours = baseline.hours_ahead;

    return {
      energy: {
        labels: hours.map(h => `${h}h`),
        datasets: [
          {
            label: 'Baseline Energy',
            data: baseline.energy,
            borderColor: 'rgb(75, 192, 192)',
            backgroundColor: 'rgba(75, 192, 192, 0.2)',
            tension: 0.1,
          },
          {
            label: 'Simulated Energy',
            data: simulated.energy,
            borderColor: 'rgb(255, 99, 132)',
            backgroundColor: 'rgba(255, 99, 132, 0.2)',
            tension: 0.1,
          },
        ],
      },
      focus: {
        labels: hours.map(h => `${h}h`),
        datasets: [
          {
            label: 'Baseline Focus',
            data: baseline.focus,
            borderColor: 'rgb(54, 162, 235)',
            backgroundColor: 'rgba(54, 162, 235, 0.2)',
            tension: 0.1,
          },
          {
            label: 'Simulated Focus',
            data: simulated.focus,
            borderColor: 'rgb(255, 206, 86)',
            backgroundColor: 'rgba(255, 206, 86, 0.2)',
            tension: 0.1,
          },
        ],
      },
      mood: {
        labels: hours.map(h => `${h}h`),
        datasets: [
          {
            label: 'Baseline Mood',
            data: baseline.mood,
            borderColor: 'rgb(153, 102, 255)',
            backgroundColor: 'rgba(153, 102, 255, 0.2)',
            tension: 0.1,
          },
          {
            label: 'Simulated Mood',
            data: simulated.mood,
            borderColor: 'rgb(255, 159, 64)',
            backgroundColor: 'rgba(255, 159, 64, 0.2)',
            tension: 0.1,
          },
        ],
      },
    };
  };

  const chartOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top' as const,
      },
      title: {
        display: true,
        text: 'Behavioral Predictions',
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        max: 1,
        title: {
          display: true,
          text: 'Level (0-1)',
        },
      },
      x: {
        title: {
          display: true,
          text: 'Hours Ahead',
        },
      },
    },
  };

  const chartData = getChartData();

  return (
    <div className="simulation-dashboard">
      <div className="dashboard-header">
        <h1>LifeTwin Behavioral Simulation</h1>
        <p>Explore how different lifestyle changes might affect your energy, focus, and mood</p>
      </div>

      <div className="dashboard-content">
        {/* Control Panel */}
        <div className="control-panel">
          <h2>Simulation Controls</h2>
          
          {/* Presets */}
          <div className="presets-section">
            <h3>Quick Presets</h3>
            <div className="preset-buttons">
              {presets.map(preset => (
                <button
                  key={preset.name}
                  className={`preset-btn ${selectedPreset === preset.name ? 'active' : ''}`}
                  onClick={() => applyPreset(preset.name)}
                  title={preset.description}
                >
                  {preset.name}
                </button>
              ))}
              <button className="reset-btn" onClick={resetParams}>
                Reset
              </button>
            </div>
          </div>

          {/* Manual Controls */}
          <div className="manual-controls">
            <h3>Custom Adjustments</h3>
            
            <div className="control-group">
              <label>
                Bedtime Shift (hours):
                <input
                  type="range"
                  min="-6"
                  max="6"
                  step="0.5"
                  value={simulationParams.bedtime_shift_hours}
                  onChange={(e) => updateParam('bedtime_shift_hours', parseFloat(e.target.value))}
                />
                <span>{simulationParams.bedtime_shift_hours > 0 ? '+' : ''}{simulationParams.bedtime_shift_hours}h</span>
              </label>
            </div>

            <div className="control-group">
              <label>
                Social Media Change (minutes):
                <input
                  type="range"
                  min="-300"
                  max="300"
                  step="15"
                  value={simulationParams.social_usage_delta_min}
                  onChange={(e) => updateParam('social_usage_delta_min', parseInt(e.target.value))}
                />
                <span>{simulationParams.social_usage_delta_min > 0 ? '+' : ''}{simulationParams.social_usage_delta_min}min</span>
              </label>
            </div>

            <div className="control-group">
              <label>
                Work App Change (minutes):
                <input
                  type="range"
                  min="-480"
                  max="480"
                  step="30"
                  value={simulationParams.work_app_delta_min}
                  onChange={(e) => updateParam('work_app_delta_min', parseInt(e.target.value))}
                />
                <span>{simulationParams.work_app_delta_min > 0 ? '+' : ''}{simulationParams.work_app_delta_min}min</span>
              </label>
            </div>

            <div className="control-group">
              <label>
                Exercise Change (minutes):
                <input
                  type="range"
                  min="-120"
                  max="120"
                  step="10"
                  value={simulationParams.exercise_delta_min}
                  onChange={(e) => updateParam('exercise_delta_min', parseInt(e.target.value))}
                />
                <span>{simulationParams.exercise_delta_min > 0 ? '+' : ''}{simulationParams.exercise_delta_min}min</span>
              </label>
            </div>

            <div className="control-group">
              <label>
                Notification Change:
                <input
                  type="range"
                  min="-100"
                  max="100"
                  step="5"
                  value={simulationParams.notification_delta}
                  onChange={(e) => updateParam('notification_delta', parseInt(e.target.value))}
                />
                <span>{simulationParams.notification_delta > 0 ? '+' : ''}{simulationParams.notification_delta}</span>
              </label>
            </div>

            <div className="control-group">
              <label>
                Screen Breaks (per hour):
                <input
                  type="range"
                  min="0"
                  max="12"
                  step="1"
                  value={simulationParams.screen_break_frequency}
                  onChange={(e) => updateParam('screen_break_frequency', parseInt(e.target.value))}
                />
                <span>{simulationParams.screen_break_frequency}</span>
              </label>
            </div>

            <div className="control-group">
              <label>
                Sleep Quality Modifier:
                <input
                  type="range"
                  min="-1"
                  max="1"
                  step="0.1"
                  value={simulationParams.sleep_quality_modifier}
                  onChange={(e) => updateParam('sleep_quality_modifier', parseFloat(e.target.value))}
                />
                <span>{simulationParams.sleep_quality_modifier > 0 ? '+' : ''}{simulationParams.sleep_quality_modifier.toFixed(1)}</span>
              </label>
            </div>
          </div>

          <button 
            className="run-simulation-btn" 
            onClick={runSimulation}
            disabled={loading}
          >
            {loading ? 'Running Simulation...' : 'Run Simulation'}
          </button>
        </div>

        {/* Results Panel */}
        <div className="results-panel">
          {simulationResult && (
            <>
              <h2>Simulation Results</h2>
              
              {/* Model Info */}
              <div className="model-info">
                <p>
                  Model Status: {simulationResult.model_info.model_available ? '✅ Available' : '⚠️ Fallback'}
                  {simulationResult.model_info.model_available && (
                    <span> (Confidence: {(simulationResult.model_info.prediction_confidence * 100).toFixed(0)}%)</span>
                  )}
                </p>
              </div>

              {/* Impact Analysis */}
              {simulationResult.impact_analysis && (
                <div className="impact-analysis">
                  <h3>Impact Analysis</h3>
                  <div className="overall-impact">
                    <p><strong>{simulationResult.impact_analysis.overall.interpretation}</strong></p>
                    <p>{simulationResult.impact_analysis.overall.recommendation}</p>
                  </div>
                  
                  <div className="metric-impacts">
                    {['energy', 'focus', 'mood'].map(metric => {
                      const impact = simulationResult.impact_analysis[metric];
                      if (!impact) return null;
                      
                      return (
                        <div key={metric} className={`metric-impact ${impact.improvement ? 'positive' : impact.degradation ? 'negative' : 'neutral'}`}>
                          <span className="metric-name">{metric.charAt(0).toUpperCase() + metric.slice(1)}:</span>
                          <span className="change-value">
                            {impact.average_change > 0 ? '+' : ''}{(impact.average_change * 100).toFixed(1)}%
                          </span>
                          <span className="change-indicator">
                            {impact.improvement ? '📈' : impact.degradation ? '📉' : '➡️'}
                          </span>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              {/* Charts */}
              <div className="charts-container">
                {chartData && (
                  <>
                    <div className="chart">
                      <h4>Energy Levels</h4>
                      <Line data={chartData.energy} options={chartOptions} />
                    </div>
                    
                    <div className="chart">
                      <h4>Focus Levels</h4>
                      <Line data={chartData.focus} options={chartOptions} />
                    </div>
                    
                    <div className="chart">
                      <h4>Mood Levels</h4>
                      <Line data={chartData.mood} options={chartOptions} />
                    </div>
                  </>
                )}
              </div>
            </>
          )}
        </div>
      </div>

      <style jsx>{`
        .simulation-dashboard {
          max-width: 1400px;
          margin: 0 auto;
          padding: 20px;
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }

        .dashboard-header {
          text-align: center;
          margin-bottom: 30px;
        }

        .dashboard-header h1 {
          color: #2c3e50;
          margin-bottom: 10px;
        }

        .dashboard-content {
          display: grid;
          grid-template-columns: 350px 1fr;
          gap: 30px;
        }

        .control-panel {
          background: #f8f9fa;
          padding: 20px;
          border-radius: 8px;
          border: 1px solid #e9ecef;
        }

        .control-panel h2 {
          margin-top: 0;
          color: #495057;
        }

        .presets-section {
          margin-bottom: 25px;
        }

        .preset-buttons {
          display: flex;
          flex-wrap: wrap;
          gap: 8px;
          margin-top: 10px;
        }

        .preset-btn, .reset-btn {
          padding: 8px 12px;
          border: 1px solid #dee2e6;
          background: white;
          border-radius: 4px;
          cursor: pointer;
          font-size: 12px;
          transition: all 0.2s;
        }

        .preset-btn:hover, .reset-btn:hover {
          background: #e9ecef;
        }

        .preset-btn.active {
          background: #007bff;
          color: white;
          border-color: #007bff;
        }

        .reset-btn {
          background: #6c757d;
          color: white;
          border-color: #6c757d;
        }

        .manual-controls h3 {
          margin-bottom: 15px;
          color: #495057;
        }

        .control-group {
          margin-bottom: 15px;
        }

        .control-group label {
          display: flex;
          flex-direction: column;
          gap: 5px;
          font-size: 14px;
          color: #495057;
        }

        .control-group input[type="range"] {
          width: 100%;
        }

        .control-group span {
          font-weight: bold;
          color: #007bff;
          text-align: center;
        }

        .run-simulation-btn {
          width: 100%;
          padding: 12px;
          background: #28a745;
          color: white;
          border: none;
          border-radius: 4px;
          font-size: 16px;
          cursor: pointer;
          margin-top: 20px;
        }

        .run-simulation-btn:hover:not(:disabled) {
          background: #218838;
        }

        .run-simulation-btn:disabled {
          background: #6c757d;
          cursor: not-allowed;
        }

        .results-panel {
          background: white;
          padding: 20px;
          border-radius: 8px;
          border: 1px solid #e9ecef;
        }

        .model-info {
          background: #f8f9fa;
          padding: 10px;
          border-radius: 4px;
          margin-bottom: 20px;
          font-size: 14px;
        }

        .impact-analysis {
          background: #f8f9fa;
          padding: 15px;
          border-radius: 4px;
          margin-bottom: 20px;
        }

        .overall-impact {
          margin-bottom: 15px;
        }

        .metric-impacts {
          display: flex;
          gap: 15px;
          flex-wrap: wrap;
        }

        .metric-impact {
          display: flex;
          align-items: center;
          gap: 8px;
          padding: 8px 12px;
          border-radius: 4px;
          font-size: 14px;
        }

        .metric-impact.positive {
          background: #d4edda;
          color: #155724;
        }

        .metric-impact.negative {
          background: #f8d7da;
          color: #721c24;
        }

        .metric-impact.neutral {
          background: #e2e3e5;
          color: #383d41;
        }

        .charts-container {
          display: grid;
          grid-template-columns: 1fr;
          gap: 20px;
        }

        .chart {
          background: white;
          padding: 15px;
          border-radius: 4px;
          border: 1px solid #e9ecef;
        }

        .chart h4 {
          margin-top: 0;
          margin-bottom: 15px;
          color: #495057;
        }

        @media (max-width: 1200px) {
          .dashboard-content {
            grid-template-columns: 1fr;
          }
        }
      `}</style>
    </div>
  );
};

export default SimulationDashboard;