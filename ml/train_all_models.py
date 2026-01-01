"""
Comprehensive ML training pipeline for LifeTwin OS.

This script orchestrates the complete ML training workflow:
1. Export data from Android database
2. Train next-app sequence model
3. Train time-series forecasting model
4. Export models to ONNX (if requested)
5. Deploy models to Android device
6. Generate training report

Usage:
    python ml/train_all_models.py --db-path /path/to/android.db
    python ml/train_all_models.py --use-demo-data  # Use demo data for testing
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import tempfile
import shutil

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MLTrainingPipeline:
    """Orchestrates the complete ML training pipeline."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.results = {
            'pipeline_start': datetime.utcnow().isoformat() + 'Z',
            'steps_completed': [],
            'models_trained': {},
            'deployment_status': {},
            'errors': []
        }
        
        # Paths
        self.data_dir = Path(config.get('data_dir', 'ml/data'))
        self.models_dir = Path(config.get('models_dir', 'ml/models'))
        self.reports_dir = Path(config.get('reports_dir', 'ml/reports'))
        
        # Create directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
    
    def step_1_export_data(self) -> bool:
        """Step 1: Export data from Android database or use demo data."""
        logger.info("Step 1: Exporting training data...")
        
        try:
            if self.config.get('use_demo_data', False):
                # Generate demo data
                success = self._generate_demo_data()
            else:
                # Export from Android database
                db_path = self.config.get('db_path')
                if not db_path:
                    raise ValueError("Database path required when not using demo data")
                
                success = self._export_android_data(db_path)
            
            if success:
                self.results['steps_completed'].append('data_export')
                logger.info("✓ Data export completed successfully")
                return True
            else:
                self.results['errors'].append("Data export failed")
                return False
                
        except Exception as e:
            logger.error(f"Data export failed: {e}")
            self.results['errors'].append(f"Data export error: {str(e)}")
            return False
    
    def step_2_train_sequence_model(self) -> bool:
        """Step 2: Train next-app sequence prediction model."""
        logger.info("Step 2: Training next-app sequence model...")
        
        try:
            # Check if data is available
            data_file = self.data_dir / "comprehensive_export.json"
            if not data_file.exists():
                data_file = self.data_dir / "summaries_export.json"
            
            if not data_file.exists():
                raise FileNotFoundError("No training data available")
            
            # Prepare training arguments
            train_args = [
                sys.executable, "-m", "ml.sequence_models.train_next_app_model",
                "--data-path", str(data_file),
                "--epochs", str(self.config.get('sequence_epochs', 20)),
                "--batch-size", str(self.config.get('sequence_batch_size', 32)),
                "--model-dir", str(self.models_dir)
            ]
            
            # Add optional arguments
            if self.config.get('use_categories', False):
                train_args.append("--use-categories")
            
            if self.config.get('export_onnx', False):
                train_args.extend(["--export-onnx", "--quantize"])
            
            # Run training
            result = subprocess.run(train_args, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Load training results
                metrics_file = self.models_dir / "next_app_model.metrics.json"
                if metrics_file.exists():
                    with open(metrics_file, 'r') as f:
                        metrics = json.load(f)
                    self.results['models_trained']['next_app'] = metrics
                
                self.results['steps_completed'].append('sequence_model_training')
                logger.info("✓ Next-app sequence model training completed")
                return True
            else:
                logger.error(f"Sequence model training failed: {result.stderr}")
                self.results['errors'].append(f"Sequence training error: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Sequence model training failed: {e}")
            self.results['errors'].append(f"Sequence training error: {str(e)}")
            return False
    
    def step_3_train_timeseries_model(self) -> bool:
        """Step 3: Train time-series forecasting model."""
        logger.info("Step 3: Training time-series forecasting model...")
        
        try:
            # Check if data is available
            data_file = self.data_dir / "comprehensive_export.json"
            if not data_file.exists():
                data_file = self.data_dir / "summaries_export.json"
            
            if not data_file.exists():
                raise FileNotFoundError("No training data available")
            
            # Prepare training arguments
            train_args = [
                sys.executable, "-m", "ml.time_series_models.train_twin",
                "--data-path", str(data_file),
                "--model-type", self.config.get('timeseries_model_type', 'rf'),
                "--model-dir", str(self.models_dir)
            ]
            
            # Add model-specific arguments
            if self.config.get('timeseries_model_type') in ['lstm', 'transformer']:
                train_args.extend([
                    "--epochs", str(self.config.get('timeseries_epochs', 100)),
                    "--batch-size", str(self.config.get('timeseries_batch_size', 32))
                ])
            
            # Run training
            result = subprocess.run(train_args, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Load training results
                metrics_file = self.models_dir / "time_series_twin.metrics.json"
                if metrics_file.exists():
                    with open(metrics_file, 'r') as f:
                        metrics = json.load(f)
                    self.results['models_trained']['time_series'] = metrics
                
                self.results['steps_completed'].append('timeseries_model_training')
                logger.info("✓ Time-series forecasting model training completed")
                return True
            else:
                logger.error(f"Time-series model training failed: {result.stderr}")
                self.results['errors'].append(f"Time-series training error: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Time-series model training failed: {e}")
            self.results['errors'].append(f"Time-series training error: {str(e)}")
            return False
    
    def step_4_deploy_models(self) -> bool:
        """Step 4: Deploy models to Android device."""
        if not self.config.get('deploy_models', False):
            logger.info("Step 4: Model deployment skipped (disabled in config)")
            return True
        
        logger.info("Step 4: Deploying models to Android device...")
        
        try:
            # Prepare deployment arguments
            deploy_args = [
                sys.executable, "-m", "ml.deployment.deploy_to_android",
                "--models-dir", str(self.models_dir),
                "--deployment-method", self.config.get('deployment_method', 'adb')
            ]
            
            # Add optional arguments
            if self.config.get('android_package'):
                deploy_args.extend(["--android-package", self.config['android_package']])
            
            if self.config.get('verify_deployment', True):
                deploy_args.append("--verify")
            
            # Run deployment
            result = subprocess.run(deploy_args, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.results['steps_completed'].append('model_deployment')
                self.results['deployment_status'] = {'success': True, 'method': self.config.get('deployment_method', 'adb')}
                logger.info("✓ Model deployment completed")
                return True
            else:
                logger.warning(f"Model deployment failed: {result.stderr}")
                self.results['deployment_status'] = {'success': False, 'error': result.stderr}
                # Don't fail the entire pipeline for deployment issues
                return True
                
        except Exception as e:
            logger.warning(f"Model deployment failed: {e}")
            self.results['deployment_status'] = {'success': False, 'error': str(e)}
            # Don't fail the entire pipeline for deployment issues
            return True
    
    def step_5_generate_report(self) -> bool:
        """Step 5: Generate comprehensive training report."""
        logger.info("Step 5: Generating training report...")
        
        try:
            self.results['pipeline_end'] = datetime.utcnow().isoformat() + 'Z'
            
            # Calculate pipeline duration
            start_time = datetime.fromisoformat(self.results['pipeline_start'].replace('Z', '+00:00'))
            end_time = datetime.fromisoformat(self.results['pipeline_end'].replace('Z', '+00:00'))
            duration = (end_time - start_time).total_seconds()
            
            # Create comprehensive report
            report = {
                'pipeline_summary': {
                    'start_time': self.results['pipeline_start'],
                    'end_time': self.results['pipeline_end'],
                    'duration_seconds': duration,
                    'steps_completed': self.results['steps_completed'],
                    'success': len(self.results['errors']) == 0,
                    'errors': self.results['errors']
                },
                'training_config': self.config,
                'models_trained': self.results['models_trained'],
                'deployment_status': self.results['deployment_status'],
                'model_files': self._get_model_files_info(),
                'recommendations': self._generate_recommendations()
            }
            
            # Save report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = self.reports_dir / f"training_report_{timestamp}.json"
            
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            # Generate human-readable summary
            summary_file = self.reports_dir / f"training_summary_{timestamp}.md"
            self._generate_markdown_summary(report, summary_file)
            
            logger.info(f"✓ Training report generated: {report_file}")
            logger.info(f"✓ Training summary generated: {summary_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return False
    
    def run_pipeline(self) -> bool:
        """Run the complete ML training pipeline."""
        logger.info("Starting ML training pipeline...")
        
        steps = [
            ("Data Export", self.step_1_export_data),
            ("Sequence Model Training", self.step_2_train_sequence_model),
            ("Time-Series Model Training", self.step_3_train_timeseries_model),
            ("Model Deployment", self.step_4_deploy_models),
            ("Report Generation", self.step_5_generate_report)
        ]
        
        for step_name, step_func in steps:
            logger.info(f"\n{'='*60}")
            logger.info(f"Starting: {step_name}")
            logger.info(f"{'='*60}")
            
            success = step_func()
            
            if not success:
                logger.error(f"Pipeline failed at step: {step_name}")
                return False
        
        logger.info(f"\n{'='*60}")
        logger.info("🎉 ML Training Pipeline Completed Successfully!")
        logger.info(f"{'='*60}")
        
        # Print summary
        self._print_pipeline_summary()
        
        return True
    
    # Helper methods
    
    def _generate_demo_data(self) -> bool:
        """Generate demo data for testing."""
        try:
            # Run backend demo export
            result = subprocess.run([
                sys.executable, "-c",
                "from backend.fastapi.scripts.demo_export import run; import asyncio; asyncio.run(run())"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("Demo data generated successfully")
                return True
            else:
                logger.error(f"Demo data generation failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Demo data generation error: {e}")
            return False
    
    def _export_android_data(self, db_path: str) -> bool:
        """Export data from Android database."""
        try:
            export_args = [
                sys.executable, "-m", "ml.data_pipeline.export_from_android",
                "--db-path", db_path,
                "--output-dir", str(self.data_dir),
                "--days-back", str(self.config.get('export_days', 30))
            ]
            
            result = subprocess.run(export_args, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("Android data exported successfully")
                return True
            else:
                logger.error(f"Android data export failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Android data export error: {e}")
            return False
    
    def _get_model_files_info(self) -> Dict[str, Any]:
        """Get information about generated model files."""
        model_files = {}
        
        for model_file in self.models_dir.glob("*"):
            if model_file.is_file():
                model_files[model_file.name] = {
                    'size_bytes': model_file.stat().st_size,
                    'size_mb': round(model_file.stat().st_size / (1024 * 1024), 2),
                    'modified': datetime.fromtimestamp(model_file.stat().st_mtime).isoformat()
                }
        
        return model_files
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on training results."""
        recommendations = []
        
        # Check model performance
        if 'next_app' in self.results['models_trained']:
            accuracy = self.results['models_trained']['next_app'].get('test_accuracy', 0)
            if accuracy < 0.3:
                recommendations.append("Next-app model accuracy is low. Consider collecting more diverse usage data or adjusting model architecture.")
            elif accuracy > 0.7:
                recommendations.append("Next-app model shows good performance. Consider deploying to production.")
        
        if 'time_series' in self.results['models_trained']:
            mae = self.results['models_trained']['time_series'].get('overall_mae', float('inf'))
            if mae > 100:  # High MAE for screen time prediction
                recommendations.append("Time-series model has high prediction error. Consider feature engineering or different model architecture.")
        
        # Check deployment
        if not self.results['deployment_status'].get('success', False):
            recommendations.append("Model deployment failed. Ensure Android device is connected and ADB is configured.")
        
        # General recommendations
        if len(self.results['steps_completed']) == 5:
            recommendations.append("All pipeline steps completed successfully. Models are ready for production use.")
        
        return recommendations
    
    def _generate_markdown_summary(self, report: Dict[str, Any], output_file: Path):
        """Generate human-readable markdown summary."""
        summary = f"""# ML Training Pipeline Report

## Summary
- **Start Time**: {report['pipeline_summary']['start_time']}
- **End Time**: {report['pipeline_summary']['end_time']}
- **Duration**: {report['pipeline_summary']['duration_seconds']:.1f} seconds
- **Success**: {'✅ Yes' if report['pipeline_summary']['success'] else '❌ No'}
- **Steps Completed**: {len(report['pipeline_summary']['steps_completed'])}/5

## Steps Completed
"""
        
        for step in report['pipeline_summary']['steps_completed']:
            summary += f"- ✅ {step.replace('_', ' ').title()}\n"
        
        if report['pipeline_summary']['errors']:
            summary += "\n## Errors\n"
            for error in report['pipeline_summary']['errors']:
                summary += f"- ❌ {error}\n"
        
        summary += "\n## Models Trained\n"
        for model_name, metrics in report['models_trained'].items():
            summary += f"\n### {model_name.replace('_', ' ').title()}\n"
            if isinstance(metrics, dict):
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        summary += f"- **{key.replace('_', ' ').title()}**: {value:.4f}\n"
        
        summary += "\n## Model Files\n"
        for filename, info in report['model_files'].items():
            summary += f"- **{filename}**: {info['size_mb']} MB\n"
        
        if report['recommendations']:
            summary += "\n## Recommendations\n"
            for rec in report['recommendations']:
                summary += f"- {rec}\n"
        
        with open(output_file, 'w') as f:
            f.write(summary)
    
    def _print_pipeline_summary(self):
        """Print pipeline summary to console."""
        print("\n" + "="*60)
        print("ML TRAINING PIPELINE SUMMARY")
        print("="*60)
        
        print(f"Steps Completed: {len(self.results['steps_completed'])}/5")
        for step in self.results['steps_completed']:
            print(f"  ✅ {step.replace('_', ' ').title()}")
        
        if self.results['errors']:
            print(f"\nErrors: {len(self.results['errors'])}")
            for error in self.results['errors']:
                print(f"  ❌ {error}")
        
        print(f"\nModels Trained: {len(self.results['models_trained'])}")
        for model_name in self.results['models_trained']:
            print(f"  📊 {model_name.replace('_', ' ').title()}")
        
        deployment_status = self.results['deployment_status']
        if deployment_status:
            status_icon = "✅" if deployment_status.get('success') else "❌"
            print(f"\nDeployment: {status_icon} {deployment_status.get('method', 'unknown')}")
        
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Run complete ML training pipeline")
    
    # Data source options
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument("--db-path", help="Path to Android SQLite database")
    data_group.add_argument("--use-demo-data", action="store_true", 
                           help="Use demo data for testing")
    
    # Training options
    parser.add_argument("--sequence-epochs", type=int, default=20,
                       help="Epochs for sequence model training")
    parser.add_argument("--sequence-batch-size", type=int, default=32,
                       help="Batch size for sequence model")
    parser.add_argument("--use-categories", action="store_true",
                       help="Use app categories instead of specific apps")
    
    parser.add_argument("--timeseries-model-type", choices=['rf', 'lstm', 'transformer'], 
                       default='rf', help="Time-series model type")
    parser.add_argument("--timeseries-epochs", type=int, default=100,
                       help="Epochs for time-series model training")
    parser.add_argument("--timeseries-batch-size", type=int, default=32,
                       help="Batch size for time-series model")
    
    # Export and deployment options
    parser.add_argument("--export-onnx", action="store_true",
                       help="Export models to ONNX format")
    parser.add_argument("--deploy-models", action="store_true",
                       help="Deploy models to Android device")
    parser.add_argument("--deployment-method", choices=['adb', 'assets'], default='adb',
                       help="Model deployment method")
    parser.add_argument("--android-package", default="com.lifetwin.mlp",
                       help="Android package name")
    
    # Pipeline options
    parser.add_argument("--export-days", type=int, default=30,
                       help="Days of data to export for training")
    parser.add_argument("--models-dir", default="ml/models",
                       help="Directory to save trained models")
    parser.add_argument("--data-dir", default="ml/data",
                       help="Directory for training data")
    parser.add_argument("--reports-dir", default="ml/reports",
                       help="Directory for training reports")
    
    args = parser.parse_args()
    
    # Create pipeline configuration
    config = vars(args)
    
    # Run pipeline
    pipeline = MLTrainingPipeline(config)
    success = pipeline.run_pipeline()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()