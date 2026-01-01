"""
Deploy trained ML models to Android device for on-device inference.

This script packages trained models and copies them to the Android app's
private storage for on-device inference.
"""

import json
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Any
import logging
import subprocess
import tempfile
import zipfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AndroidModelDeployer:
    """Deploys ML models to Android device."""
    
    def __init__(self, models_dir: str = "ml/models", android_package: str = "com.lifetwin.mlp"):
        self.models_dir = Path(models_dir)
        self.android_package = android_package
        self.device_models_path = f"/data/data/{android_package}/files/ml_models"
        
    def check_adb_connection(self) -> bool:
        """Check if ADB is available and device is connected."""
        try:
            result = subprocess.run(['adb', 'devices'], capture_output=True, text=True)
            if result.returncode != 0:
                logger.error("ADB not found. Please install Android SDK platform-tools.")
                return False
            
            devices = [line for line in result.stdout.split('\n') if '\tdevice' in line]
            if not devices:
                logger.error("No Android device connected. Please connect a device and enable USB debugging.")
                return False
            
            logger.info(f"Found {len(devices)} connected device(s)")
            return True
            
        except FileNotFoundError:
            logger.error("ADB not found. Please install Android SDK platform-tools.")
            return False
    
    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """Get list of available trained models."""
        models = {}
        
        if not self.models_dir.exists():
            logger.warning(f"Models directory not found: {self.models_dir}")
            return models
        
        # Check for next-app model
        next_app_model = self.models_dir / "next_app_model.pt"
        next_app_meta = self.models_dir / "next_app_model.json"
        next_app_vocab = self.models_dir / "vocab.json"
        
        if next_app_model.exists() and next_app_meta.exists():
            with open(next_app_meta, 'r') as f:
                metadata = json.load(f)
            
            models["next_app"] = {
                "model_file": next_app_model,
                "metadata_file": next_app_meta,
                "vocab_file": next_app_vocab if next_app_vocab.exists() else None,
                "metadata": metadata,
                "type": "sequence_prediction"
            }
        
        # Check for time-series model
        time_series_model = self.models_dir / "time_series_twin.joblib"
        time_series_meta = self.models_dir / "time_series_twin.json"
        
        if time_series_model.exists() and time_series_meta.exists():
            with open(time_series_meta, 'r') as f:
                metadata = json.load(f)
            
            models["time_series"] = {
                "model_file": time_series_model,
                "metadata_file": time_series_meta,
                "vocab_file": None,
                "metadata": metadata,
                "type": "time_series_forecasting"
            }
        
        # Check for ONNX models
        onnx_models = list(self.models_dir.glob("*.onnx"))
        for onnx_model in onnx_models:
            model_name = onnx_model.stem
            if model_name not in models:  # Don't override existing models
                models[f"{model_name}_onnx"] = {
                    "model_file": onnx_model,
                    "metadata_file": None,
                    "vocab_file": None,
                    "metadata": {"framework": "onnx", "model_name": model_name},
                    "type": "onnx_model"
                }
        
        logger.info(f"Found {len(models)} available models: {list(models.keys())}")
        return models
    
    def validate_model(self, model_info: Dict[str, Any]) -> bool:
        """Validate model files and metadata."""
        try:
            model_file = model_info["model_file"]
            if not model_file.exists():
                logger.error(f"Model file not found: {model_file}")
                return False
            
            # Check file size (should be reasonable)
            file_size_mb = model_file.stat().st_size / (1024 * 1024)
            if file_size_mb > 100:  # 100MB limit for mobile deployment
                logger.warning(f"Model file is large ({file_size_mb:.1f}MB): {model_file}")
            
            # Validate metadata
            metadata = model_info.get("metadata", {})
            required_fields = ["model_name", "framework"]
            for field in required_fields:
                if field not in metadata:
                    logger.error(f"Missing required metadata field: {field}")
                    return False
            
            logger.info(f"Model validation passed: {metadata.get('model_name')}")
            return True
            
        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            return False
    
    def create_deployment_package(self, models: Dict[str, Dict[str, Any]], 
                                 output_path: Path) -> bool:
        """Create a deployment package with all model files."""
        try:
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for model_name, model_info in models.items():
                    # Add model file
                    model_file = model_info["model_file"]
                    zipf.write(model_file, f"{model_name}/{model_file.name}")
                    
                    # Add metadata file
                    if model_info["metadata_file"]:
                        metadata_file = model_info["metadata_file"]
                        zipf.write(metadata_file, f"{model_name}/{metadata_file.name}")
                    
                    # Add vocabulary file
                    if model_info["vocab_file"]:
                        vocab_file = model_info["vocab_file"]
                        zipf.write(vocab_file, f"{model_name}/{vocab_file.name}")
                
                # Add deployment manifest
                manifest = {
                    "deployment_timestamp": __import__('datetime').datetime.utcnow().isoformat() + 'Z',
                    "models": {
                        name: {
                            "type": info["type"],
                            "metadata": info["metadata"]
                        } for name, info in models.items()
                    }
                }
                
                zipf.writestr("deployment_manifest.json", json.dumps(manifest, indent=2))
            
            logger.info(f"Created deployment package: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create deployment package: {e}")
            return False
    
    def deploy_via_adb(self, models: Dict[str, Dict[str, Any]]) -> bool:
        """Deploy models to Android device via ADB."""
        try:
            # Create device directory
            subprocess.run([
                'adb', 'shell', 'mkdir', '-p', self.device_models_path
            ], check=True)
            
            # Deploy each model
            for model_name, model_info in models.items():
                logger.info(f"Deploying {model_name}...")
                
                # Push model file
                model_file = model_info["model_file"]
                device_model_path = f"{self.device_models_path}/{model_file.name}"
                subprocess.run([
                    'adb', 'push', str(model_file), device_model_path
                ], check=True)
                
                # Push metadata file
                if model_info["metadata_file"]:
                    metadata_file = model_info["metadata_file"]
                    device_meta_path = f"{self.device_models_path}/{metadata_file.name}"
                    subprocess.run([
                        'adb', 'push', str(metadata_file), device_meta_path
                    ], check=True)
                
                # Push vocabulary file
                if model_info["vocab_file"]:
                    vocab_file = model_info["vocab_file"]
                    device_vocab_path = f"{self.device_models_path}/{vocab_file.name}"
                    subprocess.run([
                        'adb', 'push', str(vocab_file), device_vocab_path
                    ], check=True)
            
            # Set proper permissions
            subprocess.run([
                'adb', 'shell', 'chmod', '-R', '644', self.device_models_path
            ], check=True)
            
            logger.info("Model deployment completed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"ADB command failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            return False
    
    def deploy_via_assets(self, models: Dict[str, Dict[str, Any]], 
                         assets_dir: str = "mobile/app/src/main/assets/ml_models") -> bool:
        """Deploy models to Android app assets directory."""
        try:
            assets_path = Path(assets_dir)
            assets_path.mkdir(parents=True, exist_ok=True)
            
            # Clear existing models
            for existing_file in assets_path.glob("*"):
                if existing_file.is_file():
                    existing_file.unlink()
            
            # Copy model files to assets
            for model_name, model_info in models.items():
                logger.info(f"Copying {model_name} to assets...")
                
                # Copy model file
                model_file = model_info["model_file"]
                shutil.copy2(model_file, assets_path / model_file.name)
                
                # Copy metadata file
                if model_info["metadata_file"]:
                    metadata_file = model_info["metadata_file"]
                    shutil.copy2(metadata_file, assets_path / metadata_file.name)
                
                # Copy vocabulary file
                if model_info["vocab_file"]:
                    vocab_file = model_info["vocab_file"]
                    shutil.copy2(vocab_file, assets_path / vocab_file.name)
            
            # Create deployment manifest
            manifest = {
                "deployment_timestamp": __import__('datetime').datetime.utcnow().isoformat() + 'Z',
                "deployment_method": "assets",
                "models": {
                    name: {
                        "type": info["type"],
                        "files": {
                            "model": info["model_file"].name,
                            "metadata": info["metadata_file"].name if info["metadata_file"] else None,
                            "vocabulary": info["vocab_file"].name if info["vocab_file"] else None
                        },
                        "metadata": info["metadata"]
                    } for name, info in models.items()
                }
            }
            
            with open(assets_path / "deployment_manifest.json", 'w') as f:
                json.dump(manifest, f, indent=2)
            
            logger.info(f"Models deployed to assets directory: {assets_path}")
            return True
            
        except Exception as e:
            logger.error(f"Assets deployment failed: {e}")
            return False
    
    def verify_deployment(self) -> bool:
        """Verify that models were deployed successfully."""
        try:
            # Check if models directory exists on device
            result = subprocess.run([
                'adb', 'shell', 'ls', self.device_models_path
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error("Models directory not found on device")
                return False
            
            # List deployed files
            files = result.stdout.strip().split('\n')
            files = [f.strip() for f in files if f.strip()]
            
            logger.info(f"Deployed files on device: {files}")
            
            # Check for required files
            required_files = ["next_app_model.json", "time_series_twin.json"]
            missing_files = [f for f in required_files if f not in files]
            
            if missing_files:
                logger.warning(f"Missing files on device: {missing_files}")
            
            return len(missing_files) == 0
            
        except Exception as e:
            logger.error(f"Deployment verification failed: {e}")
            return False
    
    def get_deployment_status(self) -> Dict[str, Any]:
        """Get current deployment status."""
        status = {
            "adb_available": False,
            "device_connected": False,
            "models_on_device": [],
            "available_models": [],
            "deployment_timestamp": None
        }
        
        try:
            # Check ADB
            status["adb_available"] = self.check_adb_connection()
            
            if status["adb_available"]:
                status["device_connected"] = True
                
                # Check models on device
                result = subprocess.run([
                    'adb', 'shell', 'ls', self.device_models_path
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    files = result.stdout.strip().split('\n')
                    status["models_on_device"] = [f.strip() for f in files if f.strip()]
            
            # Check available models
            available_models = self.get_available_models()
            status["available_models"] = list(available_models.keys())
            
        except Exception as e:
            logger.error(f"Failed to get deployment status: {e}")
        
        return status


def main():
    parser = argparse.ArgumentParser(description="Deploy ML models to Android device")
    parser.add_argument("--models-dir", default="ml/models",
                       help="Directory containing trained models")
    parser.add_argument("--android-package", default="com.lifetwin.mlp",
                       help="Android package name")
    parser.add_argument("--deployment-method", choices=["adb", "assets", "package"], default="adb",
                       help="Deployment method")
    parser.add_argument("--assets-dir", default="mobile/app/src/main/assets/ml_models",
                       help="Assets directory for assets deployment")
    parser.add_argument("--output-package", default="ml_models_deployment.zip",
                       help="Output package file for package deployment")
    parser.add_argument("--models", nargs='+', 
                       help="Specific models to deploy (default: all available)")
    parser.add_argument("--verify", action="store_true",
                       help="Verify deployment after completion")
    parser.add_argument("--status", action="store_true",
                       help="Show deployment status and exit")
    
    args = parser.parse_args()
    
    deployer = AndroidModelDeployer(args.models_dir, args.android_package)
    
    # Show status and exit if requested
    if args.status:
        status = deployer.get_deployment_status()
        print(json.dumps(status, indent=2))
        return
    
    # Get available models
    available_models = deployer.get_available_models()
    if not available_models:
        logger.error("No trained models found. Please train models first.")
        return
    
    # Filter models if specific ones requested
    if args.models:
        models_to_deploy = {name: info for name, info in available_models.items() 
                           if name in args.models}
        if not models_to_deploy:
            logger.error(f"Requested models not found: {args.models}")
            return
    else:
        models_to_deploy = available_models
    
    # Validate models
    valid_models = {}
    for name, info in models_to_deploy.items():
        if deployer.validate_model(info):
            valid_models[name] = info
        else:
            logger.warning(f"Skipping invalid model: {name}")
    
    if not valid_models:
        logger.error("No valid models to deploy")
        return
    
    # Deploy models
    success = False
    
    if args.deployment_method == "adb":
        if not deployer.check_adb_connection():
            logger.error("ADB deployment not available")
            return
        success = deployer.deploy_via_adb(valid_models)
        
    elif args.deployment_method == "assets":
        success = deployer.deploy_via_assets(valid_models, args.assets_dir)
        
    elif args.deployment_method == "package":
        output_path = Path(args.output_package)
        success = deployer.create_deployment_package(valid_models, output_path)
    
    if success:
        logger.info("Deployment completed successfully!")
        
        # Verify deployment if requested
        if args.verify and args.deployment_method == "adb":
            if deployer.verify_deployment():
                logger.info("Deployment verification passed")
            else:
                logger.warning("Deployment verification failed")
    else:
        logger.error("Deployment failed")


if __name__ == "__main__":
    main()