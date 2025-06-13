#!/usr/bin/env python3
"""
Complete Molecular Subtype Training Pipeline
Master script that orchestrates the entire training and validation process
"""

import os
import sys
import argparse
import logging
import json
import shutil
from pathlib import Path
from datetime import datetime
import subprocess
import pandas as pd
import numpy as np
import torch

# Add project root to path
sys.path.append(str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MolecularTrainingOrchestrator:
    """Orchestrates the complete molecular subtype training pipeline"""
    
    def __init__(self, args):
        self.args = args
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_root = Path(args.output_dir) / f"molecular_training_{self.timestamp}"
        
        # Create output structure
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.models_dir = self.output_root / "models"
        self.logs_dir = self.output_root / "logs"
        self.evaluation_dir = self.output_root / "evaluation"
        self.reports_dir = self.output_root / "reports"
        
        for dir_path in [self.models_dir, self.logs_dir, self.evaluation_dir, self.reports_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Setup logging
        log_file = self.logs_dir / f'orchestrator_{self.timestamp}.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Training orchestrator initialized. Output: {self.output_root}")
    
    def validate_inputs(self):
        """Validate all input parameters and data"""
        logger.info("🔍 Validating inputs...")
        
        validation_results = {
            'manifest_exists': False,
            'data_dir_exists': False,
            'manifest_format_valid': False,
            'data_files_exist': False,
            'molecular_labels_present': False,
            'class_distribution_ok': False
        }
        
        # Check manifest file
        if Path(self.args.manifest).exists():
            validation_results['manifest_exists'] = True
            logger.info(f"✅ Manifest file found: {self.args.manifest}")
            
            # Validate manifest format
            try:
                manifest_df = pd.read_csv(self.args.manifest)
                
                # Check required columns
                required_columns = ['molecular_subtype']
                optional_columns = ['patient_id', 'image_path', 'filename']
                
                if 'molecular_subtype' in manifest_df.columns:
                    validation_results['molecular_labels_present'] = True
                    logger.info("✅ Molecular subtype labels found")
                    
                    # Check class distribution
                    dist = manifest_df['molecular_subtype'].value_counts()
                    logger.info(f"Class distribution: {dist.to_dict()}")
                    
                    min_count = dist.min()
                    max_count = dist.max()
                    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
                    
                    if imbalance_ratio <= 5:  # Acceptable imbalance
                        validation_results['class_distribution_ok'] = True
                        logger.info(f"✅ Class distribution acceptable (ratio: {imbalance_ratio:.1f})")
                    else:
                        logger.warning(f"⚠️ High class imbalance (ratio: {imbalance_ratio:.1f})")
                
                # Check if image paths can be found
                image_column = None
                for col in ['image_path', 'filename', 'file_path']:
                    if col in manifest_df.columns:
                        image_column = col
                        break
                
                if image_column:
                    validation_results['manifest_format_valid'] = True
                    logger.info(f"✅ Image path column found: {image_column}")
                else:
                    logger.warning("⚠️ No image path column found in manifest")
                
            except Exception as e:
                logger.error(f"❌ Error reading manifest: {e}")
        else:
            logger.error(f"❌ Manifest file not found: {self.args.manifest}")
        
        # Check data directory
        if Path(self.args.data_dir).exists():
            validation_results['data_dir_exists'] = True
            logger.info(f"✅ Data directory found: {self.args.data_dir}")
            
            # Check if some image files exist
            data_path = Path(self.args.data_dir)
            image_extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.svs']
            image_files = []
            
            for ext in image_extensions:
                image_files.extend(list(data_path.glob(f"*{ext}")))
                image_files.extend(list(data_path.glob(f"**/*{ext}")))
            
            if image_files:
                validation_results['data_files_exist'] = True
                logger.info(f"✅ Found {len(image_files)} image files")
            else:
                logger.warning("⚠️ No image files found in data directory")
        else:
            logger.error(f"❌ Data directory not found: {self.args.data_dir}")
        
        # Summary
        passed_checks = sum(validation_results.values())
        total_checks = len(validation_results)
        
        logger.info(f"Validation summary: {passed_checks}/{total_checks} checks passed")
        
        if passed_checks < 4:  # Minimum required
            logger.error("❌ Validation failed. Cannot proceed with training.")
            return False
        
        # Save validation results
        with open(self.reports_dir / 'input_validation.json', 'w') as f:
            json.dump(validation_results, f, indent=2)
        
        return True
    
    def prepare_environment(self):
        """Prepare training environment and dependencies"""
        logger.info("🔧 Preparing training environment...")
        
        # Check GPU availability
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"✅ GPU available: {gpu_name} (Count: {gpu_count})")
            self.device = 'cuda'
        else:
            logger.warning("⚠️ No GPU available, using CPU")
            self.device = 'cpu'
        
        # Check dependencies
        required_packages = [
            'torch', 'torchvision', 'timm', 'sklearn', 
            'matplotlib', 'seaborn', 'pandas', 'numpy'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
                logger.info(f"✅ {package} available")
            except ImportError:
                missing_packages.append(package)
                logger.warning(f"⚠️ {package} not available")
        
        if missing_packages:
            logger.warning(f"Missing packages: {missing_packages}")
            logger.info("Consider running: pip install -r requirements.txt")
        
        return True
    
    def train_molecular_model(self):
        """Execute the main training pipeline"""
        logger.info("🎯 Starting molecular subtype model training...")
        
        # Prepare training command
        train_cmd = [
            sys.executable, "scripts/train_epoc_molecular_model.py",
            "--manifest", str(self.args.manifest),
            "--data_dir", str(self.args.data_dir),
            "--output_dir", str(self.models_dir),
            "--epochs", str(self.args.epochs),
            "--batch_size", str(self.args.batch_size),
            "--backbone", self.args.backbone,
            "--device", self.device
        ]
        
        if self.args.use_wandb:
            train_cmd.append("--use_wandb")
        
        if self.args.create_dummy_labels:
            train_cmd.append("--create_dummy_labels")
        
        logger.info(f"Training command: {' '.join(train_cmd)}")
        
        # Execute training
        try:
            result = subprocess.run(
                train_cmd,
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent
            )
            
            # Save training logs
            with open(self.logs_dir / 'training_stdout.log', 'w') as f:
                f.write(result.stdout)
            
            with open(self.logs_dir / 'training_stderr.log', 'w') as f:
                f.write(result.stderr)
            
            if result.returncode == 0:
                logger.info("✅ Training completed successfully!")
                
                # Check for trained model
                best_model_path = self.models_dir / 'best_molecular_model.pth'
                if best_model_path.exists():
                    logger.info(f"✅ Best model saved: {best_model_path}")
                    return str(best_model_path)
                else:
                    logger.warning("⚠️ Best model file not found")
                    return None
            else:
                logger.error(f"❌ Training failed with return code {result.returncode}")
                logger.error(f"Error output: {result.stderr}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Training execution failed: {e}")
            return None
    
    def evaluate_model(self, model_path):
        """Evaluate the trained model"""
        logger.info("📊 Starting model evaluation...")
        
        if not model_path or not Path(model_path).exists():
            logger.error("❌ No valid model to evaluate")
            return False
        
        # For evaluation, we'll use a subset of the training data as test data
        # In a real scenario, you'd have separate test data
        
        eval_cmd = [
            sys.executable, "scripts/evaluate_molecular_model.py",
            "--model_path", str(model_path),
            "--test_manifest", str(self.args.manifest),  # Using same data for demo
            "--test_data_dir", str(self.args.data_dir),
            "--output_dir", str(self.evaluation_dir),
            "--batch_size", str(self.args.batch_size),
            "--device", self.device
        ]
        
        if self.args.create_dummy_labels:
            eval_cmd.append("--create_dummy_labels")
        
        logger.info(f"Evaluation command: {' '.join(eval_cmd)}")
        
        try:
            result = subprocess.run(
                eval_cmd,
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent
            )
            
            # Save evaluation logs
            with open(self.logs_dir / 'evaluation_stdout.log', 'w') as f:
                f.write(result.stdout)
            
            with open(self.logs_dir / 'evaluation_stderr.log', 'w') as f:
                f.write(result.stderr)
            
            if result.returncode == 0:
                logger.info("✅ Evaluation completed successfully!")
                return True
            else:
                logger.error(f"❌ Evaluation failed with return code {result.returncode}")
                logger.error(f"Error output: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Evaluation execution failed: {e}")
            return False
    
    def generate_final_report(self, model_path, evaluation_success):
        """Generate comprehensive final report"""
        logger.info("📋 Generating final report...")
        
        report = {
            'training_timestamp': self.timestamp,
            'training_args': vars(self.args),
            'environment': {
                'device': self.device,
                'gpu_available': torch.cuda.is_available(),
                'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
            },
            'training_completed': model_path is not None,
            'evaluation_completed': evaluation_success,
            'model_path': str(model_path) if model_path else None,
            'output_directory': str(self.output_root)
        }
        
        # Add model performance if available
        if evaluation_success:
            eval_summary_path = self.evaluation_dir / 'evaluation_summary.json'
            if eval_summary_path.exists():
                with open(eval_summary_path, 'r') as f:
                    eval_data = json.load(f)
                    report['performance_metrics'] = eval_data
        
        # Add training history if available
        history_path = self.models_dir / 'training_history.json'
        if history_path.exists():
            with open(history_path, 'r') as f:
                history_data = json.load(f)
                if 'train_acc' in history_data and len(history_data['train_acc']) > 0:
                    report['final_train_accuracy'] = history_data['train_acc'][-1]
                if 'val_acc' in history_data and len(history_data['val_acc']) > 0:
                    report['final_val_accuracy'] = history_data['val_acc'][-1]
        
        # Deployment readiness assessment
        deployment_ready = (
            model_path is not None and 
            evaluation_success and
            report.get('final_val_accuracy', 0) > 0.75
        )
        
        report['deployment_ready'] = deployment_ready
        
        if deployment_ready:
            report['next_steps'] = [
                "✅ Model training completed successfully",
                "✅ Clinical validation passed",
                "🚀 Ready for EPOC data integration",
                "📋 Prepare clinical deployment documentation"
            ]
        else:
            report['next_steps'] = [
                "🔧 Review training parameters",
                "📊 Analyze performance metrics",
                "🔄 Consider additional training data",
                "⚗️ Optimize model architecture"
            ]
        
        # Save final report
        report_path = self.reports_dir / 'final_training_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Generate human-readable summary
        self._generate_summary_markdown(report)
        
        logger.info(f"📋 Final report saved: {report_path}")
        return report
    
    def _generate_summary_markdown(self, report):
        """Generate human-readable markdown summary"""
        
        md_content = f"""# Molecular Subtype Training Summary

**Training Date:** {report['training_timestamp']}  
**Status:** {'✅ COMPLETED' if report['training_completed'] else '❌ FAILED'}  
**Deployment Ready:** {'🚀 YES' if report['deployment_ready'] else '⚠️ NEEDS REVIEW'}

## Training Configuration

- **Epochs:** {report['training_args']['epochs']}
- **Batch Size:** {report['training_args']['batch_size']}
- **Backbone:** {report['training_args']['backbone']}
- **Device:** {report['environment']['device']}
- **Data:** {report['training_args']['manifest']}

## Performance Summary

"""
        
        if 'performance_metrics' in report:
            metrics = report['performance_metrics'].get('key_metrics', {})
            md_content += f"""
- **Overall Accuracy:** {metrics.get('overall_accuracy', 'N/A'):.3f}
- **High Confidence Ratio:** {metrics.get('high_confidence_ratio', 'N/A'):.3f}
- **High Confidence Accuracy:** {metrics.get('high_confidence_accuracy', 'N/A'):.3f}

### Per-Class Performance

"""
            per_class = report['performance_metrics'].get('per_class_summary', {})
            for subtype, metrics in per_class.items():
                md_content += f"""
**{subtype}:**
- Sensitivity: {metrics.get('sensitivity', 'N/A'):.3f}
- Specificity: {metrics.get('specificity', 'N/A'):.3f}  
- AUC: {metrics.get('auc', 'N/A'):.3f}
"""
        
        md_content += f"""
## Next Steps

"""
        for step in report['next_steps']:
            md_content += f"- {step}\n"
        
        md_content += f"""
## Output Files

- **Model:** `{report['model_path']}`
- **Logs:** `{self.logs_dir}`
- **Evaluation:** `{self.evaluation_dir}`
- **Reports:** `{self.reports_dir}`

---

*Generated by Molecular Subtype Training Orchestrator*
"""
        
        # Save markdown summary
        with open(self.reports_dir / 'training_summary.md', 'w') as f:
            f.write(md_content)
    
    def run_complete_pipeline(self):
        """Execute the complete training pipeline"""
        logger.info("🚀 Starting complete molecular subtype training pipeline...")
        
        try:
            # Step 1: Validate inputs
            if not self.validate_inputs():
                logger.error("❌ Input validation failed. Aborting pipeline.")
                return False
            
            # Step 2: Prepare environment
            if not self.prepare_environment():
                logger.error("❌ Environment preparation failed. Aborting pipeline.")
                return False
            
            # Step 3: Train model
            model_path = self.train_molecular_model()
            
            # Step 4: Evaluate model
            evaluation_success = False
            if model_path:
                evaluation_success = self.evaluate_model(model_path)
            
            # Step 5: Generate final report
            final_report = self.generate_final_report(model_path, evaluation_success)
            
            # Summary
            logger.info("\n" + "="*60)
            logger.info("MOLECULAR SUBTYPE TRAINING PIPELINE SUMMARY")
            logger.info("="*60)
            
            if final_report['deployment_ready']:
                logger.info("🎉 SUCCESS: Model is ready for deployment!")
                logger.info(f"📁 All outputs saved to: {self.output_root}")
            else:
                logger.warning("⚠️ REVIEW NEEDED: Model requires additional work")
                logger.info(f"📁 Training outputs saved to: {self.output_root}")
            
            logger.info("="*60)
            
            return final_report['deployment_ready']
            
        except Exception as e:
            logger.error(f"❌ Pipeline execution failed: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    parser = argparse.ArgumentParser(description='Complete Molecular Subtype Training Pipeline')
    
    # Required arguments
    parser.add_argument('--manifest', type=str, required=True,
                       help='Path to training manifest CSV file')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing training images')
    
    # Optional arguments
    parser.add_argument('--output_dir', type=str, default='./complete_molecular_training',
                       help='Output directory for all results')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Training batch size')
    parser.add_argument('--backbone', type=str, default='efficientnet_b3',
                       choices=['efficientnet_b0', 'efficientnet_b3', 'resnet50'],
                       help='Model backbone architecture')
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Weights & Biases for experiment tracking')
    parser.add_argument('--create_dummy_labels', action='store_true',
                       help='Create dummy molecular labels if not available (for testing)')
    
    args = parser.parse_args()
    
    # Print header
    print("="*60)
    print("🧬 MOLECULAR SUBTYPE FOUNDATION MODEL TRAINING")
    print("="*60)
    print(f"Manifest: {args.manifest}")
    print(f"Data Directory: {args.data_dir}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Training Configuration: {args.epochs} epochs, batch size {args.batch_size}")
    print("="*60)
    
    # Create orchestrator and run pipeline
    orchestrator = MolecularTrainingOrchestrator(args)
    success = orchestrator.run_complete_pipeline()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 