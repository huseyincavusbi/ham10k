"""
Enhanced inference module with MLOps capabilities.
"""
import torch
import torch.nn.functional as F
import timm
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import Tuple, Dict, List, Optional
import time
import mlflow
from pathlib import Path

from src.utils.config import settings, CLASS_NAMES, CLASS_DESCRIPTIONS, CLASS_URGENCY
from src.utils.logging import get_logger

logger = get_logger(__name__)

class ModelMetrics:
    """Class to track model performance metrics."""
    
    def __init__(self):
        self.predictions = []
        self.confidences = []
        self.inference_times = []
        
    def add_prediction(self, prediction: str, confidence: float, inference_time: float):
        """Add a prediction result for tracking."""
        self.predictions.append(prediction)
        self.confidences.append(confidence)
        self.inference_times.append(inference_time)
        
    def get_metrics(self) -> Dict:
        """Get aggregated metrics."""
        if not self.predictions:
            return {}
            
        return {
            "total_predictions": len(self.predictions),
            "avg_confidence": np.mean(self.confidences),
            "avg_inference_time": np.mean(self.inference_times),
            "min_confidence": np.min(self.confidences),
            "max_confidence": np.max(self.confidences),
            "prediction_distribution": {
                pred: self.predictions.count(pred) for pred in set(self.predictions)
            }
        }

class SkinLesionPredictor:
    """Enhanced skin lesion predictor with MLOps capabilities."""
    
    def __init__(self, 
                 model_a_path: str = settings.MODEL_A_PATH,
                 model_b_path: str = settings.MODEL_B_PATH,
                 device: str = settings.DEVICE):
        """
        Initialize the predictor with ensemble models.
        
        Args:
            model_a_path: Path to first model weights
            model_b_path: Path to second model weights
            device: Device to run inference on
        """
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.num_classes = settings.NUM_CLASSES
        self.model_name = settings.MODEL_NAME
        
        logger.info(f"Initializing predictor on device: {self.device}")
        
        # Initialize metrics tracking
        self.metrics = ModelMetrics()
        
        # Load models
        self._load_models(model_a_path, model_b_path)
        
        # Set up transforms
        self._setup_transforms()
        
        # Class names and metadata
        self.class_names = CLASS_NAMES
        self.class_descriptions = CLASS_DESCRIPTIONS
        self.class_urgency = CLASS_URGENCY
        
        logger.info("Predictor initialized successfully")
        
    def _get_model(self, pretrained: bool = False):
        """Create a model instance."""
        return timm.create_model(
            self.model_name, 
            pretrained=pretrained, 
            num_classes=self.num_classes
        )
    
    def _load_models(self, model_a_path: str, model_b_path: str):
        """Load both ensemble models."""
        # Load Model A
        logger.info(f"Loading Model A from: {model_a_path}")
        self.model_a = self._get_model(pretrained=False)
        model_a_state = torch.load(model_a_path, map_location=self.device)
        self.model_a.load_state_dict(model_a_state)
        self.model_a.to(self.device)
        self.model_a.eval()
        
        # Load Model B
        logger.info(f"Loading Model B from: {model_b_path}")
        self.model_b = self._get_model(pretrained=False)
        model_b_state = torch.load(model_b_path, map_location=self.device)
        self.model_b.load_state_dict(model_b_state)
        self.model_b.to(self.device)
        self.model_b.eval()
        
        logger.info("Both models loaded successfully")
    
    def _setup_transforms(self):
        """Set up image preprocessing transforms."""
        self.transform = transforms.Compose([
            transforms.Resize((settings.IMG_SIZE, settings.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Test-time augmentation transforms
        self.tta_transforms = [
            transforms.Compose([
                transforms.Resize((settings.IMG_SIZE, settings.IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
            transforms.Compose([
                transforms.Resize((settings.IMG_SIZE, settings.IMG_SIZE)),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
            transforms.Compose([
                transforms.Resize((settings.IMG_SIZE, settings.IMG_SIZE)),
                transforms.RandomVerticalFlip(p=1.0),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        ]
    
    def _predict_single_model(self, image_tensor: torch.Tensor, model: torch.nn.Module) -> Tuple[str, float, torch.Tensor]:
        """Get prediction from a single model."""
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = F.softmax(output, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            predicted_class = self.class_names[predicted_idx.item()]
            
        return predicted_class, confidence.item(), probabilities
    
    def predict_with_tta(self, image: Image.Image) -> Dict:
        """
        Predict with Test-Time Augmentation for improved accuracy.
        
        Args:
            image: PIL Image to classify
            
        Returns:
            Dictionary containing ensemble prediction results
        """
        start_time = time.time()
        
        # Collect predictions from all TTA variants
        all_probs_a = []
        all_probs_b = []
        
        for transform in self.tta_transforms:
            image_tensor = transform(image).unsqueeze(0).to(self.device)
            
            # Model A prediction
            _, _, probs_a = self._predict_single_model(image_tensor, self.model_a)
            all_probs_a.append(probs_a)
            
            # Model B prediction
            _, _, probs_b = self._predict_single_model(image_tensor, self.model_b)
            all_probs_b.append(probs_b)
        
        # Average probabilities across TTA variants
        avg_probs_a = torch.mean(torch.stack(all_probs_a), dim=0)
        avg_probs_b = torch.mean(torch.stack(all_probs_b), dim=0)
        
        # Get final predictions
        confidence_a, predicted_idx_a = torch.max(avg_probs_a, 1)
        confidence_b, predicted_idx_b = torch.max(avg_probs_b, 1)
        
        pred_class_a = self.class_names[predicted_idx_a.item()]
        pred_class_b = self.class_names[predicted_idx_b.item()]
        
        # Ensemble the two models
        ensemble_probs = (avg_probs_a + avg_probs_b) / 2
        ensemble_confidence, ensemble_idx = torch.max(ensemble_probs, 1)
        ensemble_class = self.class_names[ensemble_idx.item()]
        
        inference_time = time.time() - start_time
        
        # Track metrics
        self.metrics.add_prediction(ensemble_class, ensemble_confidence.item(), inference_time)
        
        # Get all class probabilities for the ensemble
        all_class_probs = {
            self.class_names[i]: float(ensemble_probs[0][i]) 
            for i in range(len(self.class_names))
        }
        
        result = {
            "ensemble_prediction": {
                "class": ensemble_class,
                "confidence": float(ensemble_confidence.item()),
                "urgency": self.class_urgency.get(ensemble_class, "unknown"),
                "description": self.class_descriptions.get(ensemble_class, "Unknown condition")
            },
            "model_a_prediction": {
                "class": pred_class_a,
                "confidence": float(confidence_a.item())
            },
            "model_b_prediction": {
                "class": pred_class_b,
                "confidence": float(confidence_b.item())
            },
            "all_class_probabilities": all_class_probs,
            "inference_time": inference_time,
            "tta_enabled": True
        }
        
        # Log prediction with MLflow if available
        try:
            with mlflow.start_run():
                mlflow.log_metrics({
                    "confidence": ensemble_confidence.item(),
                    "inference_time": inference_time
                })
                mlflow.log_params({
                    "predicted_class": ensemble_class,
                    "model_ensemble": True,
                    "tta_enabled": True
                })
        except Exception as e:
            logger.warning(f"MLflow logging failed: {e}")
        
        logger.info(f"Prediction completed: {ensemble_class} (confidence: {ensemble_confidence.item():.3f})")
        
        return result
    
    def predict(self, image: Image.Image, use_tta: bool = True) -> Dict:
        """
        Main prediction method.
        
        Args:
            image: PIL Image to classify
            use_tta: Whether to use Test-Time Augmentation
            
        Returns:
            Dictionary containing prediction results
        """
        if use_tta:
            return self.predict_with_tta(image)
        else:
            # Simple prediction without TTA
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            start_time = time.time()
            
            # Get predictions from both models
            pred_class_a, confidence_a, _ = self._predict_single_model(image_tensor, self.model_a)
            pred_class_b, confidence_b, _ = self._predict_single_model(image_tensor, self.model_b)
            
            # Use the more confident prediction
            if confidence_a > confidence_b:
                final_class = pred_class_a
                final_confidence = confidence_a
            else:
                final_class = pred_class_b
                final_confidence = confidence_b
            
            inference_time = time.time() - start_time
            
            # Track metrics
            self.metrics.add_prediction(final_class, final_confidence, inference_time)
            
            return {
                "final_prediction": {
                    "class": final_class,
                    "confidence": final_confidence,
                    "urgency": self.class_urgency.get(final_class, "unknown"),
                    "description": self.class_descriptions.get(final_class, "Unknown condition")
                },
                "model_a_prediction": {
                    "class": pred_class_a,
                    "confidence": confidence_a
                },
                "model_b_prediction": {
                    "class": pred_class_b,
                    "confidence": confidence_b
                },
                "inference_time": inference_time,
                "tta_enabled": False
            }
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded models."""
        return {
            "model_architecture": self.model_name,
            "num_classes": self.num_classes,
            "device": str(self.device),
            "class_names": self.class_names,
            "metrics": self.metrics.get_metrics()
        }
    
    def health_check(self) -> Dict:
        """Perform a health check on the models."""
        try:
            # Create a dummy tensor to test the models
            dummy_input = torch.randn(1, 3, settings.IMG_SIZE, settings.IMG_SIZE).to(self.device)
            
            with torch.no_grad():
                output_a = self.model_a(dummy_input)
                output_b = self.model_b(dummy_input)
            
            return {
                "status": "healthy",
                "model_a_loaded": True,
                "model_b_loaded": True,
                "device": str(self.device),
                "models_responsive": True
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "device": str(self.device),
                "models_responsive": False
            }
