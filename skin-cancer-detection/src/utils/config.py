"""
Configuration management for the skin cancer detection system.
"""
import os
from pydantic_settings import BaseSettings
from typing import List, Optional

class Settings(BaseSettings):
    # API Settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 1
    API_DEBUG: bool = False
    
    # Model Paths
    MODEL_A_PATH: str = "/app/models/best_model_A_tuned.pth"
    MODEL_B_PATH: str = "/app/models/best_model_B_tuned.pth"
    
    # Model Settings
    MODEL_NAME: str = "seresnext101_32x8d"
    NUM_CLASSES: int = 7
    IMG_SIZE: int = 224
    
    # Device Settings
    DEVICE: str = "cuda" if os.getenv("CUDA_AVAILABLE") == "true" else "cpu"
    
    # LM Studio Settings
    LM_STUDIO_ENDPOINT: str = "http://localhost:1234/v1/chat/completions"
    LM_STUDIO_MODEL: str = "medgemma-4b-it"
    
    # Gemini CLI Settings
    GEMINI_CLI_PATH: str = "/usr/local/bin/gemini-cli"
    GEMINI_MODEL: str = "gemini-pro"
    
    # Monitoring Settings
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    MLFLOW_EXPERIMENT_NAME: str = "skin-cancer-detection"
    
    # Security Settings
    API_KEY_HEADER: str = "X-API-Key"
    ALLOWED_HOSTS: str = "*"
    
    # File Upload Settings
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: str = ".jpg,.jpeg,.png"
    
    # Database Settings (for future expansion)
    DATABASE_URL: Optional[str] = None
    
    # Logging Settings
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    class Config:
        env_file = ".env"
        case_sensitive = True
    
    # Property methods for easier access
    @property
    def model_name(self) -> str:
        return self.MODEL_NAME
    
    @property
    def model_a_path(self) -> str:
        return self.MODEL_A_PATH
    
    @property
    def model_b_path(self) -> str:
        return self.MODEL_B_PATH
    
    @property
    def device(self) -> str:
        if self.DEVICE == "auto":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.DEVICE
    
    @property
    def num_classes(self) -> int:
        return self.NUM_CLASSES
    
    @property
    def img_size(self) -> int:
        return self.IMG_SIZE
    
    @property
    def allowed_hosts(self) -> List[str]:
        if self.ALLOWED_HOSTS == "*":
            return ["*"]
        return [host.strip() for host in self.ALLOWED_HOSTS.split(",")]
    
    @property
    def allowed_extensions(self) -> List[str]:
        return [ext.strip() for ext in self.ALLOWED_EXTENSIONS.split(",")]

# Global settings instance
settings = Settings()

# Class names for HAM10000 dataset
CLASS_NAMES = [
    'actinic_keratoses',
    'basal_cell_carcinoma', 
    'benign_keratosis-like_lesions',
    'dermatofibroma',
    'melanocytic_nevi',
    'melanoma',
    'vascular_lesions'
]

# Medical descriptions for each class
CLASS_DESCRIPTIONS = {
    'actinic_keratoses': 'Rough, scaly patches on sun-exposed skin that can develop into cancer',
    'basal_cell_carcinoma': 'The most common type of skin cancer, typically slow-growing',
    'benign_keratosis-like_lesions': 'Non-cancerous skin growths that may look concerning but are harmless',
    'dermatofibroma': 'Common benign skin tumors that are usually harmless',
    'melanocytic_nevi': 'Common moles that are typically benign',
    'melanoma': 'A serious form of skin cancer that can spread to other parts of the body',
    'vascular_lesions': 'Lesions involving blood vessels, usually benign'
}

# Urgency levels for each class
CLASS_URGENCY = {
    'actinic_keratoses': 'moderate',
    'basal_cell_carcinoma': 'high',
    'benign_keratosis-like_lesions': 'low',
    'dermatofibroma': 'low',
    'melanocytic_nevi': 'low',
    'melanoma': 'urgent',
    'vascular_lesions': 'low'
}
