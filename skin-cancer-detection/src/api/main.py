"""Simplified FastAPI backend for the Skin Cancer Detection System.

This version removes references to non‑existent monitoring modules so the
service can run locally for end‑to‑end manual testing with the Streamlit
frontend. If you later add observability code, reintroduce those imports.
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware
from pathlib import Path
import hashlib
from PIL import Image
import io
import time
from typing import Dict, Any, Optional
import structlog

from src.models.inference import SkinLesionPredictor
from src.models.medgemma_interpreter import LocalMedicalInterpreter
from src.utils.config import settings
from src.utils.logging import setup_logging
import uvicorn

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

logger = setup_logging()

# Initialize FastAPI app
app = FastAPI(
    title="Skin Cancer Detection API",
    description="Skin cancer detection system (simplified backend)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_hosts,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.allowed_hosts
)

# Global variables for model instances
predictor: Optional[SkinLesionPredictor] = None
interpreter: Optional[LocalMedicalInterpreter] = None

# Request tracking
request_count = 0
error_count = 0

class RequestValidator:
    """Validate incoming requests"""
    
    @staticmethod
    def validate_image_file(file: UploadFile) -> None:
        """Validate uploaded image file"""
        # Check file size
        if file.size and file.size > settings.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {settings.MAX_FILE_SIZE / (1024*1024):.1f}MB"
            )
        
        # Check file extension
        if file.filename:
            file_ext = Path(file.filename).suffix.lower()
            if file_ext not in settings.allowed_extensions:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid file type. Allowed extensions: {', '.join(settings.allowed_extensions)}"
                )
        
        # Check content type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="File must be an image"
            )

async def get_predictor() -> SkinLesionPredictor:
    """Dependency to get the predictor instance"""
    global predictor
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return predictor

async def get_interpreter() -> LocalMedicalInterpreter:
    """Dependency to get the interpreter instance"""
    global interpreter
    if interpreter is None:
        raise HTTPException(status_code=503, detail="Interpreter not loaded")
    return interpreter

@app.middleware("http")
async def request_tracking_middleware(request: Request, call_next):
    """Track requests and add request ID"""
    global request_count
    request_count += 1
    
    # Generate unique request ID
    request_id = hashlib.md5(f"{time.time()}{request_count}".encode()).hexdigest()[:8]
    
    # Add request ID to headers
    request.state.request_id = request_id
    
    # Log request
    start_time = time.time()
    logger.info(f"Request {request_id}: {request.method} {request.url}")
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # Add custom headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = str(process_time)
        
        logger.info(f"Request {request_id} completed in {process_time:.3f}s")
        return response
        
    except Exception as e:
        global error_count
        error_count += 1
        process_time = time.time() - start_time
        logger.error(f"Request {request_id} failed after {process_time:.3f}s: {e}")
        raise

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    request_id = getattr(request.state, 'request_id', 'unknown')
    logger.error(f"Unhandled exception in request {request_id}: {exc}")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "request_id": request_id,
            "detail": "An unexpected error occurred. Please try again later."
        }
    )

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    global predictor, interpreter
    
    logger.info("Starting up Skin Cancer Detection API...")
    
    try:
        # Initialize predictor
        logger.info("Loading prediction models...")
        # Fallback to repo root model files if container paths not present
        model_a_path = Path(settings.MODEL_A_PATH)
        model_b_path = Path(settings.MODEL_B_PATH)
        repo_root = Path(__file__).resolve().parents[3]
        if not model_a_path.exists():
            candidate = repo_root / "best_model_A_tuned.pth"
            if candidate.exists():
                model_a_path = candidate
        if not model_b_path.exists():
            candidate = repo_root / "best_model_B_tuned.pth"
            if candidate.exists():
                model_b_path = candidate

        predictor = SkinLesionPredictor(
            model_a_path=str(model_a_path),
            model_b_path=str(model_b_path),
            device=settings.DEVICE
        )
        logger.info("Prediction models loaded successfully")
        
        # Initialize interpreter
        logger.info("Loading medical interpreter...")
        interpreter = LocalMedicalInterpreter()
        logger.info("Medical interpreter loaded successfully")
        
        # Perform health checks
        model_health = predictor.health_check()
        interpreter_health = interpreter.health_check()
        
        logger.info(f"Model health: {model_health}")
        logger.info(f"Interpreter health: {interpreter_health}")
        
        logger.info("API startup completed successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize API: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Skin Cancer Detection API...")
    
    # Log final statistics
    logger.info(f"Final stats - Requests: {request_count}, Errors: {error_count}")
    
    # If using MLflow, close tracking
    try:
        import mlflow
        mlflow.end_run()
    except:
        pass

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Skin Cancer Detection API",
        "version": "1.0.0",
        "status": "healthy",
        "endpoints": {
            "predict": "/predict - POST: Upload image for prediction",
            "interpret": "/interpret - POST: Get medical interpretation",
            "health": "/health - GET: System health check",
            "metrics": "/metrics - GET: System metrics",
            "docs": "/docs - API documentation"
        }
    }

@app.get("/health")
async def health_check(
    predictor: SkinLesionPredictor = Depends(get_predictor),
    interpreter: LocalMedicalInterpreter = Depends(get_interpreter)
):
    """Comprehensive health check endpoint"""
    try:
        model_health = predictor.health_check()
        interpreter_health = interpreter.health_check()
        
        overall_status = "healthy" if (
            model_health.get("status") == "healthy" and 
            interpreter_health.get("lm_studio_available", False) or 
            interpreter_health.get("fallback_ready", False)
        ) else "degraded"
        
        return {
            "status": overall_status,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "models": model_health,
            "interpreter": interpreter_health,
            "api": {
                "requests_processed": request_count,
                "errors_encountered": error_count,
                "error_rate": error_count / max(request_count, 1) * 100
            }
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        )

@app.get("/metrics")
async def get_metrics(predictor: SkinLesionPredictor = Depends(get_predictor)):
    """Get system metrics"""
    try:
        model_info = predictor.get_model_info()
        
        return {
            "system_metrics": {
                "total_requests": request_count,
                "total_errors": error_count,
                "error_rate_percent": (error_count / max(request_count, 1)) * 100,
                "uptime": time.time()  # You might want to track actual uptime
            },
            "model_metrics": model_info.get("metrics", {}),
            "model_info": {
                "architecture": model_info.get("model_architecture"),
                "num_classes": model_info.get("num_classes"),
                "device": model_info.get("device"),
                "class_names": model_info.get("class_names")
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve metrics")

@app.post("/predict")
async def predict_lesion(
    file: UploadFile = File(...),
    use_tta: bool = True,
    predictor: SkinLesionPredictor = Depends(get_predictor)
):
    """
    Predict skin lesion classification from uploaded image.
    
    Args:
        file: Image file to analyze
        use_tta: Whether to use Test-Time Augmentation for improved accuracy
        
    Returns:
        Prediction results with confidence scores
    """
    request_id = getattr(file, 'request_id', 'unknown')
    
    try:
        # Validate the uploaded file
        RequestValidator.validate_image_file(file)
        
        # Read and process the image
        logger.info(f"Processing image upload: {file.filename}")
        contents = await file.read()
        
        try:
            image = Image.open(io.BytesIO(contents)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")
        
        # Get prediction
        logger.info(f"Running prediction with TTA={use_tta}")
        prediction_result = predictor.predict(image, use_tta=use_tta)
        
        # Add metadata
        prediction_result["metadata"] = {
            "filename": file.filename,
            "file_size": len(contents),
            "image_size": image.size,
            "model_version": "ensemble-v1.0",
            "processed_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        logger.info(f"Prediction completed successfully")
        return prediction_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/interpret")
async def interpret_prediction(
    prediction_data: Dict[str, Any],
    interpreter: LocalMedicalInterpreter = Depends(get_interpreter)
):
    """
    Generate medical interpretation for prediction results.
    
    Args:
        prediction_data: Prediction results from /predict endpoint
        
    Returns:
        Medical interpretation and recommendations
    """
    try:
        logger.info("Generating medical interpretation...")
        
        # Validate input data
        if not prediction_data:
            raise HTTPException(status_code=400, detail="Prediction data is required")
        
        # Generate medical report
        medical_report = interpreter.generate_final_report(prediction_data)
        
        logger.info("Medical interpretation generated successfully")
        return medical_report
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Interpretation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Interpretation failed: {str(e)}")

@app.post("/analyze")
async def analyze_complete(
    file: UploadFile = File(...),
    use_tta: bool = True,
    predictor: SkinLesionPredictor = Depends(get_predictor),
    interpreter: LocalMedicalInterpreter = Depends(get_interpreter)
):
    """
    Complete analysis pipeline: prediction + medical interpretation.
    
    Args:
        file: Image file to analyze
        use_tta: Whether to use Test-Time Augmentation
        
    Returns:
        Complete analysis with prediction and medical interpretation
    """
    try:
        # Step 1: Get prediction
        logger.info("Starting complete analysis pipeline...")
        
        # Validate file
        RequestValidator.validate_image_file(file)
        
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Get prediction
        prediction_result = predictor.predict(image, use_tta=use_tta)
        
        # Step 2: Generate medical interpretation
        medical_report = interpreter.generate_final_report(prediction_result)
        
        # Combine results
        complete_analysis = {
            "prediction": prediction_result,
            "medical_report": medical_report,
            "metadata": {
                "filename": file.filename,
                "file_size": len(contents),
                "image_size": image.size,
                "analysis_type": "complete",
                "processed_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        }
        
        logger.info("Complete analysis finished successfully")
        return complete_analysis
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Complete analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

if __name__ == "__main__":
    # Streamlit frontend expects port 8002; override if not explicitly set
    port = int(settings.API_PORT) if settings.API_PORT != 8000 else 8002
    uvicorn.run(
        "src.api.main:app",
        host=settings.API_HOST,
        port=port,
        workers=settings.API_WORKERS,
        reload=settings.API_DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )
