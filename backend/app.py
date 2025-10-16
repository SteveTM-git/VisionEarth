from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn
from dotenv import load_dotenv
import os
import cv2
import numpy as np
from io import BytesIO
import base64
from pathlib import Path
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.predictor import DeforestationPredictor

load_dotenv()

app = FastAPI(
    title="VisionEarth API",
    version="1.0.0",
    description="AI-powered environmental sustainability analysis using satellite imagery"
)

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predictor
MODEL_PATH = "models/unet_real_deforestation.pth"
predictor = None

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    global predictor
    
    # Try real model first, fallback to dummy model
    if Path(MODEL_PATH).exists():
        predictor = DeforestationPredictor(MODEL_PATH)
        print("✅ Real deforestation model loaded successfully")
    elif Path("models/unet_deforestation.pth").exists():
        predictor = DeforestationPredictor("models/unet_deforestation.pth")
        print("⚠️  Using dummy model (real model not found)")
    else:
        print("⚠️  No model found. Train model first with: python3 train_real.py")



@app.get("/")
async def root():
    return {
        "message": "VisionEarth API - Environmental Sustainability AI",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "analyze": "/analyze",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    model_status = "loaded" if predictor is not None else "not loaded"
    return {
        "status": "healthy",
        "model_status": model_status
    }


@app.post("/predict")
async def predict_deforestation(file: UploadFile = File(...)):
    """
    Predict deforestation from uploaded satellite image
    
    Returns segmentation mask and statistics
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Predict
        mask, probs = predictor.predict(image)
        
        # Get statistics
        stats = predictor.get_statistics(mask)
        
        # Create visualization
        vis_image = predictor.visualize_prediction(image, mask)
        
        # Encode visualization as base64
        _, buffer = cv2.imencode('.png', vis_image)
        vis_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            "success": True,
            "statistics": stats,
            "visualization": f"data:image/png;base64,{vis_base64}",
            "image_shape": image.shape[:2]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze")
async def analyze_satellite_image(file: UploadFile = File(...)):
    """
    Comprehensive analysis of satellite image
    
    Returns detailed analysis with multiple metrics
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Predict
        mask, probs = predictor.predict(image)
        stats = predictor.get_statistics(mask)
        
        # Calculate additional metrics
        deforestation_percentage = stats.get('Deforestation', {}).get('percentage', 0)
        
        # Risk assessment
        if deforestation_percentage < 5:
            risk_level = "Low"
            risk_color = "green"
        elif deforestation_percentage < 15:
            risk_level = "Medium"
            risk_color = "yellow"
        else:
            risk_level = "High"
            risk_color = "red"
        
        # Create visualization
        vis_image = predictor.visualize_prediction(image, mask)
        _, buffer = cv2.imencode('.png', vis_image)
        vis_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            "success": True,
            "analysis": {
                "deforestation_detected": deforestation_percentage > 1,
                "deforestation_percentage": deforestation_percentage,
                "risk_level": risk_level,
                "risk_color": risk_color,
                "statistics": stats,
                "total_area_analyzed": image.shape[0] * image.shape[1]
            },
            "visualization": f"data:image/png;base64,{vis_base64}",
            "recommendations": get_recommendations(deforestation_percentage)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def get_recommendations(deforestation_percentage):
    """Generate recommendations based on deforestation level"""
    if deforestation_percentage < 5:
        return [
            "Continue monitoring the area regularly",
            "Maintain current conservation efforts",
            "Share success stories with neighboring regions"
        ]
    elif deforestation_percentage < 15:
        return [
            "Increase monitoring frequency",
            "Investigate causes of deforestation",
            "Implement targeted conservation measures",
            "Engage local communities in protection efforts"
        ]
    else:
        return [
            "⚠️ Urgent action required",
            "Deploy ground teams for verification",
            "Implement immediate conservation interventions",
            "Alert relevant environmental authorities",
            "Consider legal action against illegal activities",
            "Establish protected zones"
        ]


@app.get("/model-info")
async def get_model_info():
    """Get information about the loaded model"""
    if predictor is None:
        return {"status": "not loaded"}
    
    return {
        "status": "loaded",
        "model_path": MODEL_PATH,
        "classes": predictor.class_names,
        "device": str(predictor.device)
    }


if __name__ == "__main__":
    print("🚀 Starting VisionEarth API...")
    print("📍 Server: http://localhost:8000")
    print("📖 API Docs: http://localhost:8000/docs")
    print("🧪 Test predictions at: http://localhost:8000/docs")
    
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        log_level="info"
    )