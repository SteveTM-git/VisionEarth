from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/")
def read_root():
    return {"status": "VisionEarth API is running!"}

if __name__ == "__main__":
    print("🚀 Starting VisionEarth API...")
    print("📍 Server will run at: http://localhost:8000")
    print("📖 API docs at: http://localhost:8000/docs")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")