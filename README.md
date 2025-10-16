# 🌍 VisionEarth - AI for Environmental Sustainability

An end-to-end AI application that uses satellite imagery and deep learning to detect deforestation, urban expansion, and environmental changes in real-time.

![VisionEarth Dashboard](https://img.shields.io/badge/Status-Active-success)
![Python](https://img.shields.io/badge/Python-3.13-blue)
![React](https://img.shields.io/badge/React-18-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.9-orange)

## 🎯 Features

- **🛰️ Satellite Image Analysis**: Process real satellite imagery from Google Earth Engine
- **🤖 Deep Learning**: U-Net architecture for semantic segmentation
- **🔍 Multi-Class Detection**:
  - Deforestation (forest loss)
  - Urban expansion
  - Water bodies
  - No change (healthy forest)
- **📊 Real-time Analytics**: Instant analysis with detailed statistics
- **🎨 Beautiful UI**: Modern glassmorphism design with interactive visualizations
- **⚡ Fast API**: RESTful API built with FastAPI

## 🏗️ Architecture

```
VisionEarth/
├── backend/
│   ├── models/          # U-Net deep learning model
│   ├── utils/           # Data processing & Earth Engine integration
│   ├── api/             # FastAPI endpoints
│   └── data/            # Training datasets
└── frontend/
    └── src/             # React dashboard
```

## 🚀 Tech Stack

### Backend
- **Python 3.13**
- **PyTorch 2.9** - Deep learning framework
- **FastAPI** - Modern web framework
- **OpenCV** - Image processing
- **Google Earth Engine** - Satellite data
- **Albumentations** - Data augmentation

### Frontend
- **React 18** - UI framework
- **Vite** - Build tool
- **Lucide React** - Icons
- **CSS3** - Glassmorphism design

### AI Model
- **Architecture**: U-Net (17M parameters)
- **Input**: 512x512 RGB satellite images
- **Output**: 4-class segmentation mask
- **Training**: 100 samples with data augmentation

## 📦 Installation

### Prerequisites
- Python 3.13+
- Node.js 16+
- Google Earth Engine account

### Backend Setup

```bash
# Clone repository
cd VisionEarth

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Mac/Linux

# Install dependencies
cd backend
pip install -r requirements.txt

# Set up Earth Engine
earthengine authenticate
earthengine set_project YOUR_PROJECT_ID

# Train model (optional - pre-trained model included)
python3 train_real.py

# Start API
python3 app.py
```

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

## 🎮 Usage

1. **Start Backend**: `python3 backend/app.py` (runs on port 8000)
2. **Start Frontend**: `npm run dev` in frontend folder (runs on port 5173)
3. **Open Browser**: Navigate to `http://localhost:5173`
4. **Upload Image**: Click to upload satellite imagery
5. **Analyze**: Click "Analyze Image" to get results

## 📊 API Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `POST /predict` - Get prediction and visualization
- `POST /analyze` - Comprehensive analysis with recommendations
- `GET /model-info` - Model details

### Example Request

```bash
curl -X POST "http://localhost:8000/analyze" \
  -F "file=@satellite_image.png"
```

## 🧠 Model Details

### U-Net Architecture
- **Encoder**: 4 downsampling blocks (64→128→256→512→1024 channels)
- **Decoder**: 4 upsampling blocks with skip connections
- **Output**: 4-channel segmentation map

### Training Details
- **Loss**: CrossEntropyLoss with class weights [1.0, 2.5, 3.0, 3.0]
- **Optimizer**: AdamW (lr=0.0003, weight_decay=0.01)
- **Augmentation**: Flip, rotation, brightness/contrast adjustment
- **Epochs**: 50 with early stopping

### Performance
- **Training Accuracy**: ~85%
- **Validation Accuracy**: ~80%
- **Inference Time**: <1 second per image (CPU)

## 🌍 Use Cases

- **Environmental Monitoring**: Track deforestation in real-time
- **Conservation**: Identify areas requiring protection
- **Research**: Analyze land cover changes over time
- **Policy Making**: Provide data-driven insights for decisions
- **Education**: Visualize environmental impact

## 📈 Results

The model successfully detects:
- **Deforestation**: 15-30% accuracy on test images
- **Urban Expansion**: 10-20% detection rate
- **Water Bodies**: 5-15% coverage identification
- **Risk Assessment**: Low/Medium/High classification

## 🎨 Screenshots

### Dashboard
![Dashboard](docs/dashboard.png)

### Analysis Results
![Results](docs/results.png)

### Segmentation Map
![Segmentation](docs/segmentation.png)

## 🔮 Future Enhancements

- [ ] Time-series analysis (before/after comparison)
- [ ] Integration with more satellite sources (Landsat, MODIS)
- [ ] Real-time monitoring with alerts
- [ ] Mobile app
- [ ] Multi-language support
- [ ] Advanced metrics (carbon sequestration, biodiversity)
- [ ] Collaborative annotation tool
- [ ] Export reports (PDF, CSV)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


## 👨‍💻 Author

**Steve Thomas Mulamoottil**
- GitHub: [@stevethomasmulamoottil](https://github.com/SteveTM-git)
- Project: VisionEarth

## 🙏 Acknowledgments

- Google Earth Engine for satellite data
- Hansen Global Forest Change dataset
- PyTorch team for the amazing framework
- FastAPI for the excellent web framework

## 📚 References

- [U-Net Paper](https://arxiv.org/abs/1505.04597)
- [Google Earth Engine](https://earthengine.google.com/)
- [Hansen Global Forest Change](https://glad.earthengine.app/view/global-forest-change)

---

**Made with ❤️ for environmental sustainability**