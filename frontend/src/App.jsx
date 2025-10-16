import { useState } from 'react';
import { Upload, Satellite, AlertCircle, CheckCircle, Activity } from 'lucide-react';
import './App.css';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreview(URL.createObjectURL(file));
      setResults(null);
      setError(null);
    }
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return;

    setAnalyzing(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch('http://localhost:8000/analyze', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Analysis failed');
      }

      const data = await response.json();
      setResults(data);
    } catch (err) {
      setError(err.message || 'Error analyzing image');
    } finally {
      setAnalyzing(false);
    }
  };

  const getRiskColor = (riskLevel) => {
    switch (riskLevel?.toLowerCase()) {
      case 'low': return '#10b981';
      case 'medium': return '#f59e0b';
      case 'high': return '#ef4444';
      default: return '#6b7280';
    }
  };

  return (
    <div className="app">
      <header className="header">
        <div className="header-content">
          <div className="logo">
            <Satellite size={32} />
            <h1>VisionEarth</h1>
          </div>
          <p className="tagline">AI-Powered Environmental Sustainability Analysis</p>
        </div>
      </header>

      <main className="main">
        <div className="upload-section">
          <div className="upload-card">
            <h2>Upload Satellite Image</h2>
            <div className="upload-area">
              <input
                type="file"
                id="file-upload"
                accept="image/*"
                onChange={handleFileSelect}
                style={{ display: 'none' }}
              />
              <label htmlFor="file-upload" className="upload-label">
                <Upload size={48} />
                <p>Click to upload or drag and drop</p>
                <span>PNG, JPG, TIFF up to 10MB</span>
              </label>
            </div>

            {preview && (
              <div className="preview-section">
                <h3>Selected Image</h3>
                <img src={preview} alt="Preview" className="preview-image" />
                <button
                  onClick={handleAnalyze}
                  disabled={analyzing}
                  className="analyze-btn"
                >
                  {analyzing ? (
                    <>
                      <Activity className="spin" size={20} />
                      Analyzing...
                    </>
                  ) : (
                    <>
                      <Satellite size={20} />
                      Analyze Image
                    </>
                  )}
                </button>
              </div>
            )}
          </div>
        </div>

        {error && (
          <div className="error-card">
            <AlertCircle size={24} />
            <p>{error}</p>
          </div>
        )}

        {results && (
          <div className="results-section">
            <div className="results-header">
              <h2>Analysis Results</h2>
              <div
                className="risk-badge"
                style={{ backgroundColor: getRiskColor(results.analysis.risk_level) }}
              >
                {results.analysis.risk_level} Risk
              </div>
            </div>

            <div className="results-grid">
              <div className="result-card">
                <h3>Deforestation Detection</h3>
                <div className="metric">
                  <div className="metric-value">
                    {results.analysis.deforestation_percentage.toFixed(2)}%
                  </div>
                  <div className="metric-label">Area Affected</div>
                </div>
                {results.analysis.deforestation_detected ? (
                  <div className="status detected">
                    <AlertCircle size={20} />
                    Deforestation Detected
                  </div>
                ) : (
                  <div className="status safe">
                    <CheckCircle size={20} />
                    No Deforestation Detected
                  </div>
                )}
              </div>

              <div className="result-card">
                <h3>Land Cover Statistics</h3>
                <div className="stats-list">
                  {Object.entries(results.analysis.statistics).map(([key, value]) => (
                    <div key={key} className="stat-item">
                      <span className="stat-label">{key}</span>
                      <span className="stat-value">{value.percentage.toFixed(2)}%</span>
                    </div>
                  ))}
                </div>
              </div>

              <div className="result-card visualization-card">
                <h3>Segmentation Map</h3>
                <img
                  src={results.visualization}
                  alt="Segmentation"
                  className="visualization-image"
                />
                <div className="legend">
                  <div className="legend-item">
                    <span className="color-box" style={{ backgroundColor: '#00ff00' }}></span>
                    <span>No Change</span>
                  </div>
                  <div className="legend-item">
                    <span className="color-box" style={{ backgroundColor: '#ff0000' }}></span>
                    <span>Deforestation</span>
                  </div>
                  <div className="legend-item">
                    <span className="color-box" style={{ backgroundColor: '#808080' }}></span>
                    <span>Urban</span>
                  </div>
                  <div className="legend-item">
                    <span className="color-box" style={{ backgroundColor: '#0000ff' }}></span>
                    <span>Water</span>
                  </div>
                </div>
              </div>

              <div className="result-card recommendations-card">
                <h3>Recommendations</h3>
                <ul className="recommendations-list">
                  {results.recommendations.map((rec, index) => (
                    <li key={index}>{rec}</li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        )}
      </main>

      <footer className="footer">
        <p>
          VisionEarth © 2025 | Powered by AI for Environmental Sustainability
        </p>
      </footer>
    </div>
  );
}

export default App;