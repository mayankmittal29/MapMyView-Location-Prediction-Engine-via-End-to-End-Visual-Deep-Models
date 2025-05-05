import React, { useState } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [file, setFile] = useState(null);
  const [previewURL, setPreviewURL] = useState(null);
  const [prediction, setPrediction] = useState(null);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
    setPreviewURL(URL.createObjectURL(selectedFile));
    setPrediction(null);
  };

  const handlePredict = async () => {
    if (!file) {
      alert("Please select an image before predicting.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await axios.post("http://localhost:8000/predict/", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setPrediction(res.data);
    } catch (error) {
      console.error("Prediction failed:", error);
      alert("Prediction failed. Please try again.");
    }
  };

  const handleClear = () => {
    if (!file) {
      alert("No image selected to clear.");
      return;
    }

    setFile(null);
    setPreviewURL(null);
    setPrediction(null);
    document.getElementById("fileInput").value = null;
  };

  return (
    <>
      <div className="app-header">
        <h1 className="app-title">MapMyView: See Where You Stand Instantly</h1>
      </div>
      
      <div className="main-container">
        {/* Static Map Card - Now Larger */}
        <div className="card map-card">
          <h2>Region ID Map</h2>
          <img src="/regionmap.png" alt="Campus Map" className="map-image" />
        </div>

        {/* Prediction Card - Now Smaller */}
        <div className="card prediction-card">
          <h2>Predictor</h2>

          <input
            id="fileInput"
            type="file"
            accept="image/*"
            onChange={handleFileChange}
          />

          {previewURL && (
            <div className="preview-container">
              <img src={previewURL} alt="Preview" className="preview-image" />
            </div>
          )}

          <div className="button-group">
            <button onClick={handlePredict}>Predict</button>
            <button className="clear-button" onClick={handleClear}>Clear</button>
          </div>

          {prediction && (
            <div className="results">
              <h3>Prediction Results:</h3>
              <p><strong>Region ID:</strong> {prediction.Region_ID}</p>
              <p><strong>Latitude:</strong> {prediction.latitude}</p>
              <p><strong>Longitude:</strong> {prediction.longitude}</p>
              <p><strong>Angle:</strong> {prediction.angle}°</p>
            </div>
          )}
        </div>
      </div>
    </>
  );
}

export default App;