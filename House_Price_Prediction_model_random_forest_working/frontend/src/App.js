import React, { useState } from "react";
import HouseForm from "./components/HouseForm";
import "bootstrap/dist/css/bootstrap.min.css";

function App() {
  const [predictedPrice, setPredictedPrice] = useState(null);

  const handlePrediction = async (formData) => {
    const response = await fetch("http://127.0.0.1:8000/predict/", {  // Updated URL
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(formData),
    });

    const data = await response.json();
    setPredictedPrice(data.predicted_price);
  };

  return (
    <div className="container mt-5">
      <h1 className="text-center">House Price Prediction</h1>
      <HouseForm onPredict={handlePrediction} />
      {predictedPrice && <h2 className="text-center mt-4">Predicted Price: ${predictedPrice.toFixed(2)}</h2>}
    </div>
  );
}

export default App;
