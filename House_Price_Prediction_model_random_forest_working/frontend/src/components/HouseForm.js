import React, { useState } from "react";

function HouseForm({ onPredict }) {
  const [formData, setFormData] = useState({
    bedrooms: 3,
    bathrooms: 2.5,
    sqft_living: 1800,
    sqft_lot: 5000,
    floors: 2.0,
    waterfront: 0,
    view: 0,
    condition: 4,
    sqft_above: 1500,
    sqft_basement: 300,
    yr_built: 2005,
    yr_renovated: 0,
  });

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    onPredict(formData);
  };

  return (
    <form onSubmit={handleSubmit} className="card p-4">
      <h3 className="text-center">Enter House Details</h3>
      {Object.keys(formData).map((key) => (
        <div className="mb-3" key={key}>
          <label className="form-label">{key.replace(/_/g, " ")}:</label>
          <input
            type="number"
            className="form-control"
            name={key}
            value={formData[key]}
            onChange={handleChange}
            required
          />
        </div>
      ))}
      <button type="submit" className="btn btn-primary w-100">Predict Price</button>
    </form>
  );
}

export default HouseForm;
