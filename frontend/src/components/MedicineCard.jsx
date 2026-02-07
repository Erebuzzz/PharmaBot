import React from "react";

const CONFIDENCE_STYLES = {
  high: { bg: "#dcfce7", color: "#166534", label: "High Confidence — FDA Verified" },
  medium: { bg: "#fef3c7", color: "#92400e", label: "Medium Confidence — Dataset" },
  low: { bg: "#fee2e2", color: "#991b1b", label: "Low Confidence — Consult a Doctor" },
};

export default function MedicineCard({ medicine }) {
  if (!medicine) return null;

  return (
    <div className="medicine-card">
      <div className="medicine-card-header">
        <h3>💊 {medicine.name || medicine.brand_name}</h3>
        {medicine.generic_name && (
          <span className="generic-name">({medicine.generic_name})</span>
        )}
      </div>

      <div className="medicine-card-grid">
        {medicine.manufacturer && (
          <div className="medicine-field">
            <span className="field-label">Manufacturer</span>
            <span className="field-value">{medicine.manufacturer}</span>
          </div>
        )}
        {medicine.composition && (
          <div className="medicine-field">
            <span className="field-label">Composition</span>
            <span className="field-value">{medicine.composition}</span>
          </div>
        )}
        {medicine.disease_category && (
          <div className="medicine-field">
            <span className="field-label">Disease Category</span>
            <span className="field-value">{medicine.disease_category}</span>
          </div>
        )}
        {medicine.route && (
          <div className="medicine-field">
            <span className="field-label">Route</span>
            <span className="field-value">{medicine.route}</span>
          </div>
        )}
        {medicine.category && (
          <div className="medicine-field">
            <span className="field-label">Category</span>
            <span className="field-value">{medicine.category}</span>
          </div>
        )}
        {medicine.dosage_form && (
          <div className="medicine-field">
            <span className="field-label">Dosage Form</span>
            <span className="field-value">{medicine.dosage_form}</span>
          </div>
        )}
        {medicine.strength && (
          <div className="medicine-field">
            <span className="field-label">Strength</span>
            <span className="field-value">{medicine.strength}</span>
          </div>
        )}
        {medicine.price && (
          <div className="medicine-field">
            <span className="field-label">Price</span>
            <span className="field-value">₹{medicine.price}</span>
          </div>
        )}
        {medicine.prescription_required && (
          <div className="medicine-field">
            <span className="field-label">Prescription</span>
            <span className="field-value">{medicine.prescription_required === true || medicine.prescription_required === "True" ? "Required" : "OTC"}</span>
          </div>
        )}
        {medicine.classification && (
          <div className="medicine-field">
            <span className="field-label">Classification</span>
            <span className="field-value">{medicine.classification}</span>
          </div>
        )}
        {medicine.indication && (
          <div className="medicine-field">
            <span className="field-label">Indication</span>
            <span className="field-value">{medicine.indication}</span>
          </div>
        )}
      </div>

      {medicine.source && (
        <div className="medicine-source">
          Source: {medicine.source === "fda" ? "FDA Approved Label" :
                  medicine.source === "netmeds" ? "Netmeds Dataset" :
                  medicine.source === "az_india" ? "India A-Z Dataset" :
                  "Medicine Dataset"}
        </div>
      )}
    </div>
  );
}
