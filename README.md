# DCF Valuation Model (API-Based)

This project implements an **automated Discounted Cash Flow (DCF)** valuation model that retrieves company financial data via API and computes intrinsic share value.  
It combines **financial modelling** and **programmatic data automation**, demonstrating how valuation models can be scaled and automated in Python.

> ⚠️ **Note:**  
> The external API used for this implementation is currently inactive.  
> The project is shared **for demonstration and educational purposes only**, to showcase the logic, data flow, and structure of automated valuation modeling.

---

## 📊 Overview
The model uses two main modules:

- **`wacc_via_API.py`** → Calculates the **Weighted Average Cost of Capital (WACC)** using CAPM, company balance sheet data, and debt structure retrieved through `yfinance`.  
- **`DCF_via_API.py`** → Uses the computed WACC to estimate **Free Cash Flows (FCF)**, **Enterprise Value**, and **Fair Value per Share** through a structured DCF pipeline.

---

## ⚙️ Features
- Automated financial data retrieval through API and Yahoo Finance (`yfinance`)
- Dynamic calculation of:
  - **WACC** (risk-free rate, beta, cost of debt, market premium)
  - **Free Cash Flow projections**
  - **Terminal value** and **intrinsic share price**
- Built-in error handling for API calls and missing data
- Modular structure with separate scripts for clarity

---

## 🧠 Methodology
1. **Data retrieval** — Fetch balance sheet, cash flow, and income statement data from the API (formerly hosted at `iveurope.eu`).  
2. **WACC calculation** — Compute the company’s weighted average cost of capital using CAPM assumptions.  
3. **DCF model** — Forecast Free Cash Flows (FCF), compute present values, and derive the intrinsic share value.  
4. **Output** — The model returns both the **market price** and the **DCF-estimated fair value**.

---

## 🧰 Technologies
- **Python**
- `pandas`, `numpy`, `requests`, `yfinance`, `json`, `logging`

---

## 📁 File Structure
