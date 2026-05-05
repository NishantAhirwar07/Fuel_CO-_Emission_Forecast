# 🚗 Fuel CO₂ Emission Forecast

A machine learning project that predicts vehicle **CO₂ emissions** based on engine and fuel consumption characteristics using **Multiple Linear Regression**.



## 📌 Overview

This project analyzes the relationship between vehicle attributes — engine size, number of cylinders, and combined fuel consumption — and their carbon dioxide (CO₂) emissions. A Multiple Linear Regression model is trained and evaluated to forecast emissions from unseen vehicle data.



## 📂 Dataset

**File:** `FuelConsumptionCo2.csv`

The dataset contains fuel consumption ratings and CO₂ emission estimates for light-duty vehicles. The following features are used:

| Feature | Description |
| `CYLINDERS` | Number of engine cylinders |
| `ENGINESIZE` | Engine displacement (Liters) |
| `FUELCONSUMPTION_COMB` | Combined fuel consumption (L/100 km) |
| `CO2EMISSIONS` | Tailpipe CO₂ emissions (g/km) — **Target Variable** |


---

## 🧠 Model

**Algorithm:** Multiple Linear Regression (`sklearn.linear_model.LinearRegression`)

The model learns the following relationship:

CO2EMISSIONS = m1×CYLINDERS + m2×ENGINESIZE + m3×FUELCONSUMPTION_COMB + c


---

## 📊 Results

The model was evaluated on a 20% held-out test set:

| Metric | Value |
|---|---|
| **R² Score** | ~0.8912 |
| **RMSE** | 22.4468 |
| **MAE** | 16.1325 |

> ✅ An R² of **0.8912** means the model explains ~89% of the variance in CO₂ emissions — a strong result for a linear model.


## 📈 Visualizations

The notebook includes:
- Histograms of all features
- Scatter plots: `FUELCONSUMPTION_COMB` vs `CO2EMISSIONS`
- Scatter plots: `ENGINESIZE` vs `CO2EMISSIONS`
- Scatter plots: `CYLINDERS` vs `CO2EMISSIONS`


## 🛠️ Tech Stack

| Tool | Purpose |
| Python  | Core language |
| Pandas | Data loading & manipulation |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Scikit-learn | Model training & evaluation |
| Google Colab | Development environment |

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/fuel-co2-emission-forecast.git
cd fuel-co2-emission-forecast

### 2. Install Dependencies

```bash
pip install pandas numpy matplotlib scikit-learn
```

### 3. Run the Notebook

Open `Fuel_CO₂_EmissionForecast.ipynb` in **Google Colab** or **Jupyter Notebook** and run all cells.

> **Note:** If running locally, replace the Google Colab file upload block with:
> ```python
> df = pd.read_csv("FuelConsumptionCo2.csv")


## 📁 Project Structure
```
fuel-co2-emission-forecast/
│
├── Fuel_CO₂_EmissionForecast.ipynb     # Main notebook
├── FuelConsumptionCo2.csv              # Dataset
└── README.md                           # Project documentation 
```

