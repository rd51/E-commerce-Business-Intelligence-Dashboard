# 📊 E-Commerce Business Intelligence Dashboard v2.0

An end-to-end **enterprise-grade** interactive Business Intelligence dashboard built with Streamlit for e-commerce and sales analytics, featuring **Decision Intelligence**, **Predictive ML Models**, **Explainable AI (XAI)**, **Real-Time Analytics**, **Knowledge Graphs**, and **Executive-level BI Storytelling**.

![Dashboard Preview](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit)
![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python)
![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn)
![NetworkX](https://img.shields.io/badge/NetworkX-Knowledge%20Graphs-blue?style=for-the-badge)

## 🎯 Features Overview

### 📊 Tab 1: Executive Summary
- Real-time KPI tracking (Revenue, Profit, Orders, AOV, Active Customers)
- Year-over-Year growth comparisons with trend indicators
- Revenue trends with interactive forecasting
- Regional performance breakdown with maps
- Key executive insights and alerts

### 📈 Tab 2: Sales Analytics
- Category performance analysis with drill-down
- Channel performance comparison
- Sales heatmap by day of week
- Detailed performance metrics table
- Interactive filtering and multi-dimensional analysis

### 👥 Tab 3: Customer Intelligence
- Customer segmentation (Premium, Regular, New, At-Risk)
- Customer Lifetime Value (CLV) distribution
- RFM Analysis visualization
- Retention cohort analysis
- Churn risk identification

### 🎯 Tab 4: ML Segmentation
- **K-Means Clustering** with elbow method
- **Hierarchical Clustering** with dendrograms
- 3D cluster visualization
- Segment profiles and characteristics
- Business insights from ML

### 💎 Tab 5: CLV Analysis (NEW!)
- **Customer Lifetime Value Calculation** per customer
- CLV distribution by segment (Platinum, Gold, Silver, Bronze, Standard)
- **High-Value vs Low-Value** customer analysis
- Pareto analysis (80/20 rule)
- CLV:CAC ratio insights
- At-risk high-value customer identification

### 🤖 Tab 6: Predictive AI (NEW!)
- **Multi-model ML Pipeline**:
  - Random Forest Classifier
  - Gradient Boosting Classifier
  - Logistic Regression
  - Decision Tree Classifier
- **Churn Prediction** with probability scoring
- **Model Evaluation Metrics**:
  - Confusion Matrix visualization
  - Accuracy, Precision, Recall, F1-Score
  - ROC-AUC Score & ROC Curve
  - Cumulative Gains Chart
  - Log Loss
- **Prescriptive Recommendations**:
  - Dynamic pricing optimization
  - Discount strategy by segment
  - Targeted marketing actions

### 🔬 Tab 7: Explainable AI (NEW!)
- **Global Feature Importance** charts
- **SHAP-style Analysis**:
  - Summary plots (feature impact distribution)
  - Mean absolute SHAP values
  - Dependence plots
- **Individual Prediction Explanations**:
  - Customer-level drill-down
  - Feature contribution visualization
  - AI-generated recommendations per customer
- **Feature Correlation Heatmap**

### ⚡ Tab 8: Real-Time Analytics (NEW!)
- **Live KPI Ticker** with auto-refresh simulation
- Trend indicators (↑↓→)
- Configurable refresh rates
- Simulation modes (Normal, High Activity, Low Activity)
- Live revenue stream visualization
- Real-time orders by region
- Recent transactions feed
- System performance metrics

### 🕸️ Tab 9: Knowledge Graph (NEW!)
- **NetworkX-powered** entity relationship visualization
- Interactive graph with multiple entity types:
  - Customers
  - Products
  - Categories
  - Regions
- Configurable layouts (spring, circular, shell)
- Network statistics (nodes, edges, density)
- **Relationship strength analysis**
- Category-Region revenue matrix heatmap

### 📋 Tab 10: Decision Intelligence Center
- **Executive Summary Dashboard**
- Business Health Score
- Strategic recommendations with:
  - Priority levels (Critical, High, Medium)
  - Confidence scores
  - Expected impact
  - Timelines
- **What-If Scenario Analysis**
- Executive report generation
- Downloadable reports

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Installation

1. **Clone/Download the repository**

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the dashboard**
   ```bash
   streamlit run app.py
   ```

5. **Open in browser**
   - The dashboard will automatically open at `http://localhost:8501`

## 📁 Project Structure

```
Dashboard Presentation/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md          # Documentation
```

## 🔧 Dashboard Filters

The sidebar provides comprehensive filtering options:
- **Date Range**: Select custom time periods (2 years of data)
- **Product Categories**: Electronics, Fashion, Home & Garden, Sports, Beauty, Books, Toys
- **Regions**: North America, Europe, Asia Pacific, Latin America, Middle East
- **Sales Channels**: Website, Mobile App, Marketplace, Social Media

## 📊 Key Metrics & KPIs

| Metric | Description |
|--------|-------------|
| Total Revenue | Sum of all transaction revenues |
| Gross Profit | Revenue minus costs |
| Profit Margin | Profit as percentage of revenue |
| AOV | Average Order Value |
| Customer LTV | Customer Lifetime Value |
| Churn Rate | Percentage of at-risk customers |
| NPS Score | Net Promoter Score |

## 🔮 Predictive Analytics Features

### Revenue Forecasting
- Linear trend analysis with confidence intervals
- Seasonal adjustment factors
- 6-month forward projection

### Churn Prediction
- Multi-factor risk scoring
- Segments: Low Risk, Medium Risk, High Risk
- Key factors: Recency, Frequency, Support tickets, NPS

### Demand Forecasting
- Category-level predictions
- Inventory optimization recommendations
- Stock level alerts

## 🎨 Dashboard Components

### Visualizations Used
- **Line Charts**: Revenue trends and forecasts
- **Bar Charts**: Category and channel performance
- **Pie/Donut Charts**: Regional distribution
- **Heatmaps**: Sales patterns by time
- **Treemaps**: RFM customer segmentation
- **Scatter Plots**: Customer cohort analysis
- **Histograms**: CLV distribution

### Interactive Elements
- Date range picker
- Multi-select filters
- What-If sliders
- Expandable sections
- Downloadable reports

## 📈 Use Cases

1. **Executive Reporting**: Weekly/monthly performance reviews
2. **Strategic Planning**: Market expansion decisions
3. **Operations**: Inventory optimization
4. **Marketing**: Campaign ROI analysis
5. **Customer Success**: Churn prevention initiatives

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| **Streamlit** | Web application framework |
| **Pandas** | Data manipulation and analysis |
| **NumPy** | Numerical computations |
| **Plotly** | Interactive visualizations |
| **Scikit-learn** | ML models & evaluation |
| **NetworkX** | Knowledge graph analysis |
| **SciPy** | Statistical analysis & clustering |

## 📊 Sample Data

The dashboard includes a synthetic e-commerce dataset for demonstration:
- **50,000 transactions** over 2 years (2024-2025)
- **7 product categories** with realistic pricing
- **5 global regions** with market-appropriate distributions
- **4 sales channels** with varying performance
- **10,000 simulated customers** with behavioral metrics
- **Seasonal patterns** (holiday spikes in Nov-Dec)

## 🤖 Machine Learning Features

### Classification Models
- **Random Forest**: Ensemble method with configurable trees/depth
- **Gradient Boosting**: Sequential learning with tunable learning rate
- **Logistic Regression**: Linear model with regularization
- **Decision Tree**: Interpretable single-tree classifier

### Model Evaluation
- Confusion Matrix visualization
- Precision, Recall, F1-Score
- ROC Curve & AUC
- Cumulative Gains Chart
- Cross-validation support

### Explainable AI (XAI)
- Feature importance rankings
- SHAP-style impact analysis
- Individual prediction explanations
- Feature dependence plots

## 🎯 BI Storytelling Elements

The dashboard follows executive BI storytelling principles:

1. **Context First**: High-level KPIs before detailed analysis
2. **Comparative Insights**: YoY growth, benchmarks, trends
3. **Actionable Recommendations**: Clear next steps with expected ROI
4. **Risk Alerts**: Proactive identification of concerns
5. **Scenario Planning**: What-If analysis for decision support

## 📝 Course Information

**S.P. Jain School of Global Management**  
Data Visualization & Analytics (DVA) Course  
Dashboard Presentation Project

---

**Note**: This dashboard uses synthetic data for demonstration purposes. For production deployment, connect to actual data sources and implement proper security measures.
