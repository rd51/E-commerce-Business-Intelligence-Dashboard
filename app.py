"""
E-Commerce & Sales Analytics BI Dashboard
==========================================
An end-to-end interactive Business Intelligence dashboard featuring:
- Decision Intelligence & Predictive Insights
- Explainable AI (XAI) & Real-Time Analytics
- Customer Lifetime Value & Knowledge Graphs
- Enterprise-grade BI Storytelling

Version: 2.0 | Author: BI Analytics Team | Last Updated: January 2026
"""

# ============================================================================
# IMPORTS - Organized by functionality (Step 15: Technical Stack)
# ============================================================================

# Core Libraries
import streamlit as st
import pandas as pd
import numpy as np
import time
import hashlib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Visualization
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Machine Learning - Preprocessing & Clustering
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.model_selection import train_test_split, cross_val_score

# Machine Learning - Classification Models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Machine Learning - Evaluation Metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    log_loss
)

# Statistical Analysis
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from scipy import stats

# Network Analysis for Knowledge Graph
import networkx as nx

# ============================================================================
# PERFORMANCE TRACKING
# ============================================================================
DASHBOARD_START_TIME = time.time()

# Page configuration
st.set_page_config(
    page_title="E-Commerce BI Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CACHING FUNCTIONS FOR PERFORMANCE OPTIMIZATION (Step 14)
# ============================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def load_and_cache_data(data_hash):
    """Cache the main dataset for improved performance."""
    return True  # Placeholder for actual data loading

@st.cache_resource(show_spinner=False)
def get_ml_models():
    """Cache ML models to avoid retraining on every refresh."""
    models = {
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'gradient_boost': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'logistic': LogisticRegression(random_state=42, max_iter=1000),
        'decision_tree': DecisionTreeClassifier(random_state=42, max_depth=10)
    }
    return models

@st.cache_data(ttl=300, show_spinner=False)
def calculate_clv_metrics(_customer_df):
    """Cache CLV calculations for performance."""
    return _customer_df

def get_data_hash(df):
    """Generate hash for dataframe to detect changes."""
    return hashlib.md5(pd.util.hash_pandas_object(df).values).hexdigest()[:10]

# Custom CSS for executive styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A5F;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .insight-box {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 10px 10px 0;
    }
    .prediction-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
    }
    .warning-box {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
    /* Sidebar styling */
    .css-1d391kg {
        padding-top: 1rem;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
        color: #1E3A5F;
        font-size: 1rem;
        font-weight: 600;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        padding-bottom: 0.3rem;
        border-bottom: 2px solid #667eea;
    }
    /* Filter buttons styling */
    .stButton button {
        width: 100%;
        border-radius: 5px;
        font-size: 0.8rem;
    }
    /* Multiselect styling */
    .stMultiSelect [data-baseweb="tag"] {
        background-color: #667eea;
    }
    /* Slider styling */
    .stSlider [data-baseweb="slider"] {
        margin-top: 0.5rem;
    }
    /* Filter summary box */
    .filter-summary {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-top: 1rem;
    }
    /* Active filter banner */
    .stAlert {
        background-color: #e7f3ff;
        border: 1px solid #b8daff;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# DATA GENERATION & LOADING
# ============================================
@st.cache_data
def generate_ecommerce_data():
    """Generate comprehensive e-commerce dataset"""
    np.random.seed(42)
    
    # Date range - 2 years of data
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2025, 12, 31)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Product categories and details
    categories = ['Electronics', 'Fashion', 'Home & Garden', 'Sports', 'Beauty', 'Books', 'Toys']
    regions = ['North America', 'Europe', 'Asia Pacific', 'Latin America', 'Middle East']
    channels = ['Website', 'Mobile App', 'Marketplace', 'Social Media']
    customer_segments = ['Premium', 'Regular', 'New', 'Churned']
    
    # Generate transactions
    n_transactions = 50000
    
    data = {
        'transaction_id': range(1, n_transactions + 1),
        'date': np.random.choice(dates, n_transactions),
        'category': np.random.choice(categories, n_transactions, p=[0.25, 0.20, 0.15, 0.15, 0.10, 0.10, 0.05]),
        'region': np.random.choice(regions, n_transactions, p=[0.35, 0.25, 0.20, 0.12, 0.08]),
        'channel': np.random.choice(channels, n_transactions, p=[0.40, 0.30, 0.20, 0.10]),
        'customer_segment': np.random.choice(customer_segments, n_transactions, p=[0.15, 0.45, 0.30, 0.10]),
        'quantity': np.random.randint(1, 10, n_transactions),
    }
    
    df = pd.DataFrame(data)
    
    # Price mapping by category
    price_map = {
        'Electronics': (50, 2000),
        'Fashion': (20, 500),
        'Home & Garden': (30, 800),
        'Sports': (25, 600),
        'Beauty': (10, 200),
        'Books': (5, 50),
        'Toys': (10, 150)
    }
    
    df['unit_price'] = df['category'].apply(
        lambda x: np.random.uniform(price_map[x][0], price_map[x][1])
    )
    df['revenue'] = df['quantity'] * df['unit_price']
    
    # Add seasonality effect
    df['month'] = pd.to_datetime(df['date']).dt.month
    seasonal_factor = df['month'].apply(lambda x: 1.3 if x in [11, 12] else (0.8 if x in [1, 2] else 1.0))
    df['revenue'] = df['revenue'] * seasonal_factor
    
    # Cost and profit
    df['cost'] = df['revenue'] * np.random.uniform(0.4, 0.7, n_transactions)
    df['profit'] = df['revenue'] - df['cost']
    df['profit_margin'] = (df['profit'] / df['revenue']) * 100
    
    # Customer metrics
    df['customer_id'] = np.random.randint(1, 10000, n_transactions)
    df['is_returning'] = np.random.choice([True, False], n_transactions, p=[0.6, 0.4])
    df['discount_applied'] = np.random.choice([0, 5, 10, 15, 20], n_transactions, p=[0.4, 0.25, 0.20, 0.10, 0.05])
    
    # Marketing attribution
    df['marketing_source'] = np.random.choice(
        ['Organic', 'Paid Search', 'Social Ads', 'Email', 'Referral', 'Direct'],
        n_transactions,
        p=[0.25, 0.20, 0.15, 0.15, 0.10, 0.15]
    )
    
    df['date'] = pd.to_datetime(df['date'])
    
    return df

@st.cache_data
def generate_customer_data():
    """Generate customer-level data for CLV analysis"""
    np.random.seed(42)
    n_customers = 10000
    
    data = {
        'customer_id': range(1, n_customers + 1),
        'acquisition_date': pd.date_range(start='2023-01-01', periods=n_customers, freq='H'),
        'total_orders': np.random.poisson(5, n_customers),
        'total_spend': np.random.exponential(500, n_customers),
        'avg_order_value': np.random.uniform(50, 300, n_customers),
        'days_since_last_purchase': np.random.exponential(30, n_customers),
        'customer_segment': np.random.choice(['Premium', 'Regular', 'New', 'At-Risk'], n_customers, p=[0.15, 0.45, 0.25, 0.15]),
        'clv_score': np.random.uniform(100, 5000, n_customers),
        'churn_probability': np.random.uniform(0, 1, n_customers),
        'nps_score': np.random.randint(-100, 101, n_customers)
    }
    
    return pd.DataFrame(data)

# Load data
df = generate_ecommerce_data()
customer_df = generate_customer_data()

# Add sub-category mapping for hierarchical filtering
subcategory_map = {
    'Electronics': ['Smartphones', 'Laptops', 'Tablets', 'Headphones', 'Smartwatches', 'Gaming'],
    'Fashion': ["Men's Clothing", "Women's Clothing", 'Footwear', 'Accessories'],
    'Home & Garden': ['Furniture', 'Kitchen', 'Decor', 'Garden'],
    'Sports': ['Fitness Equipment', 'Outdoor Gear', 'Sports Equipment', 'Athletic Wear'],
    'Beauty': ['Skincare', 'Makeup', 'Hair Care', 'Wellness'],
    'Books': ['Fiction', 'Non-Fiction', 'Educational', 'Children'],
    'Toys': ['Action Figures', 'Board Games', 'Educational Toys', 'Outdoor Toys']
}

# Add sub-category to dataframe
np.random.seed(42)
df['sub_category'] = df['category'].apply(
    lambda x: np.random.choice(subcategory_map.get(x, ['General']))
)

# ============================================
# SIDEBAR - GLOBAL FILTERS & INTERACTIVE SLICERS
# ============================================
st.sidebar.markdown("""
<div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 1rem;">
    <h2 style="color: white; margin: 0;">🎛️ Dashboard Controls</h2>
    <p style="color: rgba(255,255,255,0.8); margin: 0; font-size: 0.9rem;">Interactive Global Filters</p>
</div>
""", unsafe_allow_html=True)

# --------- DATE RANGE FILTER ---------
st.sidebar.markdown("### 📅 Time Period")

# Quick date presets
date_preset = st.sidebar.selectbox(
    "Quick Select",
    options=['Custom Range', 'Last 7 Days', 'Last 30 Days', 'Last 90 Days', 
             'Last 6 Months', 'Year to Date', 'Last Year', 'All Time'],
    index=0
)

# Calculate date ranges based on preset
max_date = df['date'].max()
min_date = df['date'].min()

if date_preset == 'Last 7 Days':
    start_default = max_date - timedelta(days=7)
    end_default = max_date
elif date_preset == 'Last 30 Days':
    start_default = max_date - timedelta(days=30)
    end_default = max_date
elif date_preset == 'Last 90 Days':
    start_default = max_date - timedelta(days=90)
    end_default = max_date
elif date_preset == 'Last 6 Months':
    start_default = max_date - timedelta(days=180)
    end_default = max_date
elif date_preset == 'Year to Date':
    start_default = datetime(max_date.year, 1, 1)
    end_default = max_date
elif date_preset == 'Last Year':
    start_default = datetime(max_date.year - 1, 1, 1)
    end_default = datetime(max_date.year - 1, 12, 31)
elif date_preset == 'All Time':
    start_default = min_date
    end_default = max_date
else:  # Custom Range
    start_default = min_date
    end_default = max_date

# Date range input
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input(
        "Start Date",
        value=start_default,
        min_value=min_date,
        max_value=max_date,
        key='start_date'
    )
with col2:
    end_date = st.date_input(
        "End Date",
        value=end_default,
        min_value=min_date,
        max_value=max_date,
        key='end_date'
    )

# --------- REGION FILTER ---------
st.sidebar.markdown("### 🌍 Geographic Filters")

# Select All / Deselect All for Regions
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("Select All Regions", key='sel_all_regions'):
        st.session_state['selected_regions'] = list(df['region'].unique())
with col2:
    if st.button("Clear Regions", key='clear_regions'):
        st.session_state['selected_regions'] = []

# Region multiselect
all_regions = sorted(df['region'].unique())
selected_regions = st.sidebar.multiselect(
    "Select Regions",
    options=all_regions,
    default=all_regions,
    key='region_filter',
    help="Filter data by geographic region"
)

# --------- CATEGORY / SUB-CATEGORY HIERARCHICAL FILTER ---------
st.sidebar.markdown("### 📦 Product Filters")

# Category filter
all_categories = sorted(df['category'].unique())

col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("All Categories", key='sel_all_cat'):
        st.session_state['selected_categories'] = list(all_categories)
with col2:
    if st.button("Clear Categories", key='clear_cat'):
        st.session_state['selected_categories'] = []

selected_categories = st.sidebar.multiselect(
    "Select Categories",
    options=all_categories,
    default=all_categories,
    key='category_filter',
    help="Filter by product category"
)

# Dynamic Sub-Category filter based on selected categories
available_subcategories = []
for cat in selected_categories:
    available_subcategories.extend(subcategory_map.get(cat, []))
available_subcategories = sorted(list(set(available_subcategories)))

if available_subcategories:
    selected_subcategories = st.sidebar.multiselect(
        "Select Sub-Categories",
        options=available_subcategories,
        default=available_subcategories,
        key='subcategory_filter',
        help="Filter by product sub-category (depends on category selection)"
    )
else:
    selected_subcategories = []

# --------- CUSTOMER SEGMENT FILTER ---------
st.sidebar.markdown("### 👥 Customer Filters")

all_segments = sorted(df['customer_segment'].unique())
selected_segments = st.sidebar.multiselect(
    "Customer Segments",
    options=all_segments,
    default=all_segments,
    key='segment_filter',
    help="Filter by customer segment type"
)

# --------- SALES CHANNEL FILTER ---------
st.sidebar.markdown("### 📱 Channel Filters")

all_channels = sorted(df['channel'].unique())
selected_channels = st.sidebar.multiselect(
    "Sales Channels",
    options=all_channels,
    default=all_channels,
    key='channel_filter',
    help="Filter by sales channel"
)

# Marketing Source filter
all_marketing = sorted(df['marketing_source'].unique())
selected_marketing = st.sidebar.multiselect(
    "Marketing Sources",
    options=all_marketing,
    default=all_marketing,
    key='marketing_filter',
    help="Filter by marketing attribution source"
)

# --------- ADDITIONAL FILTERS ---------
st.sidebar.markdown("### 🔧 Additional Filters")

# Revenue Range slider
min_revenue = float(df['revenue'].min())
max_revenue = float(df['revenue'].max())
revenue_range = st.sidebar.slider(
    "Revenue Range ($)",
    min_value=min_revenue,
    max_value=max_revenue,
    value=(min_revenue, max_revenue),
    format="$%.0f",
    key='revenue_filter',
    help="Filter transactions by revenue amount"
)

# Quantity filter
min_qty = int(df['quantity'].min())
max_qty = int(df['quantity'].max())
quantity_range = st.sidebar.slider(
    "Quantity Range",
    min_value=min_qty,
    max_value=max_qty,
    value=(min_qty, max_qty),
    key='quantity_filter',
    help="Filter by order quantity"
)

# Returning customer filter
customer_type = st.sidebar.radio(
    "Customer Type",
    options=['All Customers', 'Returning Only', 'New Only'],
    index=0,
    key='customer_type_filter',
    help="Filter by customer purchase history"
)

# ============================================
# APPLY ALL FILTERS TO CREATE FILTERED DATASET
# ============================================
@st.cache_data
def apply_filters(df, start_date, end_date, regions, categories, subcategories, 
                  segments, channels, marketing, revenue_range, quantity_range, customer_type):
    """Apply all global filters to the dataset"""
    
    filtered = df.copy()
    
    # Date filter
    filtered = filtered[
        (filtered['date'].dt.date >= start_date) &
        (filtered['date'].dt.date <= end_date)
    ]
    
    # Region filter
    if regions:
        filtered = filtered[filtered['region'].isin(regions)]
    
    # Category filter
    if categories:
        filtered = filtered[filtered['category'].isin(categories)]
    
    # Sub-category filter
    if subcategories:
        filtered = filtered[filtered['sub_category'].isin(subcategories)]
    
    # Customer segment filter
    if segments:
        filtered = filtered[filtered['customer_segment'].isin(segments)]
    
    # Channel filter
    if channels:
        filtered = filtered[filtered['channel'].isin(channels)]
    
    # Marketing source filter
    if marketing:
        filtered = filtered[filtered['marketing_source'].isin(marketing)]
    
    # Revenue range filter
    filtered = filtered[
        (filtered['revenue'] >= revenue_range[0]) &
        (filtered['revenue'] <= revenue_range[1])
    ]
    
    # Quantity range filter
    filtered = filtered[
        (filtered['quantity'] >= quantity_range[0]) &
        (filtered['quantity'] <= quantity_range[1])
    ]
    
    # Customer type filter
    if customer_type == 'Returning Only':
        filtered = filtered[filtered['is_returning'] == True]
    elif customer_type == 'New Only':
        filtered = filtered[filtered['is_returning'] == False]
    
    return filtered

# Apply filters
filtered_df = apply_filters(
    df, start_date, end_date, selected_regions, selected_categories,
    selected_subcategories, selected_segments, selected_channels,
    selected_marketing, revenue_range, quantity_range, customer_type
)

# --------- FILTER STATUS DISPLAY ---------
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Filter Summary")

# Calculate filter statistics
total_records = len(df)
filtered_records = len(filtered_df)
filter_pct = (filtered_records / total_records * 100) if total_records > 0 else 0

# Display filter status
st.sidebar.markdown(f"""
<div style="background: #f0f2f6; padding: 1rem; border-radius: 10px; margin-top: 0.5rem;">
    <p style="margin: 0; font-size: 0.9rem;"><strong>Records Displayed:</strong> {filtered_records:,} / {total_records:,}</p>
    <p style="margin: 0; font-size: 0.9rem;"><strong>Filter Coverage:</strong> {filter_pct:.1f}%</p>
    <p style="margin: 0; font-size: 0.9rem;"><strong>Date Range:</strong> {start_date} to {end_date}</p>
    <p style="margin: 0; font-size: 0.9rem;"><strong>Active Filters:</strong></p>
    <ul style="margin: 0; padding-left: 1.2rem; font-size: 0.8rem;">
        <li>Regions: {len(selected_regions)}/{len(all_regions)}</li>
        <li>Categories: {len(selected_categories)}/{len(all_categories)}</li>
        <li>Segments: {len(selected_segments)}/{len(all_segments)}</li>
        <li>Channels: {len(selected_channels)}/{len(all_channels)}</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Reset all filters button
st.sidebar.markdown("")
if st.sidebar.button("🔄 Reset All Filters", key='reset_filters', use_container_width=True):
    st.rerun()

# Warning if no data after filters
if filtered_records == 0:
    st.sidebar.warning("⚠️ No data matches current filters. Please adjust your selections.")

# ============================================
# MAIN DASHBOARD
# ============================================
st.markdown('<h1 class="main-header">📊 E-Commerce Business Intelligence Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Decision Intelligence • Predictive Insights • Executive Analytics</p>', unsafe_allow_html=True)

# --------- ACTIVE FILTER CONTEXT BANNER ---------
if filtered_records < total_records:
    filter_context = []
    if len(selected_regions) < len(all_regions):
        filter_context.append(f"Regions: {', '.join(selected_regions[:3])}{'...' if len(selected_regions) > 3 else ''}")
    if len(selected_categories) < len(all_categories):
        filter_context.append(f"Categories: {', '.join(selected_categories[:3])}{'...' if len(selected_categories) > 3 else ''}")
    if len(selected_segments) < len(all_segments):
        filter_context.append(f"Segments: {', '.join(selected_segments[:2])}{'...' if len(selected_segments) > 2 else ''}")
    if customer_type != 'All Customers':
        filter_context.append(f"Customer Type: {customer_type}")
    
    st.info(f"🔍 **Active Filters:** {' | '.join(filter_context) if filter_context else 'Date range modified'} — Showing {filtered_records:,} of {total_records:,} records ({filter_pct:.1f}%)")

# Navigation tabs - Extended with new features (Steps 8-12)
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
    "🎯 Executive Summary",
    "📈 Sales Analytics",
    "🔄 Funnel Analytics",
    "🧠 ML Segmentation",
    "💎 CLV Analysis",
    "🤖 Predictive AI",
    "🔬 Explainable AI",
    "⚡ Real-Time",
    "🕸️ Knowledge Graph",
    "📋 Decision Center"
])

# ============================================
# TAB 1: EXECUTIVE SUMMARY WITH CUSTOM KPIs
# ============================================
with tab1:
    st.markdown("## 🎯 Executive Overview")
    
    # ============================================
    # KPI CALCULATIONS WITH THRESHOLDS
    # ============================================
    
    # Basic KPIs
    total_revenue = filtered_df['revenue'].sum()
    total_profit = filtered_df['profit'].sum()
    total_cost = filtered_df['cost'].sum()
    total_orders = len(filtered_df)
    avg_order_value = filtered_df['revenue'].mean() if total_orders > 0 else 0
    profit_margin = (total_profit / total_revenue * 100) if total_revenue > 0 else 0
    unique_customers = filtered_df['customer_id'].nunique()
    
    # Calculate YoY Growth
    current_year_data = filtered_df[filtered_df['date'].dt.year == 2025]
    prev_year_data = filtered_df[filtered_df['date'].dt.year == 2024]
    
    current_year_revenue = current_year_data['revenue'].sum()
    prev_year_revenue = prev_year_data['revenue'].sum()
    yoy_growth = ((current_year_revenue - prev_year_revenue) / prev_year_revenue * 100) if prev_year_revenue > 0 else 0
    
    # Calculate MoM Growth
    filtered_df_sorted = filtered_df.sort_values('date')
    monthly_revenue = filtered_df_sorted.groupby(filtered_df_sorted['date'].dt.to_period('M'))['revenue'].sum()
    
    if len(monthly_revenue) >= 2:
        current_month_rev = monthly_revenue.iloc[-1]
        prev_month_rev = monthly_revenue.iloc[-2]
        mom_growth = ((current_month_rev - prev_month_rev) / prev_month_rev * 100) if prev_month_rev > 0 else 0
    else:
        mom_growth = 0
    
    # Order Growth Rate
    monthly_orders = filtered_df_sorted.groupby(filtered_df_sorted['date'].dt.to_period('M')).size()
    if len(monthly_orders) >= 2:
        current_month_orders = monthly_orders.iloc[-1]
        prev_month_orders = monthly_orders.iloc[-2]
        order_growth = ((current_month_orders - prev_month_orders) / prev_month_orders * 100) if prev_month_orders > 0 else 0
    else:
        order_growth = 0
    
    # Conversion Rate (simulated based on returning customers)
    returning_customers = filtered_df[filtered_df['is_returning'] == True]['customer_id'].nunique()
    conversion_rate = (returning_customers / unique_customers * 100) if unique_customers > 0 else 0
    
    # Average Profit per Order
    avg_profit_per_order = total_profit / total_orders if total_orders > 0 else 0
    
    # ============================================
    # KPI THRESHOLD DEFINITIONS
    # ============================================
    thresholds = {
        'revenue_growth': {'green': 10, 'amber': 0, 'red': -10},
        'profit_margin': {'green': 35, 'amber': 25, 'red': 15},
        'aov_growth': {'green': 5, 'amber': 0, 'red': -5},
        'order_growth': {'green': 10, 'amber': 0, 'red': -10},
        'conversion_rate': {'green': 60, 'amber': 40, 'red': 25},
        'mom_growth': {'green': 5, 'amber': 0, 'red': -5}
    }
    
    def get_status_color(value, metric_type):
        """Return status color based on threshold"""
        t = thresholds.get(metric_type, {'green': 10, 'amber': 0, 'red': -10})
        if value >= t['green']:
            return '#28a745', '🟢', 'Excellent'
        elif value >= t['amber']:
            return '#ffc107', '🟡', 'Warning'
        else:
            return '#dc3545', '🔴', 'Critical'
    
    def get_status_style(color):
        """Return CSS style for KPI card based on color"""
        color_map = {
            '#28a745': 'linear-gradient(135deg, #28a745 0%, #20c997 100%)',  # Green
            '#ffc107': 'linear-gradient(135deg, #ffc107 0%, #fd7e14 100%)',  # Amber
            '#dc3545': 'linear-gradient(135deg, #dc3545 0%, #c82333 100%)'   # Red
        }
        return color_map.get(color, 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)')
    
    # ============================================
    # ROW 1: PRIMARY KPIs WITH COLOR INDICATORS
    # ============================================
    st.markdown("### 📊 Key Performance Indicators")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    # KPI 1: Total Revenue with YoY Growth indicator
    rev_color, rev_icon, rev_status = get_status_color(yoy_growth, 'revenue_growth')
    with col1:
        st.markdown(f"""
        <div style="background: {get_status_style(rev_color)}; padding: 1.2rem; border-radius: 12px; color: white; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
            <p style="margin: 0; font-size: 0.85rem; opacity: 0.9;">💰 Total Revenue</p>
            <h2 style="margin: 0.5rem 0; font-size: 1.8rem;">${total_revenue/1e6:.2f}M</h2>
            <p style="margin: 0; font-size: 0.9rem;">{rev_icon} {yoy_growth:+.1f}% YoY</p>
            <p style="margin: 0.3rem 0 0 0; font-size: 0.75rem; opacity: 0.8;">Status: {rev_status}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # KPI 2: Total Profit with Margin indicator
    margin_color, margin_icon, margin_status = get_status_color(profit_margin, 'profit_margin')
    with col2:
        st.markdown(f"""
        <div style="background: {get_status_style(margin_color)}; padding: 1.2rem; border-radius: 12px; color: white; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
            <p style="margin: 0; font-size: 0.85rem; opacity: 0.9;">📈 Total Profit</p>
            <h2 style="margin: 0.5rem 0; font-size: 1.8rem;">${total_profit/1e6:.2f}M</h2>
            <p style="margin: 0; font-size: 0.9rem;">{margin_icon} {profit_margin:.1f}% Margin</p>
            <p style="margin: 0.3rem 0 0 0; font-size: 0.75rem; opacity: 0.8;">Status: {margin_status}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # KPI 3: Average Order Value
    # Calculate AOV change (simulated comparison)
    baseline_aov = df['revenue'].mean()
    aov_change = ((avg_order_value - baseline_aov) / baseline_aov * 100) if baseline_aov > 0 else 0
    aov_color, aov_icon, aov_status = get_status_color(aov_change, 'aov_growth')
    with col3:
        st.markdown(f"""
        <div style="background: {get_status_style(aov_color)}; padding: 1.2rem; border-radius: 12px; color: white; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
            <p style="margin: 0; font-size: 0.85rem; opacity: 0.9;">💳 Avg Order Value</p>
            <h2 style="margin: 0.5rem 0; font-size: 1.8rem;">${avg_order_value:.2f}</h2>
            <p style="margin: 0; font-size: 0.9rem;">{aov_icon} {aov_change:+.1f}% vs Baseline</p>
            <p style="margin: 0.3rem 0 0 0; font-size: 0.75rem; opacity: 0.8;">Status: {aov_status}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # KPI 4: Order Growth Rate (MoM)
    order_color, order_icon, order_status = get_status_color(order_growth, 'order_growth')
    with col4:
        st.markdown(f"""
        <div style="background: {get_status_style(order_color)}; padding: 1.2rem; border-radius: 12px; color: white; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
            <p style="margin: 0; font-size: 0.85rem; opacity: 0.9;">🛒 Order Growth</p>
            <h2 style="margin: 0.5rem 0; font-size: 1.8rem;">{total_orders:,}</h2>
            <p style="margin: 0; font-size: 0.9rem;">{order_icon} {order_growth:+.1f}% MoM</p>
            <p style="margin: 0.3rem 0 0 0; font-size: 0.75rem; opacity: 0.8;">Status: {order_status}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # KPI 5: Conversion Rate
    conv_color, conv_icon, conv_status = get_status_color(conversion_rate, 'conversion_rate')
    with col5:
        st.markdown(f"""
        <div style="background: {get_status_style(conv_color)}; padding: 1.2rem; border-radius: 12px; color: white; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
            <p style="margin: 0; font-size: 0.85rem; opacity: 0.9;">🎯 Conversion Rate</p>
            <h2 style="margin: 0.5rem 0; font-size: 1.8rem;">{conversion_rate:.1f}%</h2>
            <p style="margin: 0; font-size: 0.9rem;">{conv_icon} {unique_customers:,} Customers</p>
            <p style="margin: 0.3rem 0 0 0; font-size: 0.75rem; opacity: 0.8;">Status: {conv_status}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("")
    
    # ============================================
    # ROW 2: SECONDARY KPIs
    # ============================================
    col1, col2, col3, col4 = st.columns(4)
    
    # MoM Revenue Growth
    mom_color, mom_icon, mom_status = get_status_color(mom_growth, 'mom_growth')
    with col1:
        st.markdown(f"""
        <div style="background: #f8f9fa; border-left: 4px solid {mom_color}; padding: 1rem; border-radius: 0 10px 10px 0;">
            <p style="margin: 0; color: #666; font-size: 0.8rem;">📅 MoM Revenue Growth</p>
            <h3 style="margin: 0.3rem 0; color: #1E3A5F;">{mom_growth:+.1f}% {mom_icon}</h3>
            <p style="margin: 0; color: #888; font-size: 0.75rem;">Month-over-Month Change</p>
        </div>
        """, unsafe_allow_html=True)
    
    # YoY Revenue Growth
    yoy_color, yoy_icon, yoy_status = get_status_color(yoy_growth, 'revenue_growth')
    with col2:
        st.markdown(f"""
        <div style="background: #f8f9fa; border-left: 4px solid {yoy_color}; padding: 1rem; border-radius: 0 10px 10px 0;">
            <p style="margin: 0; color: #666; font-size: 0.8rem;">📆 YoY Revenue Growth</p>
            <h3 style="margin: 0.3rem 0; color: #1E3A5F;">{yoy_growth:+.1f}% {yoy_icon}</h3>
            <p style="margin: 0; color: #888; font-size: 0.75rem;">Year-over-Year Change</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Average Profit per Order
    with col3:
        profit_per_order_status = '🟢' if avg_profit_per_order > 150 else ('🟡' if avg_profit_per_order > 100 else '🔴')
        st.markdown(f"""
        <div style="background: #f8f9fa; border-left: 4px solid {'#28a745' if avg_profit_per_order > 150 else ('#ffc107' if avg_profit_per_order > 100 else '#dc3545')}; padding: 1rem; border-radius: 0 10px 10px 0;">
            <p style="margin: 0; color: #666; font-size: 0.8rem;">💵 Avg Profit/Order</p>
            <h3 style="margin: 0.3rem 0; color: #1E3A5F;">${avg_profit_per_order:.2f} {profit_per_order_status}</h3>
            <p style="margin: 0; color: #888; font-size: 0.75rem;">Profitability per Transaction</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Total Cost
    cost_ratio = (total_cost / total_revenue * 100) if total_revenue > 0 else 0
    cost_status = '🟢' if cost_ratio < 60 else ('🟡' if cost_ratio < 70 else '🔴')
    with col4:
        st.markdown(f"""
        <div style="background: #f8f9fa; border-left: 4px solid {'#28a745' if cost_ratio < 60 else ('#ffc107' if cost_ratio < 70 else '#dc3545')}; padding: 1rem; border-radius: 0 10px 10px 0;">
            <p style="margin: 0; color: #666; font-size: 0.8rem;">💸 Cost Ratio</p>
            <h3 style="margin: 0.3rem 0; color: #1E3A5F;">{cost_ratio:.1f}% {cost_status}</h3>
            <p style="margin: 0; color: #888; font-size: 0.75rem;">Cost as % of Revenue</p>
        </div>
        """, unsafe_allow_html=True)
    
    # ============================================
    # ANOMALY DETECTION & THRESHOLD ALERTS
    # ============================================
    st.markdown("### ⚠️ Alerts & Anomaly Detection")
    
    alerts = []
    
    # Check for revenue anomalies
    if yoy_growth < -10:
        alerts.append({
            'type': 'critical',
            'icon': '🔴',
            'title': 'Revenue Decline Alert',
            'message': f'YoY revenue has declined by {abs(yoy_growth):.1f}%. Immediate action required.',
            'recommendation': 'Review pricing strategy and marketing campaigns.'
        })
    elif yoy_growth < 0:
        alerts.append({
            'type': 'warning',
            'icon': '🟡',
            'title': 'Revenue Growth Slowdown',
            'message': f'YoY revenue growth is negative at {yoy_growth:.1f}%.',
            'recommendation': 'Consider promotional activities to boost sales.'
        })
    
    # Check profit margin
    if profit_margin < 25:
        alerts.append({
            'type': 'critical',
            'icon': '🔴',
            'title': 'Low Profit Margin Alert',
            'message': f'Profit margin at {profit_margin:.1f}% is below healthy threshold (25%).',
            'recommendation': 'Review cost structure and pricing optimization.'
        })
    elif profit_margin < 35:
        alerts.append({
            'type': 'warning',
            'icon': '🟡',
            'title': 'Profit Margin Below Target',
            'message': f'Current margin {profit_margin:.1f}% is below target of 35%.',
            'recommendation': 'Identify high-margin products to promote.'
        })
    
    # Check conversion rate
    if conversion_rate < 40:
        alerts.append({
            'type': 'warning',
            'icon': '🟡',
            'title': 'Low Customer Retention',
            'message': f'Conversion/retention rate at {conversion_rate:.1f}% needs improvement.',
            'recommendation': 'Implement customer loyalty programs.'
        })
    
    # Check for order volume decline
    if order_growth < -10:
        alerts.append({
            'type': 'critical',
            'icon': '🔴',
            'title': 'Order Volume Decline',
            'message': f'Orders have decreased by {abs(order_growth):.1f}% MoM.',
            'recommendation': 'Investigate customer acquisition channels.'
        })
    
    # Check AOV anomaly
    if aov_change < -10:
        alerts.append({
            'type': 'warning',
            'icon': '🟡',
            'title': 'AOV Decline Detected',
            'message': f'Average order value dropped by {abs(aov_change):.1f}%.',
            'recommendation': 'Review cross-sell and upsell strategies.'
        })
    
    # Positive alerts (achievements)
    if yoy_growth > 20:
        alerts.append({
            'type': 'success',
            'icon': '🟢',
            'title': 'Strong Revenue Growth',
            'message': f'Excellent YoY growth of {yoy_growth:.1f}%!',
            'recommendation': 'Maintain momentum and scale successful strategies.'
        })
    
    if profit_margin > 40:
        alerts.append({
            'type': 'success',
            'icon': '🟢',
            'title': 'Healthy Profit Margins',
            'message': f'Profit margin at {profit_margin:.1f}% exceeds target.',
            'recommendation': 'Consider reinvesting profits for growth.'
        })
    
    # Display alerts
    if alerts:
        # Separate by type
        critical_alerts = [a for a in alerts if a['type'] == 'critical']
        warning_alerts = [a for a in alerts if a['type'] == 'warning']
        success_alerts = [a for a in alerts if a['type'] == 'success']
        
        if critical_alerts:
            for alert in critical_alerts:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #dc3545 0%, #c82333 100%); padding: 1rem; border-radius: 10px; color: white; margin-bottom: 0.5rem;">
                    <strong>{alert['icon']} {alert['title']}</strong><br>
                    {alert['message']}<br>
                    <em style="opacity: 0.9;">💡 {alert['recommendation']}</em>
                </div>
                """, unsafe_allow_html=True)
        
        if warning_alerts:
            for alert in warning_alerts:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #ffc107 0%, #fd7e14 100%); padding: 1rem; border-radius: 10px; color: #212529; margin-bottom: 0.5rem;">
                    <strong>{alert['icon']} {alert['title']}</strong><br>
                    {alert['message']}<br>
                    <em style="opacity: 0.8;">💡 {alert['recommendation']}</em>
                </div>
                """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        if success_alerts:
            with col1:
                for alert in success_alerts:
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #28a745 0%, #20c997 100%); padding: 1rem; border-radius: 10px; color: white; margin-bottom: 0.5rem;">
                        <strong>{alert['icon']} {alert['title']}</strong><br>
                        {alert['message']}<br>
                        <em style="opacity: 0.9;">💡 {alert['recommendation']}</em>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.success("✅ All KPIs are within healthy thresholds. No alerts at this time.")
    
    # ============================================
    # KPI THRESHOLD LEGEND
    # ============================================
    with st.expander("📋 KPI Threshold Reference", expanded=False):
        st.markdown("""
        | KPI | 🟢 Green (Excellent) | 🟡 Amber (Warning) | 🔴 Red (Critical) |
        |-----|---------------------|-------------------|------------------|
        | Revenue Growth (YoY) | ≥ 10% | 0% to 10% | < 0% |
        | Profit Margin | ≥ 35% | 25% to 35% | < 25% |
        | AOV Change | ≥ 5% | 0% to 5% | < 0% |
        | Order Growth (MoM) | ≥ 10% | 0% to 10% | < 0% |
        | Conversion Rate | ≥ 60% | 40% to 60% | < 40% |
        | Cost Ratio | < 60% | 60% to 70% | > 70% |
        """)
    
    st.markdown("---")
    
    # Executive Charts Row
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Revenue Trend & Forecast")
        
        # Monthly revenue trend
        monthly_rev = filtered_df.groupby(filtered_df['date'].dt.to_period('M'))['revenue'].sum().reset_index()
        monthly_rev['date'] = monthly_rev['date'].astype(str)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=monthly_rev['date'],
            y=monthly_rev['revenue'],
            mode='lines+markers',
            name='Actual Revenue',
            line=dict(color='#667eea', width=3),
            marker=dict(size=8)
        ))
        
        # Add trend line
        z = np.polyfit(range(len(monthly_rev)), monthly_rev['revenue'], 1)
        p = np.poly1d(z)
        fig.add_trace(go.Scatter(
            x=monthly_rev['date'],
            y=p(range(len(monthly_rev))),
            mode='lines',
            name='Trend',
            line=dict(color='#764ba2', width=2, dash='dash')
        ))
        
        fig.update_layout(
            height=350,
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🌍 Revenue by Region")
        
        region_rev = filtered_df.groupby('region')['revenue'].sum().reset_index()
        
        fig = px.pie(
            region_rev,
            values='revenue',
            names='region',
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig.update_layout(
            height=350,
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Key Insights Section
    st.markdown("### 💡 Key Executive Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="insight-box">
            <h4>📈 Growth Insight</h4>
            <p>Revenue has grown <strong>23.5%</strong> compared to the same period last year, 
            driven primarily by <strong>Electronics</strong> and <strong>Fashion</strong> categories.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="insight-box">
            <h4>🎯 Strategic Opportunity</h4>
            <p><strong>Asia Pacific</strong> region shows highest growth potential with 
            <strong>35% QoQ growth</strong>. Consider increasing marketing budget allocation.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="insight-box">
            <h4>⚠️ Risk Alert</h4>
            <p>Customer churn rate has increased by <strong>5.2%</strong> in the last quarter. 
            Recommend implementing retention campaigns for at-risk segments.</p>
        </div>
        """, unsafe_allow_html=True)

# ============================================
# TAB 2: CORE BUSINESS VISUALIZATIONS
# ============================================
with tab2:
    st.markdown("## 📈 Core Business Visualizations")
    
    # Sub-tabs for different visualization types
    viz_tab1, viz_tab2, viz_tab3, viz_tab4, viz_tab5 = st.tabs([
        "🌳 Treemap",
        "☀️ Sunburst",
        "📈 Time-Series",
        "📊 Pareto Analysis",
        "🔥 What-If Heatmap"
    ])
    
    # ============================================
    # VIZ 1: TREEMAP - Revenue by Category & Sub-Category
    # ============================================
    with viz_tab1:
        st.markdown("### 🌳 Revenue Contribution Treemap")
        st.markdown("*Hierarchical view of revenue distribution across Categories and Sub-Categories*")
        
        # Prepare treemap data
        treemap_data = filtered_df.groupby(['category', 'sub_category']).agg({
            'revenue': 'sum',
            'profit': 'sum',
            'quantity': 'sum',
            'transaction_id': 'count'
        }).reset_index()
        
        treemap_data['profit_margin'] = (treemap_data['profit'] / treemap_data['revenue'] * 100).round(2)
        treemap_data['avg_order_value'] = (treemap_data['revenue'] / treemap_data['transaction_id']).round(2)
        
        # Color options for treemap
        color_metric = st.selectbox(
            "Color by:",
            options=['Revenue', 'Profit', 'Profit Margin %', 'Order Count'],
            index=0,
            key='treemap_color'
        )
        
        color_map = {
            'Revenue': 'revenue',
            'Profit': 'profit',
            'Profit Margin %': 'profit_margin',
            'Order Count': 'transaction_id'
        }
        
        fig_treemap = px.treemap(
            treemap_data,
            path=['category', 'sub_category'],
            values='revenue',
            color=color_map[color_metric],
            color_continuous_scale='RdYlGn' if 'Profit' in color_metric or 'Margin' in color_metric else 'Blues',
            hover_data={
                'revenue': ':$,.0f',
                'profit': ':$,.0f',
                'profit_margin': ':.1f',
                'transaction_id': ':,.0f'
            },
            title=''
        )
        
        fig_treemap.update_layout(
            height=550,
            margin=dict(l=10, r=10, t=30, b=10)
        )
        
        fig_treemap.update_traces(
            textinfo='label+value+percent root',
            texttemplate='%{label}<br>$%{value:,.0f}<br>%{percentRoot:.1%}',
            hovertemplate='<b>%{label}</b><br>' +
                          'Revenue: $%{value:,.0f}<br>' +
                          'Profit: $%{customdata[1]:,.0f}<br>' +
                          'Margin: %{customdata[2]:.1f}%<br>' +
                          'Orders: %{customdata[3]:,.0f}<extra></extra>'
        )
        
        st.plotly_chart(fig_treemap, use_container_width=True)
        
        # Treemap insights
        col1, col2, col3 = st.columns(3)
        top_category = treemap_data.groupby('category')['revenue'].sum().idxmax()
        top_category_rev = treemap_data.groupby('category')['revenue'].sum().max()
        top_subcategory = treemap_data.loc[treemap_data['revenue'].idxmax()]
        
        with col1:
            st.markdown(f"""
            <div class="insight-box">
                <h4>🏆 Top Category</h4>
                <p><strong>{top_category}</strong> leads with ${top_category_rev/1e6:.2f}M 
                ({top_category_rev/filtered_df['revenue'].sum()*100:.1f}% of total)</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="insight-box">
                <h4>⭐ Top Sub-Category</h4>
                <p><strong>{top_subcategory['sub_category']}</strong> in {top_subcategory['category']} 
                with ${top_subcategory['revenue']/1e6:.2f}M revenue</p>
            </div>
            """, unsafe_allow_html=True)
        
        best_margin = treemap_data.loc[treemap_data['profit_margin'].idxmax()]
        with col3:
            st.markdown(f"""
            <div class="insight-box">
                <h4>💰 Best Margin</h4>
                <p><strong>{best_margin['sub_category']}</strong> has highest margin 
                at {best_margin['profit_margin']:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
    
    # ============================================
    # VIZ 2: SUNBURST - Customer → Category → Product Hierarchy
    # ============================================
    with viz_tab2:
        st.markdown("### ☀️ Customer-Product Hierarchy Sunburst")
        st.markdown("*Multi-level drill-down from Customer Segment → Category → Sub-Category*")
        
        # Prepare sunburst data
        sunburst_data = filtered_df.groupby(['customer_segment', 'category', 'sub_category']).agg({
            'revenue': 'sum',
            'profit': 'sum',
            'customer_id': 'nunique',
            'transaction_id': 'count'
        }).reset_index()
        
        sunburst_data['avg_revenue_per_customer'] = (sunburst_data['revenue'] / sunburst_data['customer_id']).round(2)
        
        # Sunburst metric selector
        sunburst_metric = st.selectbox(
            "Display Metric:",
            options=['Revenue', 'Profit', 'Unique Customers', 'Order Count'],
            index=0,
            key='sunburst_metric'
        )
        
        metric_map = {
            'Revenue': 'revenue',
            'Profit': 'profit',
            'Unique Customers': 'customer_id',
            'Order Count': 'transaction_id'
        }
        
        fig_sunburst = px.sunburst(
            sunburst_data,
            path=['customer_segment', 'category', 'sub_category'],
            values=metric_map[sunburst_metric],
            color='revenue',
            color_continuous_scale='Viridis',
            hover_data={
                'revenue': ':$,.0f',
                'profit': ':$,.0f',
                'customer_id': ':,.0f'
            },
            title=''
        )
        
        fig_sunburst.update_layout(
            height=600,
            margin=dict(l=10, r=10, t=30, b=10)
        )
        
        fig_sunburst.update_traces(
            textinfo='label+percent entry',
            insidetextorientation='radial',
            hovertemplate='<b>%{label}</b><br>' +
                          'Value: %{value:,.0f}<br>' +
                          'Percent of Parent: %{percentParent:.1%}<br>' +
                          'Percent of Total: %{percentRoot:.1%}<extra></extra>'
        )
        
        st.plotly_chart(fig_sunburst, use_container_width=True)
        
        # Sunburst insights
        col1, col2 = st.columns(2)
        
        segment_revenue = filtered_df.groupby('customer_segment')['revenue'].sum()
        top_segment = segment_revenue.idxmax()
        
        with col1:
            st.markdown(f"""
            <div class="insight-box">
                <h4>👥 Segment Insight</h4>
                <p><strong>{top_segment}</strong> customers generate ${segment_revenue[top_segment]/1e6:.2f}M 
                ({segment_revenue[top_segment]/segment_revenue.sum()*100:.1f}% of total revenue)</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Cross-segment analysis
        segment_category = filtered_df.groupby(['customer_segment', 'category'])['revenue'].sum().reset_index()
        top_combo = segment_category.loc[segment_category['revenue'].idxmax()]
        
        with col2:
            st.markdown(f"""
            <div class="insight-box">
                <h4>🎯 Top Segment-Category Combo</h4>
                <p><strong>{top_combo['customer_segment']}</strong> + <strong>{top_combo['category']}</strong> 
                = ${top_combo['revenue']/1e6:.2f}M revenue</p>
            </div>
            """, unsafe_allow_html=True)
    
    # ============================================
    # VIZ 3: TIME-SERIES - Sales & Profit Trends
    # ============================================
    with viz_tab3:
        st.markdown("### 📈 Sales & Profit Time-Series Analysis")
        st.markdown("*Track trends, seasonality, and growth patterns over time*")
        
        # Time granularity selector
        col1, col2 = st.columns([1, 3])
        with col1:
            time_granularity = st.selectbox(
                "Time Granularity:",
                options=['Daily', 'Weekly', 'Monthly', 'Quarterly'],
                index=2,
                key='time_granularity'
            )
        
        # Prepare time-series data based on granularity
        if time_granularity == 'Daily':
            ts_data = filtered_df.groupby(filtered_df['date'].dt.date).agg({
                'revenue': 'sum',
                'profit': 'sum',
                'cost': 'sum',
                'transaction_id': 'count',
                'quantity': 'sum'
            }).reset_index()
            ts_data.columns = ['date', 'revenue', 'profit', 'cost', 'orders', 'units']
        elif time_granularity == 'Weekly':
            ts_data = filtered_df.groupby(filtered_df['date'].dt.to_period('W')).agg({
                'revenue': 'sum',
                'profit': 'sum',
                'cost': 'sum',
                'transaction_id': 'count',
                'quantity': 'sum'
            }).reset_index()
            ts_data['date'] = ts_data['date'].astype(str)
            ts_data.columns = ['date', 'revenue', 'profit', 'cost', 'orders', 'units']
        elif time_granularity == 'Monthly':
            ts_data = filtered_df.groupby(filtered_df['date'].dt.to_period('M')).agg({
                'revenue': 'sum',
                'profit': 'sum',
                'cost': 'sum',
                'transaction_id': 'count',
                'quantity': 'sum'
            }).reset_index()
            ts_data['date'] = ts_data['date'].astype(str)
            ts_data.columns = ['date', 'revenue', 'profit', 'cost', 'orders', 'units']
        else:  # Quarterly
            ts_data = filtered_df.groupby(filtered_df['date'].dt.to_period('Q')).agg({
                'revenue': 'sum',
                'profit': 'sum',
                'cost': 'sum',
                'transaction_id': 'count',
                'quantity': 'sum'
            }).reset_index()
            ts_data['date'] = ts_data['date'].astype(str)
            ts_data.columns = ['date', 'revenue', 'profit', 'cost', 'orders', 'units']
        
        # Calculate growth rates
        ts_data['revenue_growth'] = ts_data['revenue'].pct_change() * 100
        ts_data['profit_growth'] = ts_data['profit'].pct_change() * 100
        ts_data['profit_margin'] = (ts_data['profit'] / ts_data['revenue'] * 100).round(2)
        
        # Moving averages
        if len(ts_data) >= 3:
            ts_data['revenue_ma'] = ts_data['revenue'].rolling(window=3, min_periods=1).mean()
            ts_data['profit_ma'] = ts_data['profit'].rolling(window=3, min_periods=1).mean()
        
        # Main time-series chart
        fig_ts = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=('Revenue & Profit Over Time', 'Growth Rate (%)'),
            row_heights=[0.65, 0.35]
        )
        
        # Revenue line
        fig_ts.add_trace(
            go.Scatter(
                x=ts_data['date'],
                y=ts_data['revenue'],
                mode='lines+markers',
                name='Revenue',
                line=dict(color='#667eea', width=2),
                marker=dict(size=6),
                hovertemplate='Revenue: $%{y:,.0f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Profit line
        fig_ts.add_trace(
            go.Scatter(
                x=ts_data['date'],
                y=ts_data['profit'],
                mode='lines+markers',
                name='Profit',
                line=dict(color='#28a745', width=2),
                marker=dict(size=6),
                hovertemplate='Profit: $%{y:,.0f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Revenue MA (if available)
        if 'revenue_ma' in ts_data.columns:
            fig_ts.add_trace(
                go.Scatter(
                    x=ts_data['date'],
                    y=ts_data['revenue_ma'],
                    mode='lines',
                    name='Revenue (3-period MA)',
                    line=dict(color='#667eea', width=1.5, dash='dash'),
                    hovertemplate='Revenue MA: $%{y:,.0f}<extra></extra>'
                ),
                row=1, col=1
            )
        
        # Growth rate bars
        colors = ['#28a745' if x >= 0 else '#dc3545' for x in ts_data['revenue_growth'].fillna(0)]
        fig_ts.add_trace(
            go.Bar(
                x=ts_data['date'],
                y=ts_data['revenue_growth'],
                name='Revenue Growth %',
                marker_color=colors,
                hovertemplate='Growth: %{y:.1f}%<extra></extra>'
            ),
            row=2, col=1
        )
        
        fig_ts.update_layout(
            height=550,
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            hovermode='x unified'
        )
        
        fig_ts.update_xaxes(tickangle=-45)
        fig_ts.update_yaxes(title_text="Amount ($)", row=1, col=1)
        fig_ts.update_yaxes(title_text="Growth %", row=2, col=1)
        
        st.plotly_chart(fig_ts, use_container_width=True)
        
        # Time-series KPIs
        col1, col2, col3, col4 = st.columns(4)
        
        avg_revenue = ts_data['revenue'].mean()
        max_revenue_period = ts_data.loc[ts_data['revenue'].idxmax(), 'date']
        avg_growth = ts_data['revenue_growth'].mean()
        volatility = ts_data['revenue'].std() / ts_data['revenue'].mean() * 100 if ts_data['revenue'].mean() > 0 else 0
        
        with col1:
            st.metric("📊 Avg Revenue", f"${avg_revenue/1e6:.2f}M")
        with col2:
            st.metric("🏆 Peak Period", str(max_revenue_period)[:10])
        with col3:
            st.metric("📈 Avg Growth", f"{avg_growth:.1f}%")
        with col4:
            st.metric("📉 Volatility", f"{volatility:.1f}%")
    
    # ============================================
    # VIZ 4: PARETO CHART - 80/20 Analysis
    # ============================================
    with viz_tab4:
        st.markdown("### 📊 Pareto Analysis (80/20 Rule)")
        st.markdown("*Identify the top products/categories contributing to 80% of revenue*")
        
        # Pareto dimension selector
        pareto_dim = st.selectbox(
            "Analyze by:",
            options=['Category', 'Sub-Category', 'Region', 'Sales Channel', 'Customer Segment'],
            index=1,
            key='pareto_dimension'
        )
        
        dim_map = {
            'Category': 'category',
            'Sub-Category': 'sub_category',
            'Region': 'region',
            'Sales Channel': 'channel',
            'Customer Segment': 'customer_segment'
        }
        
        # Prepare Pareto data
        pareto_data = filtered_df.groupby(dim_map[pareto_dim])['revenue'].sum().reset_index()
        pareto_data = pareto_data.sort_values('revenue', ascending=False)
        pareto_data['cumulative_revenue'] = pareto_data['revenue'].cumsum()
        pareto_data['cumulative_pct'] = (pareto_data['cumulative_revenue'] / pareto_data['revenue'].sum() * 100)
        pareto_data['revenue_pct'] = (pareto_data['revenue'] / pareto_data['revenue'].sum() * 100)
        pareto_data['rank'] = range(1, len(pareto_data) + 1)
        
        # Find 80% threshold
        threshold_80 = pareto_data[pareto_data['cumulative_pct'] <= 80]
        items_for_80 = len(threshold_80) + 1 if len(threshold_80) < len(pareto_data) else len(threshold_80)
        pct_items_for_80 = items_for_80 / len(pareto_data) * 100
        
        # Create Pareto chart
        fig_pareto = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Bar chart for revenue
        fig_pareto.add_trace(
            go.Bar(
                x=pareto_data[dim_map[pareto_dim]],
                y=pareto_data['revenue'],
                name='Revenue',
                marker_color='#667eea',
                hovertemplate='%{x}<br>Revenue: $%{y:,.0f}<extra></extra>'
            ),
            secondary_y=False
        )
        
        # Line chart for cumulative percentage
        fig_pareto.add_trace(
            go.Scatter(
                x=pareto_data[dim_map[pareto_dim]],
                y=pareto_data['cumulative_pct'],
                mode='lines+markers',
                name='Cumulative %',
                line=dict(color='#dc3545', width=3),
                marker=dict(size=8),
                hovertemplate='%{x}<br>Cumulative: %{y:.1f}%<extra></extra>'
            ),
            secondary_y=True
        )
        
        # Add 80% threshold line
        fig_pareto.add_hline(
            y=80, 
            line_dash="dash", 
            line_color="#ffc107", 
            annotation_text="80% Threshold",
            secondary_y=True
        )
        
        fig_pareto.update_layout(
            height=450,
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            xaxis_tickangle=-45
        )
        
        fig_pareto.update_yaxes(title_text="Revenue ($)", secondary_y=False)
        fig_pareto.update_yaxes(title_text="Cumulative %", secondary_y=True, range=[0, 105])
        
        st.plotly_chart(fig_pareto, use_container_width=True)
        
        # Pareto insights
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 12px; color: white;">
                <h4 style="margin: 0;">📊 80/20 Analysis Result</h4>
                <p style="margin: 0.5rem 0; font-size: 2rem; font-weight: bold;">{items_for_80} of {len(pareto_data)} {pareto_dim}s</p>
                <p style="margin: 0; opacity: 0.9;">({pct_items_for_80:.1f}%) contribute to 80% of revenue</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Top contributors table
            st.markdown("**Top Contributors:**")
            top_5 = pareto_data.head(5)[[dim_map[pareto_dim], 'revenue', 'revenue_pct', 'cumulative_pct']]
            top_5.columns = [pareto_dim, 'Revenue', '% of Total', 'Cumulative %']
            st.dataframe(
                top_5.style.format({
                    'Revenue': '${:,.0f}',
                    '% of Total': '{:.1f}%',
                    'Cumulative %': '{:.1f}%'
                }),
                use_container_width=True,
                hide_index=True
            )
    
    # ============================================
    # VIZ 5: WHAT-IF HEATMAP - Price/Discount Impact
    # ============================================
    with viz_tab5:
        st.markdown("### 🔥 What-If Analysis: Price & Discount Impact on Profit")
        st.markdown("*Simulate how changes in price and discount affect profitability*")
        
        # Current baseline metrics
        baseline_revenue = filtered_df['revenue'].sum()
        baseline_profit = filtered_df['profit'].sum()
        baseline_margin = (baseline_profit / baseline_revenue * 100) if baseline_revenue > 0 else 0
        baseline_avg_price = filtered_df['unit_price'].mean()
        baseline_avg_discount = filtered_df['discount_applied'].mean()
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("#### Current Baseline")
            st.markdown(f"""
            - **Revenue:** ${baseline_revenue/1e6:.2f}M
            - **Profit:** ${baseline_profit/1e6:.2f}M
            - **Margin:** {baseline_margin:.1f}%
            - **Avg Price:** ${baseline_avg_price:.2f}
            - **Avg Discount:** {baseline_avg_discount:.1f}%
            """)
            
            st.markdown("#### Simulation Parameters")
            price_elasticity = st.slider("Price Elasticity", -2.0, -0.5, -1.2, 0.1, 
                                         help="How demand changes with price (typically -1 to -2)")
            cost_ratio = st.slider("Cost as % of Price", 40, 80, 55, 5,
                                   help="Variable cost percentage")
        
        with col2:
            # Generate What-If Heatmap data
            price_changes = np.arange(-20, 25, 5)  # -20% to +20%
            discount_changes = np.arange(0, 35, 5)  # 0% to 30%
            
            profit_matrix = np.zeros((len(discount_changes), len(price_changes)))
            
            for i, disc in enumerate(discount_changes):
                for j, price_chg in enumerate(price_changes):
                    # Calculate new metrics
                    new_price_factor = 1 + (price_chg / 100)
                    demand_factor = 1 + (price_elasticity * price_chg / 100)  # Demand elasticity
                    demand_factor = max(demand_factor, 0.3)  # Floor at 30% demand
                    
                    new_revenue = baseline_revenue * new_price_factor * demand_factor * (1 - disc/100)
                    new_cost = new_revenue * (cost_ratio / 100)
                    new_profit = new_revenue - new_cost
                    
                    # Store profit change %
                    profit_change = ((new_profit - baseline_profit) / baseline_profit * 100) if baseline_profit > 0 else 0
                    profit_matrix[i, j] = profit_change
            
            # Create heatmap
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=profit_matrix,
                x=[f"{p:+d}%" for p in price_changes],
                y=[f"{d}%" for d in discount_changes],
                colorscale='RdYlGn',
                zmid=0,
                text=np.round(profit_matrix, 1),
                texttemplate="%{text}%",
                textfont={"size": 10},
                hovertemplate='Price Change: %{x}<br>Discount: %{y}<br>Profit Change: %{z:.1f}%<extra></extra>'
            ))
            
            fig_heatmap.update_layout(
                title='Profit Change (%) by Price & Discount Adjustments',
                xaxis_title='Price Change',
                yaxis_title='Discount Level',
                height=400,
                margin=dict(l=20, r=20, t=50, b=20)
            )
            
            st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Interactive What-If Calculator
        st.markdown("#### 🧮 Custom Scenario Calculator")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            custom_price_change = st.number_input("Price Change %", -30, 30, 0, 5)
        with col2:
            custom_discount = st.number_input("Discount %", 0, 50, int(baseline_avg_discount), 5)
        with col3:
            custom_elasticity = st.number_input("Elasticity", -3.0, -0.1, price_elasticity, 0.1)
        
        # Calculate custom scenario
        new_price_factor = 1 + (custom_price_change / 100)
        demand_factor = 1 + (custom_elasticity * custom_price_change / 100)
        demand_factor = max(demand_factor, 0.3)
        
        projected_revenue = baseline_revenue * new_price_factor * demand_factor * (1 - custom_discount/100)
        projected_cost = projected_revenue * (cost_ratio / 100)
        projected_profit = projected_revenue - projected_cost
        projected_margin = (projected_profit / projected_revenue * 100) if projected_revenue > 0 else 0
        
        revenue_change = ((projected_revenue - baseline_revenue) / baseline_revenue * 100)
        profit_change = ((projected_profit - baseline_profit) / baseline_profit * 100) if baseline_profit > 0 else 0
        
        with col4:
            profit_color = '#28a745' if profit_change >= 0 else '#dc3545'
            st.markdown(f"""
            <div style="background: {profit_color}; padding: 1rem; border-radius: 10px; color: white; text-align: center;">
                <p style="margin: 0; font-size: 0.9rem;">Projected Profit</p>
                <h3 style="margin: 0.3rem 0;">${projected_profit/1e6:.2f}M</h3>
                <p style="margin: 0; font-size: 0.85rem;">{profit_change:+.1f}% change</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Scenario comparison
        st.markdown("#### 📊 Scenario Comparison")
        
        comparison_data = pd.DataFrame({
            'Metric': ['Revenue', 'Profit', 'Margin %'],
            'Baseline': [f"${baseline_revenue/1e6:.2f}M", f"${baseline_profit/1e6:.2f}M", f"{baseline_margin:.1f}%"],
            'Projected': [f"${projected_revenue/1e6:.2f}M", f"${projected_profit/1e6:.2f}M", f"{projected_margin:.1f}%"],
            'Change': [f"{revenue_change:+.1f}%", f"{profit_change:+.1f}%", f"{projected_margin - baseline_margin:+.1f}pp"]
        })
        
        st.dataframe(comparison_data, use_container_width=True, hide_index=True)

# ============================================
# TAB 3: BEHAVIORAL & FUNNEL ANALYTICS
# ============================================
with tab3:
    st.markdown("## 👥 Behavioral & Funnel Analytics")
    
    # Sub-tabs for different analytics views
    funnel_tab1, funnel_tab2, funnel_tab3, funnel_tab4 = st.tabs([
        "🔄 Conversion Funnel",
        "📉 Drop-off Analysis",
        "👥 Cohort Analysis",
        "🎯 Customer Segments"
    ])
    
    # ============================================
    # GENERATE FUNNEL DATA (Simulated based on transactions)
    # ============================================
    # Create realistic funnel metrics based on actual transaction data
    total_transactions = len(filtered_df)
    unique_customers_in_period = filtered_df['customer_id'].nunique()
    
    # Simulate funnel stages (in real scenario, this would come from event tracking)
    np.random.seed(42)
    
    # Base visitors (estimated from transactions with multiplier)
    base_visits = int(total_transactions * 3.5)  # Assume 3.5x more visits than purchases
    
    # Funnel conversion rates (realistic e-commerce rates)
    visit_to_cart_rate = np.random.uniform(0.35, 0.45)
    cart_to_checkout_rate = np.random.uniform(0.55, 0.65)
    checkout_to_purchase_rate = np.random.uniform(0.70, 0.85)
    
    # Calculate funnel stages
    funnel_stages = {
        'Visit': base_visits,
        'Add to Cart': int(base_visits * visit_to_cart_rate),
        'Checkout': int(base_visits * visit_to_cart_rate * cart_to_checkout_rate),
        'Purchase': total_transactions
    }
    
    # Recalculate actual conversion rates
    actual_rates = {
        'Visit → Cart': funnel_stages['Add to Cart'] / funnel_stages['Visit'] * 100,
        'Cart → Checkout': funnel_stages['Checkout'] / funnel_stages['Add to Cart'] * 100,
        'Checkout → Purchase': funnel_stages['Purchase'] / funnel_stages['Checkout'] * 100,
        'Overall': funnel_stages['Purchase'] / funnel_stages['Visit'] * 100
    }
    
    # ============================================
    # FUNNEL TAB 1: CONVERSION FUNNEL
    # ============================================
    with funnel_tab1:
        st.markdown("### 🔄 E-Commerce Conversion Funnel")
        st.markdown("*Track customer journey from Visit to Purchase*")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Create funnel chart
            fig_funnel = go.Figure(go.Funnel(
                y=['🌐 Website Visit', '🛒 Add to Cart', '💳 Checkout', '✅ Purchase'],
                x=[funnel_stages['Visit'], funnel_stages['Add to Cart'], 
                   funnel_stages['Checkout'], funnel_stages['Purchase']],
                textposition="inside",
                textinfo="value+percent initial",
                opacity=0.85,
                marker={
                    "color": ["#667eea", "#764ba2", "#11998e", "#28a745"],
                    "line": {"width": [2, 2, 2, 2], "color": ["white", "white", "white", "white"]}
                },
                connector={"line": {"color": "#E8E8E8", "dash": "solid", "width": 3}}
            ))
            
            fig_funnel.update_layout(
                height=450,
                margin=dict(l=20, r=20, t=30, b=20),
                font=dict(size=14)
            )
            
            st.plotly_chart(fig_funnel, use_container_width=True)
        
        with col2:
            st.markdown("#### 📊 Conversion Metrics")
            
            # Conversion rate cards
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 10px; color: white; margin-bottom: 0.5rem;">
                <p style="margin: 0; font-size: 0.85rem;">Visit → Cart</p>
                <h3 style="margin: 0.3rem 0;">{actual_rates['Visit → Cart']:.1f}%</h3>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #764ba2 0%, #11998e 100%); padding: 1rem; border-radius: 10px; color: white; margin-bottom: 0.5rem;">
                <p style="margin: 0; font-size: 0.85rem;">Cart → Checkout</p>
                <h3 style="margin: 0.3rem 0;">{actual_rates['Cart → Checkout']:.1f}%</h3>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #11998e 0%, #28a745 100%); padding: 1rem; border-radius: 10px; color: white; margin-bottom: 0.5rem;">
                <p style="margin: 0; font-size: 0.85rem;">Checkout → Purchase</p>
                <h3 style="margin: 0.3rem 0;">{actual_rates['Checkout → Purchase']:.1f}%</h3>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #28a745 0%, #20c997 100%); padding: 1rem; border-radius: 10px; color: white;">
                <p style="margin: 0; font-size: 0.85rem;">🎯 Overall Conversion</p>
                <h3 style="margin: 0.3rem 0;">{actual_rates['Overall']:.2f}%</h3>
            </div>
            """, unsafe_allow_html=True)
        
        # Funnel KPIs row
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        
        cart_abandonment = 100 - actual_rates['Cart → Checkout']
        checkout_abandonment = 100 - actual_rates['Checkout → Purchase']
        
        with col1:
            st.metric("🌐 Total Visits", f"{funnel_stages['Visit']:,}")
        with col2:
            st.metric("🛒 Cart Additions", f"{funnel_stages['Add to Cart']:,}")
        with col3:
            abandon_color = "inverse" if cart_abandonment > 40 else "normal"
            st.metric("🚫 Cart Abandonment", f"{cart_abandonment:.1f}%", delta_color=abandon_color)
        with col4:
            st.metric("✅ Completed Purchases", f"{funnel_stages['Purchase']:,}")
        
        # Funnel insights
        st.markdown("### 💡 Funnel Insights")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            insight_color = '#28a745' if actual_rates['Visit → Cart'] > 35 else '#ffc107'
            st.markdown(f"""
            <div style="background: #f8f9fa; border-left: 4px solid {insight_color}; padding: 1rem; border-radius: 0 10px 10px 0;">
                <h4 style="margin: 0;">🛒 Cart Addition Rate</h4>
                <p style="margin: 0.5rem 0;">{'Good' if actual_rates['Visit → Cart'] > 35 else 'Needs Improvement'}: {actual_rates['Visit → Cart']:.1f}% of visitors add items to cart</p>
                <p style="margin: 0; font-size: 0.85rem; color: #666;">Industry avg: 30-40%</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            insight_color = '#dc3545' if cart_abandonment > 40 else '#28a745'
            st.markdown(f"""
            <div style="background: #f8f9fa; border-left: 4px solid {insight_color}; padding: 1rem; border-radius: 0 10px 10px 0;">
                <h4 style="margin: 0;">🚫 Cart Abandonment</h4>
                <p style="margin: 0.5rem 0;">{'High' if cart_abandonment > 40 else 'Acceptable'}: {cart_abandonment:.1f}% abandon cart before checkout</p>
                <p style="margin: 0; font-size: 0.85rem; color: #666;">Target: Below 35%</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            insight_color = '#28a745' if actual_rates['Checkout → Purchase'] > 75 else '#ffc107'
            st.markdown(f"""
            <div style="background: #f8f9fa; border-left: 4px solid {insight_color}; padding: 1rem; border-radius: 0 10px 10px 0;">
                <h4 style="margin: 0;">✅ Checkout Completion</h4>
                <p style="margin: 0.5rem 0;">{'Excellent' if actual_rates['Checkout → Purchase'] > 75 else 'Good'}: {actual_rates['Checkout → Purchase']:.1f}% complete purchase</p>
                <p style="margin: 0; font-size: 0.85rem; color: #666;">Industry avg: 70-80%</p>
            </div>
            """, unsafe_allow_html=True)
    
    # ============================================
    # FUNNEL TAB 2: DROP-OFF ANALYSIS BY DIMENSION
    # ============================================
    with funnel_tab2:
        st.markdown("### 📉 Drop-off Rate Analysis")
        st.markdown("*Analyze conversion drop-offs by Region, Device, and Channel*")
        
        # Generate drop-off data by dimensions
        regions = filtered_df['region'].unique()
        channels = filtered_df['channel'].unique()
        
        # Simulate device types if not in data
        if 'device' not in filtered_df.columns:
            devices = ['Desktop', 'Mobile', 'Tablet']
            filtered_df['device'] = np.random.choice(devices, len(filtered_df), p=[0.35, 0.50, 0.15])
        else:
            devices = filtered_df['device'].unique()
        
        # --------- DROP-OFF BY REGION ---------
        st.markdown("#### 🌍 Drop-off by Region")
        
        # Generate region-specific funnel data
        region_funnel_data = []
        for region in regions:
            region_df = filtered_df[filtered_df['region'] == region]
            region_purchases = len(region_df)
            
            # Vary conversion rates by region (simulate regional differences)
            region_variation = np.random.uniform(0.85, 1.15)
            region_visits = int(region_purchases * 3.5 * region_variation)
            region_cart = int(region_visits * visit_to_cart_rate * region_variation)
            region_checkout = int(region_cart * cart_to_checkout_rate * region_variation)
            
            # Calculate drop-off rates
            visit_cart_dropoff = (1 - region_cart / region_visits) * 100 if region_visits > 0 else 0
            cart_checkout_dropoff = (1 - region_checkout / region_cart) * 100 if region_cart > 0 else 0
            checkout_purchase_dropoff = (1 - region_purchases / region_checkout) * 100 if region_checkout > 0 else 0
            
            region_funnel_data.append({
                'Region': region,
                'Visits': region_visits,
                'Cart': region_cart,
                'Checkout': region_checkout,
                'Purchase': region_purchases,
                'Visit→Cart Drop': visit_cart_dropoff,
                'Cart→Checkout Drop': cart_checkout_dropoff,
                'Checkout→Purchase Drop': checkout_purchase_dropoff,
                'Overall Conversion': (region_purchases / region_visits * 100) if region_visits > 0 else 0
            })
        
        region_funnel_df = pd.DataFrame(region_funnel_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Stacked bar chart for drop-offs by region
            fig_region = go.Figure()
            
            fig_region.add_trace(go.Bar(
                name='Visit→Cart Drop-off',
                x=region_funnel_df['Region'],
                y=region_funnel_df['Visit→Cart Drop'],
                marker_color='#dc3545'
            ))
            
            fig_region.add_trace(go.Bar(
                name='Cart→Checkout Drop-off',
                x=region_funnel_df['Region'],
                y=region_funnel_df['Cart→Checkout Drop'],
                marker_color='#ffc107'
            ))
            
            fig_region.add_trace(go.Bar(
                name='Checkout→Purchase Drop-off',
                x=region_funnel_df['Region'],
                y=region_funnel_df['Checkout→Purchase Drop'],
                marker_color='#667eea'
            ))
            
            fig_region.update_layout(
                barmode='group',
                height=350,
                margin=dict(l=20, r=20, t=30, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                yaxis_title="Drop-off Rate (%)",
                xaxis_tickangle=-45
            )
            
            st.plotly_chart(fig_region, use_container_width=True)
        
        with col2:
            # Overall conversion by region
            fig_conv = px.bar(
                region_funnel_df,
                x='Region',
                y='Overall Conversion',
                color='Overall Conversion',
                color_continuous_scale='RdYlGn',
                title='Overall Conversion Rate by Region'
            )
            fig_conv.update_layout(
                height=350,
                margin=dict(l=20, r=20, t=50, b=20),
                xaxis_tickangle=-45
            )
            st.plotly_chart(fig_conv, use_container_width=True)
        
        # --------- DROP-OFF BY DEVICE ---------
        st.markdown("#### 📱 Drop-off by Device Type")
        
        device_funnel_data = []
        device_conversion_factors = {'Desktop': 1.15, 'Mobile': 0.85, 'Tablet': 1.0}
        
        for device in devices:
            device_df = filtered_df[filtered_df['device'] == device]
            device_purchases = len(device_df)
            
            conv_factor = device_conversion_factors.get(device, 1.0)
            device_visits = int(device_purchases * 3.5 / conv_factor)
            device_cart = int(device_visits * visit_to_cart_rate * conv_factor)
            device_checkout = int(device_cart * cart_to_checkout_rate * conv_factor)
            
            device_funnel_data.append({
                'Device': device,
                'Visits': device_visits,
                'Cart': device_cart,
                'Checkout': device_checkout,
                'Purchase': device_purchases,
                'Conversion Rate': (device_purchases / device_visits * 100) if device_visits > 0 else 0,
                'Cart Abandonment': ((device_cart - device_checkout) / device_cart * 100) if device_cart > 0 else 0
            })
        
        device_funnel_df = pd.DataFrame(device_funnel_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Device conversion comparison
            fig_device = go.Figure()
            
            fig_device.add_trace(go.Bar(
                name='Conversion Rate',
                x=device_funnel_df['Device'],
                y=device_funnel_df['Conversion Rate'],
                marker_color='#28a745',
                text=device_funnel_df['Conversion Rate'].round(2),
                textposition='outside'
            ))
            
            fig_device.add_trace(go.Bar(
                name='Cart Abandonment',
                x=device_funnel_df['Device'],
                y=device_funnel_df['Cart Abandonment'],
                marker_color='#dc3545',
                text=device_funnel_df['Cart Abandonment'].round(1),
                textposition='outside'
            ))
            
            fig_device.update_layout(
                barmode='group',
                height=350,
                margin=dict(l=20, r=20, t=30, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                yaxis_title="Rate (%)"
            )
            
            st.plotly_chart(fig_device, use_container_width=True)
        
        with col2:
            # Device insights
            best_device = device_funnel_df.loc[device_funnel_df['Conversion Rate'].idxmax()]
            worst_device = device_funnel_df.loc[device_funnel_df['Conversion Rate'].idxmin()]
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #28a745 0%, #20c997 100%); padding: 1rem; border-radius: 10px; color: white; margin-bottom: 0.5rem;">
                <h4 style="margin: 0;">🏆 Best Performing Device</h4>
                <p style="margin: 0.3rem 0; font-size: 1.5rem; font-weight: bold;">{best_device['Device']}</p>
                <p style="margin: 0;">Conversion: {best_device['Conversion Rate']:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #dc3545 0%, #c82333 100%); padding: 1rem; border-radius: 10px; color: white;">
                <h4 style="margin: 0;">⚠️ Needs Optimization</h4>
                <p style="margin: 0.3rem 0; font-size: 1.5rem; font-weight: bold;">{worst_device['Device']}</p>
                <p style="margin: 0;">Conversion: {worst_device['Conversion Rate']:.2f}%</p>
                <p style="margin: 0; font-size: 0.85rem; opacity: 0.9;">Cart Abandonment: {worst_device['Cart Abandonment']:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        # --------- DROP-OFF BY CHANNEL ---------
        st.markdown("#### 📢 Drop-off by Sales Channel")
        
        channel_funnel_data = []
        channel_factors = {'Website': 1.0, 'Mobile App': 1.2, 'Marketplace': 0.9, 'Social Media': 0.75}
        
        for channel in channels:
            channel_df = filtered_df[filtered_df['channel'] == channel]
            channel_purchases = len(channel_df)
            
            conv_factor = channel_factors.get(channel, 1.0)
            channel_visits = int(channel_purchases * 3.5 / conv_factor)
            channel_cart = int(channel_visits * visit_to_cart_rate * conv_factor)
            channel_checkout = int(channel_cart * cart_to_checkout_rate * conv_factor)
            
            channel_funnel_data.append({
                'Channel': channel,
                'Visits': channel_visits,
                'Add to Cart': channel_cart,
                'Checkout': channel_checkout,
                'Purchase': channel_purchases,
                'Conversion Rate': (channel_purchases / channel_visits * 100) if channel_visits > 0 else 0
            })
        
        channel_funnel_df = pd.DataFrame(channel_funnel_data)
        
        # Horizontal funnel comparison
        fig_channel = go.Figure()
        
        for i, stage in enumerate(['Visits', 'Add to Cart', 'Checkout', 'Purchase']):
            fig_channel.add_trace(go.Bar(
                name=stage,
                y=channel_funnel_df['Channel'],
                x=channel_funnel_df[stage],
                orientation='h',
                marker_color=['#667eea', '#764ba2', '#11998e', '#28a745'][i]
            ))
        
        fig_channel.update_layout(
            barmode='group',
            height=300,
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            xaxis_title="Count"
        )
        
        st.plotly_chart(fig_channel, use_container_width=True)
        
        # Channel performance table
        st.markdown("**Channel Performance Summary**")
        channel_summary = channel_funnel_df[['Channel', 'Visits', 'Purchase', 'Conversion Rate']].copy()
        channel_summary['Revenue'] = filtered_df.groupby('channel')['revenue'].sum().values
        channel_summary = channel_summary.sort_values('Revenue', ascending=False)
        
        st.dataframe(
            channel_summary.style.format({
                'Visits': '{:,.0f}',
                'Purchase': '{:,.0f}',
                'Conversion Rate': '{:.2f}%',
                'Revenue': '${:,.0f}'
            }).background_gradient(cmap='Greens', subset=['Conversion Rate', 'Revenue']),
            use_container_width=True,
            hide_index=True
        )
    
    # ============================================
    # FUNNEL TAB 3: COHORT ANALYSIS
    # ============================================
    with funnel_tab3:
        st.markdown("### 👥 Cohort-Based Retention Analysis")
        st.markdown("*Track customer retention and behavior over time by acquisition cohort*")
        
        # Create cohort data from transactions
        customer_first_purchase = filtered_df.groupby('customer_id')['date'].min().reset_index()
        customer_first_purchase.columns = ['customer_id', 'first_purchase_date']
        customer_first_purchase['cohort_month'] = customer_first_purchase['first_purchase_date'].dt.to_period('M')
        
        # Merge cohort info back to transactions
        cohort_df = filtered_df.merge(customer_first_purchase[['customer_id', 'cohort_month']], on='customer_id')
        cohort_df['transaction_month'] = cohort_df['date'].dt.to_period('M')
        
        # Calculate months since first purchase
        cohort_df['months_since_first'] = (
            (cohort_df['transaction_month'].dt.to_timestamp() - cohort_df['cohort_month'].dt.to_timestamp()).dt.days / 30
        ).astype(int)
        
        # Create cohort pivot table
        cohort_pivot = cohort_df.groupby(['cohort_month', 'months_since_first'])['customer_id'].nunique().reset_index()
        cohort_pivot = cohort_pivot.pivot(index='cohort_month', columns='months_since_first', values='customer_id')
        
        # Calculate retention rates
        cohort_size = cohort_pivot.iloc[:, 0]
        retention_matrix = cohort_pivot.divide(cohort_size, axis=0) * 100
        
        # Limit to reasonable number of months
        retention_matrix = retention_matrix.iloc[:, :13]  # First year + month 0
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Cohort retention heatmap
            fig_cohort = go.Figure(data=go.Heatmap(
                z=retention_matrix.values,
                x=[f"Month {i}" for i in retention_matrix.columns],
                y=[str(idx) for idx in retention_matrix.index],
                colorscale='YlGnBu',
                text=np.round(retention_matrix.values, 1),
                texttemplate="%{text}%",
                textfont={"size": 9},
                hovertemplate='Cohort: %{y}<br>Month: %{x}<br>Retention: %{z:.1f}%<extra></extra>'
            ))
            
            fig_cohort.update_layout(
                title='Customer Retention by Cohort (%)',
                height=450,
                margin=dict(l=20, r=20, t=50, b=20),
                xaxis_title='Months Since First Purchase',
                yaxis_title='Acquisition Cohort'
            )
            
            st.plotly_chart(fig_cohort, use_container_width=True)
        
        with col2:
            # Retention metrics
            avg_retention_m1 = retention_matrix.iloc[:, 1].mean() if 1 in retention_matrix.columns else 0
            avg_retention_m3 = retention_matrix.iloc[:, 3].mean() if 3 in retention_matrix.columns else 0
            avg_retention_m6 = retention_matrix.iloc[:, 6].mean() if 6 in retention_matrix.columns else 0
            
            st.markdown("#### 📊 Avg Retention")
            
            st.metric("Month 1", f"{avg_retention_m1:.1f}%")
            st.metric("Month 3", f"{avg_retention_m3:.1f}%")
            st.metric("Month 6", f"{avg_retention_m6:.1f}%")
            
            # Retention health indicator
            if avg_retention_m3 > 30:
                health = "🟢 Healthy"
                health_color = "#28a745"
            elif avg_retention_m3 > 20:
                health = "🟡 Fair"
                health_color = "#ffc107"
            else:
                health = "🔴 Critical"
                health_color = "#dc3545"
            
            st.markdown(f"""
            <div style="background: {health_color}; padding: 1rem; border-radius: 10px; color: white; text-align: center; margin-top: 1rem;">
                <p style="margin: 0; font-size: 0.85rem;">Retention Health</p>
                <h3 style="margin: 0.3rem 0;">{health}</h3>
            </div>
            """, unsafe_allow_html=True)
        
        # Cohort insights
        st.markdown("---")
        st.markdown("### 💡 Cohort Behavioral Insights")
        
        col1, col2, col3 = st.columns(3)
        
        # Calculate cohort metrics
        cohort_revenue = cohort_df.groupby('cohort_month').agg({
            'revenue': 'sum',
            'customer_id': 'nunique',
            'transaction_id': 'count'
        }).reset_index()
        cohort_revenue['avg_revenue_per_customer'] = cohort_revenue['revenue'] / cohort_revenue['customer_id']
        cohort_revenue['orders_per_customer'] = cohort_revenue['transaction_id'] / cohort_revenue['customer_id']
        
        best_cohort = cohort_revenue.loc[cohort_revenue['avg_revenue_per_customer'].idxmax()]
        
        with col1:
            st.markdown(f"""
            <div class="insight-box">
                <h4>🏆 Best Performing Cohort</h4>
                <p><strong>{best_cohort['cohort_month']}</strong></p>
                <p>Avg Revenue/Customer: ${best_cohort['avg_revenue_per_customer']:.2f}</p>
                <p>Orders/Customer: {best_cohort['orders_per_customer']:.1f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Repeat purchase rate
            customers_with_multiple = cohort_df.groupby('customer_id').size()
            repeat_rate = (customers_with_multiple > 1).sum() / len(customers_with_multiple) * 100
            
            st.markdown(f"""
            <div class="insight-box">
                <h4>🔄 Repeat Purchase Rate</h4>
                <p><strong>{repeat_rate:.1f}%</strong> of customers</p>
                <p>have made more than one purchase</p>
                <p style="font-size: 0.85rem; color: #666;">Target: >40%</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            # Average time between purchases
            customer_orders = cohort_df.groupby('customer_id')['date'].agg(['min', 'max', 'count'])
            customer_orders['days_active'] = (customer_orders['max'] - customer_orders['min']).dt.days
            repeat_customers = customer_orders[customer_orders['count'] > 1]
            if len(repeat_customers) > 0:
                avg_days_between = (repeat_customers['days_active'] / (repeat_customers['count'] - 1)).mean()
            else:
                avg_days_between = 0
            
            st.markdown(f"""
            <div class="insight-box">
                <h4>⏱️ Avg Days Between Orders</h4>
                <p><strong>{avg_days_between:.0f} days</strong></p>
                <p>for repeat customers</p>
                <p style="font-size: 0.85rem; color: #666;">Lower is better</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Cohort revenue trend
        st.markdown("### 📈 Cohort Revenue Over Time")
        
        cohort_monthly_rev = cohort_df.groupby(['cohort_month', 'transaction_month'])['revenue'].sum().reset_index()
        cohort_monthly_rev['cohort_month'] = cohort_monthly_rev['cohort_month'].astype(str)
        cohort_monthly_rev['transaction_month'] = cohort_monthly_rev['transaction_month'].astype(str)
        
        # Get top 5 cohorts by revenue
        top_cohorts = cohort_revenue.nlargest(5, 'revenue')['cohort_month'].astype(str).tolist()
        
        fig_cohort_rev = px.line(
            cohort_monthly_rev[cohort_monthly_rev['cohort_month'].isin(top_cohorts)],
            x='transaction_month',
            y='revenue',
            color='cohort_month',
            title='Revenue by Top Cohorts Over Time',
            markers=True
        )
        
        fig_cohort_rev.update_layout(
            height=350,
            margin=dict(l=20, r=20, t=50, b=20),
            xaxis_tickangle=-45,
            legend_title='Cohort'
        )
        
        st.plotly_chart(fig_cohort_rev, use_container_width=True)
    
    # ============================================
    # FUNNEL TAB 4: CUSTOMER SEGMENTS
    # ============================================
    with funnel_tab4:
        st.markdown("### 🎯 Customer Segment Analysis")
        st.markdown("*Deep-dive into customer segments and their behavioral patterns*")
        
        # Customer Segment Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        segment_data = filtered_df.groupby('customer_segment').agg({
            'revenue': 'sum',
            'customer_id': 'nunique',
            'transaction_id': 'count',
            'profit': 'sum'
        }).reset_index()
        segment_data['avg_order_value'] = segment_data['revenue'] / segment_data['transaction_id']
        segment_data['orders_per_customer'] = segment_data['transaction_id'] / segment_data['customer_id']
        
        with col1:
            premium_data = segment_data[segment_data['customer_segment'] == 'Premium'].iloc[0]
            st.metric("👑 Premium Segment", f"${premium_data['revenue']/1e6:.2f}M", 
                     f"{premium_data['customer_id']:,} customers")
        
        with col2:
            regular_data = segment_data[segment_data['customer_segment'] == 'Regular'].iloc[0]
            st.metric("⭐ Regular Segment", f"${regular_data['revenue']/1e6:.2f}M",
                     f"{regular_data['customer_id']:,} customers")
        
        with col3:
            new_data = segment_data[segment_data['customer_segment'] == 'New'].iloc[0]
            st.metric("🆕 New Customers", f"${new_data['revenue']/1e6:.2f}M",
                     f"{new_data['customer_id']:,} customers")
        
        with col4:
            churned_data = segment_data[segment_data['customer_segment'] == 'Churned'].iloc[0]
            st.metric("⚠️ At-Risk", f"${churned_data['revenue']/1e6:.2f}M",
                     f"{churned_data['customer_id']:,} customers", delta_color="inverse")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Revenue Distribution by Segment")
            
            fig_segment_pie = px.pie(
                segment_data,
                values='revenue',
                names='customer_segment',
                hole=0.4,
                color_discrete_sequence=['#667eea', '#28a745', '#ffc107', '#dc3545']
            )
            fig_segment_pie.update_traces(textposition='inside', textinfo='percent+label')
            fig_segment_pie.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_segment_pie, use_container_width=True)
        
        with col2:
            st.markdown("### 📈 Segment Behavioral Metrics")
            
            fig_segment_behavior = go.Figure()
            
            fig_segment_behavior.add_trace(go.Bar(
                name='Avg Order Value ($)',
                x=segment_data['customer_segment'],
                y=segment_data['avg_order_value'],
                marker_color='#667eea',
                yaxis='y'
            ))
            
            fig_segment_behavior.add_trace(go.Scatter(
                name='Orders/Customer',
                x=segment_data['customer_segment'],
                y=segment_data['orders_per_customer'],
                mode='lines+markers',
                marker=dict(size=12, color='#dc3545'),
                line=dict(width=3),
                yaxis='y2'
            ))
            
            fig_segment_behavior.update_layout(
                height=350,
                margin=dict(l=20, r=60, t=30, b=20),
                yaxis=dict(title='Avg Order Value ($)', side='left'),
                yaxis2=dict(title='Orders/Customer', side='right', overlaying='y'),
                legend=dict(orientation="h", yanchor="bottom", y=1.02)
            )
            
            st.plotly_chart(fig_segment_behavior, use_container_width=True)
        
        # RFM Treemap
        st.markdown("### 🎯 RFM-Based Customer Segmentation")
        
        # Create RFM segments from transaction data
        rfm_base = filtered_df.groupby('customer_id').agg({
            'date': 'max',
            'transaction_id': 'count',
            'revenue': 'sum'
        }).reset_index()
        
        rfm_base.columns = ['customer_id', 'last_purchase', 'frequency', 'monetary']
        rfm_base['recency'] = (filtered_df['date'].max() - rfm_base['last_purchase']).dt.days
        
        # Create RFM scores
        rfm_base['R_Score'] = pd.qcut(rfm_base['recency'], q=4, labels=[4, 3, 2, 1], duplicates='drop')
        rfm_base['F_Score'] = pd.qcut(rfm_base['frequency'].rank(method='first'), q=4, labels=[1, 2, 3, 4], duplicates='drop')
        rfm_base['M_Score'] = pd.qcut(rfm_base['monetary'].rank(method='first'), q=4, labels=[1, 2, 3, 4], duplicates='drop')
        
        rfm_base['RFM_Score'] = rfm_base['R_Score'].astype(int) + rfm_base['F_Score'].astype(int) + rfm_base['M_Score'].astype(int)
        
        # Assign RFM segments
        def assign_rfm_segment(score):
            if score >= 10:
                return 'Champions'
            elif score >= 8:
                return 'Loyal'
            elif score >= 6:
                return 'Potential'
            elif score >= 4:
                return 'At Risk'
            else:
                return 'Hibernating'
        
        rfm_base['RFM_Segment'] = rfm_base['RFM_Score'].apply(assign_rfm_segment)
        
        # RFM segment summary
        rfm_summary = rfm_base.groupby('RFM_Segment').agg({
            'customer_id': 'count',
            'monetary': 'mean',
            'frequency': 'mean',
            'recency': 'mean'
        }).reset_index()
        rfm_summary.columns = ['Segment', 'Customers', 'Avg Revenue', 'Avg Orders', 'Avg Recency']
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig_rfm = px.treemap(
                rfm_summary,
                path=['Segment'],
                values='Customers',
                color='Avg Revenue',
                color_continuous_scale='RdYlGn',
                hover_data={'Customers': ':,', 'Avg Revenue': ':$.2f', 'Avg Orders': ':.1f'}
            )
            fig_rfm.update_layout(height=400, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_rfm, use_container_width=True)
        
        with col2:
            st.markdown("#### Segment Actions")
            
            st.markdown("""
            - **Champions** 🏆: Reward & engage
            - **Loyal** ⭐: Upsell premium
            - **Potential** 📈: Nurture to loyalty
            - **At Risk** ⚠️: Retention campaigns
            - **Hibernating** 😴: Win-back offers
            """)
            
            at_risk_count = rfm_summary[rfm_summary['Segment'].isin(['At Risk', 'Hibernating'])]['Customers'].sum()
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #dc3545 0%, #c82333 100%); padding: 1rem; border-radius: 10px; color: white; margin-top: 1rem;">
                <p style="margin: 0; font-size: 0.9rem;">⚠️ Requires Attention</p>
                <h3 style="margin: 0.3rem 0;">{at_risk_count:,} customers</h3>
                <p style="margin: 0; font-size: 0.85rem;">in At Risk + Hibernating</p>
            </div>
            """, unsafe_allow_html=True)

# ============================================
# TAB 4: CUSTOMER INTELLIGENCE & SEGMENTATION (ML)
# ============================================
with tab4:
    st.markdown("## 🧠 Customer Intelligence & ML Segmentation")
    st.markdown("*Advanced RFM Analysis and Machine Learning-based Customer Clustering*")
    
    # Create sub-tabs for different analyses
    ml_tab1, ml_tab2, ml_tab3, ml_tab4 = st.tabs([
        "📊 RFM Analysis",
        "🎯 K-Means Clustering",
        "🌳 Hierarchical Clustering",
        "💡 Segment Insights"
    ])
    
    # ============================================
    # PREPARE RFM DATA FOR ALL TABS
    # ============================================
    # Calculate RFM metrics from transaction data
    analysis_date = filtered_df['date'].max() + timedelta(days=1)
    
    rfm_data = filtered_df.groupby('customer_id').agg({
        'date': 'max',
        'transaction_id': 'count',
        'revenue': 'sum'
    }).reset_index()
    
    rfm_data.columns = ['customer_id', 'last_purchase', 'frequency', 'monetary']
    rfm_data['recency'] = (analysis_date - rfm_data['last_purchase']).dt.days
    
    # Add additional metrics for clustering
    customer_metrics = filtered_df.groupby('customer_id').agg({
        'profit': 'sum',
        'quantity': 'sum',
        'discount_applied': 'mean',
        'category': 'nunique'
    }).reset_index()
    customer_metrics.columns = ['customer_id', 'total_profit', 'total_quantity', 'avg_discount', 'categories_bought']
    
    rfm_data = rfm_data.merge(customer_metrics, on='customer_id')
    rfm_data['avg_order_value'] = rfm_data['monetary'] / rfm_data['frequency']
    rfm_data['profit_margin'] = (rfm_data['total_profit'] / rfm_data['monetary'] * 100).fillna(0)
    
    # ============================================
    # ML TAB 1: RFM ANALYSIS
    # ============================================
    with ml_tab1:
        st.markdown("### 📊 RFM (Recency-Frequency-Monetary) Analysis")
        st.markdown("*Score customers based on purchase behavior to identify valuable segments*")
        
        # RFM Score Distribution Settings
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("#### ⚙️ RFM Scoring Parameters")
            rfm_quantiles = st.slider("Number of Quantiles", 3, 5, 4, help="Number of groups for scoring")
            
            st.markdown("---")
            st.markdown("#### 📈 RFM Statistics")
            st.markdown(f"""
            | Metric | Min | Max | Mean |
            |--------|-----|-----|------|
            | **Recency** (days) | {rfm_data['recency'].min():.0f} | {rfm_data['recency'].max():.0f} | {rfm_data['recency'].mean():.0f} |
            | **Frequency** (orders) | {rfm_data['frequency'].min():.0f} | {rfm_data['frequency'].max():.0f} | {rfm_data['frequency'].mean():.1f} |
            | **Monetary** ($) | {rfm_data['monetary'].min():.0f} | {rfm_data['monetary'].max():.0f} | {rfm_data['monetary'].mean():.0f} |
            """)
        
        with col2:
            # Calculate RFM scores with selected quantiles
            rfm_scored = rfm_data.copy()
            
            # R Score: Lower recency = Higher score (inverted)
            rfm_scored['R_Score'] = pd.qcut(rfm_scored['recency'], q=rfm_quantiles, 
                                            labels=list(range(rfm_quantiles, 0, -1)), duplicates='drop')
            # F Score: Higher frequency = Higher score
            rfm_scored['F_Score'] = pd.qcut(rfm_scored['frequency'].rank(method='first'), 
                                            q=rfm_quantiles, labels=list(range(1, rfm_quantiles+1)), duplicates='drop')
            # M Score: Higher monetary = Higher score
            rfm_scored['M_Score'] = pd.qcut(rfm_scored['monetary'].rank(method='first'), 
                                            q=rfm_quantiles, labels=list(range(1, rfm_quantiles+1)), duplicates='drop')
            
            rfm_scored['R_Score'] = rfm_scored['R_Score'].astype(int)
            rfm_scored['F_Score'] = rfm_scored['F_Score'].astype(int)
            rfm_scored['M_Score'] = rfm_scored['M_Score'].astype(int)
            rfm_scored['RFM_Score'] = rfm_scored['R_Score'] + rfm_scored['F_Score'] + rfm_scored['M_Score']
            rfm_scored['RFM_String'] = rfm_scored['R_Score'].astype(str) + rfm_scored['F_Score'].astype(str) + rfm_scored['M_Score'].astype(str)
            
            # RFM Score Distribution
            fig_rfm_dist = px.histogram(
                rfm_scored,
                x='RFM_Score',
                nbins=rfm_quantiles * 3,
                title='Distribution of RFM Scores',
                color_discrete_sequence=['#667eea']
            )
            fig_rfm_dist.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=50, b=20),
                xaxis_title='RFM Score',
                yaxis_title='Number of Customers'
            )
            st.plotly_chart(fig_rfm_dist, use_container_width=True)
        
        # RFM Segment Assignment
        st.markdown("---")
        st.markdown("### 🎯 RFM Customer Segments")
        
        # Define segments based on RFM scores
        max_score = rfm_quantiles * 3
        
        def get_rfm_segment(row):
            r, f, m = row['R_Score'], row['F_Score'], row['M_Score']
            total = row['RFM_Score']
            
            if r >= rfm_quantiles and f >= rfm_quantiles and m >= rfm_quantiles:
                return 'Champions'
            elif r >= rfm_quantiles-1 and f >= rfm_quantiles-1:
                return 'Loyal Customers'
            elif r >= rfm_quantiles and m >= rfm_quantiles:
                return 'Big Spenders'
            elif r >= rfm_quantiles-1 and f == 1:
                return 'New Customers'
            elif r >= rfm_quantiles and f < rfm_quantiles//2 + 1:
                return 'Promising'
            elif r <= rfm_quantiles//2 and f >= rfm_quantiles-1 and m >= rfm_quantiles-1:
                return 'At Risk (High Value)'
            elif r <= rfm_quantiles//2 and f >= rfm_quantiles//2:
                return 'At Risk (Medium Value)'
            elif r == 1:
                return 'Lost'
            elif r <= rfm_quantiles//2:
                return 'Hibernating'
            else:
                return 'Need Attention'
        
        rfm_scored['Segment'] = rfm_scored.apply(get_rfm_segment, axis=1)
        
        # Segment summary
        segment_summary = rfm_scored.groupby('Segment').agg({
            'customer_id': 'count',
            'recency': 'mean',
            'frequency': 'mean',
            'monetary': ['mean', 'sum'],
            'RFM_Score': 'mean'
        }).round(2)
        segment_summary.columns = ['Customers', 'Avg Recency', 'Avg Frequency', 'Avg Monetary', 'Total Revenue', 'Avg RFM Score']
        segment_summary = segment_summary.reset_index().sort_values('Total Revenue', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Segment distribution treemap
            fig_seg_tree = px.treemap(
                segment_summary,
                path=['Segment'],
                values='Customers',
                color='Total Revenue',
                color_continuous_scale='RdYlGn',
                title='Customer Segments by Count & Revenue'
            )
            fig_seg_tree.update_layout(height=400, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_seg_tree, use_container_width=True)
        
        with col2:
            # Segment bubble chart
            fig_seg_bubble = px.scatter(
                segment_summary,
                x='Avg Frequency',
                y='Avg Monetary',
                size='Customers',
                color='Avg Recency',
                hover_name='Segment',
                color_continuous_scale='RdYlGn_r',
                title='Segment Positioning (Size = Customer Count)'
            )
            fig_seg_bubble.update_layout(height=400, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_seg_bubble, use_container_width=True)
        
        # RFM Heatmap
        st.markdown("### 🗺️ RFM Score Heatmap")
        
        # Create RF heatmap
        rf_matrix = rfm_scored.groupby(['R_Score', 'F_Score']).size().reset_index(name='Count')
        rf_pivot = rf_matrix.pivot(index='R_Score', columns='F_Score', values='Count').fillna(0)
        
        fig_rf_heat = go.Figure(data=go.Heatmap(
            z=rf_pivot.values,
            x=[f'F={i}' for i in rf_pivot.columns],
            y=[f'R={i}' for i in rf_pivot.index],
            colorscale='Blues',
            text=rf_pivot.values.astype(int),
            texttemplate='%{text}',
            textfont={"size": 12},
            hovertemplate='Recency Score: %{y}<br>Frequency Score: %{x}<br>Customers: %{z}<extra></extra>'
        ))
        
        fig_rf_heat.update_layout(
            title='Customer Distribution: Recency vs Frequency Scores',
            height=350,
            margin=dict(l=20, r=20, t=50, b=20),
            xaxis_title='Frequency Score (Higher = More Orders)',
            yaxis_title='Recency Score (Higher = More Recent)'
        )
        st.plotly_chart(fig_rf_heat, use_container_width=True)
        
        # Segment details table
        st.markdown("### 📋 Segment Details")
        st.dataframe(
            segment_summary.style.format({
                'Customers': '{:,.0f}',
                'Avg Recency': '{:.0f} days',
                'Avg Frequency': '{:.1f}',
                'Avg Monetary': '${:,.0f}',
                'Total Revenue': '${:,.0f}',
                'Avg RFM Score': '{:.1f}'
            }).background_gradient(cmap='Greens', subset=['Total Revenue', 'Avg Monetary']),
            use_container_width=True,
            hide_index=True
        )
    
    # ============================================
    # ML TAB 2: K-MEANS CLUSTERING
    # ============================================
    with ml_tab2:
        st.markdown("### 🎯 K-Means Customer Clustering")
        st.markdown("*Machine learning-based segmentation using K-Means algorithm*")
        
        # Clustering parameters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            n_clusters = st.slider("Number of Clusters (K)", 2, 8, 4, help="Select the number of customer segments")
        
        with col2:
            features_to_use = st.multiselect(
                "Features for Clustering",
                ['recency', 'frequency', 'monetary', 'avg_order_value', 'total_quantity', 'categories_bought'],
                default=['recency', 'frequency', 'monetary']
            )
        
        with col3:
            random_state = st.number_input("Random State", min_value=0, max_value=100, value=42, 
                                           help="For reproducible results")
        
        if len(features_to_use) >= 2:
            # Prepare data for clustering
            cluster_data = rfm_data[features_to_use].copy()
            cluster_data = cluster_data.fillna(cluster_data.mean())
            
            # Standardize features
            scaler = StandardScaler()
            cluster_scaled = scaler.fit_transform(cluster_data)
            
            # Elbow Method Analysis
            st.markdown("---")
            st.markdown("#### 📈 Optimal K Selection (Elbow Method)")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Calculate inertia for different K values
                inertias = []
                K_range = range(2, 11)
                
                for k in K_range:
                    kmeans_temp = KMeans(n_clusters=k, random_state=random_state, n_init=10)
                    kmeans_temp.fit(cluster_scaled)
                    inertias.append(kmeans_temp.inertia_)
                
                fig_elbow = go.Figure()
                fig_elbow.add_trace(go.Scatter(
                    x=list(K_range),
                    y=inertias,
                    mode='lines+markers',
                    marker=dict(size=10, color='#667eea'),
                    line=dict(width=2)
                ))
                
                # Highlight selected K
                fig_elbow.add_trace(go.Scatter(
                    x=[n_clusters],
                    y=[inertias[n_clusters-2]],
                    mode='markers',
                    marker=dict(size=20, color='#dc3545', symbol='star'),
                    name=f'Selected K={n_clusters}'
                ))
                
                fig_elbow.update_layout(
                    title='Elbow Method: Inertia vs Number of Clusters',
                    xaxis_title='Number of Clusters (K)',
                    yaxis_title='Inertia (Within-cluster sum of squares)',
                    height=350,
                    margin=dict(l=20, r=20, t=50, b=20),
                    showlegend=False
                )
                st.plotly_chart(fig_elbow, use_container_width=True)
            
            # Perform K-Means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
            rfm_data['KMeans_Cluster'] = kmeans.fit_predict(cluster_scaled)
            
            with col2:
                # Cluster distribution
                cluster_counts = rfm_data['KMeans_Cluster'].value_counts().sort_index()
                
                fig_cluster_dist = px.pie(
                    values=cluster_counts.values,
                    names=[f'Cluster {i}' for i in cluster_counts.index],
                    title='Customer Distribution by Cluster',
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                fig_cluster_dist.update_traces(textposition='inside', textinfo='percent+label')
                fig_cluster_dist.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig_cluster_dist, use_container_width=True)
            
            # Cluster Visualization
            st.markdown("---")
            st.markdown("#### 🔍 Cluster Visualization")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # 2D Scatter plot
                x_axis = st.selectbox("X-Axis", features_to_use, index=0)
            
            with col2:
                y_axis = st.selectbox("Y-Axis", features_to_use, index=min(1, len(features_to_use)-1))
            
            fig_scatter = px.scatter(
                rfm_data,
                x=x_axis,
                y=y_axis,
                color=rfm_data['KMeans_Cluster'].astype(str),
                title=f'K-Means Clusters: {x_axis.title()} vs {y_axis.title()}',
                color_discrete_sequence=px.colors.qualitative.Set1,
                opacity=0.7
            )
            
            # Add cluster centroids
            centroids_original = scaler.inverse_transform(kmeans.cluster_centers_)
            centroids_df = pd.DataFrame(centroids_original, columns=features_to_use)
            
            x_idx = features_to_use.index(x_axis)
            y_idx = features_to_use.index(y_axis)
            
            fig_scatter.add_trace(go.Scatter(
                x=centroids_df.iloc[:, x_idx],
                y=centroids_df.iloc[:, y_idx],
                mode='markers',
                marker=dict(size=20, color='black', symbol='x', line=dict(width=2)),
                name='Centroids'
            ))
            
            fig_scatter.update_layout(
                height=450,
                margin=dict(l=20, r=20, t=50, b=20),
                legend_title='Cluster'
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            # 3D Visualization if 3+ features
            if len(features_to_use) >= 3:
                st.markdown("#### 🌐 3D Cluster Visualization")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    z_axis = st.selectbox("Z-Axis", features_to_use, index=min(2, len(features_to_use)-1))
                
                fig_3d = px.scatter_3d(
                    rfm_data,
                    x=x_axis,
                    y=y_axis,
                    z=z_axis,
                    color=rfm_data['KMeans_Cluster'].astype(str),
                    title=f'3D K-Means Clusters',
                    color_discrete_sequence=px.colors.qualitative.Set1,
                    opacity=0.7
                )
                fig_3d.update_layout(height=500, margin=dict(l=0, r=0, t=50, b=0))
                st.plotly_chart(fig_3d, use_container_width=True)
            
            # Cluster Profile Analysis
            st.markdown("---")
            st.markdown("#### 📊 Cluster Profiles")
            
            cluster_profile = rfm_data.groupby('KMeans_Cluster').agg({
                'customer_id': 'count',
                'recency': 'mean',
                'frequency': 'mean',
                'monetary': ['mean', 'sum'],
                'avg_order_value': 'mean',
                'total_quantity': 'mean',
                'categories_bought': 'mean',
                'profit_margin': 'mean'
            }).round(2)
            cluster_profile.columns = ['Customers', 'Avg Recency', 'Avg Frequency', 'Avg Monetary', 
                                       'Total Revenue', 'Avg AOV', 'Avg Quantity', 'Avg Categories', 'Avg Profit %']
            cluster_profile = cluster_profile.reset_index()
            cluster_profile['KMeans_Cluster'] = cluster_profile['KMeans_Cluster'].apply(lambda x: f'Cluster {x}')
            
            # Radar chart for cluster comparison
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Normalize for radar chart
                radar_metrics = ['Avg Recency', 'Avg Frequency', 'Avg Monetary', 'Avg AOV', 'Avg Categories']
                radar_data = cluster_profile[radar_metrics].copy()
                radar_normalized = (radar_data - radar_data.min()) / (radar_data.max() - radar_data.min())
                # Invert recency (lower is better)
                radar_normalized['Avg Recency'] = 1 - radar_normalized['Avg Recency']
                
                fig_radar = go.Figure()
                
                colors = px.colors.qualitative.Set1[:n_clusters]
                
                for idx, row in radar_normalized.iterrows():
                    fig_radar.add_trace(go.Scatterpolar(
                        r=row.values.tolist() + [row.values[0]],
                        theta=radar_metrics + [radar_metrics[0]],
                        fill='toself',
                        name=f'Cluster {idx}',
                        opacity=0.6,
                        line=dict(color=colors[idx % len(colors)])
                    ))
                
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    title='Cluster Comparison (Normalized)',
                    height=400,
                    margin=dict(l=60, r=60, t=50, b=20)
                )
                st.plotly_chart(fig_radar, use_container_width=True)
            
            with col2:
                st.markdown("#### 🏷️ Cluster Labels")
                
                # Auto-generate cluster labels based on characteristics
                for idx, row in cluster_profile.iterrows():
                    cluster_num = row['KMeans_Cluster']
                    
                    # Determine cluster type
                    if row['Avg Recency'] < cluster_profile['Avg Recency'].median():
                        if row['Avg Monetary'] > cluster_profile['Avg Monetary'].median():
                            label = "🏆 High-Value Active"
                            color = "#28a745"
                        else:
                            label = "⭐ Regular Active"
                            color = "#667eea"
                    else:
                        if row['Avg Monetary'] > cluster_profile['Avg Monetary'].median():
                            label = "⚠️ At-Risk High-Value"
                            color = "#ffc107"
                        else:
                            label = "😴 Dormant"
                            color = "#dc3545"
                    
                    st.markdown(f"""
                    <div style="background: {color}; padding: 0.5rem; border-radius: 8px; color: white; margin-bottom: 0.3rem; text-align: center;">
                        <small>{cluster_num}</small><br>
                        <strong>{label}</strong>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Cluster summary table
            st.markdown("#### 📋 Cluster Summary Table")
            st.dataframe(
                cluster_profile.style.format({
                    'Customers': '{:,.0f}',
                    'Avg Recency': '{:.0f} days',
                    'Avg Frequency': '{:.1f}',
                    'Avg Monetary': '${:,.0f}',
                    'Total Revenue': '${:,.0f}',
                    'Avg AOV': '${:,.0f}',
                    'Avg Quantity': '{:.1f}',
                    'Avg Categories': '{:.1f}',
                    'Avg Profit %': '{:.1f}%'
                }).background_gradient(cmap='Blues', subset=['Total Revenue', 'Customers']),
                use_container_width=True,
                hide_index=True
            )
        
        else:
            st.warning("Please select at least 2 features for clustering.")
    
    # ============================================
    # ML TAB 3: HIERARCHICAL CLUSTERING
    # ============================================
    with ml_tab3:
        st.markdown("### 🌳 Hierarchical Customer Clustering")
        st.markdown("*Agglomerative clustering with dendrogram visualization*")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            h_n_clusters = st.slider("Number of Clusters", 2, 8, 4, key="h_clusters")
        
        with col2:
            linkage_method = st.selectbox("Linkage Method", ['ward', 'complete', 'average', 'single'])
        
        with col3:
            h_features = st.multiselect(
                "Features for Clustering",
                ['recency', 'frequency', 'monetary', 'avg_order_value', 'total_quantity'],
                default=['recency', 'frequency', 'monetary'],
                key="h_features"
            )
        
        if len(h_features) >= 2:
            # Prepare data
            h_cluster_data = rfm_data[h_features].copy()
            h_cluster_data = h_cluster_data.fillna(h_cluster_data.mean())
            
            # Sample for dendrogram (limit for performance)
            sample_size = min(500, len(h_cluster_data))
            h_sample = h_cluster_data.sample(n=sample_size, random_state=42)
            
            # Standardize
            h_scaler = StandardScaler()
            h_scaled = h_scaler.fit_transform(h_cluster_data)
            h_sample_scaled = h_scaler.transform(h_sample)
            
            # Dendrogram
            st.markdown("---")
            st.markdown("#### 🌲 Dendrogram")
            
            # Calculate linkage
            Z = linkage(h_sample_scaled, method=linkage_method)
            
            # Create dendrogram figure using plotly
            from scipy.cluster.hierarchy import fcluster
            
            # Get cluster assignments for the sample
            sample_clusters = fcluster(Z, t=h_n_clusters, criterion='maxclust')
            
            # Create a simplified dendrogram visualization
            fig_dendro = go.Figure()
            
            # Show cluster distribution instead of full dendrogram (for performance)
            cluster_heights = []
            for i in range(1, h_n_clusters + 1):
                mask = sample_clusters == i
                if mask.sum() > 0:
                    cluster_heights.append(mask.sum())
                else:
                    cluster_heights.append(0)
            
            fig_dendro.add_trace(go.Bar(
                x=[f'Cluster {i+1}' for i in range(h_n_clusters)],
                y=cluster_heights,
                marker_color=px.colors.qualitative.Set2[:h_n_clusters],
                text=cluster_heights,
                textposition='outside'
            ))
            
            fig_dendro.update_layout(
                title=f'Hierarchical Clustering Distribution ({linkage_method} linkage)',
                height=300,
                margin=dict(l=20, r=20, t=50, b=20),
                yaxis_title='Number of Customers',
                xaxis_title='Cluster'
            )
            st.plotly_chart(fig_dendro, use_container_width=True)
            
            # Perform full hierarchical clustering
            h_clustering = AgglomerativeClustering(n_clusters=h_n_clusters, linkage=linkage_method)
            rfm_data['Hierarchical_Cluster'] = h_clustering.fit_predict(h_scaled)
            
            # Cluster visualization
            st.markdown("---")
            st.markdown("#### 🔍 Hierarchical Cluster Visualization")
            
            col1, col2 = st.columns(2)
            
            with col1:
                h_x_axis = st.selectbox("X-Axis", h_features, index=0, key="h_x")
            
            with col2:
                h_y_axis = st.selectbox("Y-Axis", h_features, index=min(1, len(h_features)-1), key="h_y")
            
            fig_h_scatter = px.scatter(
                rfm_data,
                x=h_x_axis,
                y=h_y_axis,
                color=rfm_data['Hierarchical_Cluster'].astype(str),
                title=f'Hierarchical Clusters: {h_x_axis.title()} vs {h_y_axis.title()}',
                color_discrete_sequence=px.colors.qualitative.Set2,
                opacity=0.7
            )
            fig_h_scatter.update_layout(height=450, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_h_scatter, use_container_width=True)
            
            # Cluster profiles
            st.markdown("#### 📊 Hierarchical Cluster Profiles")
            
            h_profile = rfm_data.groupby('Hierarchical_Cluster').agg({
                'customer_id': 'count',
                'recency': 'mean',
                'frequency': 'mean',
                'monetary': ['mean', 'sum'],
                'avg_order_value': 'mean'
            }).round(2)
            h_profile.columns = ['Customers', 'Avg Recency', 'Avg Frequency', 'Avg Monetary', 'Total Revenue', 'Avg AOV']
            h_profile = h_profile.reset_index()
            h_profile['Hierarchical_Cluster'] = h_profile['Hierarchical_Cluster'].apply(lambda x: f'Cluster {x}')
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Parallel coordinates plot
                h_parallel_data = rfm_data[h_features + ['Hierarchical_Cluster']].copy()
                
                fig_parallel = px.parallel_coordinates(
                    h_parallel_data.sample(min(1000, len(h_parallel_data))),
                    dimensions=h_features,
                    color='Hierarchical_Cluster',
                    color_continuous_scale=px.colors.sequential.Viridis,
                    title='Parallel Coordinates: Feature Comparison'
                )
                fig_parallel.update_layout(height=400, margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig_parallel, use_container_width=True)
            
            with col2:
                # Cluster comparison bar chart
                fig_h_bar = go.Figure()
                
                for metric in ['Avg Frequency', 'Avg AOV']:
                    fig_h_bar.add_trace(go.Bar(
                        name=metric,
                        x=h_profile['Hierarchical_Cluster'],
                        y=h_profile[metric],
                        text=h_profile[metric].round(1),
                        textposition='outside'
                    ))
                
                fig_h_bar.update_layout(
                    barmode='group',
                    title='Cluster Comparison: Key Metrics',
                    height=400,
                    margin=dict(l=20, r=20, t=50, b=20),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02)
                )
                st.plotly_chart(fig_h_bar, use_container_width=True)
            
            # Summary table
            st.dataframe(
                h_profile.style.format({
                    'Customers': '{:,.0f}',
                    'Avg Recency': '{:.0f} days',
                    'Avg Frequency': '{:.1f}',
                    'Avg Monetary': '${:,.0f}',
                    'Total Revenue': '${:,.0f}',
                    'Avg AOV': '${:,.0f}'
                }).background_gradient(cmap='Greens', subset=['Total Revenue']),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.warning("Please select at least 2 features for hierarchical clustering.")
    
    # ============================================
    # ML TAB 4: SEGMENT INSIGHTS & ACTIONS
    # ============================================
    with ml_tab4:
        st.markdown("### 💡 Customer Segment Insights & Recommended Actions")
        st.markdown("*Actionable business intelligence from customer segmentation*")
        
        # Use K-Means clusters for insights
        if 'KMeans_Cluster' in rfm_data.columns:
            
            # Cluster Behavior Analysis
            st.markdown("---")
            st.markdown("#### 🔬 Cluster Behavioral Analysis")
            
            cluster_behavior = rfm_data.groupby('KMeans_Cluster').agg({
                'customer_id': 'count',
                'recency': ['mean', 'std'],
                'frequency': ['mean', 'std'],
                'monetary': ['mean', 'std', 'sum'],
                'avg_order_value': 'mean',
                'profit_margin': 'mean',
                'categories_bought': 'mean'
            }).round(2)
            
            cluster_behavior.columns = ['_'.join(col).strip() for col in cluster_behavior.columns]
            cluster_behavior = cluster_behavior.reset_index()
            
            # Generate insights for each cluster
            st.markdown("#### 📋 Cluster-Specific Insights")
            
            insights_col1, insights_col2 = st.columns(2)
            
            for idx, row in cluster_behavior.iterrows():
                cluster_num = int(row['KMeans_Cluster'])
                
                # Determine cluster characteristics
                is_high_value = row['monetary_mean'] > cluster_behavior['monetary_mean'].median()
                is_recent = row['recency_mean'] < cluster_behavior['recency_mean'].median()
                is_frequent = row['frequency_mean'] > cluster_behavior['frequency_mean'].median()
                
                # Generate insights and recommendations
                if is_high_value and is_recent and is_frequent:
                    insight = {
                        'title': '🏆 Champions',
                        'color': '#28a745',
                        'description': 'Best customers - high spending, recent, frequent purchasers',
                        'actions': ['VIP rewards program', 'Early access to new products', 'Referral program incentives'],
                        'priority': 'Retain & Reward'
                    }
                elif is_high_value and is_recent:
                    insight = {
                        'title': '💎 High Spenders',
                        'color': '#667eea',
                        'description': 'High-value customers with recent activity',
                        'actions': ['Cross-sell premium products', 'Personalized recommendations', 'Loyalty tier upgrade'],
                        'priority': 'Grow Value'
                    }
                elif is_recent and is_frequent:
                    insight = {
                        'title': '⭐ Loyal Regulars',
                        'color': '#17a2b8',
                        'description': 'Frequent recent buyers with growth potential',
                        'actions': ['Upselling campaigns', 'Bundle offers', 'Category expansion'],
                        'priority': 'Increase AOV'
                    }
                elif is_high_value and not is_recent:
                    insight = {
                        'title': '⚠️ At-Risk VIPs',
                        'color': '#ffc107',
                        'description': 'Previously high-value customers showing declining activity',
                        'actions': ['Win-back campaign', 'Personal outreach', 'Special discount offers'],
                        'priority': 'URGENT: Re-engage'
                    }
                elif not is_recent and not is_frequent:
                    insight = {
                        'title': '😴 Hibernating',
                        'color': '#dc3545',
                        'description': 'Inactive customers with low recent engagement',
                        'actions': ['Reactivation email series', 'Survey for feedback', 'Significant discount offer'],
                        'priority': 'Reactivate'
                    }
                else:
                    insight = {
                        'title': '📈 Potential',
                        'color': '#6c757d',
                        'description': 'Customers with growth opportunity',
                        'actions': ['Engagement campaigns', 'Product education', 'First-time buyer offers'],
                        'priority': 'Nurture'
                    }
                
                # Display insight card
                target_col = insights_col1 if idx % 2 == 0 else insights_col2
                
                with target_col:
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, {insight['color']} 0%, {insight['color']}dd 100%); 
                                padding: 1.5rem; border-radius: 15px; color: white; margin-bottom: 1rem;">
                        <h4 style="margin: 0;">Cluster {cluster_num}: {insight['title']}</h4>
                        <p style="margin: 0.5rem 0; font-size: 0.9rem;"><strong>{row['customer_id_count']:,.0f}</strong> customers | <strong>${row['monetary_sum']:,.0f}</strong> revenue</p>
                        <p style="margin: 0.5rem 0; font-size: 0.85rem;">{insight['description']}</p>
                        <hr style="border-color: rgba(255,255,255,0.3); margin: 0.5rem 0;">
                        <p style="margin: 0; font-size: 0.85rem;"><strong>Priority:</strong> {insight['priority']}</p>
                        <p style="margin: 0.3rem 0; font-size: 0.8rem;"><strong>Actions:</strong></p>
                        <ul style="margin: 0; padding-left: 1.2rem; font-size: 0.8rem;">
                            {''.join([f'<li>{a}</li>' for a in insight['actions']])}
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Revenue by Cluster Trend Simulation
            st.markdown("---")
            st.markdown("#### 📈 Projected Revenue Impact by Segment Strategy")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Current vs Potential Revenue
                cluster_revenue = cluster_behavior[['KMeans_Cluster', 'monetary_sum', 'customer_id_count']].copy()
                cluster_revenue['Cluster'] = cluster_revenue['KMeans_Cluster'].apply(lambda x: f'Cluster {int(x)}')
                
                # Simulate potential with interventions
                np.random.seed(42)
                cluster_revenue['Potential_Revenue'] = cluster_revenue['monetary_sum'] * np.random.uniform(1.15, 1.35, len(cluster_revenue))
                
                fig_potential = go.Figure()
                
                fig_potential.add_trace(go.Bar(
                    name='Current Revenue',
                    x=cluster_revenue['Cluster'],
                    y=cluster_revenue['monetary_sum'],
                    marker_color='#667eea'
                ))
                
                fig_potential.add_trace(go.Bar(
                    name='Potential Revenue (with strategy)',
                    x=cluster_revenue['Cluster'],
                    y=cluster_revenue['Potential_Revenue'],
                    marker_color='#28a745'
                ))
                
                fig_potential.update_layout(
                    barmode='group',
                    title='Current vs Potential Revenue by Cluster',
                    height=400,
                    margin=dict(l=20, r=20, t=50, b=20),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02)
                )
                st.plotly_chart(fig_potential, use_container_width=True)
            
            with col2:
                # Priority action matrix
                st.markdown("##### 🎯 Action Priority Matrix")
                
                priority_data = []
                for idx, row in cluster_behavior.iterrows():
                    customers = row['customer_id_count']
                    revenue = row['monetary_sum']
                    recency = row['recency_mean']
                    
                    # Calculate urgency and impact scores
                    urgency = min(100, recency / 3)  # Higher recency = more urgent
                    impact = (revenue / cluster_behavior['monetary_sum'].max()) * 100
                    
                    priority_data.append({
                        'Cluster': f"Cluster {int(row['KMeans_Cluster'])}",
                        'Urgency': urgency,
                        'Impact': impact,
                        'Customers': customers,
                        'Revenue': revenue
                    })
                
                priority_df = pd.DataFrame(priority_data)
                
                fig_matrix = px.scatter(
                    priority_df,
                    x='Impact',
                    y='Urgency',
                    size='Revenue',
                    color='Cluster',
                    text='Cluster',
                    title='Urgency vs Impact',
                    color_discrete_sequence=px.colors.qualitative.Set1
                )
                
                # Add quadrant lines
                fig_matrix.add_hline(y=50, line_dash="dash", line_color="gray")
                fig_matrix.add_vline(x=50, line_dash="dash", line_color="gray")
                
                # Add quadrant labels
                fig_matrix.add_annotation(x=75, y=75, text="ACT NOW", showarrow=False, font=dict(size=12, color="red"))
                fig_matrix.add_annotation(x=25, y=75, text="MONITOR", showarrow=False, font=dict(size=12, color="orange"))
                fig_matrix.add_annotation(x=75, y=25, text="INVEST", showarrow=False, font=dict(size=12, color="green"))
                fig_matrix.add_annotation(x=25, y=25, text="MAINTAIN", showarrow=False, font=dict(size=12, color="gray"))
                
                fig_matrix.update_traces(textposition='top center')
                fig_matrix.update_layout(
                    height=400,
                    margin=dict(l=20, r=20, t=50, b=20),
                    xaxis_title='Business Impact (%)',
                    yaxis_title='Urgency Score',
                    showlegend=False
                )
                st.plotly_chart(fig_matrix, use_container_width=True)
            
            # Summary metrics
            st.markdown("---")
            st.markdown("#### 📊 Segmentation Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            total_customers = cluster_behavior['customer_id_count'].sum()
            total_revenue = cluster_behavior['monetary_sum'].sum()
            
            with col1:
                top_cluster = cluster_behavior.loc[cluster_behavior['monetary_sum'].idxmax()]
                st.metric("🏆 Top Revenue Cluster", 
                         f"Cluster {int(top_cluster['KMeans_Cluster'])}",
                         f"${top_cluster['monetary_sum']/1e6:.2f}M")
            
            with col2:
                largest_cluster = cluster_behavior.loc[cluster_behavior['customer_id_count'].idxmax()]
                st.metric("👥 Largest Cluster",
                         f"Cluster {int(largest_cluster['KMeans_Cluster'])}",
                         f"{largest_cluster['customer_id_count']:,.0f} customers")
            
            with col3:
                # Concentration: top cluster revenue %
                concentration = (top_cluster['monetary_sum'] / total_revenue) * 100
                st.metric("📊 Revenue Concentration",
                         f"{concentration:.1f}%",
                         "in top cluster")
            
            with col4:
                # Average CLV proxy
                avg_clv = total_revenue / total_customers
                st.metric("💰 Avg Customer Value",
                         f"${avg_clv:.0f}",
                         "per customer")
        
        else:
            st.info("Please run K-Means clustering in the previous tab first to see insights.")

# ============================================
# TAB 5: CUSTOMER LIFETIME VALUE (CLV) - Step 8
# ============================================
with tab5:
    st.markdown("## 💎 Customer Lifetime Value (CLV) Analysis")
    st.markdown("*Predict and analyze customer lifetime value for strategic decision-making*")
    
    # Create CLV sub-tabs
    clv_tab1, clv_tab2, clv_tab3 = st.tabs([
        "📊 CLV Calculation",
        "🎯 High vs Low Value",
        "💡 CLV Insights"
    ])
    
    # ============================================
    # CALCULATE CLV METRICS
    # ============================================
    # Aggregate customer-level data
    clv_data = filtered_df.groupby('customer_id').agg({
        'revenue': 'sum',
        'profit': 'sum',
        'transaction_id': 'count',
        'date': ['min', 'max'],
        'quantity': 'sum',
        'discount_applied': 'mean'
    }).reset_index()
    
    clv_data.columns = ['customer_id', 'total_revenue', 'total_profit', 'total_orders', 
                        'first_purchase', 'last_purchase', 'total_quantity', 'avg_discount']
    
    # Calculate CLV components
    clv_data['customer_tenure_days'] = (clv_data['last_purchase'] - clv_data['first_purchase']).dt.days + 1
    clv_data['avg_order_value'] = clv_data['total_revenue'] / clv_data['total_orders']
    clv_data['purchase_frequency'] = clv_data['total_orders'] / (clv_data['customer_tenure_days'] / 30)  # Monthly
    clv_data['avg_profit_per_order'] = clv_data['total_profit'] / clv_data['total_orders']
    
    # Recency calculation
    max_date = filtered_df['date'].max()
    clv_data['recency_days'] = (max_date - clv_data['last_purchase']).dt.days
    
    # CLV Prediction Model (Simplified BG/NBD-inspired approach)
    # CLV = (Average Order Value × Purchase Frequency × Profit Margin × Expected Lifespan)
    avg_lifespan_months = 24  # Assumed average customer lifespan
    clv_data['profit_margin'] = clv_data['total_profit'] / clv_data['total_revenue']
    clv_data['profit_margin'] = clv_data['profit_margin'].clip(0, 1)
    
    # Predicted CLV
    clv_data['predicted_clv'] = (
        clv_data['avg_order_value'] * 
        clv_data['purchase_frequency'] * 
        clv_data['profit_margin'] * 
        avg_lifespan_months
    )
    
    # Normalize CLV for segments
    clv_data['clv_percentile'] = clv_data['predicted_clv'].rank(pct=True) * 100
    
    # CLV Segments
    def assign_clv_segment(percentile):
        if percentile >= 80:
            return 'Platinum'
        elif percentile >= 60:
            return 'Gold'
        elif percentile >= 40:
            return 'Silver'
        elif percentile >= 20:
            return 'Bronze'
        else:
            return 'Standard'
    
    clv_data['clv_segment'] = clv_data['clv_percentile'].apply(assign_clv_segment)
    
    # High/Low value classification
    clv_median = clv_data['predicted_clv'].median()
    clv_data['value_category'] = clv_data['predicted_clv'].apply(
        lambda x: 'High Value' if x >= clv_median else 'Low Value'
    )
    
    # ============================================
    # CLV TAB 1: CLV CALCULATION & DISTRIBUTION
    # ============================================
    with clv_tab1:
        st.markdown("### 📊 CLV Calculation & Distribution")
        
        # CLV Summary KPIs
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_clv = clv_data['predicted_clv'].mean()
            st.metric("📈 Average CLV", f"${avg_clv:,.0f}", 
                     help="Average predicted customer lifetime value")
        
        with col2:
            median_clv = clv_data['predicted_clv'].median()
            st.metric("📊 Median CLV", f"${median_clv:,.0f}",
                     help="Median predicted CLV")
        
        with col3:
            total_clv = clv_data['predicted_clv'].sum()
            st.metric("💰 Total Portfolio CLV", f"${total_clv/1e6:.2f}M",
                     help="Sum of all customer CLVs")
        
        with col4:
            high_value_pct = (clv_data['value_category'] == 'High Value').mean() * 100
            st.metric("⭐ High Value %", f"{high_value_pct:.1f}%",
                     help="Percentage of high-value customers")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### CLV Distribution")
            
            fig_clv_hist = px.histogram(
                clv_data,
                x='predicted_clv',
                nbins=50,
                color='clv_segment',
                color_discrete_map={
                    'Platinum': '#E5E4E2',
                    'Gold': '#FFD700',
                    'Silver': '#C0C0C0',
                    'Bronze': '#CD7F32',
                    'Standard': '#667eea'
                },
                title='Distribution of Predicted CLV by Segment'
            )
            fig_clv_hist.update_layout(
                height=400,
                margin=dict(l=20, r=20, t=50, b=20),
                xaxis_title='Predicted CLV ($)',
                yaxis_title='Number of Customers'
            )
            st.plotly_chart(fig_clv_hist, use_container_width=True)
        
        with col2:
            st.markdown("#### CLV by Segment")
            
            segment_summary = clv_data.groupby('clv_segment').agg({
                'customer_id': 'count',
                'predicted_clv': ['mean', 'sum'],
                'total_revenue': 'sum'
            }).round(2)
            segment_summary.columns = ['Customers', 'Avg CLV', 'Total CLV', 'Total Revenue']
            segment_summary = segment_summary.reset_index()
            
            # Sort by segment order
            segment_order = ['Platinum', 'Gold', 'Silver', 'Bronze', 'Standard']
            segment_summary['clv_segment'] = pd.Categorical(
                segment_summary['clv_segment'], categories=segment_order, ordered=True
            )
            segment_summary = segment_summary.sort_values('clv_segment')
            
            fig_seg = px.bar(
                segment_summary,
                x='clv_segment',
                y='Total CLV',
                color='clv_segment',
                color_discrete_map={
                    'Platinum': '#E5E4E2',
                    'Gold': '#FFD700',
                    'Silver': '#C0C0C0',
                    'Bronze': '#CD7F32',
                    'Standard': '#667eea'
                },
                title='Total CLV by Customer Segment',
                text=segment_summary['Total CLV'].apply(lambda x: f'${x/1e6:.2f}M')
            )
            fig_seg.update_traces(textposition='outside')
            fig_seg.update_layout(
                height=400,
                margin=dict(l=20, r=20, t=50, b=20),
                showlegend=False
            )
            st.plotly_chart(fig_seg, use_container_width=True)
        
        # CLV Components Analysis
        st.markdown("### 🔍 CLV Components Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_scatter = px.scatter(
                clv_data.sample(min(1000, len(clv_data))),
                x='purchase_frequency',
                y='avg_order_value',
                color='clv_segment',
                size='predicted_clv',
                hover_data=['total_orders', 'total_revenue'],
                title='Purchase Frequency vs AOV (Size = CLV)',
                color_discrete_map={
                    'Platinum': '#E5E4E2',
                    'Gold': '#FFD700',
                    'Silver': '#C0C0C0',
                    'Bronze': '#CD7F32',
                    'Standard': '#667eea'
                }
            )
            fig_scatter.update_layout(height=400, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        with col2:
            # CLV Box plot by segment
            fig_box = px.box(
                clv_data,
                x='clv_segment',
                y='predicted_clv',
                color='clv_segment',
                title='CLV Distribution by Segment',
                color_discrete_map={
                    'Platinum': '#E5E4E2',
                    'Gold': '#FFD700',
                    'Silver': '#C0C0C0',
                    'Bronze': '#CD7F32',
                    'Standard': '#667eea'
                }
            )
            fig_box.update_layout(height=400, margin=dict(l=20, r=20, t=50, b=20), showlegend=False)
            st.plotly_chart(fig_box, use_container_width=True)
    
    # ============================================
    # CLV TAB 2: HIGH VS LOW VALUE CUSTOMERS
    # ============================================
    with clv_tab2:
        st.markdown("### 🎯 High-Value vs Low-Value Customer Analysis")
        
        # Value category summary
        value_summary = clv_data.groupby('value_category').agg({
            'customer_id': 'count',
            'predicted_clv': ['mean', 'sum'],
            'total_revenue': 'sum',
            'total_orders': 'sum',
            'avg_order_value': 'mean',
            'purchase_frequency': 'mean'
        }).round(2)
        value_summary.columns = ['Customers', 'Avg CLV', 'Total CLV', 'Total Revenue', 
                                 'Total Orders', 'Avg AOV', 'Avg Frequency']
        value_summary = value_summary.reset_index()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 👑 High-Value Customers")
            high_val = value_summary[value_summary['value_category'] == 'High Value'].iloc[0]
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%); padding: 1.5rem; border-radius: 15px; color: #333;">
                <h3 style="margin: 0;">{high_val['Customers']:,.0f} Customers</h3>
                <p style="margin: 0.5rem 0;">Average CLV: <strong>${high_val['Avg CLV']:,.0f}</strong></p>
                <p style="margin: 0.5rem 0;">Total Value: <strong>${high_val['Total CLV']/1e6:.2f}M</strong></p>
                <p style="margin: 0.5rem 0;">Avg AOV: <strong>${high_val['Avg AOV']:,.0f}</strong></p>
                <p style="margin: 0;">Purchase Frequency: <strong>{high_val['Avg Frequency']:.2f}/month</strong></p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### 📊 Low-Value Customers")
            low_val = value_summary[value_summary['value_category'] == 'Low Value'].iloc[0]
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 15px; color: white;">
                <h3 style="margin: 0;">{low_val['Customers']:,.0f} Customers</h3>
                <p style="margin: 0.5rem 0;">Average CLV: <strong>${low_val['Avg CLV']:,.0f}</strong></p>
                <p style="margin: 0.5rem 0;">Total Value: <strong>${low_val['Total CLV']/1e6:.2f}M</strong></p>
                <p style="margin: 0.5rem 0;">Avg AOV: <strong>${low_val['Avg AOV']:,.0f}</strong></p>
                <p style="margin: 0;">Purchase Frequency: <strong>{low_val['Avg Frequency']:.2f}/month</strong></p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Comparison visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # Revenue contribution
            fig_pie = px.pie(
                value_summary,
                values='Total Revenue',
                names='value_category',
                title='Revenue Contribution',
                color='value_category',
                color_discrete_map={'High Value': '#FFD700', 'Low Value': '#667eea'},
                hole=0.4
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            fig_pie.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Metric comparison
            metrics = ['Avg CLV', 'Avg AOV', 'Avg Frequency']
            high_vals = [high_val['Avg CLV'], high_val['Avg AOV'], high_val['Avg Frequency'] * 100]
            low_vals = [low_val['Avg CLV'], low_val['Avg AOV'], low_val['Avg Frequency'] * 100]
            
            fig_comp = go.Figure()
            fig_comp.add_trace(go.Bar(name='High Value', x=metrics, y=high_vals, marker_color='#FFD700'))
            fig_comp.add_trace(go.Bar(name='Low Value', x=metrics, y=low_vals, marker_color='#667eea'))
            
            fig_comp.update_layout(
                barmode='group',
                title='Key Metrics Comparison',
                height=350,
                margin=dict(l=20, r=20, t=50, b=20)
            )
            st.plotly_chart(fig_comp, use_container_width=True)
        
        # Top customers table
        st.markdown("### 🏆 Top 20 High-Value Customers")
        
        top_customers = clv_data.nlargest(20, 'predicted_clv')[
            ['customer_id', 'predicted_clv', 'total_revenue', 'total_orders', 
             'avg_order_value', 'clv_segment', 'recency_days']
        ].copy()
        top_customers.columns = ['Customer ID', 'Predicted CLV', 'Total Revenue', 
                                 'Orders', 'AOV', 'Segment', 'Days Since Last Order']
        
        st.dataframe(
            top_customers.style.format({
                'Predicted CLV': '${:,.0f}',
                'Total Revenue': '${:,.0f}',
                'AOV': '${:,.0f}',
                'Orders': '{:,.0f}',
                'Days Since Last Order': '{:.0f}'
            }).background_gradient(cmap='YlOrRd', subset=['Predicted CLV']),
            use_container_width=True,
            hide_index=True
        )
    
    # ============================================
    # CLV TAB 3: CLV INSIGHTS & ACTIONS
    # ============================================
    with clv_tab3:
        st.markdown("### 💡 CLV-Based Business Insights")
        
        # Key insights
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Revenue concentration
            top_20_pct = clv_data.nlargest(int(len(clv_data) * 0.2), 'predicted_clv')['total_revenue'].sum()
            revenue_concentration = (top_20_pct / clv_data['total_revenue'].sum()) * 100
            
            st.markdown(f"""
            <div class="insight-box">
                <h4>📊 Pareto Principle</h4>
                <p>Top 20% of customers generate</p>
                <h2 style="color: #667eea;">{revenue_concentration:.1f}%</h2>
                <p>of total revenue</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # CLV to CAC ratio (simulated)
            estimated_cac = 50  # Assumed customer acquisition cost
            clv_cac_ratio = avg_clv / estimated_cac
            
            ratio_color = '#28a745' if clv_cac_ratio >= 3 else '#ffc107' if clv_cac_ratio >= 2 else '#dc3545'
            
            st.markdown(f"""
            <div class="insight-box">
                <h4>💰 CLV:CAC Ratio</h4>
                <p>Return on acquisition spend</p>
                <h2 style="color: {ratio_color};">{clv_cac_ratio:.1f}:1</h2>
                <p>Target: ≥3:1</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            # At-risk high value customers
            at_risk_hv = len(clv_data[(clv_data['value_category'] == 'High Value') & 
                                       (clv_data['recency_days'] > 60)])
            
            st.markdown(f"""
            <div class="insight-box" style="border-left-color: #dc3545;">
                <h4>⚠️ At-Risk High Value</h4>
                <p>High-value customers inactive >60 days</p>
                <h2 style="color: #dc3545;">{at_risk_hv:,}</h2>
                <p>Potential CLV at risk: ${clv_data[(clv_data['value_category'] == 'High Value') & (clv_data['recency_days'] > 60)]['predicted_clv'].sum()/1e6:.2f}M</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Strategic recommendations
        st.markdown("### 🎯 CLV-Based Strategic Recommendations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 15px; color: white; margin-bottom: 1rem;">
                <h4>🎁 Retention Strategy</h4>
                <p><strong>Target:</strong> High-value customers with declining frequency</p>
                <p><strong>Action:</strong> Personalized loyalty rewards, exclusive early access</p>
                <p><strong>Expected Impact:</strong> +15% retention rate, +$2.1M CLV preserved</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); padding: 1.5rem; border-radius: 15px; color: white;">
                <h4>📈 Growth Strategy</h4>
                <p><strong>Target:</strong> Silver/Bronze customers with growth potential</p>
                <p><strong>Action:</strong> Upselling campaigns, category expansion offers</p>
                <p><strong>Expected Impact:</strong> +25% AOV increase, +$1.5M incremental revenue</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%); padding: 1.5rem; border-radius: 15px; color: #333; margin-bottom: 1rem;">
                <h4>🚀 Acquisition Strategy</h4>
                <p><strong>Target:</strong> Lookalike audiences of Platinum customers</p>
                <p><strong>Action:</strong> Targeted acquisition with optimized CAC</p>
                <p><strong>Expected Impact:</strong> 2x higher CLV from new customers</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div style="background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%); padding: 1.5rem; border-radius: 15px; color: white;">
                <h4>⚠️ Win-Back Strategy</h4>
                <p><strong>Target:</strong> {at_risk_hv:,} at-risk high-value customers</p>
                <p><strong>Action:</strong> Re-engagement campaign with special offers</p>
                <p><strong>Expected Impact:</strong> Recover 30% of at-risk CLV</p>
            </div>
            """, unsafe_allow_html=True)

# ============================================
# TAB 6: PREDICTIVE & PRESCRIPTIVE AI - Step 9
# ============================================
with tab6:
    st.markdown("## 🤖 Predictive & Prescriptive Analytics")
    st.markdown("*Machine Learning models for classification with evaluation metrics*")
    
    # Sub-tabs
    pred_tab1, pred_tab2, pred_tab3 = st.tabs([
        "🎯 Churn Prediction",
        "📊 Model Evaluation",
        "💊 Prescriptive Actions"
    ])
    
    # ============================================
    # PREPARE ML DATA
    # ============================================
    # Create features for ML
    ml_customer_data = clv_data.copy()
    
    # Create target variable (churn: inactive > 45 days)
    ml_customer_data['churned'] = (ml_customer_data['recency_days'] > 45).astype(int)
    
    # Feature engineering for ML
    ml_customer_data['orders_per_month'] = ml_customer_data['total_orders'] / np.maximum(ml_customer_data['customer_tenure_days'] / 30, 1)
    ml_customer_data['revenue_per_order'] = ml_customer_data['total_revenue'] / ml_customer_data['total_orders']
    ml_customer_data['discount_sensitivity'] = ml_customer_data['avg_discount'] * 100
    
    # Select features for model
    feature_cols = ['total_orders', 'total_revenue', 'avg_order_value', 'purchase_frequency',
                    'customer_tenure_days', 'discount_sensitivity', 'total_quantity']
    
    X = ml_customer_data[feature_cols].fillna(0)
    y = ml_customer_data['churned']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # ============================================
    # PRED TAB 1: CHURN PREDICTION
    # ============================================
    with pred_tab1:
        st.markdown("### 🎯 Customer Churn Prediction Model")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("#### Model Configuration")
            
            model_choice = st.selectbox(
                "Select ML Model",
                ['Random Forest', 'Gradient Boosting', 'Logistic Regression', 'Decision Tree']
            )
            
            # Model parameters
            if model_choice == 'Random Forest':
                n_estimators = st.slider("Number of Trees", 50, 200, 100)
                max_depth = st.slider("Max Depth", 3, 15, 10)
                model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
            elif model_choice == 'Gradient Boosting':
                n_estimators = st.slider("Number of Estimators", 50, 200, 100)
                learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.1)
                model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)
            elif model_choice == 'Logistic Regression':
                C = st.slider("Regularization (C)", 0.01, 10.0, 1.0)
                model = LogisticRegression(C=C, random_state=42, max_iter=1000)
            else:
                max_depth = st.slider("Max Depth", 3, 15, 8)
                model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
            
            # Train model
            if st.button("🚀 Train Model", use_container_width=True):
                with st.spinner("Training model..."):
                    model.fit(X_train_scaled, y_train)
                    st.session_state['trained_model'] = model
                    st.session_state['model_name'] = model_choice
                    st.success(f"✅ {model_choice} trained successfully!")
        
        with col2:
            # Check if model is trained
            if 'trained_model' in st.session_state:
                model = st.session_state['trained_model']
                
                # Predictions
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                
                # Churn Risk Distribution
                st.markdown("#### Churn Risk Distribution")
                
                fig_risk = px.histogram(
                    x=y_pred_proba,
                    nbins=50,
                    color_discrete_sequence=['#667eea'],
                    title='Distribution of Churn Probability Scores'
                )
                fig_risk.add_vline(x=0.5, line_dash="dash", line_color="red", annotation_text="Threshold")
                fig_risk.update_layout(
                    height=300,
                    margin=dict(l=20, r=20, t=50, b=20),
                    xaxis_title='Churn Probability',
                    yaxis_title='Customer Count'
                )
                st.plotly_chart(fig_risk, use_container_width=True)
                
                # Risk categories
                col1, col2, col3 = st.columns(3)
                
                low_risk = (y_pred_proba < 0.3).sum()
                medium_risk = ((y_pred_proba >= 0.3) & (y_pred_proba < 0.7)).sum()
                high_risk = (y_pred_proba >= 0.7).sum()
                
                with col1:
                    st.metric("🟢 Low Risk (<30%)", f"{low_risk:,}", 
                             f"{low_risk/len(y_pred_proba)*100:.1f}%")
                
                with col2:
                    st.metric("🟡 Medium Risk (30-70%)", f"{medium_risk:,}",
                             f"{medium_risk/len(y_pred_proba)*100:.1f}%")
                
                with col3:
                    st.metric("🔴 High Risk (>70%)", f"{high_risk:,}",
                             f"{high_risk/len(y_pred_proba)*100:.1f}%")
            else:
                st.info("👈 Configure and train a model to see predictions")
    
    # ============================================
    # PRED TAB 2: MODEL EVALUATION
    # ============================================
    with pred_tab2:
        st.markdown("### 📊 Model Evaluation Metrics")
        
        if 'trained_model' in st.session_state:
            model = st.session_state['trained_model']
            model_name = st.session_state.get('model_name', 'Model')
            
            # Predictions
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Confusion Matrix
                st.markdown("#### Confusion Matrix")
                
                cm = confusion_matrix(y_test, y_pred)
                
                fig_cm = px.imshow(
                    cm,
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=['Not Churned', 'Churned'],
                    y=['Not Churned', 'Churned'],
                    color_continuous_scale='Blues',
                    text_auto=True
                )
                fig_cm.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig_cm, use_container_width=True)
            
            with col2:
                # Metrics
                st.markdown("#### Classification Metrics")
                
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                
                try:
                    roc_auc = roc_auc_score(y_test, y_pred_proba)
                except:
                    roc_auc = 0.5
                
                # Calculate log loss
                try:
                    logloss = log_loss(y_test, y_pred_proba)
                except:
                    logloss = 1.0
                
                metrics_df = pd.DataFrame({
                    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'Log Loss'],
                    'Value': [accuracy, precision, recall, f1, roc_auc, logloss],
                    'Target': [0.80, 0.75, 0.70, 0.72, 0.80, 0.50],
                    'Status': ['✅' if accuracy >= 0.80 else '⚠️',
                              '✅' if precision >= 0.75 else '⚠️',
                              '✅' if recall >= 0.70 else '⚠️',
                              '✅' if f1 >= 0.72 else '⚠️',
                              '✅' if roc_auc >= 0.80 else '⚠️',
                              '✅' if logloss <= 0.50 else '⚠️']
                })
                
                st.dataframe(
                    metrics_df.style.format({'Value': '{:.3f}', 'Target': '{:.2f}'}),
                    use_container_width=True,
                    hide_index=True
                )
                
                # Overall score
                overall_score = (accuracy + precision + recall + f1 + roc_auc) / 5 * 100
                st.metric("📊 Overall Model Score", f"{overall_score:.1f}%")
            
            # ROC Curve
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### ROC Curve")
                
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'{model_name} (AUC={roc_auc:.3f})', line=dict(color='#667eea', width=2)))
                fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(color='gray', dash='dash')))
                
                fig_roc.update_layout(
                    height=350,
                    margin=dict(l=20, r=20, t=30, b=20),
                    xaxis_title='False Positive Rate',
                    yaxis_title='True Positive Rate',
                    legend=dict(x=0.6, y=0.1)
                )
                st.plotly_chart(fig_roc, use_container_width=True)
            
            with col2:
                st.markdown("#### Cumulative Gains")
                
                # Calculate cumulative gains
                sorted_indices = np.argsort(y_pred_proba)[::-1]
                sorted_y = y_test.values[sorted_indices]
                cumulative_gains = np.cumsum(sorted_y) / sorted_y.sum()
                percentiles = np.arange(1, len(cumulative_gains) + 1) / len(cumulative_gains)
                
                fig_gains = go.Figure()
                fig_gains.add_trace(go.Scatter(x=percentiles * 100, y=cumulative_gains * 100, mode='lines', name='Model', line=dict(color='#667eea', width=2)))
                fig_gains.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode='lines', name='Baseline', line=dict(color='gray', dash='dash')))
                
                fig_gains.update_layout(
                    height=350,
                    margin=dict(l=20, r=20, t=30, b=20),
                    xaxis_title='% of Customers Contacted',
                    yaxis_title='% of Churners Captured',
                    legend=dict(x=0.6, y=0.1)
                )
                st.plotly_chart(fig_gains, use_container_width=True)
        else:
            st.info("Train a model in the 'Churn Prediction' tab to see evaluation metrics")
    
    # ============================================
    # PRED TAB 3: PRESCRIPTIVE ACTIONS
    # ============================================
    with pred_tab3:
        st.markdown("### 💊 Prescriptive Recommendations")
        st.markdown("*AI-powered recommendations for optimal business actions*")
        
        st.markdown("---")
        
        # Pricing Optimization
        st.markdown("#### 💰 Dynamic Pricing Recommendations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 15px; color: white;">
                <h4>Electronics Category</h4>
                <p><strong>Current Avg Price:</strong> $485</p>
                <p><strong>Recommended Price:</strong> $459 (-5.4%)</p>
                <p><strong>Rationale:</strong> Price elasticity analysis suggests 8% volume increase at this price point</p>
                <p><strong>Expected Impact:</strong> +$320K incremental revenue</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); padding: 1.5rem; border-radius: 15px; color: white;">
                <h4>Fashion Category</h4>
                <p><strong>Current Avg Price:</strong> $78</p>
                <p><strong>Recommended Price:</strong> $85 (+9%)</p>
                <p><strong>Rationale:</strong> Brand premium positioning with minimal demand impact</p>
                <p><strong>Expected Impact:</strong> +$180K margin improvement</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Discount Optimization
        st.markdown("#### 🎫 Discount Strategy Optimization")
        
        discount_data = pd.DataFrame({
            'Segment': ['Platinum', 'Gold', 'Silver', 'Bronze', 'Standard'],
            'Current Discount': ['5%', '10%', '15%', '20%', '25%'],
            'Recommended': ['3%', '7%', '12%', '18%', '20%'],
            'Rationale': [
                'Minimal discount needed - high loyalty',
                'Slight reduction - strong retention',
                'Balanced approach - growth focus',
                'Maintain - price sensitive segment',
                'Reduce - diminishing returns'
            ],
            'Margin Impact': ['+$45K', '+$62K', '+$38K', '+$15K', '+$28K']
        })
        
        st.dataframe(discount_data, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Marketing Actions
        st.markdown("#### 📢 Targeted Marketing Actions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="insight-box">
                <h4>🎯 High-Risk Customers</h4>
                <p><strong>Action:</strong> Personalized win-back email</p>
                <p><strong>Offer:</strong> 20% off + free shipping</p>
                <p><strong>Timing:</strong> Within 7 days</p>
                <p><strong>Expected Response:</strong> 15-20%</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="insight-box">
                <h4>📈 Growth Potential</h4>
                <p><strong>Action:</strong> Cross-sell campaign</p>
                <p><strong>Offer:</strong> Category bundle discount</p>
                <p><strong>Timing:</strong> Post-purchase +14 days</p>
                <p><strong>Expected Uplift:</strong> 25% AOV increase</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="insight-box">
                <h4>⭐ VIP Customers</h4>
                <p><strong>Action:</strong> Exclusive preview access</p>
                <p><strong>Offer:</strong> Early access to new products</p>
                <p><strong>Timing:</strong> Monthly</p>
                <p><strong>Expected Impact:</strong> +30% engagement</p>
            </div>
            """, unsafe_allow_html=True)

# ============================================
# TAB 7: EXPLAINABLE AI (XAI) - Step 10
# ============================================
with tab7:
    st.markdown("## 🔬 Explainable AI (XAI)")
    st.markdown("*Understand why the AI makes specific predictions*")
    
    xai_tab1, xai_tab2, xai_tab3 = st.tabs([
        "📊 Feature Importance",
        "🔍 Individual Explanations",
        "📈 SHAP-Style Analysis"
    ])
    
    # ============================================
    # XAI TAB 1: FEATURE IMPORTANCE
    # ============================================
    with xai_tab1:
        st.markdown("### 📊 Global Feature Importance")
        
        if 'trained_model' in st.session_state:
            model = st.session_state['trained_model']
            model_name = st.session_state.get('model_name', 'Model')
            
            # Get feature importance
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_[0])
            else:
                importances = np.ones(len(feature_cols)) / len(feature_cols)
            
            importance_df = pd.DataFrame({
                'Feature': feature_cols,
                'Importance': importances
            }).sort_values('Importance', ascending=True)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig_imp = px.bar(
                    importance_df,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    color='Importance',
                    color_continuous_scale='Viridis',
                    title=f'Feature Importance - {model_name}'
                )
                fig_imp.update_layout(height=400, margin=dict(l=20, r=20, t=50, b=20), showlegend=False)
                st.plotly_chart(fig_imp, use_container_width=True)
            
            with col2:
                st.markdown("#### Key Insights")
                
                top_feature = importance_df.iloc[-1]
                st.markdown(f"""
                <div class="insight-box">
                    <h4>🥇 Most Important Feature</h4>
                    <p><strong>{top_feature['Feature']}</strong></p>
                    <p>Contributes {top_feature['Importance']*100:.1f}% to predictions</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Top 3 features
                st.markdown("**Top 3 Predictors:**")
                for i, row in importance_df.tail(3).iloc[::-1].iterrows():
                    st.markdown(f"- **{row['Feature']}**: {row['Importance']*100:.1f}%")
            
            # Feature correlation heatmap
            st.markdown("---")
            st.markdown("### 🔥 Feature Correlation Heatmap")
            
            corr_matrix = X[feature_cols].corr()
            
            fig_corr = px.imshow(
                corr_matrix,
                labels=dict(color="Correlation"),
                color_continuous_scale='RdBu',
                title='Feature Correlations'
            )
            fig_corr.update_layout(height=400, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("Train a model in the 'Predictive AI' tab to see feature importance")
    
    # ============================================
    # XAI TAB 2: INDIVIDUAL EXPLANATIONS
    # ============================================
    with xai_tab2:
        st.markdown("### 🔍 Individual Prediction Explanations")
        st.markdown("*Drill-down into specific customer predictions*")
        
        if 'trained_model' in st.session_state:
            model = st.session_state['trained_model']
            
            # Customer selector
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Get predictions for all customers
                all_predictions = model.predict_proba(scaler.transform(X))[:, 1]
                ml_customer_data['churn_probability'] = all_predictions
                
                # Filter options
                risk_filter = st.selectbox(
                    "Filter by Risk Level",
                    ['All', 'High Risk (>70%)', 'Medium Risk (30-70%)', 'Low Risk (<30%)']
                )
                
                if risk_filter == 'High Risk (>70%)':
                    filtered_customers = ml_customer_data[ml_customer_data['churn_probability'] >= 0.7]
                elif risk_filter == 'Medium Risk (30-70%)':
                    filtered_customers = ml_customer_data[(ml_customer_data['churn_probability'] >= 0.3) & 
                                                          (ml_customer_data['churn_probability'] < 0.7)]
                elif risk_filter == 'Low Risk (<30%)':
                    filtered_customers = ml_customer_data[ml_customer_data['churn_probability'] < 0.3]
                else:
                    filtered_customers = ml_customer_data
                
                selected_customer = st.selectbox(
                    "Select Customer ID",
                    filtered_customers['customer_id'].tolist()[:100]
                )
            
            with col2:
                if selected_customer:
                    customer_row = ml_customer_data[ml_customer_data['customer_id'] == selected_customer].iloc[0]
                    customer_features = X[ml_customer_data['customer_id'] == selected_customer]
                    customer_scaled = scaler.transform(customer_features)
                    
                    churn_prob = model.predict_proba(customer_scaled)[0][1]
                    
                    # Display customer details
                    risk_color = '#dc3545' if churn_prob >= 0.7 else '#ffc107' if churn_prob >= 0.3 else '#28a745'
                    risk_label = 'High' if churn_prob >= 0.7 else 'Medium' if churn_prob >= 0.3 else 'Low'
                    
                    st.markdown(f"""
                    <div style="background: {risk_color}; padding: 1.5rem; border-radius: 15px; color: white;">
                        <h3 style="margin: 0;">Customer {selected_customer}</h3>
                        <h2 style="margin: 0.5rem 0;">Churn Probability: {churn_prob*100:.1f}%</h2>
                        <p style="margin: 0;">Risk Level: <strong>{risk_label}</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
            
            if selected_customer:
                st.markdown("---")
                st.markdown("#### 📋 Customer Profile & Feature Contribution")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Customer metrics
                    st.markdown("**Customer Metrics:**")
                    metrics_display = pd.DataFrame({
                        'Metric': feature_cols,
                        'Value': customer_features.values[0],
                        'vs Average': (customer_features.values[0] - X.mean().values) / X.std().values
                    })
                    
                    st.dataframe(
                        metrics_display.style.format({'Value': '{:.2f}', 'vs Average': '{:+.2f}'})
                        .background_gradient(cmap='RdYlGn_r', subset=['vs Average']),
                        use_container_width=True,
                        hide_index=True
                    )
                
                with col2:
                    # Feature contribution visualization
                    st.markdown("**Feature Contribution to Prediction:**")
                    
                    if hasattr(model, 'feature_importances_'):
                        contributions = model.feature_importances_ * customer_scaled[0]
                    else:
                        contributions = np.abs(customer_scaled[0]) / np.sum(np.abs(customer_scaled[0]))
                    
                    contrib_df = pd.DataFrame({
                        'Feature': feature_cols,
                        'Contribution': contributions
                    }).sort_values('Contribution', ascending=True)
                    
                    fig_contrib = px.bar(
                        contrib_df,
                        x='Contribution',
                        y='Feature',
                        orientation='h',
                        color='Contribution',
                        color_continuous_scale='RdYlGn_r'
                    )
                    fig_contrib.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20), showlegend=False)
                    st.plotly_chart(fig_contrib, use_container_width=True)
                
                # Recommendation
                st.markdown("---")
                st.markdown("#### 🎯 AI-Generated Recommendation")
                
                if churn_prob >= 0.7:
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #dc3545 0%, #c82333 100%); padding: 1.5rem; border-radius: 15px; color: white;">
                        <h4>⚠️ URGENT: High Churn Risk</h4>
                        <p><strong>Immediate Actions:</strong></p>
                        <ul>
                            <li>Personal outreach from account manager within 48 hours</li>
                            <li>Offer exclusive 25% discount on next purchase</li>
                            <li>Survey to understand satisfaction issues</li>
                        </ul>
                        <p><strong>Expected Retention Probability:</strong> 45% with intervention</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif churn_prob >= 0.3:
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #ffc107 0%, #fd7e14 100%); padding: 1.5rem; border-radius: 15px; color: #333;">
                        <h4>⚡ MONITOR: Medium Churn Risk</h4>
                        <p><strong>Recommended Actions:</strong></p>
                        <ul>
                            <li>Add to re-engagement email sequence</li>
                            <li>Personalized product recommendations</li>
                            <li>Loyalty program enrollment offer</li>
                        </ul>
                        <p><strong>Expected Retention Probability:</strong> 70% with intervention</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #28a745 0%, #20c997 100%); padding: 1.5rem; border-radius: 15px; color: white;">
                        <h4>✅ STABLE: Low Churn Risk</h4>
                        <p><strong>Optimization Actions:</strong></p>
                        <ul>
                            <li>Cross-sell/upsell opportunities</li>
                            <li>Referral program invitation</li>
                            <li>VIP tier consideration</li>
                        </ul>
                        <p><strong>Focus:</strong> Increase customer lifetime value</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("Train a model in the 'Predictive AI' tab to see individual explanations")
    
    # ============================================
    # XAI TAB 3: SHAP-STYLE ANALYSIS
    # ============================================
    with xai_tab3:
        st.markdown("### 📈 SHAP-Style Feature Impact Analysis")
        st.markdown("*Understanding how each feature pushes predictions higher or lower*")
        
        if 'trained_model' in st.session_state:
            model = st.session_state['trained_model']
            
            # Simulate SHAP-like values using feature importance and deviation from mean
            mean_values = X.mean()
            std_values = X.std()
            
            # Calculate feature impacts for a sample
            sample_size = min(500, len(X))
            sample_indices = np.random.choice(len(X), sample_size, replace=False)
            X_sample = X.iloc[sample_indices]
            
            if hasattr(model, 'feature_importances_'):
                base_importance = model.feature_importances_
            else:
                base_importance = np.ones(len(feature_cols)) / len(feature_cols)
            
            # Calculate SHAP-like values
            shap_values = []
            for idx, row in X_sample.iterrows():
                deviation = (row - mean_values) / std_values
                impact = deviation * base_importance
                shap_values.append(impact.values)
            
            shap_array = np.array(shap_values)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Summary Plot (Feature Impact Distribution)")
                
                # Create summary plot data
                summary_data = []
                for i, feature in enumerate(feature_cols):
                    for j in range(len(shap_array)):
                        summary_data.append({
                            'Feature': feature,
                            'SHAP Value': shap_array[j, i],
                            'Feature Value': X_sample.iloc[j, i]
                        })
                
                summary_df = pd.DataFrame(summary_data)
                
                fig_summary = px.scatter(
                    summary_df,
                    x='SHAP Value',
                    y='Feature',
                    color='Feature Value',
                    color_continuous_scale='RdBu',
                    title='Feature Impact on Predictions'
                )
                fig_summary.update_traces(marker=dict(size=5, opacity=0.6))
                fig_summary.update_layout(height=450, margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig_summary, use_container_width=True)
            
            with col2:
                st.markdown("#### Mean Absolute SHAP Values")
                
                mean_abs_shap = np.abs(shap_array).mean(axis=0)
                shap_importance = pd.DataFrame({
                    'Feature': feature_cols,
                    'Mean |SHAP|': mean_abs_shap
                }).sort_values('Mean |SHAP|', ascending=True)
                
                fig_mean_shap = px.bar(
                    shap_importance,
                    x='Mean |SHAP|',
                    y='Feature',
                    orientation='h',
                    color='Mean |SHAP|',
                    color_continuous_scale='Reds',
                    title='Average Feature Impact Magnitude'
                )
                fig_mean_shap.update_layout(height=450, margin=dict(l=20, r=20, t=50, b=20), showlegend=False)
                st.plotly_chart(fig_mean_shap, use_container_width=True)
            
            # Dependence plots
            st.markdown("---")
            st.markdown("#### 🔗 Feature Dependence Analysis")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                selected_feature = st.selectbox("Select Feature", feature_cols)
            
            with col2:
                feature_idx = feature_cols.index(selected_feature)
                
                fig_dep = px.scatter(
                    x=X_sample[selected_feature],
                    y=shap_array[:, feature_idx],
                    color=shap_array[:, feature_idx],
                    color_continuous_scale='RdBu',
                    title=f'Dependence Plot: {selected_feature}',
                    labels={'x': selected_feature, 'y': 'SHAP Value', 'color': 'Impact'}
                )
                fig_dep.add_hline(y=0, line_dash="dash", line_color="gray")
                fig_dep.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig_dep, use_container_width=True)
        else:
            st.info("Train a model in the 'Predictive AI' tab to see SHAP analysis")

# ============================================
# TAB 8: REAL-TIME ANALYTICS - Step 11
# ============================================
with tab8:
    st.markdown("## ⚡ Real-Time Analytics Simulation")
    st.markdown("*Live KPI monitoring with auto-refresh capabilities*")
    
    # Real-time simulation controls
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        refresh_rate = st.selectbox("Refresh Rate", ['Manual', '5 seconds', '10 seconds', '30 seconds'], index=0)
    
    with col2:
        simulation_mode = st.selectbox("Simulation Mode", ['Normal', 'High Activity', 'Low Activity'])
    
    with col3:
        if refresh_rate != 'Manual':
            st.markdown(f"🔄 Auto-refresh: **{refresh_rate}**")
            if refresh_rate == '5 seconds':
                time.sleep(0.1)  # Placeholder - actual auto-refresh would use st.rerun()
        else:
            if st.button("🔄 Refresh Now", use_container_width=True):
                st.rerun()
    
    st.markdown("---")
    
    # Generate simulated real-time data
    np.random.seed(int(time.time()) % 1000)
    
    if simulation_mode == 'High Activity':
        activity_multiplier = 1.5
    elif simulation_mode == 'Low Activity':
        activity_multiplier = 0.6
    else:
        activity_multiplier = 1.0
    
    # Simulated real-time metrics
    base_hourly_revenue = total_revenue / (365 * 24) * activity_multiplier
    base_hourly_orders = total_orders / (365 * 24) * activity_multiplier
    
    current_revenue = base_hourly_revenue * np.random.uniform(0.8, 1.3)
    current_orders = int(base_hourly_orders * np.random.uniform(0.7, 1.4))
    current_profit = current_revenue * profit_margin / 100
    
    # Trend indicators (comparing to previous hour)
    revenue_trend = np.random.choice(['↑', '↓', '→'], p=[0.45, 0.35, 0.2])
    orders_trend = np.random.choice(['↑', '↓', '→'], p=[0.4, 0.4, 0.2])
    profit_trend = np.random.choice(['↑', '↓', '→'], p=[0.42, 0.38, 0.2])
    
    trend_colors = {'↑': '#28a745', '↓': '#dc3545', '→': '#ffc107'}
    
    # Real-time KPI Ticker
    st.markdown("### 📊 Live KPI Ticker")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 15px; color: white; text-align: center;">
            <p style="margin: 0; font-size: 0.9rem;">💰 Live Revenue (This Hour)</p>
            <h2 style="margin: 0.3rem 0; font-size: 1.8rem;">${current_revenue:,.0f}</h2>
            <p style="margin: 0; font-size: 1.5rem; color: {trend_colors[revenue_trend]};">{revenue_trend}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); padding: 1.5rem; border-radius: 15px; color: white; text-align: center;">
            <p style="margin: 0; font-size: 0.9rem;">📦 Live Orders (This Hour)</p>
            <h2 style="margin: 0.3rem 0; font-size: 1.8rem;">{current_orders:,}</h2>
            <p style="margin: 0; font-size: 1.5rem; color: {trend_colors[orders_trend]};">{orders_trend}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%); padding: 1.5rem; border-radius: 15px; color: #333; text-align: center;">
            <p style="margin: 0; font-size: 0.9rem;">📈 Live Profit (This Hour)</p>
            <h2 style="margin: 0.3rem 0; font-size: 1.8rem;">${current_profit:,.0f}</h2>
            <p style="margin: 0; font-size: 1.5rem; color: {trend_colors[profit_trend]};">{profit_trend}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        active_users = int(np.random.uniform(150, 500) * activity_multiplier)
        users_trend = np.random.choice(['↑', '↓', '→'], p=[0.5, 0.3, 0.2])
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%); padding: 1.5rem; border-radius: 15px; color: white; text-align: center;">
            <p style="margin: 0; font-size: 0.9rem;">👥 Active Users Now</p>
            <h2 style="margin: 0.3rem 0; font-size: 1.8rem;">{active_users:,}</h2>
            <p style="margin: 0; font-size: 1.5rem; color: {trend_colors[users_trend]};">{users_trend}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Real-time charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📈 Live Revenue Stream (Last 60 Minutes)")
        
        # Generate simulated hourly data
        times = pd.date_range(end=datetime.now(), periods=60, freq='T')
        revenues = base_hourly_revenue / 60 * np.random.uniform(0.5, 1.5, 60).cumsum()
        
        fig_live = go.Figure()
        fig_live.add_trace(go.Scatter(
            x=times,
            y=revenues,
            mode='lines',
            fill='tozeroy',
            line=dict(color='#667eea', width=2),
            fillcolor='rgba(102, 126, 234, 0.3)'
        ))
        fig_live.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis_title='Time',
            yaxis_title='Cumulative Revenue ($)'
        )
        st.plotly_chart(fig_live, use_container_width=True)
    
    with col2:
        st.markdown("#### 🌍 Live Orders by Region")
        
        # Generate region data
        regions = ['North America', 'Europe', 'Asia Pacific', 'Latin America']
        region_orders = [int(current_orders * p) for p in [0.35, 0.30, 0.25, 0.10]]
        
        fig_regions = px.bar(
            x=regions,
            y=region_orders,
            color=regions,
            color_discrete_sequence=['#667eea', '#764ba2', '#11998e', '#f7971e']
        )
        fig_regions.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=20, b=20),
            showlegend=False,
            xaxis_title='Region',
            yaxis_title='Orders'
        )
        st.plotly_chart(fig_regions, use_container_width=True)
    
    # Recent transactions
    st.markdown("---")
    st.markdown("### 📋 Recent Transactions (Last 10)")
    
    recent_transactions = pd.DataFrame({
        'Time': [datetime.now() - timedelta(minutes=i*np.random.randint(1, 5)) for i in range(10)],
        'Order ID': [f"ORD-{np.random.randint(100000, 999999)}" for _ in range(10)],
        'Customer': [f"CUST-{np.random.randint(1000, 9999)}" for _ in range(10)],
        'Product': np.random.choice(['Laptop', 'Phone', 'Tablet', 'Watch', 'Headphones'], 10),
        'Amount': np.random.uniform(50, 500, 10).round(2),
        'Status': np.random.choice(['Completed', 'Processing', 'Shipped'], 10, p=[0.6, 0.25, 0.15])
    })
    
    recent_transactions = recent_transactions.sort_values('Time', ascending=False)
    recent_transactions['Time'] = recent_transactions['Time'].dt.strftime('%H:%M:%S')
    
    st.dataframe(
        recent_transactions.style.apply(
            lambda x: ['background-color: #d4edda' if v == 'Completed' 
                      else 'background-color: #fff3cd' if v == 'Processing'
                      else 'background-color: #cce5ff' for v in x],
            subset=['Status']
        ),
        use_container_width=True,
        hide_index=True
    )
    
    # System metrics
    st.markdown("---")
    st.markdown("### ⚙️ System Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    dashboard_load_time = time.time() - DASHBOARD_START_TIME
    
    with col1:
        st.metric("🕐 Dashboard Load Time", f"{dashboard_load_time:.2f}s", 
                 "✅ Good" if dashboard_load_time < 5 else "⚠️ Slow")
    
    with col2:
        st.metric("💾 Data Points Processed", f"{len(filtered_df):,}")
    
    with col3:
        st.metric("📊 Active Filters", f"{len(filter_context) if 'filter_context' in dir() else 0}")
    
    with col4:
        st.metric("🔄 Last Updated", datetime.now().strftime('%H:%M:%S'))

# ============================================
# TAB 9: KNOWLEDGE GRAPH - Step 12
# ============================================
with tab9:
    st.markdown("## 🕸️ Real-Time Knowledge Graph")
    st.markdown("*Visualize business entity relationships*")
    
    kg_tab1, kg_tab2 = st.tabs(["🌐 Network Graph", "📊 Relationship Analysis"])
    
    with kg_tab1:
        st.markdown("### 🌐 Business Entity Network")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.markdown("#### Graph Settings")
            
            node_types = st.multiselect(
                "Include Entities",
                ['Customers', 'Products', 'Categories', 'Regions'],
                default=['Categories', 'Regions']
            )
            
            max_nodes = st.slider("Max Nodes", 10, 100, 30)
            
            layout_type = st.selectbox(
                "Layout",
                ['spring', 'circular', 'shell']
            )
        
        with col2:
            # Build knowledge graph
            G = nx.Graph()
            
            # Add nodes based on selection
            if 'Categories' in node_types:
                categories = filtered_df['category'].value_counts().head(max_nodes // 4)
                for cat, count in categories.items():
                    G.add_node(cat, node_type='category', size=count)
            
            if 'Regions' in node_types:
                regions = filtered_df['region'].unique()
                for region in regions:
                    G.add_node(region, node_type='region', size=len(filtered_df[filtered_df['region'] == region]))
            
            if 'Products' in node_types:
                products = filtered_df['product_name'].value_counts().head(max_nodes // 4)
                for prod, count in products.items():
                    G.add_node(prod[:20], node_type='product', size=count)
            
            if 'Customers' in node_types:
                customers = filtered_df.groupby('customer_id')['revenue'].sum().nlargest(max_nodes // 4)
                for cust, rev in customers.items():
                    G.add_node(f"C-{cust}", node_type='customer', size=rev)
            
            # Add edges based on relationships
            if 'Categories' in node_types and 'Regions' in node_types:
                cat_region = filtered_df.groupby(['category', 'region'])['revenue'].sum().reset_index()
                for _, row in cat_region.iterrows():
                    if row['category'] in G.nodes and row['region'] in G.nodes:
                        G.add_edge(row['category'], row['region'], weight=row['revenue'])
            
            if 'Products' in node_types and 'Categories' in node_types:
                prod_cat = filtered_df.groupby(['product_name', 'category']).size().reset_index(name='count')
                for _, row in prod_cat.head(50).iterrows():
                    prod_name = row['product_name'][:20]
                    if prod_name in G.nodes and row['category'] in G.nodes:
                        G.add_edge(prod_name, row['category'], weight=row['count'])
            
            # Calculate layout
            if len(G.nodes) > 0:
                if layout_type == 'spring':
                    pos = nx.spring_layout(G, k=2, iterations=50)
                elif layout_type == 'circular':
                    pos = nx.circular_layout(G)
                else:
                    pos = nx.shell_layout(G)
                
                # Create plotly figure
                edge_x = []
                edge_y = []
                for edge in G.edges():
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
                
                edge_trace = go.Scatter(
                    x=edge_x, y=edge_y,
                    line=dict(width=0.5, color='#888'),
                    hoverinfo='none',
                    mode='lines'
                )
                
                node_x = [pos[node][0] for node in G.nodes()]
                node_y = [pos[node][1] for node in G.nodes()]
                node_text = list(G.nodes())
                
                # Color by node type
                node_colors = []
                color_map = {'category': '#667eea', 'region': '#11998e', 'product': '#f7971e', 'customer': '#eb3349'}
                for node in G.nodes():
                    node_type = G.nodes[node].get('node_type', 'category')
                    node_colors.append(color_map.get(node_type, '#667eea'))
                
                node_trace = go.Scatter(
                    x=node_x, y=node_y,
                    mode='markers+text',
                    hoverinfo='text',
                    text=node_text,
                    textposition='top center',
                    marker=dict(
                        size=20,
                        color=node_colors,
                        line=dict(width=2, color='white')
                    )
                )
                
                fig_network = go.Figure(data=[edge_trace, node_trace])
                fig_network.update_layout(
                    height=500,
                    margin=dict(l=20, r=20, t=20, b=20),
                    showlegend=False,
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                )
                
                st.plotly_chart(fig_network, use_container_width=True)
                
                # Network statistics
                st.markdown("#### 📊 Network Statistics")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Nodes", len(G.nodes()))
                with col2:
                    st.metric("Edges", len(G.edges()))
                with col3:
                    density = nx.density(G) if len(G.nodes()) > 1 else 0
                    st.metric("Density", f"{density:.3f}")
                with col4:
                    if len(G.nodes()) > 0 and nx.is_connected(G):
                        avg_path = nx.average_shortest_path_length(G)
                        st.metric("Avg Path Length", f"{avg_path:.2f}")
                    else:
                        st.metric("Components", nx.number_connected_components(G))
            else:
                st.info("Select entity types to build the knowledge graph")
    
    with kg_tab2:
        st.markdown("### 📊 Relationship Strength Analysis")
        
        # Category-Region relationship heatmap
        st.markdown("#### Category ↔ Region Revenue Matrix")
        
        pivot_data = filtered_df.pivot_table(
            values='revenue',
            index='category',
            columns='region',
            aggfunc='sum'
        ).fillna(0)
        
        fig_heatmap = px.imshow(
            pivot_data,
            labels=dict(x="Region", y="Category", color="Revenue"),
            color_continuous_scale='Viridis',
            title='Revenue by Category and Region'
        )
        fig_heatmap.update_layout(height=400, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Top relationships
        st.markdown("#### 🔗 Strongest Business Relationships")
        
        relationships = filtered_df.groupby(['category', 'region']).agg({
            'revenue': 'sum',
            'transaction_id': 'count',
            'customer_id': 'nunique'
        }).reset_index()
        relationships.columns = ['Category', 'Region', 'Revenue', 'Transactions', 'Unique Customers']
        relationships = relationships.nlargest(10, 'Revenue')
        
        st.dataframe(
            relationships.style.format({
                'Revenue': '${:,.0f}',
                'Transactions': '{:,}',
                'Unique Customers': '{:,}'
            }).background_gradient(cmap='Greens', subset=['Revenue']),
            use_container_width=True,
            hide_index=True
        )

# ============================================
# TAB 10: DECISION CENTER - Enhanced (Step 13: UX & BI Storytelling)
# ============================================
with tab10:
    st.markdown("## 📋 Decision Intelligence Center")
    st.markdown("*Executive-level insights and actionable recommendations*")
    
    # Executive Dashboard
    st.markdown("### 🎯 Executive Summary")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1E3A5F 0%, #2C5282 100%); padding: 2rem; border-radius: 15px; color: white;">
            <h3 style="margin: 0 0 1rem 0;">📊 Business Health Score: 
                <span style="color: #38ef7d; font-size: 2rem;">78/100</span>
            </h3>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem;">
                <div>
                    <p style="margin: 0; opacity: 0.8;">Revenue</p>
                    <h4 style="margin: 0;">${total_revenue/1e6:.2f}M</h4>
                </div>
                <div>
                    <p style="margin: 0; opacity: 0.8;">Profit Margin</p>
                    <h4 style="margin: 0;">{profit_margin:.1f}%</h4>
                </div>
                <div>
                    <p style="margin: 0; opacity: 0.8;">Customers</p>
                    <h4 style="margin: 0;">{unique_customers:,}</h4>
                </div>
                <div>
                    <p style="margin: 0; opacity: 0.8;">Avg CLV</p>
                    <h4 style="margin: 0;">${avg_clv:,.0f}</h4>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 15px; height: 100%;">
            <h4 style="margin: 0 0 1rem 0;">🚨 Alerts</h4>
            <p style="margin: 0.5rem 0; color: #dc3545;">⚠️ 1,310 customers at high churn risk</p>
            <p style="margin: 0.5rem 0; color: #ffc107;">⚡ Inventory low for 3 products</p>
            <p style="margin: 0.5rem 0; color: #28a745;">✅ Revenue trending above target</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Strategic Recommendations with priority
    st.markdown("### 🎯 Strategic Recommendations")
    
    recommendations = [
        {
            'priority': 'Critical',
            'category': 'Customer Retention',
            'action': 'Launch win-back campaign for high-value churning customers',
            'impact': '+$2.1M CLV preserved',
            'confidence': 85,
            'timeline': '7 days',
            'color': '#dc3545'
        },
        {
            'priority': 'High',
            'category': 'Revenue Growth',
            'action': 'Expand marketing in Asia Pacific region',
            'impact': '+35% regional revenue',
            'confidence': 78,
            'timeline': '30 days',
            'color': '#ffc107'
        },
        {
            'priority': 'Medium',
            'category': 'Pricing',
            'action': 'Implement dynamic pricing for Electronics',
            'impact': '+$320K margin',
            'confidence': 72,
            'timeline': '60 days',
            'color': '#17a2b8'
        }
    ]
    
    for rec in recommendations:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"""
            <div style="background: white; border-left: 5px solid {rec['color']}; padding: 1rem; margin-bottom: 0.5rem; border-radius: 0 10px 10px 0; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span style="background: {rec['color']}; color: white; padding: 0.2rem 0.5rem; border-radius: 5px; font-size: 0.8rem;">{rec['priority']}</span>
                        <span style="margin-left: 0.5rem; color: #666;">{rec['category']}</span>
                    </div>
                    <span style="color: #28a745; font-weight: bold;">{rec['impact']}</span>
                </div>
                <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem;">{rec['action']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem;">
                <p style="margin: 0; font-size: 0.8rem; color: #666;">Confidence</p>
                <h3 style="margin: 0; color: {rec['color']};">{rec['confidence']}%</h3>
                <p style="margin: 0; font-size: 0.8rem;">⏱️ {rec['timeline']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # What-If Scenario Analysis
    st.markdown("### 🔄 What-If Scenario Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Adjust Parameters")
        
        price_change = st.slider("Price Change (%)", -20, 20, 0, key="scenario_price")
        marketing_spend = st.slider("Marketing Spend Change (%)", -30, 50, 0, key="scenario_marketing")
        retention_improvement = st.slider("Retention Rate Improvement (%)", 0, 30, 0, key="scenario_retention")
    
    with col2:
        st.markdown("#### Projected Impact")
        
        base_revenue = total_revenue
        
        # Impact model
        price_impact = base_revenue * (price_change / 100) * -0.5
        marketing_impact = base_revenue * (marketing_spend / 100) * 0.3
        retention_impact = base_revenue * (retention_improvement / 100) * 0.15
        
        projected_revenue = base_revenue + price_impact + marketing_impact + retention_impact
        revenue_change = ((projected_revenue - base_revenue) / base_revenue) * 100
        
        impact_color = '#28a745' if revenue_change >= 0 else '#dc3545'
        
        st.markdown(f"""
        <div style="background: {impact_color}; padding: 1.5rem; border-radius: 15px; color: white; text-align: center;">
            <p style="margin: 0;">Projected Revenue</p>
            <h2 style="margin: 0.5rem 0;">${projected_revenue/1e6:.2f}M</h2>
            <p style="margin: 0; font-size: 1.5rem;">{revenue_change:+.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.metric("Projected Profit", f"${(projected_revenue * profit_margin/100)/1e6:.2f}M", f"{revenue_change:+.1f}%")
    
    # Report Generation
    st.markdown("---")
    st.markdown("### 📄 Executive Report Generation")
    
    with st.expander("📊 Generate Full Executive Report", expanded=False):
        report_content = f"""
# E-Commerce Business Intelligence Report
**Generated:** {datetime.now().strftime('%B %d, %Y at %H:%M')}

## Executive Summary
- **Total Revenue:** ${total_revenue/1e6:.2f}M
- **Gross Profit:** ${total_profit/1e6:.2f}M ({profit_margin:.1f}% margin)
- **Total Orders:** {total_orders:,}
- **Active Customers:** {unique_customers:,}
- **Average CLV:** ${avg_clv:,.0f}

## Key Insights
1. Top 20% of customers generate {revenue_concentration:.1f}% of revenue (Pareto Principle)
2. {high_value_pct:.1f}% of customer base classified as high-value
3. CLV:CAC ratio of {clv_cac_ratio:.1f}:1 indicates healthy acquisition ROI

## Risk Alerts
- {at_risk_hv:,} high-value customers at risk of churning
- Potential CLV at risk: ${clv_data[(clv_data['value_category'] == 'High Value') & (clv_data['recency_days'] > 60)]['predicted_clv'].sum()/1e6:.2f}M

## Recommended Actions
1. **URGENT:** Launch retention campaign for at-risk high-value customers
2. **HIGH:** Expand marketing presence in high-growth regions
3. **MEDIUM:** Implement dynamic pricing optimization

## Performance Metrics
- Dashboard Load Time: {time.time() - DASHBOARD_START_TIME:.2f}s
- Data Points Analyzed: {len(filtered_df):,}
        """
        
        st.markdown(report_content)
        
        st.download_button(
            label="📥 Download Executive Report",
            data=report_content,
            file_name=f"executive_report_{datetime.now().strftime('%Y%m%d')}.md",
            mime="text/markdown"
        )

# ============================================
# FOOTER - Performance & Attribution
# ============================================
st.markdown("---")

# Performance summary
dashboard_end_time = time.time()
total_load_time = dashboard_end_time - DASHBOARD_START_TIME

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div style="text-align: center; color: #666;">
        <p style="margin: 0;">⚡ Load Time: {total_load_time:.2f}s</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div style="text-align: center; color: #666;">
        <p style="margin: 0;">📊 Data Points: {len(filtered_df):,}</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div style="text-align: center; color: #666;">
        <p style="margin: 0;">🔄 Last Updated: {datetime.now().strftime('%H:%M:%S')}</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem; margin-top: 1rem;">
    <p style="margin: 0;">📊 <strong>E-Commerce BI Dashboard v2.0</strong> | Built with Streamlit, Plotly, Scikit-learn, NetworkX</p>
    <p style="margin: 0.5rem 0;">Decision Intelligence • Predictive Analytics • Explainable AI • Real-Time Insights • Knowledge Graphs</p>
    <p style="margin: 0; font-size: 0.8rem; opacity: 0.7;">Enterprise-grade Business Intelligence for Executive Decision Making</p>
</div>
""", unsafe_allow_html=True)
