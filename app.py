"""
========================================================================================
UNLOX ACADEMY - MACHINE LEARNING AUGUST 2026 BATCH
MAJOR PROJECT 2: INDIAN PRE-OWNED CAR VALUATION & PREDICTION SUITE
========================================================================================
Student Name : Rhydham Shukla
Batch        : ML (Machine Learning) 2026 August Batch
Institution  : Unlox Academy
Architecture : Streamlit + LightGBM / XGBoost / Random Forest Stacking Ensemble
Features     : Single Valuation, Bulk Batch Prediction, Market Insights & Fair Valuation Bands
========================================================================================
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import joblib

# Page Configuration
st.set_page_config(
    page_title="AutoValuate AI | Indian Pre-Owned Car Valuation Engine",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .hero-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 32px 36px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
    }
    .hero-title {
        font-size: 32px;
        font-weight: 800;
        background: linear-gradient(90deg, #60a5fa, #a78bfa, #38bdf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 15px;
        line-height: 1.6;
        margin-bottom: 12px;
    }
    .badge-pill {
        display: inline-block;
        background: rgba(99, 102, 241, 0.15);
        color: #818cf8;
        border: 1px solid rgba(99, 102, 241, 0.3);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        margin-right: 8px;
    }
    .metric-card {
        background: #1e293b;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #6366f1;
    }
    .metric-value {
        font-size: 26px;
        font-weight: 800;
        color: #f8fafc;
        margin-top: 6px;
    }
    .metric-label {
        font-size: 12px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }
    .price-box {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
        border: 2px solid #6366f1;
        border-radius: 16px;
        padding: 28px;
        text-align: center;
        box-shadow: 0 10px 25px -5px rgba(99, 102, 241, 0.3);
        margin: 20px 0;
    }
    .price-text {
        font-size: 42px;
        font-weight: 800;
        color: #38bdf8;
        margin: 8px 0;
    }
    .band-text {
        font-size: 15px;
        color: #c7d2fe;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Cache model loading
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "Rhydham_Shukla_Car_Price_Model.joblib")
    if not os.path.exists(model_path):
        st.error(f"Model file not found at: {model_path}. Please run train_car_price_model.py first.")
        st.stop()
    return joblib.load(model_path)

payload = load_model()
lgb_model = payload['final_lgb']
xgb_model = payload['final_xgb']
rf_model = payload['final_rf']
feature_names = payload['feature_names']
impute_dict = payload['impute_dict']
dummy_cols = payload['dummy_cols']
metrics = payload.get('metrics', {})

# Header Banner
st.markdown("""
<div class="hero-header">
    <div class="hero-title">🚗 AutoValuate AI: Indian Pre-Owned Car Valuation Engine</div>
    <div class="hero-subtitle">
        High-precision automotive pricing intelligence leveraging a multi-model stacking ensemble (LightGBM + XGBoost + Random Forest) 
        trained on 53,000+ Indian vehicle market transactions.
    </div>
    <div>
        <span class="badge-pill">Unlox Academy ML Capstone</span>
        <span class="badge-pill">Author: Rhydham Shukla</span>
        <span class="badge-pill">Batch: August 2026</span>
        <span class="badge-pill">R² Score: 0.9301</span>
        <span class="badge-pill">5-Fold CV Verified</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Single Vehicle Valuation",
    "📁 Bulk Batch Prediction",
    "📊 Market Insights & EDA",
    "🔬 Model Architecture & Metrics"
])

# TAB 1: Single Vehicle Valuation
with tab1:
    st.subheader("Interactive Valuation Calculator")
    st.caption("Adjust vehicle characteristics to obtain real-time market value estimation with fair price confidence banding.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("##### 🚘 Core Specifications")
        maker = st.selectbox("Manufacturer (Maker)", ['bmw', 'skoda', 'audi', 'toyota', 'nissan', 'hyundai', 'fiat', 'maserati'], index=0)
        
        # Filter models dynamically based on dataset ground truth
        model_options = {
            'bmw': ['x1', 'x3', 'x5'],
            'skoda': ['octavia', 'superb', 'rapid', 'yeti', 'roomster', 'citigo'],
            'audi': ['q3', 'q5', 'q7', 'coupe', 'tt'],
            'toyota': ['yaris', 'auris', 'avensis', 'aygo'],
            'nissan': ['qashqai', 'micra', 'juke'],
            'hyundai': ['i30', 'coupe'],
            'fiat': ['panda', 'coupe'],
            'maserati': ['coupe']
        }
        selected_models = model_options.get(maker, ['x1'])
        car_model = st.selectbox("Model", selected_models, index=0)
        location = st.selectbox("Registered Location", ['Bangalore', 'Kochi', 'Coimbatore', 'Chennai', 'Hyderabad', 'Pune', 'Mumbai', 'Delhi', 'Kolkata', 'Ahmedabad', 'Jaipur'])
        body_type = st.selectbox("Body Type", ['sedan_suv', 'compact', 'van'])

    with col2:
        st.markdown("##### ⚙️ Drivetrain & Metrics")
        fuel_type = st.selectbox("Fuel Type", ['diesel', 'petrol'])
        transmission = st.selectbox("Transmission", ['man', 'auto'])
        owner_type = st.selectbox("Owner Sequence", ['First', 'Second', 'Third', 'Fourth & Above'])
        vroom_rating = st.slider("Vroom Audit Rating (1-5 Stars)", min_value=1, max_value=5, value=3)

    with col3:
        st.markdown("##### ⏱️ Age & Engine Capacity")
        age_of_car = st.slider("Age of Vehicle (Years)", min_value=0, max_value=25, value=4)
        manufacture_year = 2026 - age_of_car
        distance = st.number_input("Odometer Reading (Kilometers Driven)", min_value=500, max_value=500000, value=45000, step=5000)
        engine_displacement = st.slider("Engine Displacement (cc)", min_value=800, max_value=5000, value=1968, step=50)
        engine_power = st.slider("Engine Power (kW / BHP equivalent)", min_value=40, max_value=450, value=110, step=5)
        seat_count = st.selectbox("Seat Count", [4.0, 5.0, 7.0, 8.0], index=1)
        door_count = st.selectbox("Door Count", [2.0, 4.0, 5.0], index=1)

    # Compute Feature Transformation
    input_dict = {
        'Maker': maker,
        'model': car_model,
        'Location': location,
        'Owner Type': owner_type,
        'transmission': transmission,
        'fuel_type': fuel_type,
        'body_type': body_type,
        'Distance': float(distance),
        'manufacture_year': float(manufacture_year),
        'Age of car': int(age_of_car),
        'engine_displacement': int(engine_displacement),
        'engine_power': float(engine_power),
        'door_count': float(door_count),
        'seat_count': float(seat_count),
        'Vroom Audit Rating': int(vroom_rating),
        'Power_to_Displacement': float(engine_power) / (float(engine_displacement) + 1e-5),
        'Annual_Mileage': float(distance) / (float(age_of_car) + 1.0),
        'Age_Squared': float(age_of_car ** 2),
        'Log_Distance': np.log1p(float(distance)),
        'Log_Displacement': np.log1p(float(engine_displacement)),
        'Is_Luxury': 1 if maker.lower() in ['audi', 'bmw', 'mercedes', 'volvo', 'maserati'] else 0,
        'Audit_Age_Ratio': float(vroom_rating) / (float(age_of_car) + 1.0)
    }

    # Construct feature row with exact alignment to training features
    feature_row = pd.DataFrame(0.0, index=[0], columns=feature_names)
    
    # Fill numeric columns
    num_cols = ['Distance', 'manufacture_year', 'Age of car', 'engine_displacement', 'engine_power',
                'door_count', 'seat_count', 'Vroom Audit Rating', 'Power_to_Displacement', 
                'Annual_Mileage', 'Age_Squared', 'Log_Distance', 'Log_Displacement', 'Is_Luxury', 'Audit_Age_Ratio']
    for c in num_cols:
        if c in feature_names:
            feature_row[c] = input_dict[c]

    # Populate categorical dummy columns directly (prevents single-row drop_first bug)
    active_dummies = [
        f'Maker_{maker.lower()}',
        f'model_{car_model.lower()}',
        f'Location_{location}',
        f'Owner Type_{owner_type}',
        f'transmission_{transmission.lower()}',
        f'fuel_type_{fuel_type.lower()}',
        f'body_type_{body_type.lower()}'
    ]
    for col in active_dummies:
        if col in feature_names:
            feature_row[col] = 1.0

    # Predict with Blended Ensemble
    pred_lgb = np.expm1(lgb_model.predict(feature_row)[0])
    pred_xgb = np.expm1(xgb_model.predict(feature_row)[0])
    pred_rf = np.expm1(rf_model.predict(feature_row)[0])
    final_pred = 0.45 * pred_lgb + 0.40 * pred_xgb + 0.15 * pred_rf

    # Dynamic Fair Valuation Band (±5%)
    lower_band = final_pred * 0.95
    upper_band = final_pred * 1.05

    st.markdown(f"""
    <div class="price-box">
        <div style="font-size: 14px; text-transform: uppercase; letter-spacing: 0.1em; color: #a5b4fc; font-weight: 700;">
            Estimated Fair Market Value
        </div>
        <div class="price-text">₹ {final_pred:,.0f}</div>
        <div class="band-text">
            Fair Price Range (±5% Band): <b>₹ {lower_band:,.0f}</b> — <b>₹ {upper_band:,.0f}</b>
        </div>
        <div style="margin-top: 10px; font-size: 13px; color: #94a3b8;">
            Ensemble Consensus: LightGBM (₹ {pred_lgb:,.0f}) | XGBoost (₹ {pred_xgb:,.0f}) | Random Forest (₹ {pred_rf:,.0f})
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Key Driver Metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Annual Usage</div>
            <div class="metric-value">{input_dict['Annual_Mileage']:,.0f} km/yr</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Power Density</div>
            <div class="metric-value">{input_dict['Power_to_Displacement']*100:.2f} kW/L</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Segment Tier</div>
            <div class="metric-value">{'Premium Luxury' if input_dict['Is_Luxury'] else 'Standard Value'}</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Audit Score</div>
            <div class="metric-value">{'⭐' * vroom_rating}</div>
        </div>
        """, unsafe_allow_html=True)

# TAB 2: Bulk Batch Prediction
with tab2:
    st.subheader("📁 Bulk Batch Prediction & Scoring")
    st.write("Upload an unlabelled test dataset (matching `Cap_Test_Data_2025.csv` format) to generate instant valuation predictions for thousands of vehicles simultaneously.")

    uploaded_file = st.file_uploader("Upload CSV File for Bulk Inference", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            batch_df.columns = batch_df.columns.str.strip()
            st.success(f"Successfully loaded {len(batch_df):,} vehicles from `{uploaded_file.name}`.")
            st.dataframe(batch_df.head(5))

            if st.button("🚀 Run Stacking Ensemble Prediction on Full Batch"):
                with st.spinner("Processing feature engineering & multi-model inference..."):
                    # Preprocess batch
                    d = batch_df.copy()
                    d['Distance'] = d['Distance'].fillna(impute_dict['Distance'])
                    d['manufacture_year'] = d['manufacture_year'].fillna(impute_dict['manufacture_year'])
                    d['engine_power'] = d['engine_power'].fillna(impute_dict['engine_power'])
                    d['door_count'] = d['door_count'].fillna(impute_dict['door_count'])
                    d['seat_count'] = d['seat_count'].fillna(impute_dict['seat_count'])
                    d['Location'] = d['Location'].fillna(impute_dict['Location'])
                    d['body_type'] = d['body_type'].fillna(impute_dict['body_type'])

                    d['Power_to_Displacement'] = d['engine_power'] / (d['engine_displacement'] + 1e-5)
                    d['Annual_Mileage'] = d['Distance'] / (d['Age of car'] + 1.0)
                    d['Age_Squared'] = (d['Age of car'] ** 2).astype(float)
                    d['Log_Distance'] = np.log1p(d['Distance'])
                    d['Log_Displacement'] = np.log1p(d['engine_displacement'])
                    d['Is_Luxury'] = d['Maker'].astype(str).str.lower().isin(['audi', 'bmw', 'mercedes', 'volvo']).astype(int)
                    d['Audit_Age_Ratio'] = d['Vroom Audit Rating'] / (d['Age of car'] + 1.0)

                    # Encode
                    cat_cols = ['Maker', 'model', 'Location', 'Owner Type', 'transmission', 'fuel_type', 'body_type']
                    num_cols = ['Distance', 'manufacture_year', 'Age of car', 'engine_displacement', 'engine_power',
                                'door_count', 'seat_count', 'Vroom Audit Rating', 'Power_to_Displacement', 
                                'Annual_Mileage', 'Age_Squared', 'Log_Distance', 'Log_Displacement', 'Is_Luxury', 'Audit_Age_Ratio']

                    batch_cat_dummies = pd.get_dummies(d[cat_cols], drop_first=True, dtype=float)
                    
                    # Create aligned matrix
                    X_batch = pd.DataFrame(0.0, index=d.index, columns=feature_names)
                    for col in num_cols:
                        if col in feature_names:
                            X_batch[col] = d[col].values
                    for col in batch_cat_dummies.columns:
                        if col in feature_names:
                            X_batch[col] = batch_cat_dummies[col].values

                    p_lgb = np.expm1(lgb_model.predict(X_batch))
                    p_xgb = np.expm1(xgb_model.predict(X_batch))
                    p_rf = np.expm1(rf_model.predict(X_batch))
                    batch_preds = 0.45 * p_lgb + 0.40 * p_xgb + 0.15 * p_rf

                    result_df = pd.DataFrame({
                        'ID': batch_df['ID'] if 'ID' in batch_df.columns else range(len(batch_df)),
                        'Predicted_Price_INR': np.round(batch_preds, 2),
                        'Lower_Bound_5pct': np.round(batch_preds * 0.95, 2),
                        'Upper_Bound_5pct': np.round(batch_preds * 1.05, 2)
                    })

                    st.markdown("### 📋 Prediction Results (First 10 Rows)")
                    st.dataframe(result_df.head(10))

                    csv_data = result_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="⬇️ Download Batch Predictions (CSV)",
                        data=csv_data,
                        file_name="AutoValuate_Batch_Predictions.csv",
                        mime="text/csv"
                    )
        except Exception as e:
            st.error(f"Error processing batch file: {e}")
    else:
        st.info("💡 You can upload `Cap_Test_Data_2025.csv` to generate the complete Kaggle-style submission file.")

# TAB 3: Market Insights & Visualizations
with tab3:
    st.subheader("📊 Indian Automotive Market Insights & EDA")
    st.write("Visual patterns distilled from 53,000+ Indian second-hand car transactions.")

    fig_dir = os.path.join(os.path.dirname(__file__), "figures")
    
    col_a, col_b = st.columns(2)
    with col_a:
        p_dist = os.path.join(fig_dir, "eda_price_distribution.png")
        if os.path.exists(p_dist):
            st.image(p_dist, caption="Price Distribution & Log-Transformed Normalization", use_container_width=True)
        
        p_box = os.path.join(fig_dir, "eda_maker_transmission_boxplot.png")
        if os.path.exists(p_box):
            st.image(p_box, caption="Price Distributions by Manufacturer & Transmission", use_container_width=True)

    with col_b:
        p_heat = os.path.join(fig_dir, "eda_correlation_heatmap.png")
        if os.path.exists(p_heat):
            st.image(p_heat, caption="Feature Correlation Matrix with Price", use_container_width=True)

        p_feat = os.path.join(fig_dir, "feature_importance_top15.png")
        if os.path.exists(p_feat):
            st.image(p_feat, caption="Top 15 Most Important Valuation Drivers (LightGBM)", use_container_width=True)

# TAB 4: Model Architecture & Metrics
with tab4:
    st.subheader("🔬 Machine Learning Methodology & Metrics")
    
    m_col1, m_col2 = st.columns([1, 1])

    with m_col1:
        st.markdown("""
        #### 🏛️ Architectural Formulation
        The prediction engine uses a **Log-Transformed Blended Stacking Regressor** to guarantee optimal residual symmetry and avoid skewness penalties:
        
        $$\\hat{y}_{\\text{final}} = \\exp\\Big(0.45 \\cdot \\log(1 + \\hat{y}_{\\text{LGBM}}) + 0.40 \\cdot \\log(1 + \\hat{y}_{\\text{XGB}}) + 0.15 \\cdot \\log(1 + \\hat{y}_{\\text{RF}})\\Big) - 1$$
        
        ##### 🛡️ Guardrails & Idempotency:
        - **Target Filtering**: 43 extreme noise entries ($Price < ₹25,000$) eliminated.
        - **Zero-Leakage Imputation**: Missing continuous features imputed strictly with training set medians.
        - **High Cardinality**: 30+ categorical permutations encoded into aligned dummy space.
        - **5-Fold Cross Validation**: Rigid out-of-fold scoring preventing data snooping.
        """)

    with m_col2:
        st.markdown("#### 📈 Benchmark Leaderboard (5-Fold CV)")
        
        bench_data = []
        for m_name, vals in metrics.items():
            bench_data.append({
                "Model": m_name,
                "R² Score": vals.get("R2", 0.0),
                "RMSE (INR)": f"₹ {vals.get('RMSE', 0):,.0f}",
                "MAE (INR)": f"₹ {vals.get('MAE', 0):,.0f}",
                "MAPE (%)": f"{vals.get('MAPE', 0):.2f}%"
            })
        
        st.dataframe(pd.DataFrame(bench_data), use_container_width=True)

        scat_fig = os.path.join(fig_dir, "actual_vs_predicted_scatter.png")
        if os.path.exists(scat_fig):
            st.image(scat_fig, caption="Out-of-Fold Actual vs Predicted Valuation", use_container_width=True)

st.markdown("---")
st.markdown("<div style='text-align: center; color: #64748b; font-size: 13px;'>Automotive Valuation Engine | Unlox Academy ML August 2026 Batch | Developed by Rhydham Shukla</div>", unsafe_allow_html=True)
