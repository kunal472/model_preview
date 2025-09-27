# app.py
import streamlit as st
import pandas as pd
from prediction_service import PredictionService # Import your class

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="FarmSensei Pro",
    page_icon="🌾",
    layout="wide"
)

# --- LOAD PREDICTION SERVICE ---
# Use caching to load the model only once
@st.cache_resource
def load_prediction_service():
    """Loads and caches the PredictionService instance."""
    try:
        service = PredictionService(model_dir='models')
        return service
    except Exception as e:
        st.error(f"Error loading prediction service: {e}")
        return None

prediction_service = load_prediction_service()

# --- UI LAYOUT ---
st.title("🌾 FarmSensei Pro")
st.markdown("Enter your farm's environmental and soil conditions to get personalized crop recommendations.")

# --- SIDEBAR FOR USER INPUT ---
st.sidebar.header("Enter Live Farm Data")

with st.sidebar:
    st.subheader("Soil Conditions")
    n = st.number_input("Nitrogen (N) Content (kg/ha)", min_value=0, max_value=200, value=90, help="Topsoil Nitrogen content.")
    p = st.number_input("Phosphorus (P) Content (kg/ha)", min_value=0, max_value=200, value=45)
    k = st.number_input("Potassium (K) Content (kg/ha)", min_value=0, max_value=200, value=45)
    ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5, format="%.2f")

    st.subheader("Environmental Factors")
    temp = st.number_input("Average Temperature (°C)", min_value=-10.0, max_value=50.0, value=25.5, format="%.2f")
    humidity = st.number_input("Average Relative Humidity (%)", min_value=0.0, max_value=100.0, value=75.0, format="%.2f")
    rainfall = st.number_input("Total Rainfall (mm)", min_value=0.0, max_value=500.0, value=150.0, format="%.2f", help="Rainfall during the crop season.")
    
    st.subheader("Market Info (Optional)")
    market_price = st.number_input("Average Market Price (₹ per Quintal)", min_value=1000, max_value=10000, value=2500, help="Optional: Provide a target market price.")

# --- PREDICTION LOGIC ---
if st.button("🌱 Get Recommendations", type="primary"):
    if prediction_service is None:
        st.error("The prediction service is not available. Please check the logs.")
    else:
        # Prepare the features dictionary from user inputs
        live_features = {
            'topsoil_nitrogen': n,
            'P_placeholder': p,
            'K_placeholder': k,
            'avg_temp_celsius': temp,
            'avg_humidity_percent': humidity,
            'topsoil_phh2o': ph,
            'total_rainfall_mm': rainfall,
            'avg_modal_price': market_price
        }

        with st.spinner("Analyzing your farm data and running models..."):
            try:
                # Get the final recommendations
                recommendations = prediction_service.get_final_recommendations(live_features)
                
                if recommendations:
                    st.success("Here are your top 5 crop recommendations:")
                    
                    # Convert to DataFrame for better display
                    df = pd.DataFrame(recommendations)
                    
                    # Prettify column names for the UI
                    df.rename(columns={
                        'crop': 'Recommended Crop',
                        'predicted_yield_quintal_per_hectare': 'Predicted Yield (Quintal/Hectare)',
                        'sustainability_score_10': 'Sustainability Score (/10)',
                        'estimated_profit_rs_per_hectare': 'Estimated Profit (₹/Hectare)'
                    }, inplace=True)

                    # Display the results in a table
                    st.dataframe(
                        df.style.format({
                            'Predicted Yield (Quintal/Hectare)': '{:.2f}',
                            'Sustainability Score (/10)': '{:.2f}',
                            'Estimated Profit (₹/Hectare)': '₹ {:,.2f}'
                        }).highlight_max(
                            subset=['Predicted Yield (Quintal/Hectare)', 'Sustainability Score (/10)', 'Estimated Profit (₹/Hectare)'], 
                            color='#D4EDDA'
                        ),
                        use_container_width=True
                    )
                else:
                    st.warning("Could not generate recommendations for the given inputs.")
            
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")