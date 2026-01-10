import streamlit as st
from streamlit_geolocation import streamlit_geolocation
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import google.generativeai as genai
import requests

genai.configure(api_key=st.secrets["API_KEY"])
model = genai.GenerativeModel(model_name=st.secrets["MODEL_NAME"])

# Function to load pre-trained models and scaler
@st.cache_resource
def load_models_and_scaler():
    nn_model = keras.models.load_model('best_model.keras')
    xgb_model = joblib.load('xgboost_model.joblib')
    rf_model = joblib.load('random_forest_model_compressed.joblib')
    scaler = joblib.load('minmax_scaler.joblib')
    return nn_model, xgb_model, rf_model, scaler

def predict_aqi(input_values, nn_model, xgb_model, rf_model, scaler):
    input_df = pd.DataFrame([input_values])
    input_scaled = scaler.transform(input_df)
    nn_pred = nn_model.predict(input_scaled).flatten()[0]
    xgb_pred = xgb_model.predict(input_scaled)[0]
    rf_pred = rf_model.predict(input_scaled)[0]
    ensemble_pred = (nn_pred + xgb_pred + rf_pred) / 3
    return ensemble_pred

def fetch_aqi_from_coords(lat, lon):
    try:
        url = st.secrets["WEATHER_URL"].format(lat=lat, lon=lon)
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if 'list' in data and len(data['list']) > 0:
            aqi_index = data['list'][0]['main']['aqi']
            components = data['list'][0]['components']
            aqi = convert_to_aqi(aqi_index, components)
            return aqi, components
        else:
            return None, None
    except Exception as e:
        st.error(f"Error fetching AQI data: {str(e)}")
        return None, None

# OpenWeather scale for Air Quality Index levels
def convert_to_aqi(owm_index, components):
    pm25 = components.get('pm2_5', 0)
    pm10 = components.get('pm10', 0)
    no2 = components.get('no2', 0)
    so2 = components.get('so2', 0)
    co = components.get('co', 0)
    o3 = components.get('o3', 0)
    
    def calc_sub_index(conc, breakpoints):
        for (c_low, c_high, i_low, i_high) in breakpoints:
            if c_low <= conc <= c_high:
                return ((i_high - i_low) / (c_high - c_low)) * (conc - c_low) + i_low
        return breakpoints[-1][3]
    
    pm25_bp = [(0, 30, 0, 50), (31, 60, 51, 100), (61, 90, 101, 200), (91, 120, 201, 300), (121, 250, 301, 400), (251, 500, 401, 500)]
    pm10_bp = [(0, 50, 0, 50), (51, 100, 51, 100), (101, 250, 101, 200), (251, 350, 201, 300), (351, 430, 301, 400), (431, 600, 401, 500)]
    no2_bp = [(0, 40, 0, 50), (41, 80, 51, 100), (81, 180, 101, 200), (181, 280, 201, 300), (281, 400, 301, 400), (401, 800, 401, 500)]
    so2_bp = [(0, 40, 0, 50), (41, 80, 51, 100), (81, 380, 101, 200), (381, 800, 201, 300), (801, 1600, 301, 400), (1601, 2400, 401, 500)]
    co_bp = [(0, 1000, 0, 50), (1001, 2000, 51, 100), (2001, 10000, 101, 200), (10001, 17000, 201, 300), (17001, 34000, 301, 400), (34001, 50000, 401, 500)]
    o3_bp = [(0, 50, 0, 50), (51, 100, 51, 100), (101, 168, 101, 200), (169, 208, 201, 300), (209, 748, 301, 400), (749, 1000, 401, 500)]
    
    sub_indices = [
        calc_sub_index(pm25, pm25_bp),
        calc_sub_index(pm10, pm10_bp),
        calc_sub_index(no2, no2_bp),
        calc_sub_index(so2, so2_bp),
        calc_sub_index(co, co_bp),
        calc_sub_index(o3, o3_bp)
    ]
    
    return int(max(sub_indices))

def construct_prompt(aqi, answers, follow_up_query=None, detailed=False):
    if aqi <= 50:
        category = "Good"
    elif aqi <= 100:
        category = "Fair"
    elif aqi <= 200:
        category = "Moderate"
    elif aqi <= 300:
        category = "Poor"
    elif aqi <= 400:
        category = "Very Poor"
    else:
        category = "Severe"
    aqi_description = f"AQI: {aqi} ({category}) on AQI scale (0-500)"
    
    prompt = f"""
    The current Air Quality: {aqi_description}, indicating a level of air pollution. Below are the key environmental factors:

    - Traffic Intensity: {answers['traffic']}
    - Proximity of Industrial Zones: {answers['industrial']}
    - Green Cover: {answers['green_cover']}
    - Climate Change Mitigation Initiatives: {answers['climate_initiatives']}
    - Zoning and Development: {'Mixed zoning (Residential + Commercial)' if answers['zoning'] == 'Yes' else 'Single-use zoning'}

    - Significant Environmental Challenge: {answers['environmental_challenge']}
    - Recent Initiatives or Unique Aspects: {answers['area_initiatives']}

    """
    if detailed and follow_up_query:
        prompt = f"{prompt}\n\nAdditional Query: {follow_up_query}\nProvide further insights or details in response."
    else:
        prompt = f"{prompt}\n\nBased on this context, provide concise and actionable recommendations to improve air quality, enhance sustainability, and promote public health. Keep the suggestions brief and easy to understand for quick implementation."
    return prompt

def main():
    st.set_page_config(page_title="AeroSense", page_icon="🌍", layout="wide")

    st.markdown(
    """
    <div style="text-align: center;">
        <h1>🌍 AeroSense: Air Quality Intelligence</h1>
        <p>Transforming cities with AI-driven recommendations to combat air pollution, improve sustainability, and enhance public health.</p>
    </div>
    <br>
    """,
    unsafe_allow_html=True
    )

    nn_model, xgb_model, rf_model, scaler = load_models_and_scaler()

    if "aqi_predicted" not in st.session_state:
        st.session_state.aqi_predicted = False
        st.session_state.predicted_aqi = None
        st.session_state.answers = {}
        st.session_state.option = "Predict AQI from Input Parameters"
        st.session_state.show_detailed_insights = False
        st.session_state.location_filled = False
    
    option = st.radio("How would you like to proceed?", 
                      options=["Predict AQI from Input Parameters", "Fetch AQI from Location"], 
                      index=0)

    if option != st.session_state.option:
        st.session_state.aqi_predicted = False
        st.session_state.predicted_aqi = None
        st.session_state.answers = {}
        st.session_state.option = option
    
    if option == "Predict AQI from Input Parameters" and not st.session_state.aqi_predicted:
        with st.expander("Input Parameters for AQI Prediction", expanded=not st.session_state.aqi_predicted):
            col1, col2, col3 = st.columns(3)
            with col1:
                co_gt = st.number_input("CO(GT) - Carbon Monoxide", min_value=0.0, value=1.0)
                nmhc_gt = st.number_input("NMHC(GT) - Non-Methane Hydrocarbons", min_value=0.0, value=1.0)
                c6h6_gt = st.number_input("C6H6(GT) - Benzene", min_value=0.0, value=1.0)
            with col2:
                nox_gt = st.number_input("NOx(GT) - Nitrogen Oxides", min_value=0.0, value=1.0)
                no2_gt = st.number_input("NO2(GT) - Nitrogen Dioxide", min_value=0.0, value=1.0)
                pt08_o3 = st.number_input("PT08.S5(O3) - Ozone Sensor", min_value=0.0, value=1.0)
            with col3:
                temp = st.number_input("T - Temperature (°C)", min_value=-50.0, max_value=50.0, value=20.0)
                humidity = st.number_input("RH - Relative Humidity (%)", min_value=0.0, max_value=100.0, value=50.0)
                absolute_humidity = st.number_input("AH - Absolute Humidity", min_value=0.0, value=5.0)

            input_values = {
                'CO(GT)': co_gt,
                'NMHC(GT)': nmhc_gt,
                'C6H6(GT)': c6h6_gt,
                'NOx(GT)': nox_gt,
                'NO2(GT)': no2_gt,
                'PT08.S5(O3)': pt08_o3,
                'T': temp,
                'RH': humidity,
                'AH': absolute_humidity
            }

            if st.button("Predict Air Quality Index"):
                    predicted_aqi = predict_aqi(input_values, nn_model, xgb_model, rf_model, scaler)
                    st.session_state.aqi_predicted = True
                    st.session_state.predicted_aqi = predicted_aqi

    elif option == "Fetch AQI from Location":
        with st.expander("Fetch AQI from Geolocation or Manual Coordinates", expanded=not st.session_state.aqi_predicted):
            st.markdown("### 🌐 Get Air Quality Data")
            
            if 'manual_aqi_value' not in st.session_state:
                st.session_state.manual_aqi_value = 0.0
            if 'manual_lat' not in st.session_state:
                st.session_state.manual_lat = 0.0
            if 'manual_lon' not in st.session_state:
                st.session_state.manual_lon = 0.0
            if 'location_method' not in st.session_state:
                st.session_state.location_method = "auto"
            
            col_auto, col_manual = st.columns(2)
            with col_auto:
                auto_selected = st.session_state.location_method == "auto"
                if st.button(
                    "🌍 Auto-Detect Location", 
                    use_container_width=True, 
                    type="primary" if auto_selected else "secondary",
                    key="btn_auto"
                ):
                    if st.session_state.location_method != "auto":
                        st.session_state.location_method = "auto"
                        st.session_state.aqi_predicted = False
                        st.session_state.predicted_aqi = None
                        st.session_state.show_detailed_insights = False
                        st.rerun()
            
            with col_manual:
                manual_selected = st.session_state.location_method == "manual"
                if st.button(
                    "📍 Manual Coordinates", 
                    use_container_width=True, 
                    type="primary" if manual_selected else "secondary",
                    key="btn_manual"
                ):
                    if st.session_state.location_method != "manual":
                        st.session_state.location_method = "manual"
                        st.session_state.aqi_predicted = False
                        st.session_state.predicted_aqi = None
                        st.session_state.show_detailed_insights = False
                        st.rerun()
            
            st.markdown("---")
            
            if st.session_state.location_method == "auto":
                st.caption("⚠️ You must allow location access when prompted by your browser")
                
                st.markdown("#### 📍 Step 1: Get Your Location")
                location = streamlit_geolocation()
                
                if location and location.get('latitude') is not None and location.get('longitude') is not None:
                    lat = location['latitude']
                    lon = location['longitude']
                    
                    st.success(f"✅ **Location Detected!**\n\n📍 Latitude: {lat:.6f}° | Longitude: {lon:.6f}°")
                    
                    st.markdown("#### 🌍 Step 2: Fetch Air Quality")
                    if st.button("🌍 Fetch AQI for My Location", use_container_width=True, type="primary", key="fetch_geo_aqi"):
                        with st.spinner('🌍 Fetching air quality data from WeatherMap...'):
                            fetched_aqi, components_data = fetch_aqi_from_coords(lat, lon)
                            
                            if fetched_aqi is not None:
                                st.session_state.predicted_aqi = fetched_aqi
                                st.session_state.aqi_predicted = True
                                
                                st.success(f"✅ **AQI: {fetched_aqi:.2f}**")
                                
                                if components_data:
                                    st.markdown("### 📊 Air Quality Details")
                                    st.caption(f"Location: {lat:.6f}°, {lon:.6f}°")
                                    
                                    col1, col2, col3, col4 = st.columns(4)
                                    with col1:
                                        st.metric("CO", f"{components_data.get('co', 0):.2f}", delta="μg/m³", delta_color="off")
                                        st.metric("PM2.5", f"{components_data.get('pm2_5', 0):.2f}", delta="μg/m³", delta_color="off")
                                    with col2:
                                        st.metric("NO₂", f"{components_data.get('no2', 0):.2f}", delta="μg/m³", delta_color="off")
                                        st.metric("PM10", f"{components_data.get('pm10', 0):.2f}", delta="μg/m³", delta_color="off")
                                    with col3:
                                        st.metric("O₃", f"{components_data.get('o3', 0):.2f}", delta="μg/m³", delta_color="off")
                                        st.metric("SO₂", f"{components_data.get('so2', 0):.2f}", delta="μg/m³", delta_color="off")
                                    with col4:
                                        st.metric("NH₃", f"{components_data.get('nh3', 0):.2f}", delta="μg/m³", delta_color="off")
                                
                                st.markdown("---")
                                st.info("⬇️ **Scroll down to proceed with recommendations**")
                            else:
                                st.error("❌ Failed to fetch AQI data. Please check your API key.")
                else:
                    st.warning("👆 **Click the 'Get Location' button above** to detect your coordinates, then you can fetch AQI.")
            
            else:
                st.caption("Enter latitude and longitude manually or select a city from below")
                
                st.markdown("#### 🏙️ Quick Select: Major Indian Cities")
                
                cities_data = {
                    "Mumbai": (19.0760, 72.8777),
                    "Delhi": (28.6139, 77.2090),
                    "Bangalore": (12.9716, 77.5946),
                    "Hyderabad": (17.3850, 78.4867),
                    "Chennai": (13.0827, 80.2707),
                    "Kolkata": (22.5726, 88.3639),
                    "Pune": (18.5204, 73.8567),
                    "Ahmedabad": (23.0225, 72.5714),
                    "Jaipur": (26.9124, 75.7873),
                    "Lucknow": (26.8467, 80.9462),
                    "Kanpur": (26.4499, 80.3319),
                    "Nagpur": (21.1458, 79.0882)
                }
                
                city_cols = st.columns(4)
                for idx, (city, coords) in enumerate(cities_data.items()):
                    with city_cols[idx % 4]:
                        if st.button(f"📍 {city}", use_container_width=True, key=f"city_{city}"):
                            st.session_state.manual_lat = coords[0]
                            st.session_state.manual_lon = coords[1]
                            st.rerun()
                
                st.markdown("---")
                st.markdown("#### ⌨️ Manual Entry")
                
                if st.session_state.manual_lat != 0.0 or st.session_state.manual_lon != 0.0:
                    st.success(f"📍 **Selected Coordinates:** Lat: {st.session_state.manual_lat:.6f}°, Lon: {st.session_state.manual_lon:.6f}°")
                
                col1, col2 = st.columns(2)
                with col1:
                    new_lat = st.number_input(
                        "Latitude", 
                        min_value=-90.0, 
                        max_value=90.0, 
                        value=float(st.session_state.manual_lat), 
                        format="%.6f",
                        help="Range: -90 to 90"
                    )
                with col2:
                    new_lon = st.number_input(
                        "Longitude", 
                        min_value=-180.0, 
                        max_value=180.0, 
                        value=float(st.session_state.manual_lon), 
                        format="%.6f",
                        help="Range: -180 to 180"
                    )
                
                st.session_state.manual_lat = new_lat
                st.session_state.manual_lon = new_lon
                
                st.markdown("---")
                if st.button("🌍 Fetch AQI with These Coordinates", use_container_width=True, type="primary", key="fetch_manual_aqi"):
                    if new_lat != 0.0 or new_lon != 0.0:
                        with st.spinner('🌍 Fetching air quality data from WeatherMap...'):
                            fetched_aqi, components_data = fetch_aqi_from_coords(new_lat, new_lon)
                            
                            if fetched_aqi is not None:
                                st.session_state.predicted_aqi = fetched_aqi
                                st.session_state.aqi_predicted = True
                                
                                st.success(f"✅ **AQI: {fetched_aqi:.2f}**")
                                
                                if components_data:
                                    st.markdown("### 📊 Air Quality Details")
                                    st.caption(f"Location: {new_lat:.6f}°, {new_lon:.6f}°")
                                    
                                    col1, col2, col3, col4 = st.columns(4)
                                    with col1:
                                        st.metric("CO", f"{components_data.get('co', 0):.2f}", delta="μg/m³", delta_color="off")
                                        st.metric("PM2.5", f"{components_data.get('pm2_5', 0):.2f}", delta="μg/m³", delta_color="off")
                                    with col2:
                                        st.metric("NO₂", f"{components_data.get('no2', 0):.2f}", delta="μg/m³", delta_color="off")
                                        st.metric("PM10", f"{components_data.get('pm10', 0):.2f}", delta="μg/m³", delta_color="off")
                                    with col3:
                                        st.metric("O₃", f"{components_data.get('o3', 0):.2f}", delta="μg/m³", delta_color="off")
                                        st.metric("SO₂", f"{components_data.get('so2', 0):.2f}", delta="μg/m³", delta_color="off")
                                    with col4:
                                        st.metric("NH₃", f"{components_data.get('nh3', 0):.2f}", delta="μg/m³", delta_color="off")
                                
                                st.markdown("---")
                                st.info("⬇️ **Scroll down to proceed with recommendations**")
                            else:
                                st.error("❌ Failed to fetch AQI data. Please check your coordinates and try again.")
                    else:
                        st.warning("⚠️ Please enter valid coordinates (both cannot be zero)")

    if st.session_state.aqi_predicted and st.session_state.predicted_aqi is not None:
        st.markdown("---")
        st.markdown("## 🌍 Air Quality Index Result")
        
        predicted_aqi = st.session_state.predicted_aqi
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.metric("AQI", f"{int(predicted_aqi)}", delta=None)
        
        with col2:
            if predicted_aqi <= 50:
                st.success("🟢 **Good (0-50)**")
                st.caption("Minimal impact. Air quality is satisfactory.")
            elif predicted_aqi <= 100:
                st.success("🟢 **Satisfactory (51-100)**")
                st.caption("Minor breathing discomfort to sensitive people.")
            elif predicted_aqi <= 200:
                st.warning("🟡 **Moderate (101-200)**")
                st.caption("Breathing discomfort to people with lungs, asthma and heart diseases.")
            elif predicted_aqi <= 300:
                st.warning("🟠 **Poor (201-300)**")
                st.caption("Breathing discomfort on prolonged exposure. Avoid outdoor activities.")
            elif predicted_aqi <= 400:
                st.error("🔴 **Very Poor (301-400)**")
                st.caption("Respiratory illness on prolonged exposure. Avoid outdoor activities.")
            else:
                st.error("🟤 **Severe (401-500)**")
                st.caption("Affects healthy people. Serious impact on people with heart/lung disease.")

    if st.session_state.aqi_predicted:
        with st.expander("Provide Insights for Recommendations", expanded=st.session_state.aqi_predicted):
            st.session_state.answers = {}
            st.session_state.answers['traffic'] = st.selectbox("Traffic Intensity", ["Select...", "Very High", "Moderate", "Low"], index=0)
            st.session_state.answers['industrial'] = st.selectbox("Proximity to Industrial Zones", ["Select...", "Within 5 km", "Within 10 km", "More than 10 km away"], index=0)
            st.session_state.answers['green_cover'] = st.selectbox("Green Cover", ["Select...", "Sparse", "Moderate", "Dense"], index=0)
            st.session_state.answers['climate_initiatives'] = st.selectbox("Climate Change Initiatives", ["Select...", "None", "A few", "Actively implemented"], index=0)
            st.session_state.answers['zoning'] = st.radio("Mixed Zoning (Residential + Commercial)", ["Yes", "No"], index=None)
            st.session_state.answers['environmental_challenge'] = st.text_area("Significant Environmental Challenge", key="environmental_challenge", placeholder="For example, too much dust due to nearby construction or heavy traffic on main roads.")
            st.session_state.answers['area_initiatives'] = st.text_area("Recent Initiatives or Unique Aspects", key="area_initiatives", placeholder="For example, a new metro station is being built nearby.")

            if st.button("Get Recommendations"):
                all_selected = (
                    st.session_state.answers['traffic'] != "Select..." and
                    st.session_state.answers['industrial'] != "Select..." and
                    st.session_state.answers['green_cover'] != "Select..." and
                    st.session_state.answers['climate_initiatives'] != "Select..." and
                    st.session_state.answers['zoning'] is not None
                )
                if not all_selected:
                    st.error("Please select an option for all fields.")
                elif not st.session_state.answers['environmental_challenge'] or not st.session_state.answers['area_initiatives']:
                    st.error("Please fill in all text fields.")
                else:
                    prompt = construct_prompt(st.session_state.predicted_aqi, st.session_state.answers)
                    try:
                        response = model.generate_content(prompt)
                        st.write("### Recommendations")
                        st.write(response.text)
                        st.session_state.show_detailed_insights=True
                    except Exception as e:
                        st.error(f"Error generating insights: {e}")

        if st.session_state.show_detailed_insights:
            st.subheader("Ask More About These Recommendations")
            follow_up_query = st.text_input("Have a follow-up question? Ask here:", key="follow_up_query")
            
            if st.button("Get Detailed Insights"):
                if follow_up_query.strip():
                    follow_up_prompt = construct_prompt(st.session_state.predicted_aqi, st.session_state.answers, follow_up_query)
                    try:
                        follow_up_response = model.generate_content(follow_up_prompt)
                        st.write("### Detailed Insights")
                        st.write(follow_up_response.text)
                    except Exception as e:
                        st.error(f"Error generating insights: {e}")
                else:
                    st.error("Please enter a valid question.")

if __name__ == "__main__":
    main()