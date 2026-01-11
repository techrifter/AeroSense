import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import google.generativeai as genai
import requests
import re

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

def search_aqi_location(location):
    try:
        search_url = f"{st.secrets['SEARCH_URL']}/{location}"
        response = requests.get(search_url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if 'results' in data and len(data['results']) > 0:
            stations = []
            for result in data['results']:
                if 's' in result and 'a' in result['s'] and result['s']['a'] != '-':
                    station = {
                        'aqi': result['s']['a'],
                        'name': result['s']['n'][0] if result['s']['n'] else result['n'][0],
                        'url': result['s'].get('u', ''),
                        'location': result['n'][-1] if len(result['n']) > 1 else result['n'][0]
                    }
                    
                    if station['url']:
                        match = re.search(r'@(\d+)', station['url'])
                        if match:
                            station['station_id'] = 'A' + match.group(1)
                            stations.append(station)
            
            return stations
        return []
    except Exception as e:
        st.error(f"Error searching location: {str(e)}")
        return []

def fetch_aqi_from_station(station_id):
    try:
        token = st.secrets["AQI_TOKEN"]
        feed_url = f"{st.secrets['FEED_URL']}/{station_id}/?token={token}"
        response = requests.get(feed_url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data['status'] == 'ok' and 'data' in data:
            aqi_data = data['data']
            return {
                'aqi': aqi_data.get('aqi', 0),
                'city': aqi_data.get('city', {}).get('name', ''),
                'location': aqi_data.get('city', {}).get('location', ''),
                'dominentpol': aqi_data.get('dominentpol', ''),
                'components': aqi_data.get('iaqi', {}),
                'time': aqi_data.get('time', {}).get('s', '')
            }
        return None
    except Exception as e:
        st.error(f"Error fetching AQI data: {str(e)}")
        return None

def construct_prompt(aqi, answers, follow_up_query=None, detailed=False):
    if aqi <= 50:
        category = "Good"
    elif aqi <= 100:
        category = "Satisfactory"
    elif aqi <= 200:
        category = "Moderate"
    elif aqi <= 300:
        category = "Poor"
    elif aqi <= 400:
        category = "Very Poor"
    else:
        category = "Severe"
    aqi_description = f"AQI: {aqi} ({category}) on AQI scale (0-500)"
    
    zoning_text = 'Mixed zoning (Residential + Commercial)' if answers['zoning'] == 'Yes' else 'Single-use zoning'
    
    prompt = st.secrets["BASE_PROMPT"].format(
        aqi_description=aqi_description,
        traffic=answers['traffic'],
        industrial=answers['industrial'],
        green_cover=answers['green_cover'],
        climate_initiatives=answers['climate_initiatives'],
        zoning=zoning_text,
        environmental_challenge=answers['environmental_challenge'],
        area_initiatives=answers['area_initiatives']
    )
    
    if detailed and follow_up_query:
        prompt = f"{prompt}\n\n{st.secrets['FOLLOWUP_PROMPT'].format(follow_up_query=follow_up_query)}"
    else:
        prompt = f"{prompt}\n\n{st.secrets['RECOMMENDATION_PROMPT']}"
    return prompt

def get_pollutant_class(value):
    """Determine CSS class based on pollutant value (AQI sub-index)"""
    try:
        val = float(value)
        if val <= 50:
            return "aqi-card-good"
        elif val <= 100:
            return "aqi-card-moderate"
        elif val <= 150:
            return "aqi-card-usg"
        elif val <= 200:
            return "aqi-card-unhealthy"
        elif val <= 300:
            return "aqi-card-very-unhealthy"
        else:
            return "aqi-card-hazardous"
    except:
        return "aqi-card"

def main():
    st.set_page_config(page_title="AeroSense", page_icon="🌍", layout="wide")

    # Load external CSS
    with open('assets/style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    st.markdown(
    """
    <div class="main-header">
        <h1>🌍 AeroSense: Air Quality Intelligence</h1>
        <p>Transforming cities to combat air pollution, improve sustainability, and enhance public health.</p>
    </div>
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
        st.session_state.scroll_to_aqi = False
        st.session_state.scroll_to_results = False
    
    st.markdown("<br>", unsafe_allow_html=True)
    option = st.radio(
        "**How would you like to proceed?**", 
        options=["Predict AQI from Input Parameters", "Fetch AQI from Location"], 
        index=0,
        horizontal=True
    )

    if option != st.session_state.option:
        st.session_state.aqi_predicted = False
        st.session_state.predicted_aqi = None
        st.session_state.answers = {}
        st.session_state.option = option
        # Clear location-specific data when switching options
        if hasattr(st.session_state, 'aqi_city'):
            delattr(st.session_state, 'aqi_city')
        if hasattr(st.session_state, 'aqi_location'):
            delattr(st.session_state, 'aqi_location')
        if hasattr(st.session_state, 'aqi_components'):
            delattr(st.session_state, 'aqi_components')
    
    if option == "Predict AQI from Input Parameters" and not st.session_state.aqi_predicted:
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("📊 Input Parameters for AQI Prediction", expanded=not st.session_state.aqi_predicted):
            st.markdown("**Enter Environmental Parameters**")
            st.caption("Provide accurate measurements for precise AQI prediction(ML-powered, experimental)")
            st.markdown("---")
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
                    # Clear location-specific data when predicting from input
                    if hasattr(st.session_state, 'aqi_city'):
                        delattr(st.session_state, 'aqi_city')
                    if hasattr(st.session_state, 'aqi_location'):
                        delattr(st.session_state, 'aqi_location')
                    if hasattr(st.session_state, 'aqi_components'):
                        delattr(st.session_state, 'aqi_components')

    elif option == "Fetch AQI from Location":
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("🔍 Search Location for AQI Data", expanded=not st.session_state.aqi_predicted):
            st.markdown("**🌍 Find your Region**")
            st.caption("Enter a city name or location to find nearby air quality monitoring stations")
            st.markdown("<br>", unsafe_allow_html=True)
            
            if 'selected_station' not in st.session_state:
                st.session_state.selected_station = None
            if 'search_results' not in st.session_state:
                st.session_state.search_results = []
            if 'last_search_term' not in st.session_state:
                st.session_state.last_search_term = ""
            
            location_search = st.text_input(
                "Location",
                placeholder="🌆 Type a city name (e.g., Mumbai, Delhi, Bangalore) and press Enter...",
                key="location_search",
                label_visibility="collapsed"
            )
            
            st.markdown("<br>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns([2, 1, 2])
            with col2:
                search_button = st.button("🔍 Search Location", use_container_width=True, type="primary")
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Trigger search on Enter key (when location_search changes) or button click
            should_search = (search_button and location_search) or \
                           (location_search and location_search != st.session_state.last_search_term and location_search.strip() != "")
            
            if should_search:
                st.session_state.last_search_term = location_search
                with st.spinner(f'Searching for "{location_search}"...'):
                    stations = search_aqi_location(location_search)
                    if stations:
                        st.session_state.search_results = stations
                        st.session_state.scroll_to_results = True
                        st.success(f"Found {len(stations)} monitoring station(s)")
                    else:
                        st.warning("No monitoring stations found for this location. Try a different search term.")
            
            if st.session_state.search_results:
                # Add anchor for scrolling to results
                st.markdown("<div id='search-results-section'></div>", unsafe_allow_html=True)
                
                # Auto-scroll to results
                if st.session_state.get('scroll_to_results', False):
                    import streamlit.components.v1 as components
                    components.html(
                        """
                        <script>
                            window.parent.document.getElementById('search-results-section').scrollIntoView({ 
                                behavior: 'smooth', 
                                block: 'start' 
                            });
                        </script>
                        """,
                        height=0
                    )
                    st.session_state.scroll_to_results = False
                
                st.markdown("---")
                st.markdown("**📍 Available Monitoring Stations**")
                st.caption(f"{len(st.session_state.search_results)} station(s) found")
                
                for idx, station in enumerate(st.session_state.search_results[:10]):
                    col_aqi, col_name, col_btn = st.columns([1, 4, 1])
                    
                    with col_aqi:
                        aqi_val = int(station['aqi'])
                        if aqi_val <= 50:
                            st.metric("AQI", aqi_val, delta="Good", delta_color="normal")
                        elif aqi_val <= 100:
                            st.metric("AQI", aqi_val, delta="Satisfactory", delta_color="normal")
                        elif aqi_val <= 200:
                            st.metric("AQI", aqi_val, delta="Moderate", delta_color="inverse")
                        elif aqi_val <= 300:
                            st.metric("AQI", aqi_val, delta="Poor", delta_color="inverse")
                        elif aqi_val <= 400:
                            st.metric("AQI", aqi_val, delta="Very Poor", delta_color="inverse")
                        else:
                            st.metric("AQI", aqi_val, delta="Severe", delta_color="inverse")
                    
                    with col_name:
                        st.markdown(f"**{station['name']}**")
                        st.caption(station['location'])
                    
                    with col_btn:
                        if st.button("Select", key=f"select_station_{idx}", use_container_width=True):
                            st.session_state.selected_station = station
                            with st.spinner('Fetching detailed AQI data...'):
                                detailed_data = fetch_aqi_from_station(station['station_id'])
                                
                                if detailed_data:
                                    st.session_state.predicted_aqi = detailed_data['aqi']
                                    st.session_state.aqi_predicted = True
                                    st.session_state.aqi_components = detailed_data['components']
                                    st.session_state.aqi_city = detailed_data['city']
                                    st.session_state.aqi_location = detailed_data['location']
                                    st.session_state.scroll_to_aqi = True
                                    st.rerun()
                                else:
                                    st.error("Failed to fetch detailed AQI data from this station")
                    
                    st.markdown("---")

    if st.session_state.aqi_predicted and st.session_state.predicted_aqi is not None:
        # Add anchor for scrolling
        st.markdown("<div id='aqi-result-section'></div>", unsafe_allow_html=True)
        
        # Auto-scroll using components
        if st.session_state.get('scroll_to_aqi', False):
            import streamlit.components.v1 as components
            components.html(
                """
                <script>
                    window.parent.document.getElementById('aqi-result-section').scrollIntoView({ 
                        behavior: 'smooth', 
                        block: 'start' 
                    });
                </script>
                """,
                height=0
            )
            st.session_state.scroll_to_aqi = False
        
        st.markdown("---")
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Location information
        if hasattr(st.session_state, 'aqi_city') and st.session_state.aqi_city:
            st.markdown(f"### 📍 {st.session_state.aqi_city}")
            if hasattr(st.session_state, 'aqi_location') and st.session_state.aqi_location:
                st.caption(f"🗺️ {st.session_state.aqi_location}")
        else:
            st.markdown("### 🌍 Air Quality Index Result")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        predicted_aqi = st.session_state.predicted_aqi
        
        # Determine AQI color class
        if predicted_aqi <= 50:
            aqi_card_class = "aqi-card-good"
            health_class = "health-info-good"
        elif predicted_aqi <= 100:
            aqi_card_class = "aqi-card-moderate"
            health_class = "health-info-moderate"
        elif predicted_aqi <= 150:
            aqi_card_class = "aqi-card-usg"
            health_class = "health-info-usg"
        elif predicted_aqi <= 200:
            aqi_card_class = "aqi-card-unhealthy"
            health_class = "health-info-unhealthy"
        elif predicted_aqi <= 300:
            aqi_card_class = "aqi-card-very-unhealthy"
            health_class = "health-info-very-unhealthy"
        else:
            aqi_card_class = "aqi-card-hazardous"
            health_class = "health-info-hazardous"
        
        # AQI Display Card
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.markdown(f"<div class='{aqi_card_class}'>", unsafe_allow_html=True)
            # Show disclaimer only for location-fetched data (from AQICN)
            show_disclaimer = hasattr(st.session_state, 'aqi_city') or hasattr(st.session_state, 'aqi_location')
            
            if show_disclaimer:
                st.markdown("""
                    <div style="display: flex; align-items: center; justify-content: center;">
                        <div style="text-align: center;">
                            <div style="font-size: 0.875rem; color: rgba(0,0,0,0.6); font-weight: 600; margin-bottom: 4px;">Current AQI</div>
                            <div style="font-size: 2.5rem; font-weight: 700; display: inline-block;">{}</div>
                            <div class="tooltip-container" style="display: inline-block; vertical-align: super;">
                                <span class="tooltip-icon">?</span>
                                <span class="tooltip-text">
                                    <strong>ℹ️ Data Disclaimer</strong><br><br>
                                    AQI data sourced from <strong>AQICN</strong> (Real-time Air Quality Index).<br><br>
                                    <strong>⚠️ Important:</strong><br>
                                    • We do not claim ownership of this data<br>
                                    • Always refer to official sources for critical decisions
                                </span>
                            </div>
                        </div>
                    </div>
                """.format(int(predicted_aqi)), unsafe_allow_html=True)
            else:
                # No disclaimer for ML-predicted AQI
                st.metric("Current AQI", f"{int(predicted_aqi)}", delta=None)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"<div class='{health_class}'>", unsafe_allow_html=True)
            if predicted_aqi <= 50:
                st.success("🟢 **Good (0-50)**")
                st.markdown("**Health Impact:** Air quality meets standards with minimal health risk.")
                st.markdown("**Recommendation:** Perfect day for outdoor activities. No precautions needed for general population.")
            elif predicted_aqi <= 100:
                st.info("🟡 **Moderate (51-100)**")
                st.markdown("**Health Impact:** Acceptable air quality. Sensitive individuals may experience minor symptoms.")
                st.markdown("**Recommendation:** Unusually sensitive people should consider limiting extended outdoor activities.")
            elif predicted_aqi <= 150:
                st.warning("🟠 **Unhealthy for Sensitive Groups (101-150)**")
                st.markdown("**Health Impact:** Sensitive groups may experience health effects; general public less likely to be affected.")
                st.markdown("**Recommendation:** Those with respiratory conditions should reduce prolonged outdoor activity. General public can continue normal activities.")
            elif predicted_aqi <= 200:
                st.warning("🟠 **Unhealthy (151-200)**")
                st.markdown("**Health Impact:** Everyone may start experiencing health effects; sensitive groups face more serious impacts.")
                st.markdown("**Recommendation:** People with respiratory or heart conditions should avoid prolonged outdoor activity. Everyone else should limit extended outdoor exertion.")
            elif predicted_aqi <= 300:
                st.error("🔴 **Very Unhealthy (201-300)**")
                st.markdown("**Health Impact:** Health alert conditions. Everyone may experience more significant health effects.")
                st.markdown("**Recommendation:** People with respiratory or heart conditions should avoid all outdoor activity. Everyone else should severely limit outdoor exposure.")
            else:
                st.error("🟤 **Hazardous (300+)**")
                st.markdown("**Health Impact:** Emergency health warnings. Entire population at risk of serious health effects.")
                st.markdown("**Recommendation:** Everyone should avoid all outdoor physical activity. Stay indoors with air filtration if possible. Seek medical attention for any symptoms.")
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Pollutant breakdown
        if hasattr(st.session_state, 'aqi_components') and st.session_state.aqi_components:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 📊 Pollutant Breakdown")
            st.caption("Individual pollutant sub-indices contributing to overall AQI")
            components = st.session_state.aqi_components
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if 'pm25' in components:
                    pm25_val = components['pm25'].get('v', 0)
                    pm25_class = get_pollutant_class(pm25_val)
                    st.markdown(f"<div class='{pm25_class}'>", unsafe_allow_html=True)
                    st.metric("PM2.5", pm25_val if pm25_val != 0 else 'N/A', delta="Fine Particles", delta_color="off")
                    st.markdown("</div>", unsafe_allow_html=True)
                if 'pm10' in components:
                    pm10_val = components['pm10'].get('v', 0)
                    pm10_class = get_pollutant_class(pm10_val)
                    st.markdown(f"<div class='{pm10_class}'>", unsafe_allow_html=True)
                    st.metric("PM10", pm10_val if pm10_val != 0 else 'N/A', delta="Coarse Particles", delta_color="off")
                    st.markdown("</div>", unsafe_allow_html=True)
            with col2:
                if 'no2' in components:
                    no2_val = components['no2'].get('v', 0)
                    no2_class = get_pollutant_class(no2_val)
                    st.markdown(f"<div class='{no2_class}'>", unsafe_allow_html=True)
                    st.metric("NO₂", no2_val if no2_val != 0 else 'N/A', delta="Nitrogen Dioxide", delta_color="off")
                    st.markdown("</div>", unsafe_allow_html=True)
                if 'so2' in components:
                    so2_val = components['so2'].get('v', 0)
                    so2_class = get_pollutant_class(so2_val)
                    st.markdown(f"<div class='{so2_class}'>", unsafe_allow_html=True)
                    st.metric("SO₂", so2_val if so2_val != 0 else 'N/A', delta="Sulfur Dioxide", delta_color="off")
                    st.markdown("</div>", unsafe_allow_html=True)
            with col3:
                if 'o3' in components:
                    o3_val = components['o3'].get('v', 0)
                    o3_class = get_pollutant_class(o3_val)
                    st.markdown(f"<div class='{o3_class}'>", unsafe_allow_html=True)
                    st.metric("O₃", o3_val if o3_val != 0 else 'N/A', delta="Ozone", delta_color="off")
                    st.markdown("</div>", unsafe_allow_html=True)
                if 'co' in components:
                    co_val = components['co'].get('v', 0)
                    co_class = get_pollutant_class(co_val)
                    st.markdown(f"<div class='{co_class}'>", unsafe_allow_html=True)
                    st.metric("CO", co_val if co_val != 0 else 'N/A', delta="Carbon Monoxide", delta_color="off")
                    st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.aqi_predicted:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("---")
        with st.expander("🤖 Get AI-Powered Recommendations", expanded=False):
            st.markdown("**Provide Context for Personalized Recommendations**")
            st.caption("Share information about your area to receive tailored air quality improvement suggestions")
            st.markdown("---")
            st.session_state.answers = {}
            st.session_state.answers['traffic'] = st.selectbox("Traffic Intensity", ["Select...", "Very High", "Moderate", "Low"], index=0)
            st.session_state.answers['industrial'] = st.selectbox("Proximity to Industrial Zones", ["Select...", "Within 5 km", "Within 10 km", "More than 10 km away"], index=0)
            st.session_state.answers['green_cover'] = st.selectbox("Green Cover", ["Select...", "Sparse", "Moderate", "Dense"], index=0)
            st.session_state.answers['climate_initiatives'] = st.selectbox("Climate Change Initiatives", ["Select...", "None", "A few", "Actively implemented"], index=0)
            st.session_state.answers['zoning'] = st.radio("Mixed Zoning (Residential + Commercial)", ["Yes", "No"], index=None)
            st.session_state.answers['environmental_challenge'] = st.text_area("Significant Environmental Challenge", key="environmental_challenge", placeholder="For example, too much dust due to nearby construction or heavy traffic on main roads.")
            st.session_state.answers['area_initiatives'] = st.text_area("Recent Initiatives or Unique Aspects", key="area_initiatives", placeholder="For example, a new metro station is being built nearby.")

            if st.button("✨ Generate Recommendations", type="primary", use_container_width=True):
                all_selected = (
                    st.session_state.answers['traffic'] != "Select..." and
                    st.session_state.answers['industrial'] != "Select..." and
                    st.session_state.answers['green_cover'] != "Select..." and
                    st.session_state.answers['climate_initiatives'] != "Select..." and
                    st.session_state.answers['zoning'] is not None
                )
                if not all_selected:
                    st.error("⚠️ Please select an option for all dropdown fields.")
                elif not st.session_state.answers['environmental_challenge'] or not st.session_state.answers['area_initiatives']:
                    st.error("⚠️ Please fill in all text fields.")
                else:
                    with st.spinner('🔄 Analyzing environmental data and generating personalized recommendations...'):
                        prompt = construct_prompt(st.session_state.predicted_aqi, st.session_state.answers)
                        try:
                            response = model.generate_content(prompt)
                            st.markdown("<br>", unsafe_allow_html=True)
                            st.markdown("### 💡 AI-Generated Recommendations")
                            st.write(response.text)
                            st.markdown("</div>", unsafe_allow_html=True)
                            st.session_state.show_detailed_insights=True
                        except Exception as e:
                            st.error(f"❌ Error generating insights: {e}")

        if st.session_state.show_detailed_insights:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 💬 Follow-up Questions")
            st.caption("Ask specific questions about the recommendations above")
            follow_up_query = st.text_input("Have a follow-up question? Ask here:", key="follow_up_query", placeholder="e.g., How can I implement these recommendations in my community?")
            
            if st.button("🔍 Get Detailed Insights", type="primary", use_container_width=True):
                if follow_up_query.strip():
                    with st.spinner('🔄 Generating detailed insights...'):
                        follow_up_prompt = construct_prompt(st.session_state.predicted_aqi, st.session_state.answers, follow_up_query)
                        try:
                            follow_up_response = model.generate_content(follow_up_prompt)
                            st.markdown("<br>", unsafe_allow_html=True)
                            st.markdown("### 🧭 Detailed Answer")
                            st.write(follow_up_response.text)
                            st.markdown("</div>", unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"❌ Error generating insights: {e}")
                else:
                    st.error("⚠️ Please enter a valid question.")

if __name__ == "__main__":
    main()