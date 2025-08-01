# store_locator.py
import streamlit as st
import google.generativeai as genai
from geopy.geocoders import Nominatim
import folium
from streamlit_folium import st_folium
from streamlit_js_eval import get_geolocation
import os
import json
import re
from dotenv import load_dotenv
import speech_recognition as sr
from io import BytesIO

# Load environment variables from .env file
load_dotenv()

# Set Streamlit page config
st.set_page_config(page_title="Nearby Airtel Store Locator", layout="wide")


geolocator = Nominatim(user_agent="airtel_store_locator")

# Initialize session state with clear separation
if 'search_state' not in st.session_state:
    st.session_state.search_state = "init"  # init, searching, results, error
    st.session_state.stores = []
    st.session_state.current_location = None
    st.session_state.radius = 5
    st.session_state.gemini_api_key = os.getenv("GEMINI_API_KEY")
    st.session_state.gemini_configured = False
    st.session_state.gemini_model = None
    st.session_state.voice_input = ""
    st.session_state.recording = False

# Configure Gemini API using .env key
if st.session_state.gemini_api_key and not st.session_state.gemini_configured:
    try:
        genai.configure(api_key=st.session_state.gemini_api_key)
        st.session_state.gemini_configured = True
        st.session_state.gemini_model = genai.GenerativeModel('gemini-2.0-flash')
    except Exception as e:
        st.error(f"Error configuring Gemini: {e}")

# Robust JSON extraction function
def extract_json_from_text(text):
    """Extract JSON content from Gemini's response"""
    try:
        # Clean the text
        clean_text = text.strip()
        
        # Remove markdown code blocks
        clean_text = re.sub(r'```json|```', '', clean_text)
        
        # Find the first { and last } to capture the JSON
        start_idx = clean_text.find('{')
        end_idx = clean_text.rfind('}') + 1
        
        if start_idx == -1 or end_idx == 0:
            return None
            
        json_str = clean_text[start_idx:end_idx]
        return json.loads(json_str)
    except Exception as e:
        st.error(f"JSON extraction error: {e}")
        return None

# Function to call Gemini and get Airtel stores
def get_airtel_stores(lat, lon, radius):
    if not st.session_state.gemini_model:
        return []

    prompt = f"""
    Find all Airtel stores within a {radius} km radius of these coordinates: 
    Latitude: {lat}, Longitude: {lon}.
    
    Return ONLY a JSON object with the following structure:
    {{
        "stores": [
            {{
                "name": "Store name",
                "address": "Full store address",
                "latitude": 12.3456,
                "longitude": 77.6789,
                "distance_km": 1.2
            }}
        ]
    }}
    
    Important:
    1. Return ONLY the JSON object with no additional text
    2. Include exact latitude/longitude coordinates
    3. Calculate distance in km
    4. Return at least 5 stores if available
    """
    try:
        response = st.session_state.gemini_model.generate_content(prompt)
        json_text = "".join(part.text for part in response.candidates[0].content.parts)
        
        # Extract and parse JSON
        data = extract_json_from_text(json_text)
        if not data:
            return []
            
        return data.get("stores", [])
    except Exception as e:
        st.error(f"⚠️ Gemini API error: {e}")
        return []

# Convert address to coordinates
def get_coordinates_from_address(address):
    try:
        location = geolocator.geocode(address)
        return (location.latitude, location.longitude) if location else None
    except Exception as e:
        st.error(f"Geocoding error: {e}")
        return None

# Function to handle voice input
def recognize_speech():
    r = sr.Recognizer()
    
    with st.spinner("Listening... Speak now"):
        with sr.Microphone() as source:
            audio = r.listen(source)
            
        try:
            text = r.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            st.error("Could not understand audio")
        except sr.RequestError as e:
            st.error(f"Could not request results: {e}")
    
    return None

# --- UI ---
st.title("📍 Airtel Store Locator")
st.markdown("Find nearby Airtel stores using your location, voice command, or by entering an address.")

# Create two columns for input methods
col1, col2 = st.columns(2)

with col1:
    st.subheader("📍 Use My Location")
    detect_btn = st.button("Detect My Location", key="detect", use_container_width=True)

with col2:
    st.subheader("🔍 Search by Address")
    address_col, voice_col = st.columns([4, 1])
    
    with address_col:
        address = st.text_input(
            "Enter address or city:",
            value=st.session_state.voice_input,
            key="address_input",
            label_visibility="collapsed",
            placeholder="Enter address or use voice"
        )
    
    with voice_col:
        voice_btn = st.button("🎤", key="voice_btn", use_container_width=True)
    
    search_btn = st.button("Find Stores", key="search", use_container_width=True)

# Radius slider
radius = st.slider("Search radius (km):", 1, 30, st.session_state.radius, key="radius_slider")
if radius != st.session_state.radius:
    st.session_state.radius = radius
    if st.session_state.search_state == "results":
        st.session_state.search_state = "init"

# Handle voice input
if voice_btn:
    st.session_state.recording = True
    st.session_state.voice_input = recognize_speech() or ""
    st.session_state.recording = False
    st.rerun()

# Handle location detection
if detect_btn:
    st.session_state.search_state = "searching"
    try:
        location = get_geolocation()
        if location and 'coords' in location:
            st.session_state.current_location = (
                location['coords']['latitude'],
                location['coords']['longitude']
            )
            st.success("📍 Location detected successfully!")
        else:
            st.error("Could not detect location. Try again or use address.")
            st.session_state.search_state = "error"
    except Exception as e:
        st.error(f"Location error: {e}")
        st.session_state.search_state = "error"

# Handle address search
if search_btn:
    if address:
        st.session_state.search_state = "searching"
        coords = get_coordinates_from_address(address)
        if coords:
            st.session_state.current_location = coords
            st.success(f"📍 Location found: {address}")
        else:
            st.error("Could not find address. Try another.")
            st.session_state.search_state = "error"
    else:
        st.error("Please enter a valid address.")
        st.session_state.search_state = "error"

# Perform search if needed
if (st.session_state.search_state == "searching" and 
    st.session_state.current_location and
    st.session_state.gemini_configured):
    
    lat, lon = st.session_state.current_location
    with st.spinner(f"🔍 Searching for Airtel stores within {st.session_state.radius} km..."):
        st.session_state.stores = get_airtel_stores(lat, lon, st.session_state.radius)
        st.session_state.search_state = "results" if st.session_state.stores else "no_results"

# Display results
if st.session_state.search_state == "results":
    lat, lon = st.session_state.current_location
    stores = st.session_state.stores
    
    # Create map
    m = folium.Map(location=[lat, lon], zoom_start=12)
    
    # Add user location marker
    folium.Marker(
        [lat, lon], 
        popup="Your Location", 
        icon=folium.Icon(color='blue', icon='user')
    ).add_to(m)
    
    # Add store markers
    for store in stores:
        folium.Marker(
            [store["latitude"], store["longitude"]],
            popup=f"<b>{store['name']}</b><br>{store['address']}",
            icon=folium.Icon(color='red', icon='shopping-cart')
        ).add_to(m)
    
    # Fit map to show all markers
    if stores:
        bounds = [[lat, lon]] + [[s["latitude"], s["longitude"]] for s in stores]
        m.fit_bounds(bounds)
    
    # Display map
    st_folium(m, width=1200, height=500)

    # Show store list
    st.subheader(f"📌 Found {len(stores)} Airtel stores:")
    for i, store in enumerate(stores, 1):
        gmap_link = f"https://www.google.com/maps/search/?api=1&query={store['latitude']},{store['longitude']}"
        st.markdown(f"""
        <div style="border:1px solid #e0e0e0; border-radius:10px; padding:15px; margin-bottom:15px;">
            <h4>{i}. {store['name']}</h4>
            <p>📍 <b>Address:</b> {store['address']}</p>
            <p>📏 <b>Distance:</b> {store['distance_km']:.1f} km</p>
            <a href="{gmap_link}" target="_blank" style="text-decoration:none;">
                <button style="background-color:#4285F4; color:white; border:none; padding:8px 15px; border-radius:4px; cursor:pointer;">
                    🗺️ View on Google Maps
                </button>
            </a>
        </div>
        """, unsafe_allow_html=True)
    
    # Add reset button
    if st.button("New Search"):
        st.session_state.search_state = "init"
        st.session_state.current_location = None
        st.session_state.stores = []
        st.session_state.voice_input = ""
        st.rerun()

elif st.session_state.search_state == "no_results":
    st.warning("No Airtel stores found. Try a different location or increase the radius.")
    if st.button("Try Again"):
        st.session_state.search_state = "init"
        st.rerun()

elif st.session_state.search_state == "error":
    if st.button("Retry Search"):
        st.session_state.search_state = "init"
        st.rerun()

elif st.session_state.current_location and not st.session_state.gemini_configured:
    st.error("Gemini API not configured. Please set GEMINI_API_KEY in your .env file.")

elif st.session_state.search_state == "init":
    st.info("👋 Start by detecting your location, using voice, or entering an address.")

# Show recording indicator
if st.session_state.recording:
    st.warning("Recording in progress... Please speak now")