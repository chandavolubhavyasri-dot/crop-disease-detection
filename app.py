
import streamlit as st
import requests
import json
import io
import base64
import re
import numpy as np
import onnxruntime as ort

from pathlib import Path
from PIL import Image
from gtts import gTTS
from deep_translator import GoogleTranslator
from huggingface_hub import hf_hub_download


# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="CropCare AI",
    page_icon="🌱",
    layout="wide"
)

MODEL_REPO = "BiernyVR/crop-disease-classifier"
BACKGROUND_IMAGE = Path("farm_backgroundjpg.png")


# ==================================================
# 2. LANGUAGES
# ==================================================

LANGUAGES = {
    "English": {"translate": "en", "tts": "en"},
    "Hindi": {"translate": "hi", "tts": "hi"},
    "Telugu": {"translate": "te", "tts": "te"},
    "Tamil": {"translate": "ta", "tts": "ta"},
    "Bengali": {"translate": "bn", "tts": "bn"},
    "Kannada": {"translate": "kn", "tts": "kn"},
    "Malayalam": {"translate": "ml", "tts": "ml"},
    "Marathi": {"translate": "mr", "tts": "mr"},
    "Gujarati": {"translate": "gu", "tts": "gu"},
    "Punjabi": {"translate": "pa", "tts": "pa"},
}


# ==================================================
# 3. ALL INDIAN STATES AND UNION TERRITORIES
# ==================================================

INDIAN_STATES_UTS = [
    "Andhra Pradesh",
    "Arunachal Pradesh",
    "Assam",
    "Bihar",
    "Chhattisgarh",
    "Goa",
    "Gujarat",
    "Haryana",
    "Himachal Pradesh",
    "Jharkhand",
    "Karnataka",
    "Kerala",
    "Madhya Pradesh",
    "Maharashtra",
    "Manipur",
    "Meghalaya",
    "Mizoram",
    "Nagaland",
    "Odisha",
    "Punjab",
    "Rajasthan",
    "Sikkim",
    "Tamil Nadu",
    "Telangana",
    "Tripura",
    "Uttar Pradesh",
    "Uttarakhand",
    "West Bengal",
    "Andaman and Nicobar Islands",
    "Chandigarh",
    "Dadra and Nagar Haveli and Daman and Diu",
    "Delhi (National Capital Territory)",
    "Jammu and Kashmir",
    "Ladakh",
    "Lakshadweep",
    "Puducherry",
]


# ==================================================
# 4. AGRICULTURE BACKGROUND
# ==================================================

def add_background():
    if not BACKGROUND_IMAGE.exists():
        return

    with open(BACKGROUND_IMAGE, "rb") as image_file:
        encoded = base64.b64encode(
            image_file.read()
        ).decode("utf-8")

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image:
                linear-gradient(
                    rgba(255,255,255,0.88),
                    rgba(255,255,255,0.91)
                ),
                url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        h1, h2, h3 {{
            color: #1b5e20;
        }}

        div.stButton > button {{
            background-color: #2e7d32;
            color: white;
            border-radius: 9px;
            border: none;
            min-height: 42px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


add_background()


# ==================================================
# 5. LOAD THE AI MODEL
# ==================================================

@st.cache_resource(show_spinner="Loading the AI model...")
def load_model():
    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename="efficientnet_v2_s_best.onnx",
    )

    # Download the separate ONNX weights file if the model uses it.
    hf_hub_download(
        repo_id=MODEL_REPO,
        filename="efficientnet_v2_s_best.onnx.data",
    )

    classes_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename="classes.json",
    )

    with open(classes_path, "r", encoding="utf-8") as file:
        class_data = json.load(file)

    if isinstance(class_data, dict):
        classes = class_data.get("classes", class_data)
    else:
        classes = class_data

    if isinstance(classes, dict):
        classes = [
            classes[str(i)] if str(i) in classes
            else classes[i]
            for i in range(len(classes))
        ]

    session = ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"],
    )

    return session, classes


# ==================================================
# 6. PREDICT THE LEAF CONDITION
# ==================================================

def predict_disease(image):
    session, classes = load_model()

    image = image.convert("RGB").resize(
        (224, 224),
        Image.Resampling.BILINEAR,
    )

    array = np.asarray(image, dtype=np.float32) / 255.0

    mean = np.array(
        [0.485, 0.456, 0.406],
        dtype=np.float32,
    )

    std = np.array(
        [0.229, 0.224, 0.225],
        dtype=np.float32,
    )

    array = (array - mean) / std

    tensor = np.transpose(
        array,
        (2, 0, 1),
    )[None, ...].astype(np.float32)

    input_name = session.get_inputs()[0].name

    output = session.run(
        None,
        {input_name: tensor},
    )[0]

    scores = np.asarray(output).reshape(-1)

    # If the model returns probabilities, preserve them.
    if (
        np.all(scores >= 0)
        and np.all(scores <= 1)
        and np.isclose(np.sum(scores), 1.0, atol=0.02)
    ):
        probabilities = scores
    else:
        scores = scores - np.max(scores)
        probabilities = np.exp(scores)
        probabilities /= np.sum(probabilities)

    if len(classes) != len(probabilities):
        raise ValueError(
            "The number of model classes does not match its output. "
            "Check the model's classes.json file."
        )

    best_index = int(np.argmax(probabilities))
    confidence = float(probabilities[best_index]) * 100

    return str(classes[best_index]), confidence


def clean_label(label):
    label = str(label)
    label = label.replace("___", " - ")
    label = label.replace("_", " ")
    return re.sub(r"\s+", " ", label).strip()


# ==================================================
# 7. GENERAL DISEASE GUIDANCE
# ==================================================

def disease_guidance(label):
    text = label.lower()

    if "healthy" in text:
        return (
            "The model classified this leaf as healthy. Continue regular "
            "field inspections and monitor new leaves for unusual changes. "
            "This result does not guarantee that the entire plant is healthy."
        )

    if "late blight" in text:
        return (
            "Possible late blight. Check nearby plants and avoid prolonged "
            "leaf wetness. Seek local agricultural advice about approved "
            "management options for this crop."
        )

    if "early blight" in text:
        return (
            "Possible early blight. Inspect nearby leaves, remove badly "
            "affected plant debris where appropriate, and avoid unnecessary "
            "leaf wetness. Confirm suitable management with an expert."
        )

    if "rust" in text:
        return (
            "Possible rust disease. Inspect surrounding plants and monitor "
            "new leaves. Ask a local agricultural officer about suitable "
            "management for the affected crop."
        )

    if "powdery mildew" in text:
        return (
            "Possible powdery mildew. Improve airflow around plants where "
            "practical, avoid excessive crowding, and check nearby leaves. "
            "Seek crop-specific management advice."
        )

    if "bacterial" in text:
        return (
            "Possible bacterial disease. Avoid handling wet plants, clean "
            "tools between plants, and avoid splashing water from affected "
            "leaves onto healthy plants. Confirm the diagnosis first."
        )

    if "mosaic" in text or "yellow leaf curl" in text:
        return (
            "Possible viral disease. Inspect nearby plants and check for "
            "insect vectors. Do not assume a pesticide can cure a virus. "
            "Ask an agricultural expert about appropriate management."
        )

    if "leaf mold" in text or "leaf spot" in text:
        return (
            "Possible leaf mold or leaf spot. Monitor nearby leaves, improve "
            "airflow where practical, and avoid unnecessary leaf wetness. "
            "Confirm the cause before choosing a treatment."
        )

    return (
        "The AI suggests a possible plant disease. Take clear photographs "
        "of affected leaves, inspect nearby plants, and consult a local "
        "agricultural officer. Do not apply chemicals based only on this "
        "prediction."
    )


# ==================================================
# 8. WEATHER LOCATION SEARCH: OPEN-METEO
# ==================================================

def find_indian_location(place):
    response = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={
            "name": place,
            "count": 10,
            "language": "en",
            "format": "json",
        },
        timeout=15,
    )

    response.raise_for_status()

    results = response.json().get("results", [])

    for result in results:
        if result.get("country_code") == "IN":
            return result

    return None


def get_weather(latitude, longitude):
    response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": latitude,
            "longitude": longitude,
            "current": (
                "temperature_2m,"
                "relative_humidity_2m,"
                "precipitation,"
                "wind_speed_10m"
            ),
            "daily": (
                "precipitation_probability_max,"
                "precipitation_sum,"
                "temperature_2m_max,"
                "temperature_2m_min"
            ),
            "forecast_days": 1,
            "timezone": "auto",
        },
        timeout=15,
    )

    response.raise_for_status()
    return response.json()


# ==================================================
# 9. WEATHER-BASED PRECAUTIONS
# ==================================================

def weather_guidance(weather):
    current = weather.get("current", {})
    daily = weather.get("daily", {})

    temperature = current.get("temperature_2m")
    humidity = current.get("relative_humidity_2m")
    rain = current.get("precipitation")
    wind = current.get("wind_speed_10m")

    rain_chances = daily.get("precipitation_probability_max") or [None]
    rain_chance = rain_chances[0]

    advice = []

    if humidity is not None and humidity >= 85:
        advice.append(
            "Humidity is high. Inspect crops regularly for disease symptoms "
            "and avoid unnecessary leaf wetness."
        )

    if rain_chance is not None and rain_chance >= 60:
        advice.append(
            "Rain is likely. Check field drainage and avoid spraying before "
            "rain. Always follow product labels and local guidance."
        )

    if rain is not None and rain >= 10:
        advice.append(
            "Precipitation is elevated. Check for waterlogging and allow "
            "foliage to dry when possible."
        )

    if temperature is not None and temperature >= 35:
        advice.append(
            "Conditions are hot. Monitor soil moisture and irrigate according "
            "to the crop's needs, preferably during cooler hours."
        )

    if wind is not None and wind >= 20:
        advice.append(
            "Wind is strong. Avoid spraying in windy conditions to reduce "
            "spray drift."
        )

    if not advice:
        advice.append(
            "No weather threshold in this prototype triggered a warning. "
            "Continue routine crop monitoring and follow local advice."
        )

    return advice


# ==================================================
# 10. TRANSLATION AND TEXT-TO-SPEECH
# ==================================================

def make_voice_alert(text, language_name):
    language = LANGUAGES[language_name]
    translated = text
    tts_language = language["tts"]

    if language["translate"] != "en":
        translated = GoogleTranslator(
            source="auto",
            target=language["translate"],
        ).translate(text)

    audio_buffer = io.BytesIO()

    gTTS(
        text=translated,
        lang=tts_language,
        slow=False,
    ).write_to_fp(audio_buffer)

    audio_buffer.seek(0)

    return audio_buffer.getvalue(), translated


# ==================================================
# 11. SIDEBAR
# ==================================================

st.title("🌱 CropCare AI")
st.subheader(
    "AI Crop Disease Detection and Multilingual Voice Alerts"
)

st.write(
    "Upload a leaf photograph for a preliminary AI prediction, "
    "general disease guidance, and weather-based precautions."
)

st.info(
    "This prototype's model is intended for a limited set of crop-disease "
    "categories. It cannot diagnose every crop or disease. Confirm important "
    "decisions with an agricultural expert."
)

with st.sidebar:
    st.header("🌐 Farmer Settings")

    selected_language = st.selectbox(
        "Choose voice language",
        list(LANGUAGES.keys()),
    )

    selected_state = st.selectbox(
        "Select state / Union Territory",
        INDIAN_STATES_UTS,
    )

    place_name = st.text_input(
        "Town, village, or district",
        placeholder="Example: Guntur",
    )

    st.caption(
        "Enter a location in India to retrieve local weather."
    )

    st.markdown("---")
    st.subheader("🌾 Agricultural Resources")

    st.markdown(
        "[TNAU Crop Protection](https://agritech.tnau.ac.in/crop_protection/crop_prot.html)"
    )

    st.markdown(
        "[TNAU Integrated Pest Management](https://agritech.tnau.ac.in/crop_protection/crop_prot_ipm.html)"
    )

    st.markdown(
        "[TNAU Disease Management](https://agritech.tnau.ac.in/crop_protection/crop_prot_disease.html)"
    )


# ==================================================
# 12. LEAF IMAGE UPLOAD
# ==================================================

st.header("📷 Upload a Crop Leaf Photograph")

crop_options = [
    "Apple",
    "Blueberry",
    "Cherry",
    "Corn / Maize",
    "Grape",
    "Orange",
    "Peach",
    "Bell Pepper",
    "Potato",
    "Raspberry",
    "Soybean",
    "Squash",
    "Strawberry",
    "Tomato",
    "Other / Not sure",
]

selected_crop = st.selectbox(
    "Choose the crop if known",
    crop_options,
)

uploaded_file = st.file_uploader(
    "Choose a clear leaf photograph",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    try:
        leaf_image = Image.open(uploaded_file).convert("RGB")

        st.image(
            leaf_image,
            caption="Uploaded leaf photograph",
            use_container_width=True,
        )

        if st.button(
            "🔍 Analyse Leaf",
            type="primary",
            use_container_width=True,
        ):
            # Reset the previous result and audio.
            st.session_state["analysis_ready"] = False
            st.session_state.pop("voice_bytes", None)
            st.session_state.pop("voice_text", None)
            st.session_state.pop("voice_signature", None)

            with st.spinner("Analysing the leaf photograph..."):
                try:
                    label, confidence = predict_disease(leaf_image)

                    st.session_state["last_label"] = clean_label(label)
                    st.session_state["last_confidence"] = confidence
                    st.session_state["last_guidance"] = disease_guidance(label)
                    st.session_state["last_weather"] = None
                    st.session_state["last_weather_advice"] = []
                    st.session_state["last_location"] = None

                    if place_name.strip():
                        try:
                            location = find_indian_location(
                                place_name.strip()
                            )

                            if location:
                                weather = get_weather(
                                    location["latitude"],
                                    location["longitude"],
                                )

                                st.session_state["last_weather"] = weather
                                st.session_state["last_weather_advice"] = (
                                    weather_guidance(weather)
                                )
                                st.session_state["last_location"] = (
                                    location.get("name", place_name)
                                )
                            else:
                                st.warning(
                                    "No matching Indian location was found. "
                                    "Check the spelling or try a nearby town."
                                )
                        except Exception:
                            st.warning(
                                "Weather data could not be loaded. "
                                "The AI prediction will still be displayed."
                            )

                    st.session_state["analysis_ready"] = True

                except Exception as error:
                    st.error(
                        "The AI model could not analyse this image. "
                        "Check the model files and Streamlit app logs."
                    )
                    st.caption(f"Technical detail: {error}")


# ==================================================
# 13. DISPLAY PREDICTION AND WEATHER
# ==================================================

if st.session_state.get("analysis_ready", False):
    label = st.session_state["last_label"]
    confidence = st.session_state["last_confidence"]
    guidance = st.session_state["last_guidance"]

    st.markdown("---")
    st.header("📋 AI Analysis Result")

    st.metric("Predicted leaf condition", label)
    st.metric("Model confidence", f"{confidence:.1f}%")

    if confidence < 60:
        st.warning(
            "Confidence is low. Try a clearer photograph and seek expert "
            "confirmation before making crop-treatment decisions."
        )

    st.subheader("🌿 Suggested Next Steps")
    st.write(guidance)

    weather = st.session_state.get("last_weather")
    weather_advice = st.session_state.get("last_weather_advice", [])
    location_name = st.session_state.get("last_location")

    if weather:
        current = weather.get("current", {})

        st.subheader("🌦️ Local Weather")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Temperature",
                f"{current.get('temperature_2m', 'N/A')} °C",
            )

        with col2:
            st.metric(
                "Humidity",
                f"{current.get('relative_humidity_2m', 'N/A')}%",
            )

        with col3:
            st.metric(
                "Precipitation",
                f"{current.get('precipitation', 'N/A')} mm",
            )

        with col4:
            st.metric(
                "Wind speed",
                f"{current.get('wind_speed_10m', 'N/A')} km/h",
            )

        st.caption(
            f"Weather location: {location_name}. "
            f"Selected state / UT: {selected_state}."
        )

        st.subheader("🌤️ Weather-Based Precautions")

        for item in weather_advice:
            st.write(f"• {item}")

    else:
        st.caption(
            "For local weather information, enter your town or district "
            "in the sidebar and analyse the leaf again."
        )


    # ==================================================
    # 14. AUTOMATIC VOICE ALERT
    # ==================================================

    spoken_text = (
        f"CropCare AI result. The selected crop is {selected_crop}. "
        f"The predicted leaf condition is {label}. "
        f"Model confidence is {confidence:.1f} percent. "
        f"Suggested next steps: {guidance}"
    )

    if weather_advice:
        spoken_text += (
            " Weather-based precautions. "
            + " ".join(weather_advice)
        )

    voice_signature = (
        label,
        round(confidence, 1),
        selected_language,
        selected_crop,
        tuple(weather_advice),
    )

    if st.session_state.get("voice_signature") != voice_signature:
        try:
            with st.spinner("Preparing your voice alert..."):
                audio_bytes, translated_text = make_voice_alert(
                    spoken_text,
                    selected_language,
                )

                st.session_state["voice_bytes"] = audio_bytes
                st.session_state["voice_text"] = translated_text
                st.session_state["voice_signature"] = voice_signature

        except Exception:
            st.session_state.pop("voice_bytes", None)
            st.session_state.pop("voice_text", None)

            st.warning(
                "Automatic voice generation failed. Check your internet "
                "connection. You can retry using the button below."
            )

    st.subheader("🗣️ Voice Alert")

    if st.session_state.get("voice_bytes"):
        st.audio(
            st.session_state["voice_bytes"],
            format="audio/mp3",
            autoplay=True,
        )

        st.caption(
            "Automatic playback is attempted. If your browser blocks it, "
            "press Play on the audio player."
        )

        with st.expander("Read the spoken alert"):
            st.write(st.session_state.get("voice_text", ""))

    if st.button("🔊 Retry Voice Alert"):
        try:
            with st.spinner("Generating voice alert..."):
                audio_bytes, translated_text = make_voice_alert(
                    spoken_text,
                    selected_language,
                )

                st.session_state["voice_bytes"] = audio_bytes
                st.session_state["voice_text"] = translated_text
                st.session_state["voice_signature"] = voice_signature

            st.rerun()

        except Exception:
            st.error(
                "Voice generation failed. Please check your internet "
                "connection and try again."


# ==================================================
# 15. FOOTER
# ==================================================

st.markdown("---")

st.caption(
    "CropCare AI is an educational prototype. Predictions may be wrong "
    "and should not replace expert diagnosis. Weather precautions are "
    "general guidance, not a disease forecast."
)


                                
        
