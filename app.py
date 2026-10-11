
import streamlit as st
import requests
import io
import base64
import re
import json
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
    layout="wide",
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
# 3. INDIAN STATES AND UNION TERRITORIES
# ==================================================
INDIAN_STATES_UTS = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar",
    "Chhattisgarh", "Goa", "Gujarat", "Haryana",
    "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala",
    "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya",
    "Mizoram", "Nagaland", "Odisha", "Punjab", "Rajasthan",
    "Sikkim", "Tamil Nadu", "Telangana", "Tripura",
    "Uttar Pradesh", "Uttarakhand", "West Bengal",
    "Andaman and Nicobar Islands", "Chandigarh",
    "Dadra and Nagar Haveli and Daman and Diu",
    "Delhi (National Capital Territory)", "Jammu and Kashmir",
    "Ladakh", "Lakshadweep", "Puducherry",
]


# ==================================================
# 4. BACKGROUND IMAGE
# ==================================================
def add_background():
    if not BACKGROUND_IMAGE.is_file():
        return

    encoded = base64.b64encode(
        BACKGROUND_IMAGE.read_bytes()
    ).decode("utf-8")

    mime = (
        "image/jpeg"
        if BACKGROUND_IMAGE.suffix.lower() in {".jpg", ".jpeg"}
        else "image/png"
    )

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image:
                linear-gradient(
                    rgba(255,255,255,0.88),
                    rgba(255,255,255,0.91)
                ),
                url("data:{mime};base64,{encoded}");
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
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


add_background()


# ==================================================
# 5. LOAD MODEL AND LABELS
# ==================================================
@st.cache_resource(show_spinner="Loading AI model...")
def load_model():
    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename="efficientnet_v2_s_best.onnx",
    )

    classes_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename="classes.json",
    )

    with open(classes_path, "r", encoding="utf-8") as file:
        class_data = json.load(file)

    if isinstance(class_data, dict):
        if "classes" in class_data:
            classes = class_data["classes"]
        elif "class_names" in class_data:
            classes = class_data["class_names"]
        else:
            try:
                classes = [
                    class_data[key]
                    for key in sorted(
                        class_data,
                        key=lambda item: int(item),
                    )
                ]
            except (ValueError, TypeError):
                raise ValueError(
                    "Unrecognised classes.json format. Expected a "
                    "'classes' list or an index-to-label dictionary."
                )
    elif isinstance(class_data, list):
        classes = class_data
    else:
        raise ValueError(
            "classes.json must contain a list or dictionary."
        )

    if not isinstance(classes, list) or not classes:
        raise ValueError("No class labels were found in classes.json.")

    session = ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"],
    )

    return session, [str(item) for item in classes]


# ==================================================
# 6. IMAGE PREDICTION
# ==================================================
def predict_disease(image):
    session, classes = load_model()

    input_meta = session.get_inputs()[0]
    input_shape = input_meta.shape

    height = (
        input_shape[2]
        if len(input_shape) == 4
        and isinstance(input_shape[2], int)
        else 224
    )
    width = (
        input_shape[3]
        if len(input_shape) == 4
        and isinstance(input_shape[3], int)
        else 224
    )

    image = image.convert("RGB").resize(
        (width, height),
        Image.Resampling.BILINEAR,
    )

    array = np.asarray(image, dtype=np.float32) / 255.0

    # Assumes the model uses ImageNet normalization.
    mean = np.array(
        [0.485, 0.456, 0.406],
        dtype=np.float32,
    )
    std = np.array(
        [0.229, 0.224, 0.225],
        dtype=np.float32,
    )

    array = (array - mean) / std

    if (
        len(input_shape) == 4
        and input_shape[1] == 3
    ):
        tensor = np.transpose(
            array,
            (2, 0, 1),
        )[None, ...].astype(np.float32)

    elif (
        len(input_shape) == 4
        and input_shape[-1] == 3
    ):
        tensor = array[None, ...].astype(np.float32)

    else:
        raise ValueError(
            f"Unsupported model input shape: {input_shape}. "
            "Preprocessing must match the model."
        )

    output = session.run(
        None,
        {input_meta.name: tensor},
    )[0]

    scores = np.asarray(output).reshape(-1)

    if len(classes) != len(scores):
        raise ValueError(
            f"Model returned {len(scores)} scores but classes.json "
            f"contains {len(classes)} labels."
        )

    # Keep probabilities if already normalized; otherwise use softmax.
    if (
        np.all(scores >= 0)
        and np.all(scores <= 1)
        and np.isclose(float(np.sum(scores)), 1.0, atol=0.02)
    ):
        probabilities = scores
    else:
        scores = scores - np.max(scores)
        exp_scores = np.exp(scores)
        probabilities = exp_scores / np.sum(exp_scores)

    index = int(np.argmax(probabilities))

    return (
        classes[index],
        float(probabilities[index]) * 100.0,
    )


def clean_label(label):
    label = str(label).replace("___", " - ").replace("_", " ")
    return re.sub(r"\s+", " ", label).strip()


# ==================================================
# 7. GENERAL DISEASE GUIDANCE
# ==================================================
def disease_guidance(label):
    text = label.lower()

    if "healthy" in text:
        return (
            "The model classified this leaf as healthy. Continue routine "
            "monitoring. This does not guarantee the entire plant is healthy."
        )

    if "late blight" in text:
        return (
            "Possible late blight. Inspect nearby plants, avoid prolonged "
            "leaf wetness, and seek local agricultural advice about approved "
            "management options."
        )

    if "early blight" in text:
        return (
            "Possible early blight. Inspect surrounding leaves, remove "
            "affected debris where appropriate, and confirm management "
            "with an agricultural expert."
        )

    if "rust" in text:
        return (
            "Possible rust disease. Inspect nearby plants and ask a local "
            "agricultural officer about crop-specific management."
        )

    if "powdery mildew" in text:
        return (
            "Possible powdery mildew. Improve airflow where practical and "
            "monitor nearby leaves. Seek crop-specific advice."
        )

    if "bacterial" in text:
        return (
            "Possible bacterial disease. Clean tools between plants and "
            "avoid splashing water from affected leaves. Confirm the "
            "diagnosis before treatment."
        )

    if "mosaic" in text or "yellow leaf curl" in text:
        return (
            "Possible viral disease. Inspect nearby plants and check for "
            "insect vectors. Consult an agricultural expert; pesticides "
            "do not cure plant viruses."
        )

    if "leaf mold" in text or "leaf spot" in text:
        return (
            "Possible leaf mold or leaf spot. Inspect nearby leaves, improve "
            "airflow where practical, and confirm the cause before treatment."
        )

    return (
        "The model suggests a possible plant disease. Take clear photos "
        "of affected leaves and consult a local agricultural officer. "
        "Do not apply chemicals based only on this prediction."
    )


# ==================================================
# 8. OPEN-METEO WEATHER
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

    for result in response.json().get("results", []):
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
                "temperature_2m,relative_humidity_2m,"
                "precipitation,wind_speed_10m"
            ),
            "daily": (
                "precipitation_probability_max,precipitation_sum,"
                "temperature_2m_max,temperature_2m_min"
            ),
            "forecast_days": 1,
            "timezone": "auto",
        },
        timeout=15,
    )

    response.raise_for_status()
    return response.json()


# ==================================================
# 9. WEATHER PRECAUTIONS
# ==================================================
def weather_guidance(weather):
    current = weather.get("current", {})
    daily = weather.get("daily", {})

    temperature = current.get("temperature_2m")
    humidity = current.get("relative_humidity_2m")
    rain = current.get("precipitation")
    wind = current.get("wind_speed_10m")

    rain_values = (
        daily.get("precipitation_probability_max") or [None]
    )
    rain_chance = rain_values[0]

    advice = []

    if humidity is not None and humidity >= 85:
        advice.append(
            "Humidity is high. Monitor crops for disease symptoms "
            "and avoid unnecessary leaf wetness."
        )

    if rain_chance is not None and rain_chance >= 60:
        advice.append(
            "Rain is likely. Check drainage and avoid spraying before "
            "rain. Follow product labels and local advice."
        )

    if rain is not None and rain >= 10:
        advice.append(
            "Precipitation is elevated. Check for waterlogging and allow "
            "foliage to dry when possible."
        )

    if temperature is not None and temperature >= 35:
        advice.append(
            "Conditions are hot. Monitor soil moisture and irrigate "
            "according to the crop's needs."
        )

    if wind is not None and wind >= 20:
        advice.append(
            "Wind is strong. Avoid spraying in windy conditions."
        )

    if not advice:
        advice.append(
            "No weather warning was triggered by this prototype's "
            "thresholds. Continue routine crop monitoring."
        )

    return advice


# ==================================================
# 10. MULTILINGUAL VOICE
# ==================================================
def make_voice_alert(text, language_name):
    language = LANGUAGES[language_name]
    translated = text

    if language["translate"] != "en":
        translated = GoogleTranslator(
            source="auto",
            target=language["translate"],
        ).translate(text)

    audio_buffer = io.BytesIO()

    gTTS(
        text=translated,
        lang=language["tts"],
        slow=False,
    ).write_to_fp(audio_buffer)

    return audio_buffer.getvalue(), translated


# ==================================================
# 11. SIDEBAR AND MAIN UI
# ==================================================
st.title("🌱 CropCare AI")
st.subheader(
    "AI Crop Disease Detection and Multilingual Voice Alerts"
)

st.write(
    "Upload a clear leaf photograph for a preliminary AI prediction, "
    "general guidance, and local weather precautions."
)

st.info(
    "This model can only recognise its trained categories. It cannot "
    "diagnose every crop or disease. Confirm important decisions with "
    "an agricultural expert."
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
        "Enter an Indian location to retrieve weather information."
    )

    st.markdown("---")
    st.subheader("🌾 Agricultural Resources")

    st.markdown(
        "[TNAU Crop Protection]"
        "(https://agritech.tnau.ac.in/crop_protection/crop_prot.html)"
    )
    st.markdown(
        "[TNAU Integrated Pest Management]"
        "(https://agritech.tnau.ac.in/crop_protection/crop_prot_ipm.html)"
    )
    st.markdown(
        "[TNAU Disease Management]"
        "(https://agritech.tnau.ac.in/crop_protection/crop_prot_disease.html)"
    )


# ==================================================
# 12. UPLOAD AND ANALYSE
# ==================================================
crop_options = [
    "Apple", "Blueberry", "Cherry", "Corn / Maize", "Grape",
    "Orange", "Peach", "Bell Pepper", "Potato", "Raspberry",
    "Soybean", "Squash", "Strawberry", "Tomato", "Other / Not sure",
]

st.header("📷 Upload a Crop Leaf Photograph")

selected_crop = st.selectbox(
    "Choose the crop if known",
    crop_options,
)

uploaded_file = st.file_uploader(
    "Choose a JPG or PNG photograph",
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
            st.session_state["analysis_ready"] = False

            for key in (
                "voice_bytes",
                "voice_text",
                "voice_signature",
            ):
                st.session_state.pop(key, None)

            try:
                with st.spinner("Analysing the leaf photograph..."):
                    raw_label, confidence = predict_disease(leaf_image)
                    label = clean_label(raw_label)

                    st.session_state["last_label"] = label
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
                                    location.get("name", place_name.strip())
                                )
                            else:
                                st.warning(
                                    "Location not found. Check the spelling "
                                    "or enter a nearby town."
                                )

                        except requests.RequestException:
                            st.warning(
                                "Weather could not be loaded. The AI "
                                "prediction will still be shown."
                            )

                    st.session_state["analysis_ready"] = True

            except Exception as error:
                st.error(
                    "AI analysis failed. Check the model files and "
                    "model input format."
                )
                st.caption(f"Technical detail: {error}")

    except Exception:
        st.error(
            "Could not open this image. Please upload a valid JPG or PNG."
        )


# ==================================================
# 13. RESULTS AND AUTOMATIC VOICE ALERT
# ==================================================
if st.session_state.get("analysis_ready", False):
    label = st.session_state["last_label"]
    confidence = st.session_state["last_confidence"]
    guidance = st.session_state["last_guidance"]

    weather = st.session_state.get("last_weather")
    weather_advice = st.session_state.get("last_weather_advice", [])
    location_name = st.session_state.get("last_location")

    st.markdown("---")
    st.header("📋 AI Analysis Result")

    st.metric("Predicted leaf condition", label)
    st.metric("Model confidence", f"{confidence:.1f}%")

    if confidence < 60:
        st.warning(
            "Confidence is low. Try a clearer image and seek expert "
            "confirmation before taking action."
        )

    st.subheader("🌿 Suggested Next Steps")
    st.write(guidance)

    if weather:
        current = weather.get("current", {})
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
            "Enter a town or district in the sidebar and analyse the "
            "leaf again to include local weather."
        )

    # Prepare a message specific to the current analysis.
    spoken_text = (
        f"CropCare AI result. Selected crop: {selected_crop}. "
        f"Predicted condition: {label}. "
        f"Confidence: {confidence:.1f} percent. "
        f"Suggested next steps: {guidance}"
    )

    if weather_advice:
        spoken_text += (
            " Weather precautions: " + " ".join(weather_advice)
        )

    voice_signature = (
        label,
        round(confidence, 1),
        selected_language,
        selected_crop,
        tuple(weather_advice),
    )

    st.subheader("🗣️ Voice Alert")

    # Generate the alert automatically once per analysis/language.
    if st.session_state.get("voice_signature") != voice_signature:
        try:
            with st.spinner("Preparing the voice alert..."):
                audio_bytes, translated_text = make_voice_alert(
                    spoken_text,
                    selected_language,
                )

                st.session_state["voice_bytes"] = audio_bytes
                st.session_state["voice_text"] = translated_text
                st.session_state["voice_signature"] = voice_signature

        except Exception as error:
            st.session_state.pop("voice_bytes", None)
            st.session_state.pop("voice_text", None)

            st.warning(
                "Automatic voice generation failed. Check your "
                "internet connection."
            )
            st.caption(f"Technical detail: {error}")

    if st.session_state.get("voice_bytes"):
        st.audio(
            st.session_state["voice_bytes"],
            format="audio/mp3",
            autoplay=True,
        )

        st.caption(
            "Automatic playback is attempted. If your browser blocks it, "
            "tap Play on the audio player."
        )

        with st.expander("Read the spoken alert"):
            st.write(st.session_state.get("voice_text", ""))

    # Manual fallback to regenerate audio if generation failed.
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

        except Exception as error:
            st.error(
                "Voice generation failed. Check your internet connection "
                "and try again."
            )
            st.caption(f"Technical detail: {error}")


# ==================================================
# 14. FOOTER
# ==================================================
st.markdown("---")

st.caption(
    "CropCare AI is an educational prototype. Predictions can be wrong "
    "and do not replace expert diagnosis. Weather precautions are general "
    "guidance, not a disease forecast."
)
