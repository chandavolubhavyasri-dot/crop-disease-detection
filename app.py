
import io
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image
from gtts import gTTS

try:
    import tensorflow as tf
except ImportError:
    tf = None


# =============== PAGE CONFIGURATION ===============

st.set_page_config(
    page_title="CropCare AI",
    page_icon="🌱",
    layout="wide"
)

st.title("🌱 CropCare AI")
st.subheader("AI Crop Health & Multilingual Farmer Assistance")
st.write(
    "A farmer-friendly prototype for leaf analysis, disease education, "
    "weather awareness and voice assistance."
)

# =============== LANGUAGES ===============

LANGUAGES = {
    "English": {"tts": "en", "message":
        "Please check your crop regularly. If you suspect a disease, "
        "consult an agricultural expert before treatment."},
    "Hindi": {"tts": "hi", "message":
        "अपनी फसल की नियमित जाँच करें। बीमारी का संदेह होने पर उपचार से पहले कृषि विशेषज्ञ से सलाह लें।"},
    "Telugu": {"tts": "te", "message":
        "మీ పంటను క్రమం తప్పకుండా పరిశీలించండి. వ్యాధి అనుమానం ఉంటే చికిత్సకు ముందు వ్యవసాయ నిపుణుడిని సంప్రదించండి."},
    "Tamil": {"tts": "ta", "message":
        "பயிரைத் தொடர்ந்து கண்காணிக்கவும். நோய் இருப்பதாக சந்தேகித்தால் சிகிச்சைக்கு முன் வேளாண் நிபுணரை அணுகவும்."},
    "Kannada": {"tts": "kn", "message":
        "ಬೆಳೆಯನ್ನು ನಿಯಮಿತವಾಗಿ ಪರಿಶೀಲಿಸಿ. ರೋಗದ ಅನುಮಾನವಿದ್ದರೆ ಚಿಕಿತ್ಸೆಗೂ ಮೊದಲು ಕೃಷಿ ತಜ್ಞರನ್ನು ಸಂಪರ್ಕಿಸಿ."},
    "Malayalam": {"tts": "ml", "message":
        "വിള പതിവായി പരിശോധിക്കുക. രോഗം സംശയിക്കുന്നുവെങ്കിൽ ചികിത്സയ്ക്ക് മുമ്പ് കാർഷിക വിദഗ്ധനെ സമീപിക്കുക."},
    "Bengali": {"tts": "bn", "message":
        "নিয়মিত ফসল পরীক্ষা করুন। রোগের সন্দেহ হলে চিকিৎসার আগে কৃষি বিশেষজ্ঞের পরামর্শ নিন।"},
    "Marathi": {"tts": "mr", "message":
        "पिकाची नियमित तपासणी करा. रोगाचा संशय असल्यास उपचारापूर्वी कृषी तज्ज्ञांचा सल्ला घ्या."},
    "Gujarati": {"tts": "gu", "message":
        "પાકની નિયમિત તપાસ કરો. રોગની શંકા હોય તો સારવાર પહેલાં કૃષિ નિષ્ણાતની સલાહ લો."},
    "Punjabi": {"tts": "pa", "message":
        "ਫਸਲ ਦੀ ਨਿਯਮਿਤ ਜਾਂਚ ਕਰੋ। ਬਿਮਾਰੀ ਦਾ ਸ਼ੱਕ ਹੋਵੇ ਤਾਂ ਇਲਾਜ ਤੋਂ ਪਹਿਲਾਂ ਖੇਤੀ ਮਾਹਿਰ ਦੀ ਸਲਾਹ ਲਵੋ."}
}

# =============== DISEASE REFERENCE LIBRARY ===============
# General educational information, not a diagnosis.

DISEASES = {
    "Rice": {
        "Blast": (
            "Diamond-shaped leaf lesions, often with grey centres.",
            "Use locally recommended varieties and balanced crop nutrition.",
            "Ask an agricultural expert to confirm the disease."
        ),
        "Bacterial leaf blight": (
            "Yellowing or drying may begin at leaf tips or edges.",
            "Use healthy seed and locally recommended varieties.",
            "Seek local confirmation before choosing treatment."
        ),
        "Brown spot": (
            "Brown oval spots can develop on leaves.",
            "Use healthy seed and maintain balanced nutrition.",
            "Consult an agricultural expert about management."
        )
    },
    "Tomato": {
        "Early blight": (
            "Dark spots may develop concentric rings and yellowing.",
            "Rotate crops and remove infected plant debris appropriately.",
            "Confirm the disease before selecting a registered treatment."
        ),
        "Late blight": (
            "Water-soaked dark lesions may spread rapidly in wet conditions.",
            "Monitor plants frequently and avoid prolonged leaf wetness.",
            "Seek prompt local agricultural advice if suspected."
        ),
        "Bacterial spot": (
            "Small dark spots may affect leaves and fruit.",
            "Use healthy planting material and avoid handling wet plants.",
            "Get expert confirmation before choosing control measures."
        ),
        "Yellow leaf curl": (
            "Leaves may curl, yellow and become smaller.",
            "Monitor whiteflies and use recommended integrated pest management.",
            "Several causes look similar; ask an expert to confirm."
        )
    },
    "Chilli": {
        "Anthracnose": (
            "Sunken dark lesions may develop on fruit.",
            "Use healthy planting material and remove affected fruit appropriately.",
            "Confirm locally before choosing treatment."
        ),
        "Leaf curl": (
            "Leaves may curl or become distorted and plants may be stunted.",
            "Monitor insect vectors and follow local crop advice.",
            "Confirm the cause before treatment."
        )
    },
    "Cotton": {
        "Bacterial blight": (
            "Angular dark lesions may affect leaves, stems or bolls.",
            "Use healthy seed and locally recommended varieties.",
            "Consult an agricultural expert for confirmation."
        ),
        "Cotton leaf curl": (
            "Leaves may curl upward and show vein thickening.",
            "Monitor whiteflies and follow regional integrated pest management advice.",
            "Seek expert confirmation before treatment."
        )
    },
    "Maize": {
        "Northern leaf blight": (
            "Long, cigar-shaped lesions can develop on leaves.",
            "Use locally recommended varieties and rotate crops where practical.",
            "Confirm the disease with an agricultural expert."
        ),
        "Common rust": (
            "Reddish-brown raised pustules may appear on leaves.",
            "Monitor fields and use recommended resistant varieties.",
            "Ask an expert to assess severity."
        )
    },
    "Potato": {
        "Early blight": (
            "Brown leaf spots may show target-like rings.",
            "Rotate crops and remove infected debris appropriately.",
            "Confirm the disease before treatment."
        ),
        "Late blight": (
            "Dark, water-soaked lesions can spread quickly in wet weather.",
            "Monitor fields and remove infected material safely.",
            "Contact an agricultural officer promptly if suspected."
        )
    },
    "Groundnut": {
        "Early leaf spot": (
            "Brown circular leaf spots may have yellow halos.",
            "Rotate crops and monitor plants regularly.",
            "Confirm locally before applying a treatment."
        ),
        "Late leaf spot": (
            "Dark leaf spots may develop, sometimes without prominent halos.",
            "Follow recommended field sanitation and crop rotation.",
            "Seek expert confirmation and management advice."
        )
    },
    "Wheat": {
        "Leaf rust": (
            "Orange-brown pustules may appear on leaf surfaces.",
            "Use locally recommended resistant varieties and monitor fields.",
            "Ask an agricultural expert to confirm."
        ),
        "Powdery mildew": (
            "White powder-like patches may appear on leaves.",
            "Use suitable varieties and avoid excessive nitrogen.",
            "Seek locally appropriate management advice."
        )
    }
}

# =============== MODEL INTEGRATION ===============
# Required files:
# model.keras
# class_names.json
#
# class_names.json must contain class labels in exactly the same
# order as the model's output classes.
#
# Example:
# ["Tomato___Early_blight", "Tomato___healthy", "Rice___Blast"]
#
# The preprocessing below assumes RGB pixels scaled to 0..1.
# Change it if your model was trained differently.

@st.cache_resource
def load_model():
    if tf is None or not os.path.exists("model.keras"):
        return None
    try:
        return tf.keras.models.load_model("model.keras")
    except Exception:
        return None


def load_labels():
    if not os.path.exists("class_names.json"):
        return []
    try:
        with open("class_names.json", "r", encoding="utf-8") as f:
            labels = json.load(f)
        return labels if isinstance(labels, list) else []
    except Exception:
        return []


def predict_image(image):
    model = load_model()
    labels = load_labels()

    if model is None or not labels:
        return None

    try:
        shape = model.input_shape
        height = int(shape[1] or 224)
        width = int(shape[2] or 224)

        image = image.convert("RGB").resize((width, height))
        pixels = np.asarray(image, dtype=np.float32) / 255.0
        pixels = np.expand_dims(pixels, axis=0)

        output = np.asarray(model.predict(pixels, verbose=0)).reshape(-1)

        if len(output) != len(labels) or len(output) == 0:
            return None

        # This assumes model outputs class probabilities.
        # For logits, apply the correct softmax for your model.
        index = int(np.argmax(output))
        confidence = float(output[index])

        return {
            "label": str(labels[index]),
            "confidence": confidence
        }
    except Exception:
        return None


# =============== SESSION DATA ===============

if "history" not in st.session_state:
    st.session_state.history = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None


# =============== SIDEBAR ===============

with st.sidebar:
    st.header("👨‍🌾 Farmer Settings")
    language = st.selectbox("Choose language", list(LANGUAGES))
    location = st.text_input("Village / district (optional)")
    st.caption("Avoid entering private or sensitive information.")

    model_ready = load_model() is not None and len(load_labels()) > 0

    st.divider()
    st.write("**AI status**")
    if model_ready:
        st.success("Model files detected")
    else:
        st.warning("AI model not connected")


# =============== NAVIGATION ===============

page = st.radio(
    "Navigation",
    [
        "Home",
        "Leaf Detection",
        "Disease Library",
        "Weather",
        "History & Reports",
        "About"
    ],
    horizontal=True
)


# =============== HOME ===============

if page == "Home":
    st.markdown("## Welcome to CropCare AI 🌿")
    st.write(
        "Explore crop-health information, upload leaf photos when a model "
        "is configured, and access voice assistance."
    )

    a, b, c = st.columns(3)
    a.metric("Languages", "10")
    b.metric("Reference crops", len(DISEASES))
    c.metric("Saved checks", len(st.session_state.history))

    st.markdown("### Main features")
    st.markdown("""
    - Leaf photo upload and trained-model integration
    - Disease symptoms, prevention and next-step guidance
    - Multilingual voice messages
    - Current weather monitoring
    - Session history and downloadable reports
    """)

    if not model_ready:
        st.info(
            "Photo upload is available, but AI diagnosis will remain inactive "
            "until compatible trained model files are added."
        )


# =============== LEAF DETECTION ===============

elif page == "Leaf Detection":
    st.header("📷 Crop Leaf Check")

    crop = st.selectbox(
        "Which crop are you checking?",
        list(DISEASES) + ["Other / Unknown"]
    )

    upload = st.file_uploader(
        "Upload a clear leaf photo",
        type=["jpg", "jpeg", "png"]
    )

    if upload:
        try:
            image = Image.open(upload).convert("RGB")
            st.image(image, caption="Uploaded leaf", use_container_width=True)

            if st.button("Analyze photo", type="primary"):
                result = predict_image(image)

                if result is None:
                    st.session_state.last_result = None
                    st.error(
                        "AI prediction is unavailable. Add a compatible "
                        "trained model and class_names.json."
                    )
                    st.write(
                        "The disease library below is for reference only "
                        "and cannot diagnose this uploaded image."
                    )
                else:
                    label = result["label"]
                    confidence = result["confidence"]

                    st.session_state.last_result = result
                    st.subheader("Model prediction")
                    st.write("Predicted class:", label)
                    st.progress(max(0.0, min(1.0, confidence)))
                    st.write(f"Model confidence: {confidence:.1%}")

                    if confidence < 0.65:
                        st.error(
                            "Low confidence: do not rely on this prediction. "
                            "Try a clearer photo and consult an expert."
                        )
                    else:
                        st.warning(
                            "This is a model prediction, not a confirmed "
                            "diagnosis. Verify it before treatment."
                        )

                    st.session_state.history.append({
                        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "selected_crop": crop,
                        "prediction": label,
                        "confidence": round(confidence, 4),
                        "location": location
                    })

                    # Show reference details only if a known name matches.
                    matched = False
                    for crop_name, entries in DISEASES.items():
                        for disease_name, details in entries.items():
                            if disease_name.lower() in label.lower():
                                st.markdown("### Related reference information")
                                st.write("**Crop reference:**", crop_name)
                                st.write("**Disease:**", disease_name)
                                st.write("**Symptoms:**", details[0])
                                st.write("**Prevention:**", details[1])
                                st.write("**Next step:**", details[2])
                                matched = True
                                break
                        if matched:
                            break

                    if not matched:
                        st.info(
                            "No matching reference entry was found. "
                            "Consult an agricultural expert."
                        )

        except Exception:
            st.error("Unable to read this image. Please try another photo.")


# =============== DISEASE LIBRARY ===============

elif page == "Disease Library":
    st.header("📚 Crop Disease Reference Library")
    st.info(
        "These entries describe common symptoms. They are not an exhaustive "
        "database and do not confirm the disease in your plant."
    )

    selected_crop = st.selectbox("Choose crop", list(DISEASES))
    disease = st.selectbox(
        "Choose reference disease",
        list(DISEASES[selected_crop])
    )

    symptoms, prevention, next_step = DISEASES[selected_crop][disease]

    st.subheader(f"{selected_crop}: {disease}")
    st.write("**Common symptoms:**", symptoms)
    st.write("**Prevention:**", prevention)
    st.write("**Recommended next step:**", next_step)
    st.caption(
        "Ask a local agricultural expert before applying pesticides or "
        "other treatments. Always follow registered product labels."
    )


# =============== WEATHER MONITORING ===============

elif page == "Weather":
    st.header("🌦️ Local Weather Monitoring")
    st.write(
        "Weather conditions can help farmers decide when to inspect crops. "
        "These readings do not diagnose disease."
    )

    col1, col2 = st.columns(2)

    with col1:
        latitude = st.number_input(
            "Latitude",
            min_value=-90.0,
            max_value=90.0,
            value=16.5062,
            format="%.4f"
        )

    with col2:
        longitude = st.number_input(
            "Longitude",
            min_value=-180.0,
            max_value=180.0,
            value=80.6480,
            format="%.4f"
        )

    if st.button("Fetch current weather"):
        try:
            response = requests.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": latitude,
                    "longitude": longitude,
                    "current": (
                        "temperature_2m,relative_humidity_2m,"
                        "precipitation,wind_speed_10m"
                    ),
                    "timezone": "auto"
                },
                timeout=20
            )
            response.raise_for_status()
            current = response.json()["current"]

            temp = current.get("temperature_2m", 0)
            humidity = current.get("relative_humidity_2m", 0)
            rain = current.get("precipitation", 0)
            wind = current.get("wind_speed_10m", 0)

            a, b, c, d = st.columns(4)
            a.metric("Temperature", f"{temp} °C")
            b.metric("Humidity", f"{humidity}%")
            c.metric("Precipitation", f"{rain} mm")
            d.metric("Wind speed", f"{wind} km/h")

            # Illustrative indicator only; not a validated disease model.
            score = (
                int(rain >= 10)
                + int(humidity >= 85)
                + int(wind >= 40)
            )

            if score >= 2:
                st.warning(
                    "Weather caution: inspect your crops and monitor "
                    "for visible symptoms."
                )
            elif score == 1:
                st.info(
                    "Consider closer crop monitoring under changing conditions."
                )
            else:
                st.success(
                    "These simple indicators did not trigger a high caution. "
                    "Continue normal crop monitoring."
                )

            st.caption(
                "Weather from Open-Meteo. Thresholds are illustrative and "
                "are not validated disease prediction thresholds."
            )

        except Exception:
            st.error(
                "Weather could not be fetched. Check your internet connection "
                "and try again."
            )


# =============== HISTORY & REPORTS ===============

elif page == "History & Reports":
    st.header("📜 Leaf Check History")

    if not st.session_state.history:
        st.info("No saved checks in this session.")
    else:
        dataframe = pd.DataFrame(st.session_state.history)
        st.dataframe(dataframe, use_container_width=True)

        st.download_button(
            "Download CSV report",
            data=dataframe.to_csv(index=False).encode("utf-8"),
            file_name="cropcare_history.csv",
            mime="text/csv"
        )

        if st.button("Clear session history"):
            st.session_state.history = []
            st.rerun()

    st.caption(
        "History is session-only and may be lost when the session restarts. "
        "Permanent history needs a database."
    )


# =============== ABOUT ===============

elif page == "About":
    st.header("ℹ️ About CropCare AI")
    st.write(
        "CropCare AI is a prototype designed to make crop-health education "
        "more accessible through images, local-language messages and voice."
    )

    st.markdown("### Keypad-phone support")
    st.write(
        "Basic keypad phones usually cannot upload photos to this website. "
        "To receive photos from those phones, add a compatible MMS or "
        "messaging provider, a secure image-receiving backend, and a "
        "configured SMS or voice-call service."
    )

    st.markdown("### Current limitations")
    st.markdown("""
    - Model accuracy depends on the training data and testing.
    - The reference library covers selected diseases, not every crop disease.
    - Translation and voice availability may vary by language and service.
    - Weather indicators are not a validated disease forecast.
    - Session history is not permanent storage.
    """)


# =============== VOICE ASSISTANCE ===============

st.divider()
st.header("🔊 Multilingual Voice Assistance")

default_message = LANGUAGES[language]["message"]

voice_text = st.text_area(
    "Message to read aloud",
    value=default_message
)

if st.button("Generate and play voice alert"):
    try:
        audio_buffer = io.BytesIO()
        speech = gTTS(
            text=voice_text,
            lang=LANGUAGES[language]["tts"]
        )
        speech.write_to_fp(audio_buffer)
        audio_buffer.seek(0)

        st.audio(audio_buffer, format="audio/mp3")
        st.success("Voice message generated.")

    except Exception:
        st.error(
            "Voice generation failed. Check the internet connection. "
            "The speech service may not support every language."
        )

st.divider()
st.caption(
    "CropCare AI prototype • Verify disease predictions and treatment "
    "decisions with qualified agricultural experts."
)
