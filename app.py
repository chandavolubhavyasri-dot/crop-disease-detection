
import base64
import json
from datetime import datetime
from pathlib import Path

import streamlit as st
from PIL import Image
from gtts import gTTS

# ---------------- PAGE SETTINGS ----------------

st.set_page_config(
    page_title="CropCare AI",
    page_icon="🌱",
    layout="wide"
)

# ---------------- AGRICULTURE BACKGROUND ----------------

def add_agriculture_background():
    image_path = Path("farm_backgroundjpg.png")

    if image_path.exists():
        image_data = base64.b64encode(
            image_path.read_bytes()
        ).decode("utf-8")

        background_css = f"""
        <style>
        .stApp {{
            background-image:
                linear-gradient(
                    rgba(245, 250, 239, 0.88),
                    rgba(245, 250, 239, 0.88)
                ),
                url("data:image/jpeg;base64,{image_data}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        [data-testid="stSidebar"] {{
            background-color: rgba(20, 80, 45, 0.96);
        }}

        [data-testid="stSidebar"] * {{
            color: white;
        }}

        h1, h2, h3 {{
            color: #145c35;
        }}

        div[data-testid="stMetric"] {{
            background-color: rgba(255, 255, 255, 0.82);
            border: 1px solid #d7e8d0;
            padding: 14px;
            border-radius: 12px;
        }}

        div[data-testid="stVerticalBlockBorderWrapper"] {{
            border-radius: 14px;
        }}

        .stButton > button {{
            border-radius: 10px;
            font-weight: 600;
        }}
        </style>
        """
    else:
        background_css = """
        <style>
        .stApp {
            background: linear-gradient(
                135deg, #f4f9ed, #e1f0d9
            );
        }
        h1, h2, h3 {
            color: #145c35;
        }
        </style>
        """

    st.markdown(background_css, unsafe_allow_html=True)


add_agriculture_background()

# ---------------- LANGUAGES ----------------

LANGUAGES = {
    "English": {
        "code": "en",
        "message": "Please check the crop carefully and consult an agricultural expert before using any pesticide."
    },
    "Hindi": {
        "code": "hi",
        "message": "कृपया फसल की सावधानीपूर्वक जांच करें और कीटनाशक का उपयोग करने से पहले कृषि विशेषज्ञ से सलाह लें।"
    },
    "Telugu": {
        "code": "te",
        "message": "దయచేసి పంటను జాగ్రత్తగా పరిశీలించి, పురుగుమందులు వాడే ముందు వ్యవసాయ నిపుణుడిని సంప్రదించండి."
    },
    "Tamil": {
        "code": "ta",
        "message": "பயிரை கவனமாக பரிசோதித்து, பூச்சிக்கொல்லி பயன்படுத்துவதற்கு முன்பு வேளாண் நிபுணரை அணுகவும்."
    },
    "Bengali": {
        "code": "bn",
        "message": "ফসলটি ভালোভাবে পরীক্ষা করুন এবং কীটনাশক ব্যবহারের আগে কৃষি বিশেষজ্ঞের পরামর্শ নিন।"
    },
    "Kannada": {
        "code": "kn",
        "message": "ಬೆಳೆಯನ್ನು ಎಚ್ಚರಿಕೆಯಿಂದ ಪರಿಶೀಲಿಸಿ ಮತ್ತು ಕೀಟನಾಶಕ ಬಳಸುವ ಮೊದಲು ಕೃಷಿ ತಜ್ಞರನ್ನು ಸಂಪರ್ಕಿಸಿ."
    },
    "Malayalam": {
        "code": "ml",
        "message": "വിള പരിശോധിച്ച ശേഷം കീടനാശിനി ഉപയോഗിക്കുന്നതിന് മുമ്പ് കാർഷിക വിദഗ്ധനെ സമീപിക്കുക."
    },
    "Marathi": {
        "code": "mr",
        "message": "पिकाची काळजीपूर्वक तपासणी करा आणि कीटकनाशक वापरण्यापूर्वी कृषी तज्ज्ञांचा सल्ला घ्या."
    },
    "Gujarati": {
        "code": "gu",
        "message": "પાકની કાળજીપૂર્વક તપાસ કરો અને જંતુનાશકનો ઉપયોગ કરતા પહેલાં કૃષિ નિષ્ણાતની સલાહ લો."
    },
    "Punjabi": {
        "code": "pa",
        "message": "ਫਸਲ ਦੀ ਧਿਆਨ ਨਾਲ ਜਾਂਚ ਕਰੋ ਅਤੇ ਕੀਟਨਾਸ਼ਕ ਵਰਤਣ ਤੋਂ ਪਹਿਲਾਂ ਖੇਤੀਬਾੜੀ ਮਾਹਿਰ ਦੀ ਸਲਾਹ ਲਵੋ।"
    }
}

# ---------------- CROP PROTECTION STARTER LIBRARY ----------------

DISEASES = {
    "Rice": {
        "Rice Blast": {
            "signs": "Spindle-shaped spots with grey or whitish centres and dark borders on leaves.",
            "prevention": "Use healthy seed, avoid excessive nitrogen, maintain suitable spacing, and remove infected residues.",
            "source": "TNAU Crop Protection"
        },
        "Bacterial Leaf Blight": {
            "signs": "Water-soaked leaf edges that may turn yellow and dry.",
            "prevention": "Use suitable resistant varieties, maintain field sanitation, and avoid excessive nitrogen.",
            "source": "TNAU Crop Protection"
        }
    },
    "Tomato": {
        "Early Blight": {
            "signs": "Brown leaf spots that may show circular, target-like rings.",
            "prevention": "Remove infected leaves, avoid wetting foliage, rotate crops, and maintain good spacing.",
            "source": "TNAU Crop Protection"
        },
        "Leaf Curl": {
            "signs": "Leaves may curl upward or downward, become smaller, and show stunted growth.",
            "prevention": "Monitor whiteflies, remove severely affected plants when appropriate, and use recommended resistant varieties.",
            "source": "TNAU Crop Protection"
        }
    },
    "Chilli": {
        "Anthracnose / Fruit Rot": {
            "signs": "Sunken dark lesions may appear on fruits.",
            "prevention": "Use healthy seed, remove infected fruits, improve drainage, and avoid overhead irrigation.",
            "source": "TNAU Crop Protection"
        }
    },
    "Cotton": {
        "Cotton Leaf Curl": {
            "signs": "Leaves may curl, thicken, or show vein swelling.",
            "prevention": "Monitor whiteflies, remove volunteer cotton plants, and follow local integrated pest management advice.",
            "source": "TNAU Crop Protection"
        }
    },
    "Maize": {
        "Fall Armyworm": {
            "signs": "Leaves may have ragged holes and rows of feeding damage; larvae may be present in the whorl.",
            "prevention": "Inspect plants regularly, remove egg masses where practical, and follow locally approved integrated pest management guidance.",
            "source": "TNAU Crop Protection"
        }
    },
    "Potato": {
        "Late Blight": {
            "signs": "Dark, water-soaked lesions can spread rapidly in cool, wet conditions.",
            "prevention": "Use healthy seed tubers, remove infected plants, avoid prolonged leaf wetness, and seek local disease-management advice.",
            "source": "TNAU Crop Protection"
        }
    },
    "Groundnut": {
        "Leaf Spot": {
            "signs": "Brown or dark spots may appear on leaves and cause premature leaf drop.",
            "prevention": "Rotate crops, remove infected residues where practical, and use recommended varieties and locally approved management methods.",
            "source": "TNAU Crop Protection"
        }
    },
    "Wheat": {
        "Rust": {
            "signs": "Orange, yellow, or dark rust-coloured pustules may appear on leaves or stems.",
            "prevention": "Grow locally recommended resistant varieties, monitor the crop, and obtain local agricultural advice if symptoms appear.",
            "source": "TNAU Crop Protection"
        }
    }
}

# ---------------- OFFICIAL RESOURCES ----------------

TNAU_URL = "https://agritech.tnau.ac.in/crop_protection/crop_prot.html"
TNAU_IPM_URL = "https://agritech.tnau.ac.in/crop_protection/crop_prot_ipm.html"
TNAU_DISEASE_URL = "https://agritech.tnau.ac.in/crop_protection/crop_prot_disease.html"

# ---------------- LOCAL HISTORY ----------------

HISTORY_FILE = Path("cropcare_history.json")


def load_history():
    try:
        if HISTORY_FILE.exists():
            with open(HISTORY_FILE, "r", encoding="utf-8") as file:
                data = json.load(file)
                return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        pass
    return []


def save_history(history):
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as file:
            json.dump(history, file, ensure_ascii=False, indent=2)
        return True
    except OSError:
        return False


# ---------------- HEADER ----------------

st.title("🌱 CropCare AI")
st.subheader("Crop Disease Support for Every Farmer")

st.write(
    "A farmer-friendly prototype for crop photo uploads, "
    "multilingual voice assistance, disease education, "
    "and agricultural reference resources."
)

st.caption(
    "Prototype notice: photo-based AI diagnosis, live expert support, "
    "and keypad-phone messaging are not connected yet."
)

# ---------------- SIDEBAR ----------------

with st.sidebar:
    st.header("🌾 Farmer Settings")

    language = st.selectbox(
        "🌐 Select your language",
        list(LANGUAGES.keys())
    )

    crop = st.selectbox(
        "🌱 Select your crop",
        [
            "Rice", "Tomato", "Chilli", "Cotton",
            "Maize", "Potato", "Groundnut", "Wheat",
            "Other / Unknown"
        ]
    )

    st.divider()

    st.subheader("🌍 Useful Resources")
    st.markdown(f"[TNAU Crop Protection]({TNAU_URL})")
    st.markdown(f"[Integrated Pest Management]({TNAU_IPM_URL})")
    st.markdown(f"[TNAU Disease Resources]({TNAU_DISEASE_URL})")

# ---------------- MAIN TABS ----------------

tab1, tab2, tab3, tab4 = st.tabs([
    "📷 Photo Assistant",
    "📚 Crop Protection Library",
    "🛡️ IPM Guide",
    "🕘 My History"
])

# ---------------- TAB 1: PHOTO ASSISTANT ----------------

with tab1:
    st.header("📷 Crop Photo Assistant")

    st.write(
        "Upload a clear picture of an affected leaf or plant. "
        "The current version displays the photo but does not diagnose it."
    )

    photo = st.file_uploader(
        "Choose a crop photo",
        type=["jpg", "jpeg", "png"],
        key="crop_photo"
    )

    farmer_notes = st.text_area(
        "Describe what you noticed (optional)",
        placeholder="Example: yellow spots, curled leaves, or holes..."
    )

    if photo:
        try:
            image = Image.open(photo).convert("RGB")

            st.image(
                image,
                caption="Uploaded crop photo",
                use_container_width=True
            )

            st.info(
                "Your photo was uploaded successfully. A trained AI model "
                "is not connected, so no disease prediction is being made."
            )

            if st.button("Save Photo Check to History"):
                history = load_history()

                history.insert(0, {
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "crop": crop,
                    "notes": farmer_notes.strip() or "No notes provided",
                    "status": "Photo uploaded; AI diagnosis not connected"
                })

                history = history[:50]

                if save_history(history):
                    st.success("Photo-check details saved to local history.")
                else:
                    st.warning(
                        "History could not be saved. The hosting environment "
                        "may not permit writing files."
                    )

        except Exception:
            st.error("Unable to read this photo. Please try another image.")

    st.divider()
    st.subheader("🔊 Multilingual Voice Assistance")

    st.write(
        "Generate a spoken reminder in the selected language. "
        "An internet connection is required for text-to-speech."
    )

    if st.button("🔊 Generate Voice Alert"):
        selected = LANGUAGES[language]
        message = selected["message"]

        try:
            audio_buffer = __import__("io").BytesIO()

            speech = gTTS(
                text=message,
                lang=selected["code"]
            )

            speech.write_to_fp(audio_buffer)
            audio_buffer.seek(0)

            st.audio(audio_buffer.getvalue(), format="audio/mp3")
            st.success(f"Voice reminder generated in {language}.")

        except Exception:
            st.error(
                "Voice generation failed. Check your internet connection. "
                "Text-to-speech availability may vary by language."
            )

    st.caption(
        "This voice alert is a general safety reminder, not a diagnosis "
        "or a pesticide recommendation."
    )

# ---------------- TAB 2: CROP PROTECTION LIBRARY ----------------

with tab2:
    st.header("📚 Crop Disease Reference Library")

    st.write(
        "This is a small educational starter library. It is not a complete "
        "copy of TNAU information and is not an AI prediction."
    )

    available_crops = list(DISEASES.keys())
    selected_library_crop = st.selectbox(
        "Choose a crop to explore",
        available_crops
    )

    disease_options = list(DISEASES[selected_library_crop].keys())
    selected_disease = st.selectbox(
        "Choose a disease or pest",
        disease_options
    )

    info = DISEASES[selected_library_crop][selected_disease]

    st.subheader(f"🌿 {selected_disease}")

    st.markdown("**Common signs**")
    st.write(info["signs"])

    st.markdown("**General prevention practices**")
    st.write(info["prevention"])

    st.warning(
        "Symptoms can look similar across different diseases and nutrient "
        "problems. Confirm the cause before selecting a treatment."
    )

    st.markdown(f"**Reference topic:** {info['source']}")
    st.markdown(f"[Visit TNAU Crop Protection]({TNAU_URL})")

    if st.button("Save Reference to History"):
        history = load_history()

        history.insert(0, {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "crop": selected_library_crop,
            "notes": f"Viewed reference: {selected_disease}",
            "status": "Educational reference viewed; not a diagnosis"
        })

        history = history[:50]

        if save_history(history):
            st.success("Reference saved to local history.")
        else:
            st.warning("Unable to save history in this hosting environment.")

# ---------------- TAB 3: IPM GUIDE ----------------

with tab3:
    st.header("🛡️ Integrated Pest Management (IPM)")

    st.write(
        "Integrated Pest Management combines monitoring, prevention, "
        "and appropriate control methods to manage crop problems."
    )

    st.markdown("""
    **1. Inspect crops regularly**

    Check both sides of leaves, stems, fruits, and the surrounding soil.
    Record new symptoms and how quickly they spread.

    **2. Maintain field hygiene**

    Remove diseased plant material when appropriate and keep tools clean.
    Follow local advice for safe disposal.

    **3. Use good crop practices**

    Maintain suitable spacing, drainage, irrigation, and balanced nutrition.
    Rotate crops where suitable for the crop and local conditions.

    **4. Encourage safe, informed decisions**

    Identify the likely cause before choosing a treatment. Do not mix or
    apply pesticides without reading the approved label.

    **5. Ask an agricultural expert**

    Contact your local agricultural extension service or qualified
    agricultural expert if symptoms spread quickly or the cause is unclear.
    """)

    st.markdown(f"[Read TNAU IPM information]({TNAU_IPM_URL})")

    st.info(
        "Treatment choices depend on the crop, region, disease confirmation, "
        "and locally approved recommendations."
    )

# ---------------- TAB 4: HISTORY ----------------

with tab4:
    st.header("🕘 My CropCare History")

    history = load_history()

    if history:
        st.caption(
            "This history is stored in a local JSON file where file writing "
            "is permitted. It is not a secure cloud database."
        )

        for index, item in enumerate(history):
            with st.expander(
                f"{item.get('date', 'Unknown date')} — "
                f"{item.get('crop', 'Unknown crop')}"
            ):
                st.write("**Details:**", item.get("notes", ""))
                st.write("**Status:**", item.get("status", ""))

        if st.button("Clear History"):
            if save_history([]):
                st.success("History cleared.")
                st.rerun()
            else:
                st.error("Unable to clear the history file.")
    else:
        st.info(
            "No saved entries yet. Save a photo check or a library reference "
            "to see it here."
        )

# ---------------- FOOTER ----------------

st.divider()

st.markdown(
    """
    <div style="text-align:center;">
        <h4>🌱 CropCare AI — Technology for Farmer Support</h4>
        <p>Multilingual assistance • Crop education • Safer farming decisions</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.caption(
    "Educational prototype only. No trained image-classification model, "
    "expert consultation service, SMS gateway, or keypad-phone messaging "
    "service is connected. Verify disease identification and treatment "
    "with a qualified agricultural expert."
)

