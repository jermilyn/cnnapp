import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# --- CONFIGURATION ---
MODEL_PATH = 'herb_model_lab5.h5'
IMG_SIZE = (120, 120)

CLASS_NAMES = [
    'Cardiospermum halicacabum (Balloon Vine)',
    'Curcuma longa (Turmeric)',
    'Mentha (Mint)',
    'Spinacia oleracea (Palak(Spinach))',
    'Zingiber officinale (Ginger)'
]

# --- HERB INFO DATABASE ---
HERB_INFO = {
    'Cardiospermum halicacabum (Balloon Vine)': {
        'common_name': 'Balloon Vine',
        'scientific_name': 'Cardiospermum halicacabum',
        'family': 'Sapindaceae',
        'description': (
            'Balloon Vine is a climbing plant known for its inflated, balloon-like seed pods. '
            'It is widely used in traditional medicine across Asia and Africa.'
        ),
        'medicinal_uses': [
            'Treats skin conditions like eczema and psoriasis',
            'Used as an anti-inflammatory agent',
            'Relieves joint and muscle pain',
            'Helps with respiratory issues such as asthma',
        ],
        'parts_used': 'Leaves, roots, seeds',
        'caution': 'Consult a healthcare provider before use. Not recommended during pregnancy.',
    },
    'Curcuma longa (Turmeric)': {
        'common_name': 'Turmeric',
        'scientific_name': 'Curcuma longa',
        'family': 'Zingiberaceae',
        'description': (
            'Turmeric is a bright yellow spice and medicinal herb native to South Asia. '
            'Its active compound, curcumin, is a powerful antioxidant and anti-inflammatory agent.'
        ),
        'medicinal_uses': [
            'Powerful anti-inflammatory and antioxidant properties',
            'Supports liver health and detoxification',
            'Aids digestion and reduces bloating',
            'May help manage arthritis symptoms',
            'Used in wound healing and skin care',
        ],
        'parts_used': 'Rhizome (root)',
        'caution': 'High doses may cause stomach upset. Avoid before surgery due to blood-thinning effects.',
    },
    'Mentha (Mint)': {
        'common_name': 'Mint',
        'scientific_name': 'Mentha spp.',
        'family': 'Lamiaceae',
        'description': (
            'Mint is an aromatic herb with a refreshing flavor, widely used in cooking, '
            'teas, and traditional medicine for its cooling and soothing properties.'
        ),
        'medicinal_uses': [
            'Relieves indigestion, bloating, and nausea',
            'Soothes headaches when applied topically',
            'Acts as a natural decongestant for colds',
            'Freshens breath and supports oral health',
            'Has antimicrobial and antifungal properties',
        ],
        'parts_used': 'Leaves, essential oil',
        'caution': 'Peppermint oil should not be applied near the face of infants or young children.',
    },
    'Spinacia oleracea (Palak(Spinach))': {
        'common_name': 'Spinach (Palak)',
        'scientific_name': 'Spinacia oleracea',
        'family': 'Amaranthaceae',
        'description': (
            'Spinach is a nutrient-dense leafy green vegetable packed with vitamins, minerals, '
            'and antioxidants. It has been used for centuries as both food and medicine.'
        ),
        'medicinal_uses': [
            'Rich in iron — helps prevent and treat anemia',
            'High in Vitamin K for bone health',
            'Supports eye health with lutein and zeaxanthin',
            'Antioxidants help reduce oxidative stress',
            'Supports healthy blood pressure levels',
        ],
        'parts_used': 'Leaves',
        'caution': 'High in oxalates — those with kidney stones should consume in moderation.',
    },
    'Zingiber officinale (Ginger)': {
        'common_name': 'Ginger',
        'scientific_name': 'Zingiber officinale',
        'family': 'Zingiberaceae',
        'description': (
            'Ginger is a flowering plant whose rhizome (root) is widely used as a spice and '
            'traditional medicine. It has a long history of use in Asian, Indian, and Arabic herbal medicine.'
        ),
        'medicinal_uses': [
            'Highly effective against nausea and vomiting',
            'Strong anti-inflammatory and analgesic properties',
            'Aids digestion and reduces bloating',
            'May help lower blood sugar levels',
            'Supports immune function and fights infections',
        ],
        'parts_used': 'Rhizome (root)',
        'caution': 'May interact with blood-thinning medications. Use cautiously if you have a bleeding disorder.',
    },
}

# --- APP LAYOUT ---
st.set_page_config(page_title="Medicinal Herb Identifier", layout="centered")
st.title("🌿 Medicinal Herb Classifier")
st.write("Upload an image of a herb, and the AI will identify its category.")

# --- LOAD MODEL ---
@st.cache_resource
def load_herb_model():
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH)
    return None

model = load_herb_model()

if model is None:
    st.error(f"Model file '{MODEL_PATH}' not found. Please run your training script first.")
else:
    # --- IMAGE UPLOAD ---
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Herb Image', use_container_width=True)

        # --- PREDICTION LOGIC ---
        with st.spinner('Analyzing...'):
            img = image.resize(IMG_SIZE)
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)

            predictions = model.predict(img_array)
            score = tf.nn.softmax(predictions[0])

            result_class = CLASS_NAMES[np.argmax(score)]
            confidence = 100 * np.max(score)

        # --- DISPLAY RESULTS ---
        st.success(f"**Prediction:** {result_class}")
        st.info(f"**Confidence Level:** {confidence:.2f}%")
        st.progress(int(confidence))

        # --- HERB INFO PANEL ---
        st.divider()
        st.subheader("🌱 Herb Information")

        herb = HERB_INFO.get(result_class)
        if herb:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Common Name:** {herb['common_name']}")
                st.markdown(f"**Scientific Name:** *{herb['scientific_name']}*")
                st.markdown(f"**Family:** {herb['family']}")
                st.markdown(f"**Parts Used:** {herb['parts_used']}")
            with col2:
                st.markdown("**Medicinal Uses:**")
                for use in herb['medicinal_uses']:
                    st.markdown(f"- {use}")

            st.markdown(f"**Description:** {herb['description']}")

            st.warning(f"⚠️ **Caution:** {herb['caution']}")
        else:
            st.write("No additional information available for this herb.")
