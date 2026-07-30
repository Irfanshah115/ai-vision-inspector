import streamlit as st              # Web app banane ke liye
import numpy as np                   # Arrays handle karne ke liye
from PIL import Image               # User ki uploaded image read karne ke liye
import tensorflow as tf             # Trained CNN model load karne ke liye

# Page settings
st.set_page_config(page_title="AI Vision Inspector", page_icon="👁️", layout="centered")

# Title
st.title("👁️ AI Vision Inspector")
st.subheader("Deep Learning Defect Detection")
st.write("Upload an equipment surface image. AI will detect cracks or defects.")

# Trained model load karo
# Model training ke waqt seekha tha: smooth = Healthy, black line = Cracked
model = tf.keras.models.load_model('defect_detection_model.keras')
st.success("✅ AI Model Loaded! Ready to inspect.")

st.divider()
st.header("📤 Upload Equipment Image")

# File uploader — user image upload kare
# accept = sirf jpg, jpeg, png files
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    
    # ============================================================
    # STEP 1: Image ko read karo aur display karo
    # ============================================================
    image = Image.open(uploaded_file)          # PIL se image open karo
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # ============================================================
    # STEP 2: Image ko CNN format mein convert karo
    # ============================================================
    # Model ne 64x64 grayscale images pe training ki thi
    # User ki image bhi same format mein lao
    
    image = image.convert('L')                 # 'L' = Grayscale mode (color hatao)
    image = image.resize((64, 64))             # 64x64 pixels mein resize karo
    image_array = np.array(image)              # PIL image ko NumPy array mein convert
    image_array = image_array / 255.0          # Normalize: 0-255 se 0-1 mein lao
    image_array = image_array.reshape(1, 64, 64, 1)  # CNN format: (1 image, 64H, 64W, 1 channel)
    
    # ============================================================
    # STEP 3: AI Prediction
    # ============================================================
    if st.button("🔍 Inspect with AI", type="primary", use_container_width=True):
        
        prediction = model.predict(image_array)[0][0]   # Probability lo
