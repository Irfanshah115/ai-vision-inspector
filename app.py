import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models

st.set_page_config(page_title="AI Vision Inspector", page_icon="👁️", layout="centered")

st.title("👁️ AI Vision Inspector")
st.subheader("Deep Learning Defect Detection")
st.write("Upload an equipment surface image. AI detects cracks/defects using CNN.")

@st.cache_resource(show_spinner=True)
def get_trained_model():
    np.random.seed(42)

    def create_healthy():
        img = np.ones((64, 64), dtype=np.float32) * 180.0
        img += np.random.normal(0, 10, (64, 64)).astype(np.float32)
        return np.clip(img, 0, 255)

    def create_cracked():
        img = create_healthy()
        sx, sy = np.random.randint(10, 54), np.random.randint(10, 54)
        length = np.random.randint(15, 30)
        angle = np.random.choice([0, 45, 90, 135])
        for i in range(length):
            if angle == 0:      x, y = sx + i, sy
            elif angle == 90:   x, y = sx, sy + i
            elif angle == 45:   x, y = sx + i, sy + i
            else:               x, y = sx + i, sy - i
            if 0 <= x < 64 and 0 <= y < 64:
                img[x, y] = 20.0
                if x + 1 < 64: img[x+1, y] = 40.0
                if y + 1 < 64: img[x, y+1] = 40.0
        return img

    X, y = [], []
    for _ in range(200):
        X.append(create_healthy()); y.append(0)
    for _ in range(200):
        X.append(create_cracked()); y.append(1)

    X = np.array(X).reshape(-1, 64, 64, 1) / 255.0
    y = np.array(y)

    m = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    m.fit(X, y, epochs=10, batch_size=32, verbose=0)
    return m

with st.spinner("🧠 Training AI Model (~10 sec)..."):
    model = get_trained_model()

st.success("✅ AI Model Ready!")

st.divider()
st.header("📤 Upload Equipment Image")

uploaded_file = st.file_uploader("Choose an image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img_gray = image.convert('L').resize((64, 64))
    img_array = np.array(img_gray, dtype=np.float32) / 255.0
    img_array = img_array.reshape(1, 64, 64, 1)

    if st.button("🔍 Inspect with AI", type="primary", use_container_width=True):
        prob = float(model.predict(img_array, verbose=0)[0][0])

        if prob > 0.5:
            st.error(f"## 🔴 DEFECT DETECTED")
            st.write(f"**Confidence:** {prob*100:.1f}%")
            st.write("**Status:** Crack/damage detected.")
            st.write("**Action:** Schedule immediate inspection.")
        else:
            st.success(f"## 🟢 HEALTHY SURFACE")
            st.write(f"**Confidence:** {(1-prob)*100:.1f}%")
            st.write("**Status:** No defects detected.")
            st.write("**Action:** Continue normal operations.")

        st.progress(prob, text=f"Defect Probability: {prob*100:.1f}%")

st.divider()
st.caption("Built by Irfan | Zafar Iqbal ML Course 2026")
