import streamlit as st
from PIL import Image
from model.predict import predict_sign

st.set_page_config(
    page_title="AI Sign Language Recognition",
    layout="wide"
)

st.title("✋ AI Sign Language Recognition System")

st.write("""
Upload a sign language hand gesture image to predict the alphabet using Deep Learning.
""")

uploaded_file = st.file_uploader(
    "Upload Hand Sign Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Hand Sign",
        use_column_width=True
    )

    if st.button("Predict Sign"):
        prediction, confidence = predict_sign(image)

        st.subheader("📊 Prediction Result")

        st.success(f"Predicted Sign: {prediction}")
        st.info(f"Confidence Score: {confidence:.2f}%")
