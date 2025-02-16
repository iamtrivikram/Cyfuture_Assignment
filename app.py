
import streamlit as st
import cv2
import numpy as np
import joblib
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as pi
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/content/drive/My Drive/gpt2_apple_disease_model"
gpt_model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Class labels and disease info
class_labels = ["Apple Scab", "Black Rot", "Cedar Apple Rust", "Healthy"]
disease_info = {
    "Apple Scab": {
        "Cause": "Fungal infection caused by Venturia inaequalis.",
        "Precautions": "Prune infected leaves, apply fungicides, use disease-resistant varieties.",
        "Crop Management": "Remove fallen leaves, ensure proper spacing for air circulation.",
        "Fertilizers": "Use balanced fertilizers with potassium and phosphorus."
    },
    "Black Rot": {
        "Cause": "Fungus Botryosphaeria obtusa.",
        "Precautions": "Remove infected fruits, apply copper-based fungicides.",
        "Crop Management": "Avoid overhead watering, maintain soil health.",
        "Fertilizers": "Use nitrogen-based fertilizers in moderate amounts."
    },
    "Cedar Apple Rust": {
        "Cause": "Fungus Gymnosporangium juniperi-virginianae.",
        "Precautions": "Remove nearby cedar trees, apply fungicides during spring.",
        "Crop Management": "Use resistant apple varieties, prune infected branches.",
        "Fertilizers": "Apply organic compost and balanced NPK fertilizers."
    },
    "Healthy": {
        "Message": "No disease detected! Maintain good agricultural practices."
    }
}

# Function to generate chatbot response
def chatbot_response(query, disease):
    context = f"The detected disease is {disease}. {disease_info.get(disease, {}).get('Cause', 'No information available')}"
    prompt = f"{context}\nUser: {query}\nChatbot:"
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = gpt2_model.generate(inputs, max_length=150, do_sample=True, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("Chatbot:")[-1].strip()

st.title("🌿 Plant Disease Detection & Chatbot 🤖")
st.write("Upload a leaf image to detect disease and chat about its management.")

with st.spinner("Loading Model Into Memory...."):
    model_save_path = "leaf_disease_model.pkl"
    model = joblib.load(model_save_path)

uploaded_image = st.file_uploader(label="Upload Image", type=["png", "jpg", "jpeg"])

if uploaded_image is not None:
    img_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    img = cv2.imdecode(img_bytes, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resize = cv2.resize(img, (224, 224))
    st.image(img, channels="RGB")
    img_resize = pi(img_resize)
    img_reshape = img_resize[np.newaxis, ...]

    predict_img = st.button("Predict")
    
    if predict_img:
        predictions = model.predict(img_reshape)
        predicted_class = np.argmax(predictions, axis=1)[0]
        disease_name = class_labels[predicted_class]
        
        st.title(f"Prediction: {disease_name}")
        for key, value in disease_info[disease_name].items():
            st.write(f"**{key}:** {value}")
        
        user_query = st.text_input("Ask me anything about the disease")
        
        if user_query:
            response = chatbot_response(user_query, disease_name)
            st.write("🤖 Chatbot Response:", response)

if __name__ == "__main__":
    st.write("### Ready to Detect and Assist!")
