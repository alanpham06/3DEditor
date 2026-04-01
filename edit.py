import streamlit as st
import os
from glb_processor import extract_properties, extract_textures
from openai_utils import openai_3DEditor

st.title("Active 3D Editor 🛠️")

# 1. Simple File Uploader
uploaded_file = st.file_uploader("Upload your .glb model", type=["glb"])

if uploaded_file:
    # Save the uploaded file locally so your processor can see it
    with open(os.path.join("objects", uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.success(f"Loaded {uploaded_file.name}")

    # 2. The Pop-up style Text Input
    user_prompt = st.text_area("What modifications should the AI make?", 
                               placeholder="e.g. Change the texture to rusted metal...")

    if st.button("Process Model"):
        with st.spinner("AI is analyzing and modifying your 3D model..."):
            # 3. Trigger your backend logic
            json_path = extract_properties(f"objects/{uploaded_file.name}")
            textures = extract_textures(f"objects/{uploaded_file.name}")
            
            # Call your OpenAI function
            result = openai_3DEditor(user_prompt, f"objects/{uploaded_file.name}", json_path, textures)

            # Format result
            result_msg = "Done! You can find the updated model in the objects folder titled 'updated_...'\n"
            result_path = f"Path: {result}"
            
            st.balloons()
            st.success(f"{result_msg}{result_path}")