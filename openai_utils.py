import os 
import json
import base64
from openai import OpenAI
from dotenv import load_dotenv
from pygltflib import GLTF2

# Loads the OpenAI API key from the .env file for security purposes
load_dotenv()

def encode_image_to_base64(image_path):
    """Helper to convert images for the AI to process"""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

# Initialize the OpenAI client
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


def openai_3DEditor(user_request, original_glb_path, json_metadata, texture_paths=None, output_folder="objects"):
    """
    1. Sends Text + JSON + Images to GPT-4o.
    2. Receives modified JSON instructions.
    3. Rebuilds the .glb with the changes.
    """
    os.makedirs(output_folder, exist_ok=True)
    
    # 1. Build the Multimodal Message (Text + JSON + Images)
    system_prompt = """
    You are a 3D model configuration editor.
    You will receive a JSON object. You must return the EXACT same JSON structure.
    Do not change keys, do not change hierarchy, and do not remove sections. 
    ONLY update the specific values requested by the user.
    Ouput must be a valid JSON.
    """

    content = [
        {
            "type": "text", 
            "text": (
                f"User Request: {user_request}\n"
                f"Reference JSON to modify: {json.dumps(json_metadata, indent=2)}\n"
            )
        }
    ]

    # Add images if provided
    if texture_paths:
        for path in texture_paths:
            if os.path.exists(path):
                b64_img = encode_image_to_base64(path)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64_img}"}
                })

    # 2. Call OpenAI
    response = client.chat.completions.create(
        model="gpt-4o",
        response_format={ "type": "json_object"},
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": content}],
        temperature=0.2 # Lower temperature for structural accuracy
    )

    # 3. Grab the message object first
    message = response.choices[0].message

    # 4. Check if there is actually content before stripping
    if message.content:
        raw_ai_output = message.content.strip()
    else:
        # This will tell you if the AI refused to answer or just errored out
        print(f"Debug: Refusal reason: {getattr(message, 'refusal', 'Unknown')}")
        raise ValueError("The AI returned an empty response. Check your prompt!")

    output_path = f"json/updated_mask_properties.json" 
    with open(output_path, "w") as f:
        f.write(raw_ai_output)
    
    # Clean up accidental markdown if the AI includes it
    if raw_ai_output.startswith("```json"):
        raw_ai_output = raw_ai_output.replace("```json", "").replace("```", "").strip()

    try:

        updated_json_dict = json.loads(raw_ai_output)
        
        # 1. Load the ORIGINAL binary GLB (this keeps your 3D mesh safe)
        gltf = GLTF2().load(original_glb_path)
        
        # 2. Extract the new values from the AI's dictionary
        # Note: We use .get() to avoid crashing if the AI missed a key
        ai_material = updated_json_dict.get("materials", [{}])[0]
        ai_props = ai_material.get("properties", {})
        
        new_metallic = ai_props.get("metallic", 1.0)
        new_roughness = ai_props.get("roughness", 0.5)

        # 3. MANUALLY apply them to the first material in the actual model
        if gltf.materials:
            mat = gltf.materials[0]
            if mat.pbrMetallicRoughness:
                # GLTF uses 'metallicFactor' and 'roughnessFactor'
                mat.pbrMetallicRoughness.metallicFactor = float(new_metallic)
                mat.pbrMetallicRoughness.roughnessFactor = float(new_roughness)
                
                print(f"Applied Metallic: {new_metallic}")

        # 4. Save the model
        output_filename = f"updated_{os.path.basename(original_glb_path)}"
        final_path = os.path.join(output_folder, output_filename)
        gltf.save(final_path)
        
        return final_path

    except json.JSONDecodeError:
        print("Error: AI returned invalid JSON. Check the output text.")
        print(f"Raw Output: {raw_ai_output}")
        return None