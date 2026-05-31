import os
from PIL import Image
from google import genai
from google.genai import types

# 1. Initialize the Client
# Ensure you have set your API key in environment variables: export GOOGLE_API_KEY='your_key'
client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))


def get_pid_annotations(image_path, system_prompt, user_prompt):
    # Load the image
    img = Image.open(image_path)

    # 2. Configure the Model Call
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=[img, user_prompt],
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            # Force the model to return a JSON object
            response_mime_type="application/json",
            # Optional: Gemini 3 supports 'thinking' for complex reasoning
            thinking_config=types.ThinkingConfig(
                thinking_level=types.ThinkingLevel.MINIMAL
            )
        ),
    )

    # 3. Access the JSON response
    return response.text


# --- Example Usage ---
sys_instruction = "You are a specialized P&ID engineer. Detect symbols and line numbers. Output ONLY a valid JSON object."
user_input = "Identify all gate valves and line numbers in this drawing. Provide bounding boxes in [ymin, xmin, ymax, xmax] format."
image_file = "backend/sample.png"

try:
    json_output = get_pid_annotations(image_file, sys_instruction, user_input)
    print(json_output)
except Exception as e:
    print(f"Error calling Gemini API: {e}")
