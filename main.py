from transformers import AutoProcessor, AutoModelForCausalLM
import json
import os

LOCAL_DIR = "./local_models/functiongemma-270m-it"
MODEL_NAME = "google/functiongemma-270m-it"

# Ensure directory exists
os.makedirs(LOCAL_DIR, exist_ok=True)

# Load processor and model
print("Loading model...")
processor = AutoProcessor.from_pretrained(
    MODEL_NAME,
    cache_dir=LOCAL_DIR
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    cache_dir=LOCAL_DIR,
    device_map="auto"
)

# Define function schemas
tools = [
    {
        "type": "function",
        "function": {
            "name": "toggle_wifi",
            "description": "Turn WiFi on or off",
            "parameters": {
                "type": "object",
                "properties": {
                    "state": {
                        "type": "string",
                        "description": "Either 'on' or 'off'",
                        "enum": ["on", "off"]
                    }
                },
                "required": ["state"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "open_app",
            "description": "Open an application",
            "parameters": {
                "type": "object",
                "properties": {
                    "app_name": {
                        "type": "string",
                        "description": "Name of the application to open"
                    }
                },
                "required": ["app_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_volume",
            "description": "Set system volume level",
            "parameters": {
                "type": "object",
                "properties": {
                    "level": {
                        "type": "integer",
                        "description": "Volume level from 0 to 100"
                    }
                },
                "required": ["level"]
            }
        }
    }
]

# Create message with proper format
messages = [
    {
        "role": "developer",
        "content": "You are a model that can do function calling with the following functions"
    },
    {
        "role": "user",
        "content": "Turn WiFi on"
    }
]

# Apply chat template
print("Generating function call...")
inputs = processor.apply_chat_template(
    messages,
    tools=tools,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)

# Generate
outputs = model.generate(
    **inputs.to(model.device),
    pad_token_id=processor.eos_token_id,
    max_new_tokens=128
)

# Decode only the new tokens
response = processor.decode(
    outputs[0][len(inputs["input_ids"][0]):],
    skip_special_tokens=True
)

print(f"\nModel output: {response}")

