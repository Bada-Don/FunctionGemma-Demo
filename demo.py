from transformers import AutoProcessor, AutoModelForCausalLM
import re
import json
from dispatcher import dispatch

LOCAL_DIR = "./local_models/functiongemma-270m-it"
MODEL_NAME = "google/functiongemma-270m-it"

# Load processor and model
print("Loading model...")
processor = AutoProcessor.from_pretrained(MODEL_NAME, cache_dir=LOCAL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, cache_dir=LOCAL_DIR, device_map="auto")

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
                    "state": {"type": "string", "enum": ["on", "off"]}
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
                    "app_name": {"type": "string"}
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
                    "level": {"type": "integer"}
                },
                "required": ["level"]
            }
        }
    }
]

def parse_function_call(output):
    """Parse FunctionGemma output into function name and arguments"""
    # Pattern: call:function_name{arg1:<escape>value1<escape>,arg2:<escape>value2<escape>}
    match = re.search(r'call:(\w+)\{(.+?)\}', output)
    if not match:
        return None, None
    
    func_name = match.group(1)
    args_str = match.group(2)
    
    # Parse arguments
    args = {}
    
    # Pattern 1: arg:<escape>value<escape> (for strings)
    escaped_pattern = r'(\w+):<escape>([^<]+)<escape>'
    for arg_match in re.finditer(escaped_pattern, args_str):
        key = arg_match.group(1)
        value = arg_match.group(2)
        args[key] = value
    
    # Pattern 2: arg:value (for numbers, no escape tags)
    direct_pattern = r'(\w+):(\d+)'
    for arg_match in re.finditer(direct_pattern, args_str):
        key = arg_match.group(1)
        if key not in args:  # Don't overwrite escaped values
            value = int(arg_match.group(2))
            args[key] = value
    
    return func_name, args

def run_query(user_input):
    """Process user input and execute function"""
    messages = [
        {"role": "developer", "content": "You are a model that can do function calling with the following functions"},
        {"role": "user", "content": user_input}
    ]
    
    inputs = processor.apply_chat_template(
        messages, tools=tools, add_generation_prompt=True,
        return_dict=True, return_tensors="pt"
    )
    
    outputs = model.generate(
        **inputs.to(model.device),
        pad_token_id=processor.eos_token_id,
        max_new_tokens=128
    )
    
    response = processor.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
    print(f"Model output: {response}")
    
    # Parse and execute
    func_name, args = parse_function_call(response)
    if func_name:
        print(f"\nCalling: {func_name}({args})")
        result = dispatch({"name": func_name, "arguments": args})
        print(f"Result: {result}")
    else:
        print("No function call detected")

# Test examples
print("\n=== Example 1: Turn WiFi on ===")
run_query("Turn WiFi on")

print("\n=== Example 2: Open Chrome ===")
run_query("Open Chrome browser")

print("\n=== Example 3: Set volume to 50 ===")
run_query("Set volume to 50")
