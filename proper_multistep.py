"""
Proper multi-step execution using FunctionGemma's conversation format.
Based on official documentation pattern.
"""

from transformers import AutoProcessor, AutoModelForCausalLM
import re
import pyautogui
import time
import subprocess

LOCAL_DIR = "./local_models/functiongemma-270m-it"
MODEL_NAME = "google/functiongemma-270m-it"

print("Loading model...")
processor = AutoProcessor.from_pretrained(MODEL_NAME, cache_dir=LOCAL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, cache_dir=LOCAL_DIR, device_map="auto")
print("Model loaded!\n")

# Define tools as Python functions (auto-converted to schemas)
def open_app(app_name: str):
    """
    Open an application by name.
    
    Args:
        app_name: Name of the application to open (e.g., notepad, calculator)
    """
    app_map = {
        'notepad': 'notepad.exe',
        'calculator': 'calc.exe',
        'calc': 'calc.exe',
        'paint': 'mspaint.exe',
    }
    try:
        subprocess.Popen(app_map.get(app_name.lower(), app_name))
        return {"status": "success", "message": f"Opened {app_name}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def type_text(text: str):
    """
    Type text using the keyboard.
    
    Args:
        text: The text to type
    """
    time.sleep(1)  # Wait for window to be ready
    try:
        pyautogui.write(text, interval=0.05)
        return {"status": "success", "message": f"Typed: {text}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def press_key(key: str):
    """
    Press a keyboard key.
    
    Args:
        key: The key to press (e.g., enter, tab, escape)
    """
    time.sleep(0.3)
    try:
        pyautogui.press(key)
        return {"status": "success", "message": f"Pressed: {key}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def search_web(query: str):
    """
    Search the web for a query.
    
    Args:
        query: The search query
    """
    try:
        import webbrowser
        webbrowser.open(f"https://www.google.com/search?q={query}")
        return {"status": "success", "message": f"Searching for: {query}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def task_done():
    """
    Indicate that the task is complete.
    """
    return {"status": "complete", "message": "Task finished"}

# Map function names to actual functions
AVAILABLE_FUNCTIONS = {
    'open_app': open_app,
    'type_text': type_text,
    'press_key': press_key,
    'search_web': search_web,
    'task_done': task_done
}

def extract_tool_calls(text):
    """Extract function calls from model output (from official docs)"""
    def cast(v):
        try: return int(v)
        except:
            try: return float(v)
            except: return {'true': True, 'false': False}.get(v.lower(), v.strip("'\""))

    return [{
        "name": name,
        "arguments": {
            k: cast((v1 or v2).strip())
            for k, v1, v2 in re.findall(r"(\w+):(?:<escape>(.*?)<escape>|([^,}]*))", args)
        }
    } for name, args in re.findall(r"<start_function_call>call:(\w+)\{(.*?)\}<end_function_call>", text, re.DOTALL)]

def execute_complex_task(user_prompt, max_turns=10):
    """
    Execute a potentially multi-step task using conversation turns.
    The model can call multiple functions in sequence.
    """
    print(f"\n{'='*70}")
    print(f"📋 Task: {user_prompt}")
    print(f"{'='*70}\n")
    
    # Initialize conversation
    message = [
        {
            "role": "developer",
            "content": "You are a model that can do function calling with the following functions. Execute the user's task step by step. Call task_done when finished."
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]
    
    tools = list(AVAILABLE_FUNCTIONS.values())
    
    for turn in range(1, max_turns + 1):
        print(f"Turn {turn}:")
        
        # Generate model response
        inputs = processor.apply_chat_template(
            message,
            tools=tools,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        
        out = model.generate(
            **inputs.to(model.device),
            pad_token_id=processor.eos_token_id,
            max_new_tokens=256
        )
        
        generated_tokens = out[0][len(inputs["input_ids"][0]):]
        output = processor.decode(generated_tokens, skip_special_tokens=True)
        
        print(f"  🤖 Model output: {output}")
        
        # Check if model is just responding (no function call)
        if "<start_function_call>" not in output:
            print(f"  💬 Final response: {output}")
            break
        
        # Extract function calls
        calls = extract_tool_calls(output)
        
        if not calls:
            print("  ✗ No function calls detected")
            break
        
        # Add assistant's tool calls to conversation
        message.append({
            "role": "assistant",
            "tool_calls": [{"type": "function", "function": call} for call in calls]
        })
        
        # Execute each function call
        results = []
        for call in calls:
            func_name = call['name']
            func_args = call['arguments']
            
            print(f"  ⚙️  Calling: {func_name}({func_args})")
            
            if func_name in AVAILABLE_FUNCTIONS:
                result = AVAILABLE_FUNCTIONS[func_name](**func_args)
                results.append({"name": func_name, "response": result})
                print(f"     Result: {result}")
                
                # Check if task is done
                if func_name == 'task_done':
                    print("\n  ✅ Task completed!")
                    return
            else:
                results.append({
                    "name": func_name,
                    "response": {"status": "error", "message": f"Unknown function: {func_name}"}
                })
        
        # Add tool results to conversation
        message.append({
            "role": "tool",
            "content": results
        })
        
        print()
        time.sleep(0.5)
    
    print(f"{'='*70}\n")

# Test cases
print("="*70)
print("  PROPER MULTI-STEP EXECUTION")
print("  (Using FunctionGemma's conversation format)")
print("="*70)

test_tasks = [
    'Open notepad and type "Hello World"',
    'Search for Python tutorials',
    'Open calculator and then open notepad',
]

for task in test_tasks:
    execute_complex_task(task, max_turns=10)
    time.sleep(2)

print("\n" + "="*70)
print("✅ Demo complete!")
print("\n💡 Note: Success depends on whether the 270M model can reason")
print("   about multi-step sequences. It may work for simple cases")
print("   but struggle with complex planning.")
print("="*70)
