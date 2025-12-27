# FunctionGemma Local Setup

A local function calling system using Google's FunctionGemma-270m model.

## Setup

### 1. Create Virtual Environment

```cmd
python -m venv venv
venv\Scripts\activate
```

### 2. Install Dependencies

```cmd
pip install -r requirements.txt
```

### 3. HuggingFace Authentication

The FunctionGemma model is gated and requires authentication:

1. Create a HuggingFace account at https://huggingface.co/join
2. Go to https://huggingface.co/settings/tokens
3. Create a **new token** with these settings:
   - Token type: **Read** (or Fine-grained with read access)
   - **IMPORTANT**: Enable "Access content in gated repos" permission
4. Copy the token
5. Login via CLI:

```cmd
huggingface-cli login
```

Paste your token when prompted.

6. Accept the model terms at https://huggingface.co/google/functiongemma-270m-it
   - Click "Agree and access repository"

### 4. Download the Model

```cmd
python download_functiongemma.py
```

This will download the model to `./local_models/functiongemma-270m-it`

**Note:** You must also accept the model terms at https://huggingface.co/google/functiongemma-270m-it before downloading.

## Usage

### Interactive Demo (Recommended)

Chat with the model and execute commands in real-time:

```cmd
python interactive_demo.py
```

Try commands like:
- "Open notepad"
- "Type Hello World"
- "Search for Python tutorials"
- "Set volume to 50"

### Multi-Step Demo (Proper Conversation Format)

Execute complex multi-step tasks using FunctionGemma's conversation pattern:

```cmd
python proper_multistep.py
```

This demonstrates how the model can chain multiple function calls through conversation turns, like "Open notepad and type Hello World".

### Basic Demos

Run the original demos:

```cmd
python demo.py
python main.py
```

## Available Functions

- `toggle_wifi(state)` - Turn WiFi on or off
- `open_app(app_name)` - Open an application
- `set_volume(level)` - Set system volume (0-100)

See `schemas.py` for function definitions and `functions.py` for implementations.
