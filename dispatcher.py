import json
import functions

def dispatch(function_call):
    name = function_call["name"]
    args = function_call["arguments"]

    if name == "toggle_wifi":
        return functions.toggle_wifi(**args)
    elif name == "open_app":
        return functions.open_app(**args)
    elif name == "set_volume":
        return functions.set_volume(**args)
    else:
        return f"Error: Unknown function '{name}'"
