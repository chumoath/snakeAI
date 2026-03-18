# switch_model.py
import json
import os

CONFIG_FILE = "model_config.json"

def set_model(model_name):
    config = {"model_name": model_name}
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f)
    print(f"模型已切换为: {model_name}")

def get_model():
    try:
        with open(CONFIG_FILE, "r") as f:
            config = json.load(f)
            return config.get("model_name", "deepseek-r1:32b")
    except:
        return "deepseek-r1:32b"

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        set_model(sys.argv[1])
    else:
        print(f"当前模型: {get_model()}")
        print("用法: python switch_model.py <model_name>")
