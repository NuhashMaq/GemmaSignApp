from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load Gemma 3n model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b")

def gemma_translate(gesture_label):
    prompt = f"Translate this sign language gesture to natural language: {gesture_label}"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=50)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

if __name__ == "__main__":
    example = "HELLO"
    translation = gemma_translate(example)
    print(f"Gesture: {example}\nTranslation: {translation}")
