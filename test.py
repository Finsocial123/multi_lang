import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor
import os

# Define model paths and cache directory
MODEL_PATHS = {
    "ai4bharat/indictrans2-en-indic-1B": "indictrans2-en-indic-1B",
    "ai4bharat/indictrans2-indic-en-1B": "indictrans2-indic-en-1B",
    "ai4bharat/indictrans2-indic-indic-1B": "indictrans2-indic-indic-1B"
}

CACHE_DIR = os.path.join(os.path.dirname(__file__), "model_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Cache for loaded models and tokenizers
_models = {}
_tokenizers = {}

def model_exists(model_name):
    """Check if model files already exist in cache"""
    model_path = os.path.join(CACHE_DIR, MODEL_PATHS[model_name])
    return os.path.exists(model_path)

def get_model_and_tokenizer(model_name):
    """Load and cache model and tokenizer"""
    if model_name not in _models:
        print(f"Loading model: {model_name}")
        
        if model_exists(model_name):
            print(f"Using cached model from {CACHE_DIR}")
        else:
            print(f"Downloading model to {CACHE_DIR}")
            
        _tokenizers[model_name] = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=CACHE_DIR
        )
        _models[model_name] = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, 
            trust_remote_code=True,
            cache_dir=CACHE_DIR
        )
        if torch.cuda.is_available():
            _models[model_name] = _models[model_name].to("cuda")
            
        print(f"Model loaded successfully")
    return _models[model_name], _tokenizers[model_name]

def translate_text(input_text, source_lang, target_lang, max_length=10000000):  # Added max_length parameter
    """
    Translates text from source language to target language.
    Args:
        input_text (str): Input text to translate
        source_lang (str): Source language code (e.g., "hin_Deva", "eng_Latn")
        target_lang (str): Target language code (e.g., "eng_Latn", "hin_Deva")
        max_length (int, optional): Maximum length for input and output text. Defaults to 1024.
    Returns:
        str: Translated text
    """
    # Select appropriate model based on language direction
    if source_lang == "eng_Latn":
        model_name = "ai4bharat/indictrans2-en-indic-1B"
    else:
        model_name = "ai4bharat/indictrans2-indic-en-1B" if target_lang == "eng_Latn" else "ai4bharat/indictrans2-indic-indic-1B"
    
    # Get or load model and tokenizer
    model, tokenizer = get_model_and_tokenizer(model_name)
    processor = IndicProcessor(inference=True)
    
    # Process input
    batch = processor.preprocess_batch([input_text], src_lang=source_lang, tgt_lang=target_lang)
    
    # Tokenize with max_length
    inputs = tokenizer(
        batch, 
        truncation=True, 
        padding=True, 
        return_tensors="pt",
        max_length=max_length
    )
    
    if torch.cuda.is_available():
        inputs = inputs.to("cuda")
    
    # Generate translation with max_length
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            num_beams=5,
            max_length=max_length,
            max_new_tokens=max_length,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True
        )
    
    # Decode translation
    with tokenizer.as_target_tokenizer():
        translated = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    
    # Post-process
    return processor.postprocess_batch([translated], lang=target_lang)[0]

if __name__ == "__main__":
    # Example usage
    hindi_text = "नमस्ते दुनिया"
    english = translate_text(hindi_text, "hin_Deva", "asm_Beng")
    print(f"Hindi to English: {english}")
