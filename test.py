import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor
import os
import gc

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

def clear_cuda_cache():
    """Clear CUDA cache and garbage collect"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def get_model_and_tokenizer(model_name):
    """Load and cache model and tokenizer"""
    try:
        if model_name not in _models:
            clear_cuda_cache()  # Clear cache before loading new model
            
            # Load tokenizer first
            if model_name not in _tokenizers:
                _tokenizers[model_name] = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    cache_dir=CACHE_DIR,
                    local_files_only=True  # Try to load from cache first
                )
            
            # Then load model
            try:
                # First try loading from cache
                _models[model_name] = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    cache_dir=CACHE_DIR,
                    local_files_only=True
                )
            except Exception:
                # If not in cache, download and save
                print(f"Downloading model {model_name} to {CACHE_DIR}")
                _models[model_name] = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    cache_dir=CACHE_DIR
                )
            
            if torch.cuda.is_available():
                _models[model_name] = _models[model_name].to("cuda")
                
        return _models[model_name], _tokenizers[model_name]
    
    except Exception as e:
        clear_cuda_cache()
        raise Exception(f"Error loading model: {str(e)}")

def translate_text(input_text, source_lang, target_lang, max_length=512):  # Reduced default max_length
    """
    Translates text from source language to target language.
    """
    try:
        # Select appropriate model based on language direction
        if source_lang == "eng_Latn":
            model_name = "ai4bharat/indictrans2-en-indic-1B"
        else:
            model_name = "ai4bharat/indictrans2-indic-en-1B" if target_lang == "eng_Latn" else "ai4bharat/indictrans2-indic-indic-1B"
        
        # Get or load model and tokenizer
        model, tokenizer = get_model_and_tokenizer(model_name)
        processor = IndicProcessor(inference=True)
        
        # Process input in chunks if too long
        if len(input_text) > max_length:
            # Simple sentence splitting
            sentences = input_text.split('।' if source_lang != 'eng_Latn' else '.')
            chunks = []
            current_chunk = ''
            
            for sentence in sentences:
                if len(current_chunk) + len(sentence) < max_length:
                    current_chunk += sentence + ('।' if source_lang != 'eng_Latn' else '.')
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = sentence + ('।' if source_lang != 'eng_Latn' else '.')
            if current_chunk:
                chunks.append(current_chunk)
        else:
            chunks = [input_text]
        
        translated_chunks = []
        for chunk in chunks:
            # Process chunk
            batch = processor.preprocess_batch([chunk], src_lang=source_lang, tgt_lang=target_lang)
            
            # Tokenize
            inputs = tokenizer(
                batch,
                truncation=True,
                padding=True,
                return_tensors="pt",
                max_length=max_length
            )
            
            if torch.cuda.is_available():
                inputs = inputs.to("cuda")
            
            # Generate translation
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    num_beams=5,
                    max_length=max_length,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            # Decode translation
            with tokenizer.as_target_tokenizer():
                translated = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
                translated_chunks.append(translated)
            
            clear_cuda_cache()  # Clear cache after each chunk
        
        # Join chunks and post-process
        final_translation = ' '.join(translated_chunks)
        return processor.postprocess_batch([final_translation], lang=target_lang)[0]
        
    except Exception as e:
        clear_cuda_cache()
        raise Exception(f"Translation error: {str(e)}")

if __name__ == "__main__":
    # Example usage
    hindi_text = "नमस्ते दुनिया"
    english = translate_text(hindi_text, "hin_Deva", "asm_Beng")
    print(f"Hindi to English: {english}")
