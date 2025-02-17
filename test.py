from fastapi import HTTPException
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor
import os
import gc
import shutil

# Define base model directory
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Define model configurations
MODEL_CONFIGS = {
    "ai4bharat/indictrans2-en-indic-1B": "en-indic",
    "ai4bharat/indictrans2-indic-en-1B": "indic-en",
    "ai4bharat/indictrans2-indic-indic-1B": "indic-indic"
}

# Cache for loaded models and tokenizers
_models = {}
_tokenizers = {}

def get_model_dir(model_name):
    """Get the directory path for a specific model"""
    return os.path.join(MODEL_DIR, MODEL_CONFIGS[model_name])

def is_model_downloaded(model_name):
    """Check if model files exist locally"""
    model_dir = get_model_dir(model_name)
    return os.path.exists(model_dir) and any(f.endswith('.bin') for f in os.listdir(model_dir))

def download_and_save_model(model_name):
    """Download model and save to local directory"""
    print(f"Downloading model {model_name}...")
    model_dir = get_model_dir(model_name)
    
    # Download and save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.save_pretrained(model_dir)
    
    # Download and save model
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)
    model.save_pretrained(model_dir)
    
    print(f"Model saved to {model_dir}")
    return model, tokenizer

def get_model_and_tokenizer(model_name):
    """Load model and tokenizer from local directory or download if not exists"""
    try:
        if model_name not in _models:
            clear_cuda_cache()
            
            model_dir = get_model_dir(model_name)
            
            if is_model_downloaded(model_name):
                print(f"Loading model from {model_dir}")
                _tokenizers[model_name] = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
                _models[model_name] = AutoModelForSeq2SeqLM.from_pretrained(model_dir, trust_remote_code=True)
            else:
                _models[model_name], _tokenizers[model_name] = download_and_save_model(model_name)
            
            if torch.cuda.is_available():
                _models[model_name] = _models[model_name].to("cuda")
                
        return _models[model_name], _tokenizers[model_name]
    except Exception as e:
        clear_cuda_cache()
        raise Exception(f"Error loading model: {str(e)}")

def clear_cuda_cache():
    """Clear CUDA cache and garbage collect"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def translate_text(input_text, source_lang, target_lang):
    """
    Translates text from source language to target language without length restrictions.
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
        
        # Split text into sentences
        sentence_end = '।' if source_lang != 'eng_Latn' else '.'
        sentences = input_text.split(sentence_end)
        sentences = [s + sentence_end for s in sentences if s.strip()]
        
        translated_sentences = []
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            # Process sentence
            batch = processor.preprocess_batch([sentence], src_lang=source_lang, tgt_lang=target_lang)
            
            # Tokenize without length restriction
            inputs = tokenizer(
                batch,
                truncation=False,
                padding=True,
                return_tensors="pt"
            )
            
            if torch.cuda.is_available():
                inputs = inputs.to("cuda")
            
            # Generate translation
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    num_beams=5,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            # Decode translation
            with tokenizer.as_target_tokenizer():
                translated = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
                translated_sentences.append(translated)
            
            clear_cuda_cache()  # Clear cache after each sentence
        
        # Join sentences and post-process
        final_translation = ' '.join(translated_sentences)
        return processor.postprocess_batch([final_translation], lang=target_lang)[0]
        
    except torch.cuda.OutOfMemoryError:
        clear_cuda_cache()
        raise HTTPException(status_code=500, detail="CUDA error: out of memory. Please try again with smaller input.")
    except Exception as e:
        clear_cuda_cache()
        raise Exception(f"Translation error: {str(e)}")

if __name__ == "__main__":
    # Example usage
    hindi_text = "नमस्ते दुनिया"
    english = translate_text(hindi_text, "hin_Deva", "asm_Beng")
    print(f"Hindi to English: {english}")
