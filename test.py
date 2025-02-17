from fastapi import HTTPException
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor
import os
import gc

# Define model paths and required files
MODEL_CONFIGS = {
    "ai4bharat/indictrans2-en-indic-1B": {
        "dir": "indictrans2-en-indic-1B",
        "files": ["config.json", "pytorch_model.bin", "tokenizer.json", "tokenizer_config.json"]
    },
    "ai4bharat/indictrans2-indic-en-1B": {
        "dir": "indictrans2-indic-en-1B",
        "files": ["config.json", "pytorch_model.bin", "tokenizer.json", "tokenizer_config.json"]
    },
    "ai4bharat/indictrans2-indic-indic-1B": {
        "dir": "indictrans2-indic-indic-1B",
        "files": ["config.json", "pytorch_model.bin", "tokenizer.json", "tokenizer_config.json"]
    }
}

CACHE_DIR = os.path.join(os.path.dirname(__file__), "model_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Cache for loaded models and tokenizers
_models = {}
_tokenizers = {}

def is_model_cached(model_name):
    """Check if all required model files exist in cache"""
    model_dir = os.path.join(CACHE_DIR, MODEL_CONFIGS[model_name]["dir"])
    if not os.path.exists(model_dir):
        return False
    
    for file in MODEL_CONFIGS[model_name]["files"]:
        if not os.path.exists(os.path.join(model_dir, file)):
            return False
    return True

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
            
            if is_model_cached(model_name):
                print(f"Loading model from cache: {CACHE_DIR}")
                # Load from cache
                try:
                    _tokenizers[model_name] = AutoTokenizer.from_pretrained(
                        os.path.join(CACHE_DIR, MODEL_CONFIGS[model_name]["dir"]),
                        trust_remote_code=True,
                        local_files_only=True
                    )
                    
                    _models[model_name] = AutoModelForSeq2SeqLM.from_pretrained(
                        os.path.join(CACHE_DIR, MODEL_CONFIGS[model_name]["dir"]),
                        trust_remote_code=True,
                        local_files_only=True
                    )
                    print("Model loaded successfully from cache")
                except Exception as e:
                    print(f"Error loading from cache: {str(e)}")
                    raise
            else:
                print(f"Downloading model {model_name} to {CACHE_DIR}")
                # Download model and tokenizer
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
                print("Model downloaded and loaded successfully")
            
            if torch.cuda.is_available():
                _models[model_name] = _models[model_name].to("cuda")
                
        return _models[model_name], _tokenizers[model_name]
    
    except Exception as e:
        clear_cuda_cache()
        raise Exception(f"Error loading model: {str(e)}")

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
