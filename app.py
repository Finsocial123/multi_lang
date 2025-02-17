from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security.api_key import APIKeyHeader, APIKey
from pydantic import BaseModel
from test import translate_text
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
import os

API_KEY = "hindAilanguageTransl@tor!token"  # Replace with your actual API key
API_KEY_NAME = "hindi_ai_api_key"

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key_header: str = Security(api_key_header)):
    if not api_key_header:
        raise HTTPException(status_code=403, detail="API Key missing")
    if api_key_header != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key_header

app = FastAPI(
    title="Language Translation API",
    description="""API for translating text between Indian languages and English.

Supported Languages:
- English (eng_Latn)
- Hindi (hin_Deva)
- Bengali (ben_Beng)
- Gujarati (guj_Gujr)
- Kannada (kan_Knda)
- Malayalam (mal_Mlym)
- Marathi (mar_Deva)
- Nepali (npi_Deva)
- Oriya (ory_Orya)
- Punjabi (pan_Guru)
- Sanskrit (san_Deva)
- Tamil (tam_Taml)
- Telugu (tel_Telu)
- Urdu (urd_Arab)
- Assamese (asm_Beng)
- Kashmiri (kas_Arab)
- Manipuri/Meitei (mni_Mtei)
- Sindhi (snd_Arab)""",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TranslationRequest(BaseModel):
    text: str
    source_lang: str
    target_lang: str

class TranslationResponse(BaseModel):
    translated_text: str
    source_lang: str
    target_lang: str

@app.post("/translate/", response_model=TranslationResponse)
async def translate(request: TranslationRequest, api_key: APIKey = Depends(get_api_key)):
    try:
        translated = translate_text(
            request.text,
            request.source_lang,
            request.target_lang
        )
        
        return TranslationResponse(
            translated_text=translated,
            source_lang=request.source_lang,
            target_lang=request.target_lang
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint without API key requirement
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app,port=8000)
