!pip install torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from test import translate_text
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Language Translation API",
    description="API for translating text between Indian languages and English",
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
async def translate(request: TranslationRequest):
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

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
