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
- Kashmiri (kas_Arab/kas_Deva)
- Manipuri/Meitei (mni_Mtei/mni_Beng)
- Sindhi (snd_Arab/snd_Deva)""",
    version="1.0.0"
)
