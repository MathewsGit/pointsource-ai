from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import math

# Load model
MODEL_NAME = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

app = FastAPI(title="AI Section Generator")

class RequestBody(BaseModel):
    text: str
    max_sections: int = 5
    chunk_size: int = 1024  # max tokens per chunk

def chunk_text(text, tokenizer, chunk_size=1024):
    tokens = tokenizer.encode(text, return_tensors="pt")[0]
    total_tokens = tokens.size(0)
    chunks = []

    for i in range(0, total_tokens, chunk_size):
        chunk_tokens = tokens[i:i+chunk_size]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text)
    return chunks

@app.post("/generate_sections")
def generate_sections(req: RequestBody):
    text_chunks = chunk_text(req.text, tokenizer, req.chunk_size)
    results = []

    for idx, chunk in enumerate(text_chunks):
        prompt = f"Split the following text into professional academic sections with titles. Max sections: {req.max_sections}\n\nText: {chunk}"
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        outputs = model.generate(**inputs, max_length=1024)
        
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        results.append(result)

    # Combine all chunk results
    final_result = "\n\n".join(results)
    return {"sections": final_result}
