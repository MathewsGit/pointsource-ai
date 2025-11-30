from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load model
MODEL_NAME = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

app = FastAPI(title="AI Section Generator")

class RequestBody(BaseModel):
    text: str
    max_sections: int = 5

@app.post("/generate_sections")
def generate_sections(req: RequestBody):
    prompt = f"Split the following text into professional academic sections with titles. Max sections: {req.max_sections}\n\nText: {req.text}"
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    outputs = model.generate(**inputs, max_length=1024)
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"sections": result}
