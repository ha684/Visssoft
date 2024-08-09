from transformers import pipeline, AutoTokenizer
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
correction_pipeline = pipeline("text2text-generation", model="bmd1905/vietnamese-correction-v2", device=0 if torch.cuda.is_available() else -1)
tokenizer = AutoTokenizer.from_pretrained("bmd1905/vietnamese-correction-v2")
MAX_LENGTH = 256
BATCH_SIZE = 32  # Adjust based on GPU memory capacity

def split_text(text, max_length):
    tokens = tokenizer(text, return_tensors='pt', truncation=False, padding=False)
    input_ids = tokens['input_ids'][0]
    
    chunks = []
    for i in range(0, len(input_ids), max_length):
        chunk = input_ids[i:i + max_length]
        chunks.append(chunk)
        
    return chunks

def batch_predict(chunks, batch_size):
    corrected_chunks = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        batch_inputs = tokenizer.batch_decode(batch, skip_special_tokens=True)
        predictions = correction_pipeline(batch_inputs, max_length=MAX_LENGTH)
        corrected_chunks.extend([pred['generated_text'] for pred in predictions])
    return corrected_chunks

def corrector(texts):
    if isinstance(texts, str):
        texts = [texts]
    
    corrected_texts = []
    
    for text in texts:
        chunks = split_text(text, MAX_LENGTH)
        corrected_chunks = batch_predict(chunks, BATCH_SIZE)
        corrected_texts.append(' '.join(corrected_chunks))
    
    return corrected_texts if len(corrected_texts) > 1 else corrected_texts[0]
