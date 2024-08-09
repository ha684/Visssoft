# coding: utf8
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

model_path = "vin-ai/PhoGPT-4B-Chat"  

config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)  
config.init_device = "cuda"
# config.attn_config['attn_impl'] = 'flash' # If installed: this will use either Flash Attention V1 or V2 depending on what is installed

# model = AutoModelForCausalLM.from_pretrained(model_path, config=config, torch_dtype=torch.bfloat16, trust_remote_code=True)
# If your GPU does not support bfloat16:
model = AutoModelForCausalLM.from_pretrained(model_path, config=config, torch_dtype=torch.float16, trust_remote_code=True)
model.eval()  

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)  

PROMPT_TEMPLATE = "### Câu hỏi: {instruction}\n### Trả lời:"  

instruction = "Sửa lỗi chính tả:\nTriệt phá băng nhóm kướp ô tô, sử dụng \"vũ khí nóng\""

input_prompt = PROMPT_TEMPLATE.format_map({"instruction": instruction})  

input_ids = tokenizer(input_prompt, return_tensors="pt")  

outputs = model.generate(  
    inputs=input_ids["input_ids"].to("cuda"),  
    attention_mask=input_ids["attention_mask"].to("cuda"),  
    do_sample=True,  
    temperature=1.0,  
    top_k=50,  
    top_p=0.9,  
    max_new_tokens=1024,  
    eos_token_id=tokenizer.eos_token_id,  
    pad_token_id=tokenizer.pad_token_id  
)  

response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]  
response = response.split("### Trả lời:")[1]