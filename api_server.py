import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import uvicorn
import os

app = FastAPI()

# 模型路径（容器内固定路径，与 Dockerfile 解压路径一致）
BASE_MODEL_NAME = "unsloth/Llama-3.2-1B-Instruct"
ADAPTER_PATH = "/app/ecg-model-leadII"  # 注意：解压后的文件夹名可能不同，请根据实际调整

print("正在加载 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

print("正在加载基础模型...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

print("正在加载 LoRA 适配器...")
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model = model.merge_and_unload()
model.eval()
print("模型加载完成！")

class ChatRequest(BaseModel):
    ecg_data: str
    question: str

class ChatResponse(BaseModel):
    reply: str

def build_prompt(ecg_data: str, question: str) -> str:
    system_prompt = "You are an ECG assistant. Answer the user's questions based on the provided ECG data. You can call tools if needed."
    prompt = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\nECG data:\n{ecg_data}\n\nUser: {question} [/INST]"
    return prompt

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        prompt = build_prompt(request.ecg_data, request.question)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if full_response.startswith(prompt):
            reply = full_response[len(prompt):].strip()
        else:
            reply = full_response
        return ChatResponse(reply=reply)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)