# main.py
from fastapi import FastAPI, Request
from pydantic import BaseModel
from langgraph_bot import build_graph
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

chat_machine = build_graph()


class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    state = {"input": req.message}
    result = chat_machine.invoke(state)
    return ChatResponse(response=result["output"])
