# main.py
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel
from langgraph_bot import build_graph
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=FileResponse)
async def serve_index():
    return FileResponse("static/index.html")

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
