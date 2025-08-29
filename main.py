# main.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from langgraph_bot import build_graph
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import traceback

app = FastAPI(title="LangGraph Chat API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=FileResponse)
async def serve_index():
    return FileResponse("static/index.html")

# Initialize chat machine
try:
    chat_machine = build_graph()
    print("✅ LangGraph chat machine initialized successfully")
except Exception as e:
    print(f"❌ Failed to initialize LangGraph: {e}")
    chat_machine = None

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="The user's message")
    device_type: str = Field(default="desktop", description="Device type: mobile, tablet, or desktop")
    action: str = Field(default="chat", description="Action type: chat, regenerate, edit_caption, edit_hashtags")
    edit_data: dict = Field(default=None, description="Data for edit operations")

class ChatResponse(BaseModel):
    output: dict | str
    status: str = "success"

@app.post("/chat", response_model=dict)
async def chat_endpoint(req: ChatRequest):
    # Validate chat machine is available
    if chat_machine is None:
        raise HTTPException(status_code=500, detail="Chat system is not available")
    
    # Validate and clean input
    message = req.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    # Prepare state for LangGraph
    state = {
        "input": message,
        "device_type": req.device_type,
        "action": req.action
    }
    
    # Add edit data if provided
    if req.edit_data:
        state["edit_data"] = req.edit_data
    
    print(f"Received request: {req.message}")
    print(f"Device type: {req.device_type}")
    print(f"Action: {req.action}")
    print(f"State for LangGraph: {state}")
    
    try:
        # Invoke LangGraph
        result = chat_machine.invoke(state)
        print(f"✅ LangGraph result: {result}")
        
        # Extract output
        output = result.get("output", "No response generated")
        
        # Return response
        return {
            "output": output,
            "status": "success"
        }
        
    except ValueError as ve:
        print(f"❌ Validation error: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print(f"📋 Full traceback:")
        traceback.print_exc()
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error: {str(e)}"
        )

