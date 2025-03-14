import os
import asyncio
import uuid
from typing import List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import aiofiles

from app.agent.swe import FinancialPlanningAgent
from app.logger import logger

# Initialize FastAPI app
app = FastAPI(title="Financial Planning Agent UI")

# Mount static files
app.mount("/static", StaticFiles(directory="app/web/static"), name="static")

# Set up Jinja2 templates
templates = Jinja2Templates(directory="app/web/templates")

# Directory for uploads
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: dict = {}
        self.agents: dict = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.agents[client_id] = FinancialPlanningAgent()
    
    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.agents:
            del self.agents[client_id]
    
    async def send_message(self, client_id: str, message: str):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_text(message)
    
    async def process_message(self, client_id: str, message: str, file_paths: Optional[List[str]] = None):
        """Process a message with the financial planning agent and send responses via websocket."""
        if client_id not in self.agents:
            await self.send_message(client_id, "Error: Session expired. Please refresh the page.")
            return
        
        # Add file references to the message if provided
        if file_paths and len(file_paths) > 0:
            file_info = "\n\nAttached files:"
            for path in file_paths:
                file_name = os.path.basename(path)
                file_info += f"\n- {file_name}: {path}"
            message += file_info
        
        try:
            # Set up a custom handler to redirect agent outputs to the websocket
            original_stream = self.agents[client_id].llm.stream
            
            async def stream_to_websocket(chunk):
                await self.send_message(client_id, chunk)
                # Also send through the original stream method if needed
                if original_stream:
                    await original_stream(chunk)
            
            # Replace the stream method temporarily
            self.agents[client_id].llm.stream = stream_to_websocket
            
            # Run the agent
            await self.agents[client_id].run(message)
            
            # Restore the original stream method
            self.agents[client_id].llm.stream = original_stream
            
            # Signal completion
            await self.send_message(client_id, "[DONE]")
            
        except Exception as e:
            logger.exception(f"Error processing message: {str(e)}")
            await self.send_message(client_id, f"An error occurred: {str(e)}")
            await self.send_message(client_id, "[DONE]")


manager = ConnectionManager()

# API endpoints
@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload/")
async def upload_files(files: List[UploadFile] = File(...)):
    file_paths = []
    for file in files:
        # Generate a unique filename
        file_ext = os.path.splitext(file.filename)[1]
        unique_filename = f"{uuid.uuid4()}{file_ext}"
        file_path = os.path.join(UPLOAD_DIR, unique_filename)
        
        # Save the file
        async with aiofiles.open(file_path, 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)
        
        file_paths.append({
            "original_name": file.filename,
            "path": file_path
        })
    
    return JSONResponse(content={"message": "Files uploaded successfully", "files": file_paths})

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    try:
        while True:
            data = await websocket.receive_json()
            
            message = data.get("message", "")
            file_paths = data.get("files", [])
            
            # Process in background to not block the WebSocket
            asyncio.create_task(manager.process_message(client_id, message, file_paths))
    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        logger.exception(f"WebSocket error: {str(e)}")
        manager.disconnect(client_id)

# Run with: uvicorn app.web.app:app --reload 