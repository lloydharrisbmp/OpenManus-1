import os
import asyncio
import uuid
import webbrowser
import time
from pathlib import Path
from typing import List, Optional, Dict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, Request, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
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

# Directories for uploads and generated files
UPLOAD_DIR = "uploads"
CLIENT_DOCS_DIR = "client_documents"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CLIENT_DOCS_DIR, exist_ok=True)

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.agents: Dict[str, FinancialPlanningAgent] = {}
        self.conversation_history: Dict[str, List[Dict]] = {}
        self.generated_files: Dict[str, List[str]] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        if client_id not in self.agents:
            self.agents[client_id] = FinancialPlanningAgent()
        if client_id not in self.conversation_history:
            self.conversation_history[client_id] = []
        if client_id not in self.generated_files:
            self.generated_files[client_id] = []
    
    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
    
    async def send_message(self, client_id: str, message: str):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_text(message)
    
    def reset_conversation(self, client_id: str):
        """Reset the conversation for a client, keeping their connection."""
        if client_id in self.agents:
            self.agents[client_id] = FinancialPlanningAgent()
        if client_id in self.conversation_history:
            self.conversation_history[client_id] = []
    
    def track_generated_file(self, client_id: str, file_path: str):
        """Track a file generated for a specific client."""
        if client_id in self.generated_files:
            if file_path not in self.generated_files[client_id]:
                self.generated_files[client_id].append(file_path)
    
    async def check_for_new_files(self, client_id: str):
        """Check for new files in the client documents directory."""
        if not os.path.exists(CLIENT_DOCS_DIR):
            return
            
        files = [f for f in os.listdir(CLIENT_DOCS_DIR) if os.path.isfile(os.path.join(CLIENT_DOCS_DIR, f))]
        
        for file in files:
            file_path = os.path.join(CLIENT_DOCS_DIR, file)
            self.track_generated_file(client_id, file_path)
    
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
        
        # Add to conversation history
        self.conversation_history[client_id].append({"role": "user", "content": message})
        
        try:
            # Set up a custom handler to redirect agent outputs to the websocket
            original_stream = self.agents[client_id].llm.stream if hasattr(self.agents[client_id].llm, 'stream') else None
            
            response_content = []
            
            async def stream_to_websocket(chunk):
                response_content.append(chunk)
                await self.send_message(client_id, chunk)
                # Also send through the original stream method if needed
                if original_stream:
                    await original_stream(chunk)
            
            # Replace the stream method temporarily
            if hasattr(self.agents[client_id].llm, 'stream'):
                self.agents[client_id].llm.stream = stream_to_websocket
            
            # Run the agent
            await self.agents[client_id].run(message)
            
            # Restore the original stream method
            if hasattr(self.agents[client_id].llm, 'stream'):
                self.agents[client_id].llm.stream = original_stream
            
            # Add to conversation history
            self.conversation_history[client_id].append({"role": "assistant", "content": "".join(response_content)})
            
            # Check for newly generated files
            await self.check_for_new_files(client_id)
            
            # Send updated file list
            if client_id in self.generated_files and self.generated_files[client_id]:
                file_list = []
                for file_path in self.generated_files[client_id]:
                    file_name = os.path.basename(file_path)
                    file_list.append({
                        "name": file_name,
                        "path": file_path
                    })
                await self.send_message(client_id, f"[FILES]{JSONResponse(content={'files': file_list}).body.decode()}")
            
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

@app.get("/download/{file_name}")
async def download_file(file_name: str):
    # Look in both directories
    for directory in [CLIENT_DOCS_DIR, UPLOAD_DIR]:
        file_path = os.path.join(directory, file_name)
        if os.path.exists(file_path):
            return FileResponse(
                path=file_path,
                filename=file_name,
                media_type="application/octet-stream"
            )
    
    # Also check for full paths that were tracked
    for client_id, files in manager.generated_files.items():
        for file_path in files:
            if os.path.basename(file_path) == file_name and os.path.exists(file_path):
                return FileResponse(
                    path=file_path,
                    filename=file_name,
                    media_type="application/octet-stream"
                )
    
    raise HTTPException(status_code=404, detail="File not found")

@app.post("/reset/{client_id}")
async def reset_conversation(client_id: str):
    manager.reset_conversation(client_id)
    return JSONResponse(content={"message": "Conversation reset successfully"})

@app.get("/files/{client_id}")
async def get_generated_files(client_id: str):
    if client_id not in manager.generated_files:
        return JSONResponse(content={"files": []})
    
    file_list = []
    for file_path in manager.generated_files[client_id]:
        if os.path.exists(file_path):
            file_name = os.path.basename(file_path)
            file_list.append({
                "name": file_name,
                "path": file_path
            })
    
    return JSONResponse(content={"files": file_list})

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    try:
        while True:
            data = await websocket.receive_json()
            
            if "action" in data and data["action"] == "reset":
                manager.reset_conversation(client_id)
                await manager.send_message(client_id, "[RESET]")
                continue
                
            message = data.get("message", "")
            file_paths = data.get("files", [])
            
            # Process in background to not block the WebSocket
            asyncio.create_task(manager.process_message(client_id, message, file_paths))
    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        logger.exception(f"WebSocket error: {str(e)}")
        manager.disconnect(client_id)

# Function to automatically open the browser
def open_browser():
    time.sleep(1)  # Give the server a second to start
    webbrowser.open("http://localhost:8000")

# Add this function to your FastAPI app startup event
@app.on_event("startup")
async def startup_event():
    # Start the browser in a separate thread
    import threading
    threading.Thread(target=open_browser).start()

# Run with: uvicorn app.web.app:app --reload 