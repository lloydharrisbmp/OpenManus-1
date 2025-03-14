import os
import asyncio
import uuid
import webbrowser
import time
import re
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

# Mount the client_documents directory to serve generated websites
app.mount("/generated", StaticFiles(directory="client_documents"), name="generated")

# Directories for uploads and generated files
UPLOAD_DIR = "uploads"
CLIENT_DOCS_DIR = "client_documents"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CLIENT_DOCS_DIR, exist_ok=True)

# Ensure subdirectories exist
for subdir in ["markdown", "text", "websites", "reports"]:
    os.makedirs(os.path.join(CLIENT_DOCS_DIR, subdir), exist_ok=True)

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.agents: Dict[str, FinancialPlanningAgent] = {}
        self.conversation_history: Dict[str, List[Dict]] = {}
        self.generated_files: Dict[str, List[str]] = {}
        self.project_sections: Dict[str, List[str]] = {}
        self.current_sections: Dict[str, str] = {}
        self.completed_tasks: Dict[str, List[str]] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        if client_id not in self.agents:
            self.agents[client_id] = FinancialPlanningAgent()
        if client_id not in self.conversation_history:
            self.conversation_history[client_id] = []
        if client_id not in self.generated_files:
            self.generated_files[client_id] = []
        if client_id not in self.project_sections:
            self.project_sections[client_id] = []
        if client_id not in self.completed_tasks:
            self.completed_tasks[client_id] = []
    
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
        if client_id in self.project_sections:
            self.project_sections[client_id] = []
        if client_id in self.current_sections:
            self.current_sections.pop(client_id, None)
        if client_id in self.completed_tasks:
            self.completed_tasks[client_id] = []
    
    def track_generated_file(self, client_id: str, file_path: str):
        """Track a file generated for a specific client."""
        if client_id in self.generated_files:
            if file_path not in self.generated_files[client_id]:
                self.generated_files[client_id].append(file_path)
    
    async def check_for_new_files(self, client_id: str):
        """Check for new files in the client documents directory."""
        file_count_before = len(self.generated_files.get(client_id, []))
        
        # Check main directory and all subdirectories
        for root, _, files in os.walk(CLIENT_DOCS_DIR):
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.isfile(file_path):
                    self.track_generated_file(client_id, file_path)
                    
        file_count_after = len(self.generated_files.get(client_id, []))
        
        # If new files were found, send update to client
        if file_count_after > file_count_before:
            await self.send_files_update(client_id)
    
    async def send_files_update(self, client_id: str):
        """Send updated file list to the client."""
        if client_id in self.active_connections:
            files_data = await self.get_file_list(client_id)
            await self.send_message(client_id, f"[FILES]{files_data}")
    
    async def get_file_list(self, client_id: str) -> str:
        """Get formatted file list for a client."""
        if client_id not in self.generated_files:
            return "{\"files\":[]}"
        
        file_list = []
        for file_path in self.generated_files[client_id]:
            if os.path.exists(file_path):
                file_name = os.path.basename(file_path)
                file_type = self._get_file_type(file_path)
                file_list.append({
                    "name": file_name,
                    "path": file_path,
                    "type": file_type
                })
        
        import json
        return json.dumps({"files": file_list})
    
    def _get_file_type(self, file_path: str) -> str:
        """Determine the type of file based on extension or directory."""
        ext = os.path.splitext(file_path)[1].lower()
        
        if "markdown" in file_path:
            return "markdown"
        elif "websites" in file_path:
            return "website"
        elif "reports" in file_path:
            return "report"
        elif ext == ".md":
            return "markdown"
        elif ext == ".html":
            return "website"
        elif ext == ".txt":
            return "text"
        elif ext == ".pdf":
            return "document"
        else:
            return "other"
    
    def extract_sections_and_tasks(self, client_id: str, message: str):
        """Extract project sections and tasks from agent messages."""
        # Extract section headers (## Working on: Section Name)
        section_match = re.search(r'## Working on: (.+?)(?:\r?\n|$)', message)
        if section_match:
            section = section_match.group(1).strip()
            if client_id in self.project_sections and section not in self.project_sections[client_id]:
                self.project_sections[client_id].append(section)
                self.current_sections[client_id] = section
        
        # Extract completed tasks (✓ Task completed: Task Name)
        task_matches = re.findall(r'✓ Task completed: (.+?)(?:\r?\n|$)', message)
        for task in task_matches:
            task = task.strip()
            if client_id in self.completed_tasks and task not in self.completed_tasks[client_id]:
                self.completed_tasks[client_id].append(task)
    
    async def send_progress_update(self, client_id: str):
        """Send progress update to the client."""
        if client_id in self.active_connections:
            import json
            progress_data = {
                "sections": self.project_sections.get(client_id, []),
                "current_section": self.current_sections.get(client_id, None),
                "completed_tasks": self.completed_tasks.get(client_id, [])
            }
            await self.send_message(client_id, f"[PROGRESS]{json.dumps(progress_data)}")
    
    def format_website_url(self, file_path: str) -> str:
        """Format the website URL for viewing through the FastAPI server."""
        # Convert absolute path to relative path from client_documents
        rel_path = os.path.relpath(file_path, "client_documents")
        return f"/generated/{rel_path}"
    
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
        
        # Process with agent
        agent = self.agents[client_id]
        
        # Add standard sections to help with progress tracking
        sections_to_extract = ["Research", "Analysis", "Document Creation"]
        if not self.project_sections.get(client_id, []):
            self.project_sections[client_id] = sections_to_extract
            await self.send_progress_update(client_id)
        
        # Process the message
        response = await agent.process_message(message)
        
        # Extract sections and tasks from the response
        self.extract_sections_and_tasks(client_id, response)
        await self.send_progress_update(client_id)
        
        # Send thinking steps first
        if hasattr(agent, 'thinking_steps') and agent.thinking_steps:
            thinking_steps = "\n\n🤔 Thinking Process:\n" + "\n".join(agent.thinking_steps)
            await self.send_message(client_id, thinking_steps)
            agent.thinking_steps = []  # Clear thinking steps after sending
        
        # Check for new files
        await self.check_for_new_files(client_id)
        
        # Make website URLs clickable in the chat
        if "WEBSITE GENERATED SUCCESSFULLY" in response:
            # The URL is already in the response from the tool
            # Just make sure it's properly formatted for the chat
            response = response.replace("→ http://", "→ <a href='http://")
            response = response.replace(".html", ".html' target='_blank'>View Website</a>")
        
        # Send the final response
        await self.send_message(client_id, response)
        
        # Add to conversation history
        self.conversation_history[client_id].append({"role": "assistant", "content": response})


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
    # Look in all possible directories
    for root, _, files in os.walk(CLIENT_DOCS_DIR):
        for file in files:
            if file == file_name:
                file_path = os.path.join(root, file)
                return FileResponse(
                    path=file_path,
                    filename=file_name,
                    media_type="application/octet-stream"
                )
    
    # Also check the uploads directory
    file_path = os.path.join(UPLOAD_DIR, file_name)
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
            file_type = manager._get_file_type(file_path)
            file_list.append({
                "name": file_name,
                "path": file_path,
                "type": file_type
            })
    
    return JSONResponse(content={"files": file_list})

@app.get("/progress/{client_id}")
async def get_progress(client_id: str):
    progress_data = {
        "sections": manager.project_sections.get(client_id, []),
        "current_section": manager.current_sections.get(client_id, None),
        "completed_tasks": manager.completed_tasks.get(client_id, [])
    }
    return JSONResponse(content=progress_data)

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

@app.get("/view-website/{path:path}")
async def view_website(path: str):
    """Serve generated website files."""
    website_path = os.path.join("client_documents/websites", path)
    if os.path.exists(website_path):
        return FileResponse(website_path)
    raise HTTPException(status_code=404, detail="Website not found")

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