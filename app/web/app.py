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
from app.web.auth import setup_auth_routes, get_current_active_user
from app.web.api_docs import setup_api_docs
from app.web.realtime import setup_websocket_routes, start_background_tasks

# Initialize FastAPI app
app = FastAPI(title="Financial Planning Agent UI")

# Set up authentication routes
setup_auth_routes(app)

# Set up API documentation
setup_api_docs(app)

# Set up WebSocket routes for real-time updates
setup_websocket_routes(app)

# Start background tasks for real-time data
start_background_tasks(app)

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
        self.current_conversation_ids: Dict[str, str] = {}  # Track current conversation ID for each client

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        if client_id not in self.agents:
            # Create a new agent with conversation management
            agent = FinancialPlanningAgent(conversation_title="Financial Planning Session")
            self.agents[client_id] = agent
            # Store the current conversation ID
            self.current_conversation_ids[client_id] = agent.current_conversation_id
        if client_id not in self.conversation_history:
            self.conversation_history[client_id] = []
        if client_id not in self.generated_files:
            self.generated_files[client_id] = []
        if client_id not in self.project_sections:
            self.project_sections[client_id] = []
        if client_id not in self.completed_tasks:
            self.completed_tasks[client_id] = []
            
        # Send available conversations to client
        await self.send_conversations_list(client_id)
    
    async def send_conversations_list(self, client_id: str):
        """Send a list of available conversations to the client."""
        if client_id in self.active_connections and client_id in self.agents:
            agent = self.agents[client_id]
            conversations = await agent.get_available_conversations()
            import json
            await self.send_message(client_id, f"[CONVERSATIONS]{json.dumps(conversations)}")
    
    async def load_conversation(self, client_id: str, conversation_id: str):
        """Load a conversation for a client."""
        if client_id in self.agents:
            agent = self.agents[client_id]
            try:
                # Load the conversation
                await agent.load_conversation(conversation_id)
                # Update the current conversation ID
                self.current_conversation_ids[client_id] = conversation_id
                # Send the conversation history to the client
                history = agent.conversation_manager.load_conversation(conversation_id)
                import json
                await self.send_message(client_id, f"[HISTORY]{json.dumps(history)}")
                # Get files in this conversation
                await self.check_for_new_files(client_id)
                # Send success message
                await self.send_message(client_id, f"[LOAD_SUCCESS]Loaded conversation: {conversation_id}")
                return True
            except Exception as e:
                await self.send_message(client_id, f"[ERROR]Failed to load conversation: {str(e)}")
                return False
        return False
    
    async def create_new_conversation(self, client_id: str, title: str = "Financial Planning Session"):
        """Create a new conversation for a client."""
        if client_id in self.agents:
            agent = self.agents[client_id]
            try:
                # Start a new conversation
                conversation_id = agent.conversation_manager.start_new_conversation(title)
                # Update the current conversation ID
                self.current_conversation_ids[client_id] = conversation_id
                agent.current_conversation_id = conversation_id
                # Reset conversation history
                self.conversation_history[client_id] = []
                # Clear sections and tasks
                self.project_sections[client_id] = []
                self.current_sections[client_id] = None
                self.completed_tasks[client_id] = []
                # Send success message
                await self.send_message(client_id, f"[NEW_CONVERSATION]Created new conversation: {conversation_id}")
                # Send updated conversations list
                await self.send_conversations_list(client_id)
                return True
            except Exception as e:
                await self.send_message(client_id, f"[ERROR]Failed to create new conversation: {str(e)}")
                return False
        return False
    
    async def rename_conversation(self, client_id: str, conversation_id: str, new_title: str):
        """Rename a conversation."""
        if client_id in self.agents:
            agent = self.agents[client_id]
            try:
                # Rename the conversation
                agent.conversation_manager.rename_conversation(conversation_id, new_title)
                # Send success message
                await self.send_message(client_id, f"[RENAME_SUCCESS]Renamed conversation to: {new_title}")
                # Send updated conversations list
                await self.send_conversations_list(client_id)
                return True
            except Exception as e:
                await self.send_message(client_id, f"[ERROR]Failed to rename conversation: {str(e)}")
                return False
        return False
    
    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
    
    async def send_message(self, client_id: str, message: str):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_text(message)
    
    def reset_conversation(self, client_id: str):
        """Reset the conversation for a client, keeping their connection."""
        if client_id in self.agents:
            # Create a new agent with a new conversation
            self.agents[client_id] = FinancialPlanningAgent(conversation_title="Financial Planning Session")
            # Update the current conversation ID
            self.current_conversation_ids[client_id] = self.agents[client_id].current_conversation_id
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
        """Check for new files in the conversation directory."""
        if client_id not in self.agents:
            return
            
        agent = self.agents[client_id]
        current_conversation_id = self.current_conversation_ids.get(client_id, None)
        
        if current_conversation_id and agent.conversation_manager:
            # Send updated file list to client
            await self.send_files_update(client_id)
    
    async def send_files_update(self, client_id: str):
        """Send updated file list to the client."""
        if client_id in self.active_connections:
            files_data = await self.get_file_list(client_id)
            await self.send_message(client_id, f"[FILES]{files_data}")
    
    async def get_file_list(self, client_id: str) -> str:
        """Get formatted file list for a client."""
        if client_id not in self.agents:
            return "{\"files\":[]}"
        
        agent = self.agents[client_id]
        current_conversation_id = self.current_conversation_ids.get(client_id, None)
        
        file_list = []
        
        if current_conversation_id and agent.conversation_manager:
            # Get files from the conversation directory
            files = agent.conversation_manager.get_conversation_files(current_conversation_id)
            
            for file_path in files:
                if os.path.exists(file_path):
                    file_name = os.path.basename(file_path)
                    file_type = self._get_file_type(str(file_path))
                    file_list.append({
                        "name": file_name,
                        "path": str(file_path),
                        "type": file_type,
                        "conversation_id": current_conversation_id
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
        
        # Show typing indicator
        await self.send_message(client_id, "[TYPING]")
        
        # Process the message
        response = await agent.process_message(message)
        
        # Extract sections and tasks from the response
        self.extract_sections_and_tasks(client_id, response)
        await self.send_progress_update(client_id)
        
        # Send thinking steps first if available
        if hasattr(agent, 'thinking_steps') and agent.thinking_steps:
            thinking_steps = "\n\n🤔 Thinking Process:\n" + "\n".join(agent.thinking_steps)
            await self.send_message(client_id, thinking_steps)
            agent.thinking_steps = []  # Clear thinking steps after sending
        
        # Check for new files
        await self.check_for_new_files(client_id)
        
        # Send the response
        await self.send_message(client_id, response)
        
        # Update conversations list (in case this is a new conversation)
        await self.send_conversations_list(client_id)
        
        # Signal that response is complete
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
    """WebSocket endpoint for real-time communication with the agent."""
    manager = app.state.connection_manager
    await manager.connect(websocket, client_id)
    
    try:
        while True:
            data = await websocket.receive_text()
            
            # Parse commands and messages
            if data.startswith("/"):
                command_parts = data[1:].split(" ", 1)
                command = command_parts[0].lower()
                
                # Handle conversation commands
                if command == "new":
                    # Create a new conversation
                    title = command_parts[1] if len(command_parts) > 1 else "Financial Planning Session"
                    await manager.create_new_conversation(client_id, title)
                
                elif command == "load":
                    # Load a conversation
                    if len(command_parts) > 1:
                        conversation_id = command_parts[1]
                        await manager.load_conversation(client_id, conversation_id)
                    else:
                        await manager.send_message(client_id, "[ERROR]Please specify a conversation ID to load")
                
                elif command == "rename":
                    # Rename current conversation
                    if len(command_parts) > 1:
                        new_title = command_parts[1]
                        conversation_id = manager.current_conversation_ids.get(client_id)
                        if conversation_id:
                            await manager.rename_conversation(client_id, conversation_id, new_title)
                        else:
                            await manager.send_message(client_id, "[ERROR]No active conversation to rename")
                    else:
                        await manager.send_message(client_id, "[ERROR]Please specify a new title")
                
                elif command == "list":
                    # List all conversations
                    await manager.send_conversations_list(client_id)
                
                elif command == "reset":
                    # Reset conversation
                    manager.reset_conversation(client_id)
                    await manager.send_message(client_id, "[RESET]Conversation has been reset")
                    
                else:
                    await manager.send_message(client_id, f"[ERROR]Unknown command: {command}")
            
            # Handle file uploads (data starting with "FILE:")
            elif data.startswith("FILE:"):
                file_info = data[5:]  # Remove "FILE:" prefix
                import json
                try:
                    file_data = json.loads(file_info)
                    file_paths = file_data.get("paths", [])
                    message = file_data.get("message", "")
                    await manager.process_message(client_id, message, file_paths)
                except json.JSONDecodeError:
                    await manager.send_message(client_id, "[ERROR]Invalid file upload data")
            
            # Handle normal messages
            else:
                await manager.process_message(client_id, data)
                
    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        try:
            manager.disconnect(client_id)
        except:
            pass

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

# After other API endpoints, add these new endpoints for conversation management

@app.post("/conversations/new/{client_id}")
async def create_new_conversation(client_id: str, title: str = Form("Financial Planning Session")):
    """Create a new conversation for a client."""
    manager = app.state.connection_manager
    success = await manager.create_new_conversation(client_id, title)
    if success:
        return JSONResponse({"status": "success", "message": f"Created new conversation with title: {title}"})
    else:
        return JSONResponse({"status": "error", "message": "Failed to create new conversation"}, status_code=500)

@app.post("/conversations/load/{client_id}/{conversation_id}")
async def load_conversation(client_id: str, conversation_id: str):
    """Load a conversation for a client."""
    manager = app.state.connection_manager
    success = await manager.load_conversation(client_id, conversation_id)
    if success:
        return JSONResponse({"status": "success", "message": f"Loaded conversation: {conversation_id}"})
    else:
        return JSONResponse({"status": "error", "message": "Failed to load conversation"}, status_code=500)

@app.post("/conversations/rename/{client_id}/{conversation_id}")
async def rename_conversation(client_id: str, conversation_id: str, title: str = Form(...)):
    """Rename a conversation."""
    manager = app.state.connection_manager
    success = await manager.rename_conversation(client_id, conversation_id, title)
    if success:
        return JSONResponse({"status": "success", "message": f"Renamed conversation to: {title}"})
    else:
        return JSONResponse({"status": "error", "message": "Failed to rename conversation"}, status_code=500)

@app.get("/conversations/list/{client_id}")
async def list_conversations(client_id: str):
    """Get a list of conversations for a client."""
    if client_id in app.state.connection_manager.agents:
        agent = app.state.connection_manager.agents[client_id]
        conversations = await agent.get_available_conversations()
        return JSONResponse({"status": "success", "conversations": conversations})
    else:
        return JSONResponse({"status": "error", "message": "Client not found"}, status_code=404)

@app.get("/conversations/current/{client_id}")
async def get_current_conversation(client_id: str):
    """Get the current conversation ID for a client."""
    if client_id in app.state.connection_manager.current_conversation_ids:
        conversation_id = app.state.connection_manager.current_conversation_ids[client_id]
        return JSONResponse({"status": "success", "conversation_id": conversation_id})
    else:
        return JSONResponse({"status": "error", "message": "No current conversation"}, status_code=404)

# Run with: uvicorn app.web.app:app --reload 