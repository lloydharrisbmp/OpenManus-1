"""
Real-time updates module for the financial planning application.

This module provides WebSocket-based real-time updates for market data,
portfolio performance, and agent status.
"""

import asyncio
import json
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
import uuid

from fastapi import WebSocket, WebSocketDisconnect, Depends, HTTPException
from pydantic import BaseModel

from app.logger import logger
from app.web.auth import get_current_active_user as get_current_user, decode_and_validate_token
from app.services.cache_service import cache_service


class WebSocketManager:
    """
    WebSocket connection manager for handling real-time updates.
    
    This manager tracks active connections, sends updates to specific clients,
    and broadcasts messages to all connected clients.
    """
    
    def __init__(self):
        """Initialize the WebSocket manager."""
        self.active_connections: Dict[str, Dict[str, WebSocket]] = {}
        self.client_sessions: Dict[str, List[str]] = {}
        self.user_connections: Dict[str, List[str]] = {}
        self.market_data_subscribers: List[str] = []
        self.portfolio_update_subscribers: Dict[str, List[str]] = {}
        self.last_market_data: Dict[str, Any] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str, user_id: str, session_id: Optional[str] = None) -> str:
        """
        Connect a client to the WebSocket.
        
        Args:
            websocket: WebSocket connection
            client_id: Unique client identifier
            user_id: User ID for authentication
            session_id: Optional session ID (generated if not provided)
            
        Returns:
            Session ID
        """
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        await websocket.accept()
        
        # Initialize dictionaries if needed
        if client_id not in self.active_connections:
            self.active_connections[client_id] = {}
            self.client_sessions[client_id] = []
        
        if user_id not in self.user_connections:
            self.user_connections[user_id] = []
        
        # Add the connection
        self.active_connections[client_id][session_id] = websocket
        self.client_sessions[client_id].append(session_id)
        self.user_connections[user_id].append(session_id)
        
        # Send initial data if available
        if len(self.last_market_data) > 0:
            await websocket.send_json({
                "type": "market_data",
                "data": self.last_market_data,
                "timestamp": datetime.now().isoformat()
            })
        
        return session_id
    
    async def disconnect(self, client_id: str, session_id: str, user_id: Optional[str] = None):
        """
        Disconnect a client from the WebSocket.
        
        Args:
            client_id: Client identifier
            session_id: Session ID
            user_id: Optional user ID
        """
        # Remove session from client sessions
        if client_id in self.client_sessions:
            if session_id in self.client_sessions[client_id]:
                self.client_sessions[client_id].remove(session_id)
            
            # Remove from active connections
            if client_id in self.active_connections and session_id in self.active_connections[client_id]:
                del self.active_connections[client_id][session_id]
        
        # Remove from user connections
        if user_id and user_id in self.user_connections:
            if session_id in self.user_connections[user_id]:
                self.user_connections[user_id].remove(session_id)
        
        # Remove from market data subscribers
        if session_id in self.market_data_subscribers:
            self.market_data_subscribers.remove(session_id)
        
        # Remove from portfolio subscribers
        for portfolio_id in self.portfolio_update_subscribers:
            if session_id in self.portfolio_update_subscribers[portfolio_id]:
                self.portfolio_update_subscribers[portfolio_id].remove(session_id)
    
    async def send_personal_message(self, message: Any, client_id: str, session_id: str):
        """
        Send a message to a specific client session.
        
        Args:
            message: Message to send
            client_id: Client identifier
            session_id: Session ID
        """
        if client_id in self.active_connections and session_id in self.active_connections[client_id]:
            try:
                if isinstance(message, dict) or isinstance(message, list):
                    await self.active_connections[client_id][session_id].send_json(message)
                else:
                    await self.active_connections[client_id][session_id].send_text(str(message))
            except Exception as e:
                logger.error(f"Error sending message to client {client_id}, session {session_id}: {e}")
    
    async def broadcast(self, message: Any, client_id: Optional[str] = None):
        """
        Broadcast a message to all sessions of a client or all clients.
        
        Args:
            message: Message to broadcast
            client_id: Optional client ID (broadcasts to all clients if None)
        """
        if client_id:
            # Broadcast to all sessions of a specific client
            if client_id in self.active_connections:
                for session_id, connection in self.active_connections[client_id].items():
                    try:
                        if isinstance(message, dict) or isinstance(message, list):
                            await connection.send_json(message)
                        else:
                            await connection.send_text(str(message))
                    except Exception as e:
                        logger.error(f"Error broadcasting to client {client_id}, session {session_id}: {e}")
        else:
            # Broadcast to all clients
            for client_id, sessions in self.active_connections.items():
                for session_id, connection in sessions.items():
                    try:
                        if isinstance(message, dict) or isinstance(message, list):
                            await connection.send_json(message)
                        else:
                            await connection.send_text(str(message))
                    except Exception as e:
                        logger.error(f"Error broadcasting to client {client_id}, session {session_id}: {e}")
    
    async def broadcast_to_user(self, message: Any, user_id: str):
        """
        Broadcast a message to all sessions of a specific user.
        
        Args:
            message: Message to broadcast
            user_id: User ID
        """
        if user_id in self.user_connections:
            for session_id in self.user_connections[user_id]:
                # Find the client ID for this session
                for client_id, sessions in self.client_sessions.items():
                    if session_id in sessions:
                        await self.send_personal_message(message, client_id, session_id)
    
    async def broadcast_market_data(self, market_data: Dict[str, Any]):
        """
        Broadcast market data updates to subscribers.
        
        Args:
            market_data: Market data to broadcast
        """
        self.last_market_data = market_data
        
        message = {
            "type": "market_data",
            "data": market_data,
            "timestamp": datetime.now().isoformat()
        }
        
        for session_id in self.market_data_subscribers:
            # Find the client ID for this session
            for client_id, sessions in self.client_sessions.items():
                if session_id in sessions:
                    await self.send_personal_message(message, client_id, session_id)
    
    async def broadcast_portfolio_update(self, portfolio_id: str, portfolio_data: Dict[str, Any]):
        """
        Broadcast portfolio updates to subscribers.
        
        Args:
            portfolio_id: Portfolio ID
            portfolio_data: Portfolio data to broadcast
        """
        if portfolio_id in self.portfolio_update_subscribers:
            message = {
                "type": "portfolio_update",
                "portfolio_id": portfolio_id,
                "data": portfolio_data,
                "timestamp": datetime.now().isoformat()
            }
            
            for session_id in self.portfolio_update_subscribers[portfolio_id]:
                # Find the client ID for this session
                for client_id, sessions in self.client_sessions.items():
                    if session_id in sessions:
                        await self.send_personal_message(message, client_id, session_id)
    
    def subscribe_to_market_data(self, session_id: str):
        """
        Subscribe a session to market data updates.
        
        Args:
            session_id: Session ID to subscribe
        """
        if session_id not in self.market_data_subscribers:
            self.market_data_subscribers.append(session_id)
    
    def subscribe_to_portfolio(self, session_id: str, portfolio_id: str):
        """
        Subscribe a session to portfolio updates.
        
        Args:
            session_id: Session ID to subscribe
            portfolio_id: Portfolio ID to subscribe to
        """
        if portfolio_id not in self.portfolio_update_subscribers:
            self.portfolio_update_subscribers[portfolio_id] = []
        
        if session_id not in self.portfolio_update_subscribers[portfolio_id]:
            self.portfolio_update_subscribers[portfolio_id].append(session_id)
    
    def unsubscribe_from_market_data(self, session_id: str):
        """
        Unsubscribe a session from market data updates.
        
        Args:
            session_id: Session ID to unsubscribe
        """
        if session_id in self.market_data_subscribers:
            self.market_data_subscribers.remove(session_id)
    
    def unsubscribe_from_portfolio(self, session_id: str, portfolio_id: str):
        """
        Unsubscribe a session from portfolio updates.
        
        Args:
            session_id: Session ID to unsubscribe
            portfolio_id: Portfolio ID to unsubscribe from
        """
        if portfolio_id in self.portfolio_update_subscribers and session_id in self.portfolio_update_subscribers[portfolio_id]:
            self.portfolio_update_subscribers[portfolio_id].remove(session_id)


# Create global WebSocket manager
websocket_manager = WebSocketManager()


class UpdateMessage(BaseModel):
    """Message format for WebSocket updates."""
    type: str
    data: Dict[str, Any]
    timestamp: datetime = datetime.now()


# Store active connections
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, client_id: str):
        async with self._lock:
            if client_id not in self.active_connections:
                self.active_connections[client_id] = set()
            self.active_connections[client_id].add(websocket)

    async def disconnect(self, websocket: WebSocket, client_id: str):
        async with self._lock:
            if client_id in self.active_connections:
                self.active_connections[client_id].discard(websocket)
                if not self.active_connections[client_id]:
                    del self.active_connections[client_id]

    async def broadcast_to_client(self, client_id: str, message: str):
        if client_id in self.active_connections:
            for connection in self.active_connections[client_id].copy():
                try:
                    await connection.send_text(message)
                except Exception as e:
                    logger.error(f"Error broadcasting to client {client_id}: {e}")
                    await self.disconnect(connection, client_id)

manager = ConnectionManager()

async def handle_websocket_connection(websocket: WebSocket, client_id: str, user: dict):
    """Handle authenticated WebSocket connection."""
    try:
        await manager.connect(websocket, client_id)
        await websocket.send_json({
            "type": "connection_established",
            "client_id": client_id,
            "timestamp": datetime.now().isoformat()
        })

        while True:
            try:
                data = await websocket.receive_text()
                # Process received data
                try:
                    message = json.loads(data)
                    # Add message validation here
                    if not isinstance(message, dict):
                        await websocket.send_json({"error": "Invalid message format"})
                        continue

                    # Process different message types
                    message_type = message.get("type")
                    if not message_type:
                        await websocket.send_json({"error": "Message type required"})
                        continue

                    # Handle different message types
                    if message_type == "ping":
                        await websocket.send_json({"type": "pong", "timestamp": datetime.now().isoformat()})
                    else:
                        # Process other message types
                        await process_message(websocket, client_id, user, message)

                except json.JSONDecodeError:
                    await websocket.send_json({"error": "Invalid JSON format"})
                    continue

            except WebSocketDisconnect:
                break

    except Exception as e:
        logger.error(f"Error in WebSocket connection handler: {e}")
    finally:
        await manager.disconnect(websocket, client_id)

async def process_message(websocket: WebSocket, client_id: str, user: dict, message: dict):
    """Process different types of WebSocket messages."""
    try:
        message_type = message["type"]
        
        # Add your message processing logic here
        # Example:
        if message_type == "request_update":
            await websocket.send_json({
                "type": "update",
                "data": {"timestamp": datetime.now().isoformat()}
            })
        else:
            await websocket.send_json({
                "error": f"Unsupported message type: {message_type}"
            })

    except Exception as e:
        logger.error(f"Error processing message: {e}")
        await websocket.send_json({
            "error": f"Error processing message: {str(e)}"
        })

def setup_websocket_routes(app):
    """Set up WebSocket routes with enhanced security."""
    
    @app.websocket("/ws/{client_id}")
    async def websocket_endpoint(websocket: WebSocket, client_id: str):
        await websocket.accept()
        
        try:
            # Wait for authentication message with timeout
            auth_message = await asyncio.wait_for(
                websocket.receive_json(),
                timeout=10.0  # 10 second timeout for authentication
            )
            
            if not auth_message.get("token"):
                await websocket.send_json({"error": "Authentication required"})
                await websocket.close()
                return
            
            # Validate token and get user
            user = await decode_and_validate_token(auth_message.get("token"))
            if not user:
                await websocket.send_json({"error": "Invalid authentication token"})
                await websocket.close()
                return
            
            # Proceed with authenticated connection
            await handle_websocket_connection(websocket, client_id, user.dict())
            
        except asyncio.TimeoutError:
            await websocket.send_json({"error": "Authentication timeout"})
            await websocket.close()
        except WebSocketDisconnect:
            logger.info(f"Client {client_id} disconnected during authentication")
        except Exception as e:
            logger.error(f"Error in websocket endpoint: {e}")
            await websocket.send_json({"error": "Internal server error"})
            await websocket.close()


async def start_market_data_broadcaster():
    """
    Start a background task that broadcasts market data updates periodically.
    """
    while True:
        try:
            # Get cached market data or create new data
            market_data = cache_service.get("latest_market_data")
            
            if market_data:
                # Broadcast to all subscribers
                await websocket_manager.broadcast_market_data(market_data)
            
            # Wait for next update (market data every 5 minutes)
            await asyncio.sleep(300)
        except Exception as e:
            logger.error(f"Error in market data broadcaster: {e}")
            await asyncio.sleep(10)


def start_background_tasks(app):
    """
    Start background tasks for real-time updates.
    
    Args:
        app: FastAPI application
    """
    @app.on_event("startup")
    async def start_tasks():
        asyncio.create_task(start_market_data_broadcaster()) 