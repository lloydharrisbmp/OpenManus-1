"""
Real-time updates module for the financial planning application.

This module provides WebSocket-based real-time updates for market data,
portfolio performance, and agent status.
"""

import asyncio
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid

from fastapi import WebSocket, WebSocketDisconnect, Depends, HTTPException
from pydantic import BaseModel

from app.logger import logger
from app.web.auth import get_current_active_user as get_current_user
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


async def handle_websocket_connection(websocket: WebSocket, client_id: str, user):
    """
    Handle a WebSocket connection.
    
    Args:
        websocket: WebSocket connection
        client_id: Client identifier
        user: Authenticated user
    """
    session_id = await websocket_manager.connect(websocket, client_id, user.username)
    
    try:
        # Send welcome message
        welcome_message = {
            "type": "welcome",
            "message": f"Welcome, {user.full_name or user.username}!",
            "user_id": user.username,
            "session_id": session_id,
            "timestamp": datetime.now().isoformat()
        }
        await websocket.send_json(welcome_message)
        
        # Handle incoming messages
        while True:
            data = await websocket.receive_json()
            
            # Handle subscription requests
            if data.get("action") == "subscribe":
                if data.get("target") == "market_data":
                    websocket_manager.subscribe_to_market_data(session_id)
                    await websocket.send_json({
                        "type": "subscription_confirmation",
                        "target": "market_data",
                        "status": "subscribed",
                        "timestamp": datetime.now().isoformat()
                    })
                
                elif data.get("target") == "portfolio" and data.get("portfolio_id"):
                    portfolio_id = data.get("portfolio_id")
                    websocket_manager.subscribe_to_portfolio(session_id, portfolio_id)
                    await websocket.send_json({
                        "type": "subscription_confirmation",
                        "target": "portfolio",
                        "portfolio_id": portfolio_id,
                        "status": "subscribed",
                        "timestamp": datetime.now().isoformat()
                    })
            
            # Handle unsubscription requests
            elif data.get("action") == "unsubscribe":
                if data.get("target") == "market_data":
                    websocket_manager.unsubscribe_from_market_data(session_id)
                    await websocket.send_json({
                        "type": "subscription_confirmation",
                        "target": "market_data",
                        "status": "unsubscribed",
                        "timestamp": datetime.now().isoformat()
                    })
                
                elif data.get("target") == "portfolio" and data.get("portfolio_id"):
                    portfolio_id = data.get("portfolio_id")
                    websocket_manager.unsubscribe_from_portfolio(session_id, portfolio_id)
                    await websocket.send_json({
                        "type": "subscription_confirmation",
                        "target": "portfolio",
                        "portfolio_id": portfolio_id,
                        "status": "unsubscribed",
                        "timestamp": datetime.now().isoformat()
                    })
            
            # Echo any other messages with timestamp
            else:
                await websocket.send_json({
                    "type": "echo",
                    "data": data,
                    "timestamp": datetime.now().isoformat()
                })
    
    except WebSocketDisconnect:
        await websocket_manager.disconnect(client_id, session_id, user.username)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket_manager.disconnect(client_id, session_id, user.username)


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


def setup_websocket_routes(app):
    """
    Set up WebSocket routes for the application.
    
    Args:
        app: FastAPI application
    """
    @app.websocket("/ws/{client_id}")
    async def websocket_endpoint(websocket: WebSocket, client_id: str):
        """
        WebSocket endpoint requiring authentication.
        Client must send a valid JWT token in the first message.
        """
        await websocket.accept()
        
        try:
            # Wait for authentication message
            auth_data = await websocket.receive_json()
            
            if not auth_data.get("token"):
                await websocket.send_json({"error": "Authentication required"})
                await websocket.close()
                return
            
            # Validate token and get user
            try:
                # This is a simplified verification - in a real app, use proper token validation
                from app.web.auth import decode_and_validate_token
                user = await decode_and_validate_token(auth_data.get("token"))
                
                if not user:
                    await websocket.send_json({"error": "Invalid authentication token"})
                    await websocket.close()
                    return
                
                # Proceed with authenticated connection
                await handle_websocket_connection(websocket, client_id, user)
            
            except Exception as e:
                logger.error(f"Authentication error: {e}")
                await websocket.send_json({"error": "Authentication failed"})
                await websocket.close()
        
        except WebSocketDisconnect:
            logger.info(f"Client {client_id} disconnected during authentication")
        except Exception as e:
            logger.error(f"Error in websocket endpoint: {e}")


def start_background_tasks(app):
    """
    Start background tasks for real-time updates.
    
    Args:
        app: FastAPI application
    """
    @app.on_event("startup")
    async def start_tasks():
        asyncio.create_task(start_market_data_broadcaster()) 