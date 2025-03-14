"""
Database service for financial planning application.

This module provides an abstraction layer for database operations,
supporting both file-based and SQLite storage options.
"""

import json
import os
import sqlite3
import uuid
from datetime import datetime, date
from typing import Any, Dict, List, Optional, Union, Type, TypeVar
from pathlib import Path

from app.logger import logger
from app.exceptions import DocumentProcessingError

T = TypeVar('T')


class JSONEncoder(json.JSONEncoder):
    """Extended JSON encoder that handles dates and custom objects."""
    
    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if hasattr(obj, 'to_dict'):
            return obj.to_dict()
        return super().default(obj)


class DatabaseService:
    """
    Database service for storing client and financial data.
    
    Provides both file-based JSON storage and SQLite options.
    """
    
    def __init__(self, db_path: str = "database"):
        """
        Initialize the database service.
        
        Args:
            db_path: Path to the database directory or file
        """
        self.db_path = db_path
        self.sqlite_path = os.path.join(db_path, "financial_planning.db")
        
        # Create database directory if it doesn't exist
        os.makedirs(db_path, exist_ok=True)
        
        # Create subdirectories for different data types
        for subdir in ["clients", "portfolios", "reports", "analysis"]:
            os.makedirs(os.path.join(db_path, subdir), exist_ok=True)
        
        # Initialize SQLite if needed
        self._init_sqlite()
    
    def _init_sqlite(self):
        """Initialize SQLite database with required tables."""
        try:
            conn = sqlite3.connect(self.sqlite_path)
            cursor = conn.cursor()
            
            # Create clients table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS clients (
                client_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT,
                risk_profile TEXT,
                data JSON,
                created_at TEXT,
                updated_at TEXT
            )
            ''')
            
            # Create portfolios table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS portfolios (
                portfolio_id TEXT PRIMARY KEY,
                client_id TEXT,
                name TEXT NOT NULL,
                entity_type TEXT,
                risk_profile TEXT,
                data JSON,
                created_at TEXT,
                updated_at TEXT,
                FOREIGN KEY (client_id) REFERENCES clients (client_id)
            )
            ''')
            
            # Create financial goals table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS goals (
                goal_id TEXT PRIMARY KEY,
                client_id TEXT,
                goal_type TEXT,
                description TEXT,
                target_amount REAL,
                timeline_years INTEGER,
                priority INTEGER,
                current_progress REAL,
                created_at TEXT,
                updated_at TEXT,
                FOREIGN KEY (client_id) REFERENCES clients (client_id)
            )
            ''')
            
            # Create reports table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS reports (
                report_id TEXT PRIMARY KEY,
                client_id TEXT,
                title TEXT,
                report_type TEXT,
                file_path TEXT,
                format TEXT,
                created_at TEXT,
                FOREIGN KEY (client_id) REFERENCES clients (client_id)
            )
            ''')
            
            conn.commit()
            conn.close()
        except sqlite3.Error as e:
            logger.error(f"SQLite initialization error: {e}")
    
    def save_json(self, data: Dict, collection: str, doc_id: Optional[str] = None) -> str:
        """
        Save data as a JSON file.
        
        Args:
            data: Data to save
            collection: Collection/directory name
            doc_id: Document ID (generated if not provided)
            
        Returns:
            Document ID
        """
        if doc_id is None:
            doc_id = str(uuid.uuid4())
        
        file_path = os.path.join(self.db_path, collection, f"{doc_id}.json")
        
        try:
            with open(file_path, 'w') as f:
                json.dump(data, f, cls=JSONEncoder, indent=2)
        except IOError as e:
            logger.error(f"Error saving JSON to {file_path}: {e}")
            raise DocumentProcessingError(file_path, f"Failed to save JSON document: {e}")
        
        return doc_id
    
    def load_json(self, collection: str, doc_id: str) -> Optional[Dict]:
        """
        Load data from a JSON file.
        
        Args:
            collection: Collection/directory name
            doc_id: Document ID
            
        Returns:
            Loaded data or None if not found
        """
        file_path = os.path.join(self.db_path, collection, f"{doc_id}.json")
        
        if not os.path.exists(file_path):
            return None
        
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            logger.error(f"Error loading JSON from {file_path}: {e}")
            return None
    
    def list_documents(self, collection: str) -> List[str]:
        """
        List all document IDs in a collection.
        
        Args:
            collection: Collection/directory name
            
        Returns:
            List of document IDs
        """
        collection_path = os.path.join(self.db_path, collection)
        
        if not os.path.exists(collection_path):
            return []
        
        return [
            file_name.replace('.json', '')
            for file_name in os.listdir(collection_path)
            if file_name.endswith('.json')
        ]
    
    def delete_document(self, collection: str, doc_id: str) -> bool:
        """
        Delete a document.
        
        Args:
            collection: Collection/directory name
            doc_id: Document ID
            
        Returns:
            True if successful, False otherwise
        """
        file_path = os.path.join(self.db_path, collection, f"{doc_id}.json")
        
        if not os.path.exists(file_path):
            return False
        
        try:
            os.remove(file_path)
            return True
        except OSError as e:
            logger.error(f"Error deleting {file_path}: {e}")
            return False
    
    def save_client(self, client_data: Dict) -> str:
        """
        Save client data to SQLite.
        
        Args:
            client_data: Client data
            
        Returns:
            Client ID
        """
        client_id = client_data.get('client_id', str(uuid.uuid4()))
        client_data['client_id'] = client_id
        now = datetime.now().isoformat()
        
        try:
            conn = sqlite3.connect(self.sqlite_path)
            cursor = conn.cursor()
            
            # Check if client exists
            cursor.execute("SELECT client_id FROM clients WHERE client_id = ?", (client_id,))
            exists = cursor.fetchone() is not None
            
            if exists:
                # Update
                cursor.execute(
                    "UPDATE clients SET name = ?, email = ?, risk_profile = ?, data = ?, updated_at = ? WHERE client_id = ?",
                    (
                        client_data.get('name', ''),
                        client_data.get('email'),
                        client_data.get('risk_profile', 'moderate'),
                        json.dumps(client_data, cls=JSONEncoder),
                        now,
                        client_id
                    )
                )
            else:
                # Insert
                cursor.execute(
                    "INSERT INTO clients (client_id, name, email, risk_profile, data, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        client_id,
                        client_data.get('name', ''),
                        client_data.get('email'),
                        client_data.get('risk_profile', 'moderate'),
                        json.dumps(client_data, cls=JSONEncoder),
                        now,
                        now
                    )
                )
            
            conn.commit()
            conn.close()
            
            # Also save as JSON for backward compatibility
            self.save_json(client_data, "clients", client_id)
            
            return client_id
        except sqlite3.Error as e:
            logger.error(f"SQLite error saving client: {e}")
            # Fall back to JSON
            return self.save_json(client_data, "clients", client_id)
    
    def get_client(self, client_id: str) -> Optional[Dict]:
        """
        Get client data from SQLite.
        
        Args:
            client_id: Client ID
            
        Returns:
            Client data or None if not found
        """
        try:
            conn = sqlite3.connect(self.sqlite_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT data FROM clients WHERE client_id = ?", (client_id,))
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return json.loads(result[0])
            
            # Try JSON fallback
            return self.load_json("clients", client_id)
        except sqlite3.Error as e:
            logger.error(f"SQLite error getting client: {e}")
            # Fall back to JSON
            return self.load_json("clients", client_id)
    
    def save_portfolio(self, portfolio_data: Dict) -> str:
        """
        Save portfolio data to SQLite.
        
        Args:
            portfolio_data: Portfolio data
            
        Returns:
            Portfolio ID
        """
        portfolio_id = portfolio_data.get('portfolio_id', str(uuid.uuid4()))
        portfolio_data['portfolio_id'] = portfolio_id
        now = datetime.now().isoformat()
        
        try:
            conn = sqlite3.connect(self.sqlite_path)
            cursor = conn.cursor()
            
            # Check if portfolio exists
            cursor.execute("SELECT portfolio_id FROM portfolios WHERE portfolio_id = ?", (portfolio_id,))
            exists = cursor.fetchone() is not None
            
            if exists:
                # Update
                cursor.execute(
                    "UPDATE portfolios SET name = ?, client_id = ?, entity_type = ?, risk_profile = ?, data = ?, updated_at = ? WHERE portfolio_id = ?",
                    (
                        portfolio_data.get('name', ''),
                        portfolio_data.get('client_id'),
                        portfolio_data.get('entity_type', 'individual'),
                        portfolio_data.get('risk_profile', 'moderate'),
                        json.dumps(portfolio_data, cls=JSONEncoder),
                        now,
                        portfolio_id
                    )
                )
            else:
                # Insert
                cursor.execute(
                    "INSERT INTO portfolios (portfolio_id, client_id, name, entity_type, risk_profile, data, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        portfolio_id,
                        portfolio_data.get('client_id'),
                        portfolio_data.get('name', ''),
                        portfolio_data.get('entity_type', 'individual'),
                        portfolio_data.get('risk_profile', 'moderate'),
                        json.dumps(portfolio_data, cls=JSONEncoder),
                        now,
                        now
                    )
                )
            
            conn.commit()
            conn.close()
            
            # Also save as JSON for backward compatibility
            self.save_json(portfolio_data, "portfolios", portfolio_id)
            
            return portfolio_id
        except sqlite3.Error as e:
            logger.error(f"SQLite error saving portfolio: {e}")
            # Fall back to JSON
            return self.save_json(portfolio_data, "portfolios", portfolio_id)
    
    def get_client_portfolios(self, client_id: str) -> List[Dict]:
        """
        Get all portfolios for a client.
        
        Args:
            client_id: Client ID
            
        Returns:
            List of portfolio data
        """
        try:
            conn = sqlite3.connect(self.sqlite_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT data FROM portfolios WHERE client_id = ?", (client_id,))
            results = cursor.fetchall()
            conn.close()
            
            return [json.loads(row[0]) for row in results]
        except sqlite3.Error as e:
            logger.error(f"SQLite error getting client portfolios: {e}")
            # Try to find any JSON portfolios for this client
            portfolios = []
            for portfolio_id in self.list_documents("portfolios"):
                portfolio = self.load_json("portfolios", portfolio_id)
                if portfolio and portfolio.get('client_id') == client_id:
                    portfolios.append(portfolio)
            return portfolios

    def count_tenant_clients(self, tenant_id: str) -> int:
        """
        Count the number of clients for a tenant.
        
        Args:
            tenant_id: Tenant ID
            
        Returns:
            Number of clients
        """
        try:
            conn = sqlite3.connect(self.sqlite_path)
            cursor = conn.cursor()
            
            # Execute query to count clients with the given tenant_id
            cursor.execute("""
            SELECT COUNT(*) FROM clients 
            WHERE JSON_EXTRACT(data, '$.tenant_id') = ?
            """, (tenant_id,))
            
            result = cursor.fetchone()
            conn.close()
            
            return result[0] if result else 0
        except sqlite3.Error as e:
            logger.error(f"SQLite error counting tenant clients: {e}")
            
            # Fall back to JSON count
            count = 0
            for client_id in self.list_documents("clients"):
                client = self.load_json("clients", client_id)
                if client and client.get('tenant_id') == tenant_id:
                    count += 1
            return count

    def get_tenant_clients(self, tenant_id: str) -> List[Dict]:
        """
        Get all clients for a tenant.
        
        Args:
            tenant_id: Tenant ID
            
        Returns:
            List of client data
        """
        try:
            conn = sqlite3.connect(self.sqlite_path)
            cursor = conn.cursor()
            
            # Execute query to get clients with the given tenant_id
            cursor.execute("""
            SELECT data FROM clients 
            WHERE JSON_EXTRACT(data, '$.tenant_id') = ?
            """, (tenant_id,))
            
            results = cursor.fetchall()
            conn.close()
            
            return [json.loads(row[0]) for row in results]
        except sqlite3.Error as e:
            logger.error(f"SQLite error getting tenant clients: {e}")
            
            # Fall back to JSON search
            clients = []
            for client_id in self.list_documents("clients"):
                client = self.load_json("clients", client_id)
                if client and client.get('tenant_id') == tenant_id:
                    clients.append(client)
            return clients

    def save_client_with_tenant(self, client_data: Dict, tenant_id: str) -> str:
        """
        Save client data with tenant association.
        
        Args:
            client_data: Client data
            tenant_id: Tenant ID
            
        Returns:
            Client ID
        """
        # Add tenant_id to client data
        client_data['tenant_id'] = tenant_id
        
        # Use existing save_client method
        return self.save_client(client_data)

    def save_portfolio_with_tenant(self, portfolio_data: Dict, tenant_id: str) -> str:
        """
        Save portfolio data with tenant association.
        
        Args:
            portfolio_data: Portfolio data
            tenant_id: Tenant ID
            
        Returns:
            Portfolio ID
        """
        # Add tenant_id to portfolio data
        portfolio_data['tenant_id'] = tenant_id
        
        # Use existing save_portfolio method
        return self.save_portfolio(portfolio_data)

    def get_tenant_resources(self, tenant_id: str, resource_type: str) -> List[Dict]:
        """
        Get all resources of a specific type for a tenant.
        
        Args:
            tenant_id: Tenant ID
            resource_type: Type of resource (clients, portfolios, reports)
            
        Returns:
            List of resource data
        """
        if resource_type == "clients":
            return self.get_tenant_clients(tenant_id)
        
        try:
            conn = sqlite3.connect(self.sqlite_path)
            cursor = conn.cursor()
            
            # Execute query based on resource type
            if resource_type == "portfolios":
                cursor.execute("""
                SELECT data FROM portfolios 
                WHERE JSON_EXTRACT(data, '$.tenant_id') = ?
                """, (tenant_id,))
            elif resource_type == "reports":
                cursor.execute("""
                SELECT data FROM reports 
                WHERE JSON_EXTRACT(data, '$.tenant_id') = ?
                """, (tenant_id,))
            else:
                conn.close()
                logger.error(f"Unknown resource type: {resource_type}")
                return []
            
            results = cursor.fetchall()
            conn.close()
            
            return [json.loads(row[0]) for row in results]
        except sqlite3.Error as e:
            logger.error(f"SQLite error getting tenant resources: {e}")
            
            # Fall back to JSON search
            resources = []
            for resource_id in self.list_documents(resource_type):
                resource = self.load_json(resource_type, resource_id)
                if resource and resource.get('tenant_id') == tenant_id:
                    resources.append(resource)
            return resources


# Global instance
db_service = DatabaseService() 