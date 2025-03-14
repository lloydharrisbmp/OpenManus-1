"""
Multi-tenancy service for the financial planning application.

This module provides functionality for managing multiple tenants (financial advisory firms)
and their users within the system.
"""

import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime
import os
import json

from pydantic import BaseModel, Field

from app.logger import logger
from app.exceptions import ConfigurationError
from app.dependency_manager import DependencyManager
from app.services.database import db_service


class TenantPlan(str):
    """Subscription plan types for tenants."""
    BASIC = "basic"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"


class Tenant(BaseModel):
    """
    Tenant model representing a financial advisory firm.
    
    Each tenant can have multiple users (advisors) and clients.
    """
    tenant_id: str
    name: str
    plan_type: str = TenantPlan.BASIC
    domain: Optional[str] = None
    contact_email: Optional[str] = None
    logo_url: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    is_active: bool = True
    max_users: int = 5
    max_clients: int = 50
    features: Dict[str, bool] = Field(default_factory=dict)
    custom_branding: bool = False
    custom_domain: bool = False
    api_keys: List[str] = Field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert tenant to dictionary."""
        return {
            "tenant_id": self.tenant_id,
            "name": self.name,
            "plan_type": self.plan_type,
            "domain": self.domain,
            "contact_email": self.contact_email,
            "logo_url": self.logo_url,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "is_active": self.is_active,
            "max_users": self.max_users,
            "max_clients": self.max_clients,
            "features": self.features,
            "custom_branding": self.custom_branding,
            "custom_domain": self.custom_domain,
            "api_keys": self.api_keys
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Tenant":
        """Create tenant from dictionary."""
        if isinstance(data.get("created_at"), str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if isinstance(data.get("updated_at"), str):
            data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        return cls(**data)


class TenantUser(BaseModel):
    """
    User association with a tenant.
    
    Links a user to a specific tenant with role information.
    """
    tenant_id: str
    user_id: str
    role: str
    permissions: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    is_active: bool = True


class TenantService:
    """
    Service for managing tenants in the system.
    
    Handles tenant creation, retrieval, and user-tenant associations.
    """
    
    def __init__(self, data_dir: str = "tenants"):
        """
        Initialize the tenant service.
        
        Args:
            data_dir: Directory to store tenant data
        """
        self.data_dir = data_dir
        self.tenants_file = os.path.join(data_dir, "tenants.json")
        self.tenant_users_file = os.path.join(data_dir, "tenant_users.json")
        self.tenants: Dict[str, Tenant] = {}
        self.tenant_users: Dict[str, List[TenantUser]] = {}
        self.user_tenants: Dict[str, List[str]] = {}
        
        # Create directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Load existing data
        self._load_data()
    
    def _load_data(self):
        """Load tenant data from files."""
        # Load tenants
        if os.path.exists(self.tenants_file):
            try:
                with open(self.tenants_file, 'r') as f:
                    tenant_data = json.load(f)
                    for tenant_id, data in tenant_data.items():
                        self.tenants[tenant_id] = Tenant.from_dict(data)
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Error loading tenants data: {e}")
        
        # Load tenant users
        if os.path.exists(self.tenant_users_file):
            try:
                with open(self.tenant_users_file, 'r') as f:
                    self.tenant_users = json.load(f)
                    
                    # Build user to tenants mapping
                    for tenant_id, users in self.tenant_users.items():
                        for user in users:
                            user_id = user["user_id"]
                            if user_id not in self.user_tenants:
                                self.user_tenants[user_id] = []
                            if tenant_id not in self.user_tenants[user_id]:
                                self.user_tenants[user_id].append(tenant_id)
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Error loading tenant users data: {e}")
    
    def _save_tenants(self):
        """Save tenants to file."""
        try:
            with open(self.tenants_file, 'w') as f:
                tenant_data = {tenant_id: tenant.to_dict() for tenant_id, tenant in self.tenants.items()}
                json.dump(tenant_data, f, indent=2)
        except IOError as e:
            logger.error(f"Error saving tenants data: {e}")
    
    def _save_tenant_users(self):
        """Save tenant users to file."""
        try:
            with open(self.tenant_users_file, 'w') as f:
                json.dump(self.tenant_users, f, indent=2)
        except IOError as e:
            logger.error(f"Error saving tenant users data: {e}")
    
    def create_tenant(self, name: str, plan_type: str = TenantPlan.BASIC, **kwargs) -> Tenant:
        """
        Create a new tenant.
        
        Args:
            name: Tenant name
            plan_type: Subscription plan type
            **kwargs: Additional tenant properties
            
        Returns:
            Created tenant
        """
        tenant_id = str(uuid.uuid4())
        
        # Set default features based on plan type
        features = {}
        custom_branding = False
        custom_domain = False
        max_users = 5
        max_clients = 50
        
        if plan_type == TenantPlan.PROFESSIONAL:
            features = {
                "portfolio_optimization": True,
                "tax_optimization": True,
                "report_generation": True,
                "client_portal": True
            }
            custom_branding = True
            max_users = 20
            max_clients = 200
        
        elif plan_type == TenantPlan.ENTERPRISE:
            features = {
                "portfolio_optimization": True,
                "tax_optimization": True,
                "report_generation": True,
                "client_portal": True,
                "api_access": True,
                "white_label": True,
                "dedicated_support": True
            }
            custom_branding = True
            custom_domain = True
            max_users = 50
            max_clients = 500
        
        else:  # Basic plan
            features = {
                "portfolio_optimization": True,
                "tax_optimization": False,
                "report_generation": True,
                "client_portal": False
            }
        
        # Create tenant
        tenant = Tenant(
            tenant_id=tenant_id,
            name=name,
            plan_type=plan_type,
            features=features,
            custom_branding=custom_branding,
            custom_domain=custom_domain,
            max_users=max_users,
            max_clients=max_clients,
            **kwargs
        )
        
        # Save tenant
        self.tenants[tenant_id] = tenant
        self._save_tenants()
        
        # Initialize tenant users list
        self.tenant_users[tenant_id] = []
        self._save_tenant_users()
        
        return tenant
    
    def get_tenant(self, tenant_id: str) -> Optional[Tenant]:
        """
        Get a tenant by ID.
        
        Args:
            tenant_id: Tenant ID
            
        Returns:
            Tenant or None if not found
        """
        return self.tenants.get(tenant_id)
    
    def update_tenant(self, tenant_id: str, **kwargs) -> Optional[Tenant]:
        """
        Update a tenant.
        
        Args:
            tenant_id: Tenant ID
            **kwargs: Properties to update
            
        Returns:
            Updated tenant or None if not found
        """
        tenant = self.get_tenant(tenant_id)
        if not tenant:
            return None
        
        # Update tenant properties
        for key, value in kwargs.items():
            if hasattr(tenant, key):
                setattr(tenant, key, value)
        
        # Update timestamp
        tenant.updated_at = datetime.now()
        
        # Save changes
        self._save_tenants()
        
        return tenant
    
    def delete_tenant(self, tenant_id: str) -> bool:
        """
        Delete a tenant.
        
        Args:
            tenant_id: Tenant ID
            
        Returns:
            True if successful, False otherwise
        """
        if tenant_id not in self.tenants:
            return False
        
        # Remove tenant
        del self.tenants[tenant_id]
        
        # Remove tenant users
        if tenant_id in self.tenant_users:
            # First update user_tenants mapping
            for user in self.tenant_users[tenant_id]:
                user_id = user["user_id"]
                if user_id in self.user_tenants and tenant_id in self.user_tenants[user_id]:
                    self.user_tenants[user_id].remove(tenant_id)
            
            # Then remove tenant users
            del self.tenant_users[tenant_id]
        
        # Save changes
        self._save_tenants()
        self._save_tenant_users()
        
        return True
    
    def add_user_to_tenant(self, tenant_id: str, user_id: str, role: str, permissions: List[str] = None) -> bool:
        """
        Add a user to a tenant.
        
        Args:
            tenant_id: Tenant ID
            user_id: User ID
            role: User role in the tenant
            permissions: Optional list of permissions
            
        Returns:
            True if successful, False otherwise
        """
        if tenant_id not in self.tenants:
            return False
        
        # Initialize tenant users list if needed
        if tenant_id not in self.tenant_users:
            self.tenant_users[tenant_id] = []
        
        # Check if user already exists in tenant
        for user in self.tenant_users[tenant_id]:
            if user["user_id"] == user_id:
                # Update role and permissions
                user["role"] = role
                if permissions:
                    user["permissions"] = permissions
                self._save_tenant_users()
                return True
        
        # Add new user to tenant
        tenant_user = TenantUser(
            tenant_id=tenant_id,
            user_id=user_id,
            role=role,
            permissions=permissions or []
        )
        
        self.tenant_users[tenant_id].append(tenant_user.dict())
        
        # Update user_tenants mapping
        if user_id not in self.user_tenants:
            self.user_tenants[user_id] = []
        if tenant_id not in self.user_tenants[user_id]:
            self.user_tenants[user_id].append(tenant_id)
        
        # Save changes
        self._save_tenant_users()
        
        return True
    
    def remove_user_from_tenant(self, tenant_id: str, user_id: str) -> bool:
        """
        Remove a user from a tenant.
        
        Args:
            tenant_id: Tenant ID
            user_id: User ID
            
        Returns:
            True if successful, False otherwise
        """
        if tenant_id not in self.tenant_users:
            return False
        
        # Find and remove user
        for i, user in enumerate(self.tenant_users[tenant_id]):
            if user["user_id"] == user_id:
                self.tenant_users[tenant_id].pop(i)
                
                # Update user_tenants mapping
                if user_id in self.user_tenants and tenant_id in self.user_tenants[user_id]:
                    self.user_tenants[user_id].remove(tenant_id)
                
                # Save changes
                self._save_tenant_users()
                return True
        
        return False
    
    def get_user_tenants(self, user_id: str) -> List[Tenant]:
        """
        Get all tenants for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            List of tenants
        """
        tenant_ids = self.user_tenants.get(user_id, [])
        return [self.tenants[tenant_id] for tenant_id in tenant_ids if tenant_id in self.tenants]
    
    def get_tenant_users(self, tenant_id: str) -> List[Dict[str, Any]]:
        """
        Get all users for a tenant.
        
        Args:
            tenant_id: Tenant ID
            
        Returns:
            List of tenant users
        """
        return self.tenant_users.get(tenant_id, [])
    
    def get_user_role_in_tenant(self, tenant_id: str, user_id: str) -> Optional[str]:
        """
        Get a user's role in a tenant.
        
        Args:
            tenant_id: Tenant ID
            user_id: User ID
            
        Returns:
            User role or None if not found
        """
        if tenant_id not in self.tenant_users:
            return None
        
        for user in self.tenant_users[tenant_id]:
            if user["user_id"] == user_id:
                return user["role"]
        
        return None
    
    def get_tenant_by_domain(self, domain: str) -> Optional[Tenant]:
        """
        Get a tenant by domain.
        
        Args:
            domain: Tenant domain
            
        Returns:
            Tenant or None if not found
        """
        for tenant in self.tenants.values():
            if tenant.domain == domain:
                return tenant
        
        return None
    
    def check_tenant_limit(self, tenant_id: str, limit_type: str) -> bool:
        """
        Check if a tenant has reached a limit.
        
        Args:
            tenant_id: Tenant ID
            limit_type: Type of limit to check (users, clients)
            
        Returns:
            True if limit is not reached, False otherwise
        """
        tenant = self.get_tenant(tenant_id)
        if not tenant:
            return False
        
        if limit_type == "users":
            current_users = len(self.get_tenant_users(tenant_id))
            return current_users < tenant.max_users
        
        elif limit_type == "clients":
            # Count clients associated with this tenant
            client_count = db_service.count_tenant_clients(tenant_id)
            return client_count < tenant.max_clients
        
        return True


# Create global tenant service
tenant_service = TenantService()

# Register with dependency manager
DependencyManager.register(TenantService, lambda: tenant_service) 