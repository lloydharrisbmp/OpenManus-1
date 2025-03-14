"""
Dependency management system for the application.

This module provides a centralized way to manage dependencies and services
throughout the application, making it easier to maintain, test, and extend.
"""

from typing import Dict, Any, Type, TypeVar, Optional, cast
import inspect

T = TypeVar('T')

class DependencyManager:
    """
    A simple dependency injection container to manage application dependencies.
    
    Example usage:
        # Register a dependency
        DependencyManager.register(DatabaseService, lambda: DatabaseService(config))
        
        # Get a dependency
        db_service = DependencyManager.get(DatabaseService)
    """
    
    _instances: Dict[Type, Any] = {}
    _factories: Dict[Type, Any] = {}
    
    @classmethod
    def register(cls, dependency_type: Type[T], factory=None) -> None:
        """
        Register a dependency with the container.
        
        Args:
            dependency_type: The type to register
            factory: Optional factory function to create the instance
        """
        if factory is None:
            # If no factory provided, use the class constructor
            cls._factories[dependency_type] = dependency_type
        else:
            cls._factories[dependency_type] = factory
    
    @classmethod
    def get(cls, dependency_type: Type[T]) -> T:
        """
        Get an instance of the requested dependency.
        
        Args:
            dependency_type: The type to retrieve
            
        Returns:
            An instance of the requested type
            
        Raises:
            KeyError: If the dependency is not registered
        """
        # Return existing instance if available
        if dependency_type in cls._instances:
            return cast(T, cls._instances[dependency_type])
        
        # Create new instance if factory exists
        if dependency_type in cls._factories:
            factory = cls._factories[dependency_type]
            instance = factory()
            cls._instances[dependency_type] = instance
            return cast(T, instance)
        
        raise KeyError(f"Dependency {dependency_type.__name__} not registered")
    
    @classmethod
    def clear(cls) -> None:
        """Clear all registered dependencies (useful for testing)."""
        cls._instances.clear()
        cls._factories.clear()


def inject(dependency_type: Type[T]) -> T:
    """
    Decorator for automatically injecting dependencies into methods or functions.
    
    Example:
        class MyService:
            @inject(DatabaseService)
            def process_data(self, db_service):
                # Use db_service
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            if dependency_type.__name__ not in kwargs:
                kwargs[dependency_type.__name__.lower()] = DependencyManager.get(dependency_type)
            return func(*args, **kwargs)
        return wrapper
    return decorator
