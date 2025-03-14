from typing import Dict, Type, Optional
from pydantic import BaseModel

_model_registry: Dict[str, Type[BaseModel]] = {}

def register_model(name: str, model_cls: Type[BaseModel]) -> None:
    _model_registry[name] = model_cls

def get_model_by_name(name: str) -> Optional[Type[BaseModel]]:
    return _model_registry.get(name) 