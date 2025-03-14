"""
API documentation for the financial planning application.

This module configures the OpenAPI schema for the application and
provides the Swagger UI interface.
"""

from fastapi.openapi.utils import get_openapi
from fastapi import FastAPI
from fastapi.openapi.docs import get_swagger_ui_html

from app.schema import (
    ClientProfile,
    Portfolio,
    TaxOptimizationRequest,
    TaxOptimizationResponse,
    PortfolioOptimizationRequest,
    PortfolioOptimizationResponse,
    MarketAnalysisRequest,
    MarketAnalysisResponse,
    ReportGenerationRequest,
    ReportGenerationResponse,
    WebsiteGenerationRequest,
    WebsiteGenerationResponse,
)


def custom_openapi_schema(app: FastAPI):
    """
    Generate a custom OpenAPI schema for the application.
    
    Args:
        app: FastAPI application instance
        
    Returns:
        OpenAPI schema
    """
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="Australian Financial Planning API",
        version="1.0.0",
        description="API for the Australian Financial Planning application, providing financial planning, portfolio optimization, and tax strategies for high net worth individuals.",
        routes=app.routes,
    )
    
    # Add security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "bearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
        }
    }
    
    # Apply security globally
    openapi_schema["security"] = [{"bearerAuth": []}]
    
    # Add custom examples for request/response bodies
    # Client example
    openapi_schema["components"]["examples"] = {
        "ClientProfile": {
            "value": {
                "client_id": "c01b785c-87f5-4d91-ac06-a0c2ca8c1c66",
                "name": "John Smith",
                "email": "john.smith@example.com",
                "risk_profile": "moderate",
                "annual_income": 250000,
                "has_spouse": True,
                "number_of_dependents": 2
            }
        },
        "Portfolio": {
            "value": {
                "portfolio_id": "p7834b51-1cd4-42c7-9c88-71c630e3be8f",
                "name": "Growth Portfolio",
                "entity_type": "individual",
                "risk_profile": "moderate_aggressive",
                "holdings": [
                    {
                        "asset_code": "ASX:VAS",
                        "asset_type": "equity",
                        "quantity": 1000,
                        "purchase_price": 85.50,
                        "current_value": 92000
                    }
                ]
            }
        },
        "TaxOptimizationResponse": {
            "value": {
                "tax_summary": {
                    "total_income": 300000,
                    "total_tax": 95350,
                    "marginal_rate": 0.45,
                    "effective_rate": 0.318
                },
                "optimization_strategies": [
                    {
                        "strategy": "Income splitting",
                        "potential_savings": 15000,
                        "complexity": "medium"
                    }
                ],
                "recommended_actions": [
                    "Consider establishing a family trust to distribute income",
                    "Maximize concessional superannuation contributions"
                ],
                "estimated_tax_savings": 18500
            }
        }
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


def setup_api_docs(app: FastAPI):
    """
    Set up API documentation for the application.
    
    Args:
        app: FastAPI application instance
    """
    @app.get("/api/docs", include_in_schema=False)
    async def get_documentation():
        """Get custom Swagger UI documentation."""
        return get_swagger_ui_html(
            openapi_url="/api/openapi.json",
            title="Australian Financial Planning API Documentation",
            swagger_favicon_url="/static/images/favicon.ico",
        )
    
    @app.get("/api/openapi.json", include_in_schema=False)
    async def get_openapi_schema():
        """Get OpenAPI schema."""
        return custom_openapi_schema(app)
    
    # Override default schema
    app.openapi = lambda: custom_openapi_schema(app) 