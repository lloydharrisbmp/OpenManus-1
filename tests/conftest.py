"""
Pytest configuration and fixtures for the financial planning application.
"""
import os
import pytest
import asyncio
from typing import Generator, AsyncGenerator
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from app.web.app import app
from app.web.auth import get_test_token
from app.services.enhanced_database import EnhancedDatabaseService
from app.services.market_data import MarketDataService
from app.services.compliance_service import ComplianceService

# Test database URL
TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL", "sqlite+aiosqlite:///./test.db")

# Create test engine
engine = create_async_engine(TEST_DATABASE_URL, echo=True)
TestingSessionLocal = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
async def test_app() -> Generator:
    """Create a FastAPI TestClient instance that uses the test database."""
    # Configure test settings
    app.dependency_overrides = {}  # Reset any overrides
    yield app

@pytest.fixture(scope="session")
async def client() -> Generator:
    """Create a FastAPI TestClient instance."""
    with TestClient(app) as client:
        yield client

@pytest.fixture(scope="session")
async def auth_headers() -> dict:
    """Get authentication headers for testing."""
    token = get_test_token()
    return {"Authorization": f"Bearer {token}"}

@pytest.fixture(scope="function")
async def db_session() -> AsyncGenerator:
    """Create a fresh database session for each test."""
    async with TestingSessionLocal() as session:
        yield session
        await session.rollback()
        await session.close()

@pytest.fixture(scope="function")
async def enhanced_db() -> AsyncGenerator:
    """Create an EnhancedDatabaseService instance for testing."""
    test_db_path = "test_database"
    db_service = EnhancedDatabaseService(
        db_path=test_db_path,
        concurrency_limit=3,
        cleanup_interval=3600
    )
    yield db_service
    # Cleanup after tests
    await db_service.cleanup()

@pytest.fixture(scope="function")
def market_data_service() -> MarketDataService:
    """Create a MarketDataService instance for testing."""
    return MarketDataService(
        api_keys={"test": "test_key"},
        cache_duration=1,  # Short cache duration for testing
        rate_limit=100  # High rate limit for testing
    )

@pytest.fixture(scope="function")
def compliance_service() -> ComplianceService:
    """Create a ComplianceService instance for testing."""
    return ComplianceService(
        data_dir="test_compliance",
        concurrency_limit=3,
        cleanup_interval=3600
    )

@pytest.fixture(scope="function")
async def test_client(auth_headers) -> AsyncGenerator:
    """Create an authenticated test client."""
    with TestClient(app) as client:
        client.headers.update(auth_headers)
        yield client

@pytest.fixture(scope="session", autouse=True)
async def setup_test_environment():
    """Set up test environment before running tests."""
    # Set test environment variables
    os.environ["ENVIRONMENT"] = "test"
    os.environ["LOG_LEVEL"] = "DEBUG"
    
    # Create test directories
    test_dirs = ["test_database", "test_compliance", "test_uploads", "test_logs"]
    for dir_name in test_dirs:
        os.makedirs(dir_name, exist_ok=True)
    
    yield
    
    # Cleanup after all tests
    for dir_name in test_dirs:
        if os.path.exists(dir_name):
            for root, dirs, files in os.walk(dir_name, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(dir_name)

# Test utilities
def assert_successful_response(response, status_code=200):
    """Assert that a response is successful and return its JSON content."""
    assert response.status_code == status_code
    return response.json()

def assert_error_response(response, status_code, error_message=None):
    """Assert that a response is an error with the expected status code and message."""
    assert response.status_code == status_code
    if error_message:
        assert error_message in response.json()["detail"]

async def create_test_user(client, username="testuser", password="testpass"):
    """Create a test user and return the authentication token."""
    response = await client.post(
        "/register",
        json={"username": username, "password": password}
    )
    assert response.status_code == 200
    
    response = await client.post(
        "/token",
        data={"username": username, "password": password}
    )
    assert response.status_code == 200
    return response.json()["access_token"] 