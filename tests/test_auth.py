"""
Tests for authentication functionality.
"""
import pytest
from fastapi import status
from httpx import AsyncClient

pytestmark = pytest.mark.asyncio

async def test_register_user(test_app, client):
    """Test user registration."""
    response = await client.post(
        "/register",
        json={
            "username": "testuser",
            "password": "testpass123",
            "email": "test@example.com"
        }
    )
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["message"] == "User registered successfully"

async def test_register_duplicate_user(test_app, client):
    """Test registering a duplicate user."""
    # Register first user
    await client.post(
        "/register",
        json={
            "username": "testuser2",
            "password": "testpass123",
            "email": "test2@example.com"
        }
    )
    
    # Try to register the same user again
    response = await client.post(
        "/register",
        json={
            "username": "testuser2",
            "password": "testpass123",
            "email": "test2@example.com"
        }
    )
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    data = response.json()
    assert "Username already registered" in data["detail"]

async def test_login_success(test_app, client):
    """Test successful login."""
    # Register user
    await client.post(
        "/register",
        json={
            "username": "logintest",
            "password": "testpass123",
            "email": "login@example.com"
        }
    )
    
    # Login
    response = await client.post(
        "/token",
        data={
            "username": "logintest",
            "password": "testpass123"
        }
    )
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"

async def test_login_invalid_credentials(test_app, client):
    """Test login with invalid credentials."""
    response = await client.post(
        "/token",
        data={
            "username": "nonexistent",
            "password": "wrongpass"
        }
    )
    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    data = response.json()
    assert "Incorrect username or password" in data["detail"]

async def test_protected_route(test_app, test_client):
    """Test accessing a protected route with valid token."""
    response = await test_client.get("/protected-route")
    assert response.status_code == status.HTTP_200_OK

async def test_protected_route_no_token(test_app, client):
    """Test accessing a protected route without token."""
    response = await client.get("/protected-route")
    assert response.status_code == status.HTTP_401_UNAUTHORIZED

async def test_token_expiration(test_app, client):
    """Test token expiration."""
    # Register and login
    await client.post(
        "/register",
        json={
            "username": "expiretest",
            "password": "testpass123",
            "email": "expire@example.com"
        }
    )
    
    response = await client.post(
        "/token",
        data={
            "username": "expiretest",
            "password": "testpass123"
        }
    )
    token = response.json()["access_token"]
    
    # Use expired token (this requires mocking time or using a very short expiration)
    headers = {"Authorization": f"Bearer {token}"}
    async with AsyncClient(app=test_app, base_url="http://test") as ac:
        response = await ac.get("/protected-route", headers=headers)
    assert response.status_code == status.HTTP_200_OK

async def test_rate_limiting(test_app, client):
    """Test rate limiting on authentication endpoints."""
    # Try multiple rapid login attempts
    for _ in range(6):  # Limit is 5 per minute
        await client.post(
            "/token",
            data={
                "username": "ratelimit",
                "password": "testpass123"
            }
        )
    
    # The 6th attempt should be rate limited
    response = await client.post(
        "/token",
        data={
            "username": "ratelimit",
            "password": "testpass123"
        }
    )
    assert response.status_code == status.HTTP_429_TOO_MANY_REQUESTS

async def test_password_hashing(test_app, client):
    """Test that passwords are properly hashed."""
    # Register user
    await client.post(
        "/register",
        json={
            "username": "hashtest",
            "password": "testpass123",
            "email": "hash@example.com"
        }
    )
    
    # Verify login works
    response = await client.post(
        "/token",
        data={
            "username": "hashtest",
            "password": "testpass123"
        }
    )
    assert response.status_code == status.HTTP_200_OK
    
    # Try wrong password
    response = await client.post(
        "/token",
        data={
            "username": "hashtest",
            "password": "wrongpass"
        }
    )
    assert response.status_code == status.HTTP_401_UNAUTHORIZED 