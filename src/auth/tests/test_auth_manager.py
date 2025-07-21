"""
Tests for AuthManager

Comprehensive tests for the main authentication manager.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from ..auth_manager import AuthManager, AuthConfig
from ..models import User, AuthProvider, UserStatus
from ..exceptions import (
    AuthenticationError,
    InvalidCredentialsError,
    AccountLockedError,
    TenantInactiveError
)


@pytest.fixture
async def auth_manager():
    """Create AuthManager for testing."""
    config = AuthConfig()
    config.max_failed_attempts = 3
    config.lockout_duration_minutes = 5
    
    manager = AuthManager(config)
    await manager.initialize()
    
    yield manager
    
    await manager.cleanup()


@pytest.fixture
async def test_user(auth_manager):
    """Create test user."""
    user = await auth_manager.create_user(
        email="test@example.com",
        username="testuser",
        full_name="Test User",
        password="TestPassword123!",
        tenant_id="test_tenant"
    )
    return user


class TestAuthManager:
    """Test cases for AuthManager."""
    
    @pytest.mark.asyncio
    async def test_create_user(self, auth_manager):
        """Test user creation."""
        user = await auth_manager.create_user(
            email="new@example.com",
            username="newuser",
            full_name="New User",
            password="Password123!",
            tenant_id="tenant1"
        )
        
        assert user.email == "new@example.com"
        assert user.username == "newuser"
        assert user.full_name == "New User"
        assert user.tenant_id == "tenant1"
        assert user.auth_provider == AuthProvider.LOCAL
        assert user.status == UserStatus.ACTIVE
        assert user.password_hash is not None
    
    @pytest.mark.asyncio
    async def test_authenticate_success(self, auth_manager, test_user):
        """Test successful authentication."""
        credentials = {
            'email': 'test@example.com',
            'password': 'TestPassword123!',
            'ip_address': '127.0.0.1',
            'user_agent': 'Test Agent'
        }
        
        user, session = await auth_manager.authenticate(credentials)
        
        assert user.id == test_user.id
        assert user.email == test_user.email
        assert session is not None
        assert session.user_id == user.id
        assert session.is_valid()
    
    @pytest.mark.asyncio
    async def test_authenticate_invalid_password(self, auth_manager, test_user):
        """Test authentication with invalid password."""
        credentials = {
            'email': 'test@example.com',
            'password': 'WrongPassword',
            'ip_address': '127.0.0.1'
        }
        
        with pytest.raises(InvalidCredentialsError):
            await auth_manager.authenticate(credentials)
    
    @pytest.mark.asyncio
    async def test_authenticate_nonexistent_user(self, auth_manager):
        """Test authentication with nonexistent user."""
        credentials = {
            'email': 'nonexistent@example.com',
            'password': 'Password123!',
            'ip_address': '127.0.0.1'
        }
        
        with pytest.raises(InvalidCredentialsError):
            await auth_manager.authenticate(credentials)
    
    @pytest.mark.asyncio
    async def test_account_lockout(self, auth_manager, test_user):
        """Test account lockout after multiple failed attempts."""
        credentials = {
            'email': 'test@example.com',
            'password': 'WrongPassword',
            'ip_address': '127.0.0.1'
        }
        
        # Make multiple failed attempts
        for _ in range(auth_manager.config.max_failed_attempts):
            with pytest.raises(InvalidCredentialsError):
                await auth_manager.authenticate(credentials)
        
        # Next attempt should be account locked
        with pytest.raises(AccountLockedError):
            await auth_manager.authenticate(credentials)
        
        # Even correct password should fail when locked
        credentials['password'] = 'TestPassword123!'
        with pytest.raises(AccountLockedError):
            await auth_manager.authenticate(credentials)
    
    @pytest.mark.asyncio
    async def test_jwt_token_creation_validation(self, auth_manager, test_user):
        """Test JWT token creation and validation."""
        # Create session
        session = await auth_manager.session_manager.create_session(
            user_id=test_user.id,
            tenant_id=test_user.tenant_id
        )
        
        # Create JWT token
        token = await auth_manager.create_jwt_token(test_user, session)
        assert token is not None
        assert isinstance(token, str)
        
        # Validate token
        payload = await auth_manager.validate_jwt_token(token)
        assert payload['user_id'] == test_user.id
        assert payload['session_id'] == session.id
        assert payload['tenant_id'] == test_user.tenant_id
    
    @pytest.mark.asyncio
    async def test_session_validation(self, auth_manager, test_user):
        """Test session validation."""
        # Create session
        session = await auth_manager.session_manager.create_session(
            user_id=test_user.id,
            tenant_id=test_user.tenant_id,
            ip_address='127.0.0.1'
        )
        
        # Validate session
        validated_user, validated_session = await auth_manager.validate_session(session.token)
        
        assert validated_user.id == test_user.id
        assert validated_session.id == session.id
    
    @pytest.mark.asyncio
    async def test_logout(self, auth_manager, test_user):
        """Test logout functionality."""
        # Create session
        session = await auth_manager.session_manager.create_session(
            user_id=test_user.id,
            tenant_id=test_user.tenant_id
        )
        
        # Logout
        await auth_manager.logout(session.token)
        
        # Session should be invalid
        invalid_session = await auth_manager.session_manager.get_session(session.token)
        assert invalid_session is None
    
    @pytest.mark.asyncio
    async def test_user_update(self, auth_manager, test_user):
        """Test user update."""
        updates = {
            'full_name': 'Updated Name',
            'status': UserStatus.INACTIVE
        }
        
        updated_user = await auth_manager.update_user(test_user.id, updates)
        
        assert updated_user.full_name == 'Updated Name'
        assert updated_user.status == UserStatus.INACTIVE
        assert updated_user.updated_at > test_user.updated_at
    
    @pytest.mark.asyncio
    async def test_user_deletion(self, auth_manager, test_user):
        """Test user deletion."""
        user_id = test_user.id
        
        # Delete user
        await auth_manager.delete_user(user_id)
        
        # User should not exist
        deleted_user = await auth_manager.get_user(user_id)
        assert deleted_user is None
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, auth_manager):
        """Test rate limiting functionality."""
        # Mock rate limit config
        auth_manager.config.rate_limit_enabled = True
        auth_manager.config.rate_limit_requests_per_minute = 2
        
        identifier = 'test_user'
        
        # First two requests should pass
        assert await auth_manager._check_rate_limit(identifier) is True
        assert await auth_manager._check_rate_limit(identifier) is True
        
        # Third request should fail
        assert await auth_manager._check_rate_limit(identifier) is False
    
    @pytest.mark.asyncio
    async def test_tenant_validation(self, auth_manager):
        """Test tenant validation during authentication."""
        # Create user in tenant1
        user = await auth_manager.create_user(
            email="tenant1@example.com",
            username="tenant1user",
            full_name="Tenant 1 User",
            password="Password123!",
            tenant_id="tenant1"
        )
        
        credentials = {
            'email': 'tenant1@example.com',
            'password': 'Password123!',
            'ip_address': '127.0.0.1'
        }
        
        # Should work with matching tenant
        auth_user, session = await auth_manager.authenticate(
            credentials, tenant_id="tenant1"
        )
        assert auth_user.id == user.id
        
        # Should fail with different tenant
        with pytest.raises(InvalidCredentialsError):
            await auth_manager.authenticate(credentials, tenant_id="tenant2")


@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()