"""
Main Authentication Manager

Coordinates all authentication providers and manages the authentication flow.
"""

import asyncio
import logging
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timedelta
import jwt
import secrets

from .models import User, Session, AuthProvider, UserStatus, Tenant
from .session_manager import SessionManager
from .rbac_manager import RBACManager
from .oauth_provider import OAuthProvider
from .saml_provider import SAMLProvider
from .api_key_manager import ApiKeyManager
from .tenant_manager import TenantManager
from .audit_logger import AuthAuditLogger
from .exceptions import (
    AuthenticationError,
    AuthorizationError,
    InvalidCredentialsError,
    AccountLockedError,
    TenantInactiveError,
    SessionExpiredError
)

logger = logging.getLogger(__name__)


class AuthConfig:
    """Authentication configuration."""
    def __init__(self):
        # JWT settings
        self.jwt_secret = secrets.token_urlsafe(64)
        self.jwt_algorithm = "HS256"
        self.jwt_expiry_hours = 24
        
        # Session settings
        self.session_timeout_minutes = 30
        self.max_sessions_per_user = 5
        self.allow_concurrent_sessions = True
        
        # Security settings
        self.max_failed_attempts = 5
        self.lockout_duration_minutes = 30
        self.require_mfa_for_admin = True
        self.password_min_length = 12
        self.password_require_complexity = True
        
        # OAuth settings
        self.oauth_providers = {}
        
        # SAML settings
        self.saml_providers = {}
        
        # API key settings
        self.api_key_header = "X-API-Key"
        self.api_key_query_param = "api_key"
        
        # Rate limiting
        self.rate_limit_enabled = True
        self.rate_limit_requests_per_minute = 60


class AuthManager:
    """Main authentication manager coordinating all auth providers."""
    
    def __init__(self, config: AuthConfig = None):
        self.config = config or AuthConfig()
        
        # Initialize managers
        self.session_manager = SessionManager(self.config)
        self.rbac_manager = RBACManager()
        self.oauth_provider = OAuthProvider(self.config)
        self.saml_provider = SAMLProvider(self.config)
        self.api_key_manager = ApiKeyManager(self.config)
        self.tenant_manager = TenantManager()
        self.audit_logger = AuthAuditLogger()
        
        # User storage (in production, use a database)
        self._users: Dict[str, User] = {}
        self._users_by_email: Dict[str, User] = {}
        
        # Rate limiting tracking
        self._rate_limit_tracker: Dict[str, List[datetime]] = {}
        
        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
    
    async def initialize(self):
        """Initialize authentication manager."""
        logger.info("Initializing authentication manager")
        
        try:
            # Initialize sub-managers
            await self.session_manager.initialize()
            await self.rbac_manager.initialize()
            await self.tenant_manager.initialize()
            await self.audit_logger.initialize()
            
            # Initialize OAuth providers
            await self.oauth_provider.initialize()
            
            # Initialize SAML providers
            await self.saml_provider.initialize()
            
            # Start background tasks
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            
            logger.info("Authentication manager initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize authentication manager: {e}")
            return False
    
    async def authenticate(
        self,
        credentials: Dict[str, Any],
        auth_type: AuthProvider = AuthProvider.LOCAL,
        tenant_id: Optional[str] = None
    ) -> Tuple[User, Session]:
        """
        Authenticate a user with various providers.
        
        Args:
            credentials: Authentication credentials
            auth_type: Type of authentication provider
            tenant_id: Tenant ID for multi-tenant authentication
            
        Returns:
            Tuple of (User, Session)
        """
        try:
            # Check rate limiting
            if not await self._check_rate_limit(credentials.get('identifier', '')):
                raise AuthenticationError("Rate limit exceeded")
            
            # Validate tenant if provided
            if tenant_id:
                tenant = await self.tenant_manager.get_tenant(tenant_id)
                if not tenant or not tenant.is_active():
                    raise TenantInactiveError("Tenant is not active")
            
            # Authenticate based on provider type
            user = None
            
            if auth_type == AuthProvider.LOCAL:
                user = await self._authenticate_local(credentials, tenant_id)
            elif auth_type == AuthProvider.OAUTH:
                user = await self.oauth_provider.authenticate(credentials, tenant_id)
            elif auth_type == AuthProvider.SAML:
                user = await self.saml_provider.authenticate(credentials, tenant_id)
            elif auth_type == AuthProvider.API_KEY:
                user = await self.api_key_manager.authenticate(credentials.get('api_key'))
            else:
                raise AuthenticationError(f"Unsupported auth provider: {auth_type}")
            
            if not user:
                raise InvalidCredentialsError("Authentication failed")
            
            # Check user status
            if user.status == UserStatus.LOCKED:
                raise AccountLockedError("Account is locked")
            elif user.status != UserStatus.ACTIVE:
                raise AuthenticationError(f"Account is {user.status.value}")
            
            # Create session
            session = await self.session_manager.create_session(
                user_id=user.id,
                tenant_id=user.tenant_id,
                ip_address=credentials.get('ip_address'),
                user_agent=credentials.get('user_agent')
            )
            
            # Update user login info
            user.last_login_at = datetime.utcnow()
            user.last_login_ip = credentials.get('ip_address')
            user.failed_login_attempts = 0
            
            # Log authentication event
            await self.audit_logger.log_authentication(
                user_id=user.id,
                tenant_id=user.tenant_id,
                auth_type=auth_type,
                success=True,
                ip_address=credentials.get('ip_address')
            )
            
            return user, session
            
        except Exception as e:
            # Log failed authentication
            await self.audit_logger.log_authentication(
                user_id=credentials.get('identifier'),
                tenant_id=tenant_id,
                auth_type=auth_type,
                success=False,
                error=str(e),
                ip_address=credentials.get('ip_address')
            )
            raise
    
    async def _authenticate_local(self, credentials: Dict[str, Any], tenant_id: Optional[str]) -> Optional[User]:
        """Authenticate using local credentials."""
        email = credentials.get('email')
        username = credentials.get('username')
        password = credentials.get('password')
        
        if not password:
            raise InvalidCredentialsError("Password is required")
        
        # Find user by email or username
        user = None
        if email:
            user = self._users_by_email.get(email)
        elif username:
            # Find by username (implement username index in production)
            user = next((u for u in self._users.values() if u.username == username), None)
        
        if not user:
            raise InvalidCredentialsError("Invalid credentials")
        
        # Verify tenant
        if tenant_id and user.tenant_id != tenant_id:
            raise InvalidCredentialsError("Invalid credentials")
        
        # Check account lock
        if user.failed_login_attempts >= self.config.max_failed_attempts:
            if user.status != UserStatus.LOCKED:
                user.status = UserStatus.LOCKED
                await self.audit_logger.log_security_event(
                    "account_locked",
                    user_id=user.id,
                    reason="max_failed_attempts"
                )
            raise AccountLockedError("Account is locked due to failed login attempts")
        
        # Verify password
        if not user.verify_password(password):
            user.failed_login_attempts += 1
            raise InvalidCredentialsError("Invalid credentials")
        
        return user
    
    async def validate_session(self, session_token: str) -> Tuple[User, Session]:
        """Validate a session token and return user and session."""
        try:
            # Get session
            session = await self.session_manager.get_session(session_token)
            if not session:
                raise SessionExpiredError("Invalid session")
            
            # Get user
            user = await self.get_user(session.user_id)
            if not user:
                raise AuthenticationError("User not found")
            
            # Validate user status
            if not user.is_active():
                raise AuthenticationError("User account is not active")
            
            # Validate tenant
            if user.tenant_id:
                tenant = await self.tenant_manager.get_tenant(user.tenant_id)
                if not tenant or not tenant.is_active():
                    raise TenantInactiveError("Tenant is not active")
            
            # Refresh session
            await self.session_manager.refresh_session(session_token)
            
            return user, session
            
        except Exception as e:
            logger.error(f"Session validation failed: {e}")
            raise
    
    async def create_jwt_token(self, user: User, session: Session) -> str:
        """Create a JWT token for the user."""
        payload = {
            'user_id': user.id,
            'session_id': session.id,
            'tenant_id': user.tenant_id,
            'roles': [role.name for role in user.roles],
            'exp': datetime.utcnow() + timedelta(hours=self.config.jwt_expiry_hours),
            'iat': datetime.utcnow(),
            'iss': 'voice-agent-platform'
        }
        
        return jwt.encode(payload, self.config.jwt_secret, algorithm=self.config.jwt_algorithm)
    
    async def validate_jwt_token(self, token: str) -> Dict[str, Any]:
        """Validate a JWT token and return the payload."""
        try:
            payload = jwt.decode(
                token,
                self.config.jwt_secret,
                algorithms=[self.config.jwt_algorithm],
                options={"verify_exp": True}
            )
            return payload
        except jwt.ExpiredSignatureError:
            raise SessionExpiredError("Token has expired")
        except jwt.InvalidTokenError as e:
            raise AuthenticationError(f"Invalid token: {e}")
    
    async def logout(self, session_token: str):
        """Logout a user by invalidating their session."""
        session = await self.session_manager.get_session(session_token)
        if session:
            await self.session_manager.invalidate_session(session_token)
            await self.audit_logger.log_authentication(
                user_id=session.user_id,
                tenant_id=session.tenant_id,
                auth_type=AuthProvider.LOCAL,
                event_type="logout",
                success=True
            )
    
    async def create_user(
        self,
        email: str,
        username: str,
        full_name: str,
        password: Optional[str] = None,
        tenant_id: Optional[str] = None,
        roles: Optional[List[str]] = None,
        auth_provider: AuthProvider = AuthProvider.LOCAL
    ) -> User:
        """Create a new user."""
        # Check if user already exists
        if email in self._users_by_email:
            raise ValueError(f"User with email {email} already exists")
        
        # Create user
        user = User(
            email=email,
            username=username,
            full_name=full_name,
            auth_provider=auth_provider,
            tenant_id=tenant_id
        )
        
        # Set password for local auth
        if auth_provider == AuthProvider.LOCAL and password:
            user.set_password(password)
        
        # Assign roles
        if roles:
            for role_name in roles:
                role = await self.rbac_manager.get_role_by_name(role_name)
                if role:
                    user.add_role(role)
        
        # Store user
        self._users[user.id] = user
        self._users_by_email[email] = user
        
        # Log user creation
        await self.audit_logger.log_user_management(
            action="create_user",
            user_id=user.id,
            tenant_id=tenant_id,
            details={"email": email, "roles": roles}
        )
        
        return user
    
    async def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        return self._users.get(user_id)
    
    async def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        return self._users_by_email.get(email)
    
    async def update_user(self, user_id: str, updates: Dict[str, Any]) -> Optional[User]:
        """Update user information."""
        user = self._users.get(user_id)
        if not user:
            return None
        
        # Update allowed fields
        allowed_fields = ['full_name', 'status', 'profile', 'preferences']
        for field in allowed_fields:
            if field in updates:
                setattr(user, field, updates[field])
        
        user.updated_at = datetime.utcnow()
        
        # Log update
        await self.audit_logger.log_user_management(
            action="update_user",
            user_id=user_id,
            details=updates
        )
        
        return user
    
    async def delete_user(self, user_id: str):
        """Delete a user."""
        user = self._users.get(user_id)
        if user:
            # Remove from indices
            del self._users[user_id]
            if user.email in self._users_by_email:
                del self._users_by_email[user.email]
            
            # Invalidate all user sessions
            await self.session_manager.invalidate_user_sessions(user_id)
            
            # Log deletion
            await self.audit_logger.log_user_management(
                action="delete_user",
                user_id=user_id
            )
    
    async def _check_rate_limit(self, identifier: str) -> bool:
        """Check rate limiting for an identifier."""
        if not self.config.rate_limit_enabled:
            return True
        
        now = datetime.utcnow()
        window_start = now - timedelta(minutes=1)
        
        # Get or create request history
        if identifier not in self._rate_limit_tracker:
            self._rate_limit_tracker[identifier] = []
        
        # Remove old requests
        self._rate_limit_tracker[identifier] = [
            ts for ts in self._rate_limit_tracker[identifier]
            if ts > window_start
        ]
        
        # Check limit
        if len(self._rate_limit_tracker[identifier]) >= self.config.rate_limit_requests_per_minute:
            return False
        
        # Add current request
        self._rate_limit_tracker[identifier].append(now)
        return True
    
    async def _cleanup_loop(self):
        """Background task to clean up expired sessions and rate limit data."""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                # Clean up expired sessions
                await self.session_manager.cleanup_expired_sessions()
                
                # Clean up old rate limit data
                cutoff = datetime.utcnow() - timedelta(minutes=5)
                for identifier in list(self._rate_limit_tracker.keys()):
                    self._rate_limit_tracker[identifier] = [
                        ts for ts in self._rate_limit_tracker[identifier]
                        if ts > cutoff
                    ]
                    if not self._rate_limit_tracker[identifier]:
                        del self._rate_limit_tracker[identifier]
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup task error: {e}")
    
    async def cleanup(self):
        """Clean up resources."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        await self.session_manager.cleanup()
        await self.audit_logger.cleanup()