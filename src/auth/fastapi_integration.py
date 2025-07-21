"""
FastAPI Integration for Authentication System

Provides FastAPI middleware, dependencies, and endpoints for authentication.
"""

import logging
from typing import Optional, Dict, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException, Depends, Request, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse
import jwt

from .auth_manager import AuthManager, AuthConfig
from .session_manager import SessionManager, SessionMiddleware
from .rbac_manager import RBACManager
from .tenant_manager import TenantManager, TenantMiddleware
from .api_key_manager import ApiKeyManager, ApiKeyAuthMiddleware
from .admin_interface import AdminInterface
from .models import User, Session, PermissionType
from .exceptions import (
    AuthenticationError,
    AuthorizationError,
    SessionExpiredError,
    InvalidTokenError,
    TenantInactiveError
)

logger = logging.getLogger(__name__)


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for authentication."""
    
    def __init__(self, app, auth_manager: AuthManager):
        super().__init__(app)
        self.auth_manager = auth_manager
        self.session_middleware = SessionMiddleware(auth_manager.session_manager)
        self.api_key_middleware = ApiKeyAuthMiddleware(auth_manager.api_key_manager)
        self.tenant_middleware = TenantMiddleware(auth_manager.tenant_manager)
        
        # Paths that don't require authentication
        self.public_paths = {
            "/", "/health", "/docs", "/redoc", "/openapi.json",
            "/auth/login", "/auth/oauth", "/auth/saml", "/auth/register"
        }
    
    async def dispatch(self, request: Request, call_next):
        # Skip authentication for public paths
        if any(request.url.path.startswith(path) for path in self.public_paths):
            return await call_next(request)
        
        # Try to authenticate request
        user = None
        session = None
        tenant_context = None
        
        try:
            # Extract request info
            ip_address = request.client.host if request.client else None
            user_agent = request.headers.get("user-agent")
            
            # Try session authentication first
            session = await self.session_middleware.process_request(
                dict(request.headers),
                dict(request.cookies),
                ip_address,
                user_agent
            )
            
            if session:
                user = await self.auth_manager.get_user(session.user_id)
            
            # Try API key authentication if no session
            if not user:
                user = await self.api_key_middleware.authenticate_request(
                    dict(request.headers),
                    dict(request.query_params),
                    ip_address,
                    request.headers.get("origin")
                )
            
            # Resolve tenant context
            if user:
                tenant_context = await self.tenant_middleware.resolve_tenant(
                    tenant_header=request.headers.get("x-tenant-id"),
                    tenant_param=request.query_params.get("tenant"),
                    user_tenant_id=user.tenant_id
                )
            
            # Add to request state
            request.state.user = user
            request.state.session = session
            request.state.tenant_context = tenant_context
            
        except Exception as e:
            logger.debug(f"Authentication failed: {e}")
        
        # Continue with request
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        
        return response


# FastAPI dependencies
security = HTTPBearer(auto_error=False)


async def get_current_user(request: Request) -> Optional[User]:
    """Get current authenticated user."""
    return getattr(request.state, 'user', None)


async def require_authentication(request: Request) -> User:
    """Require user to be authenticated."""
    user = await get_current_user(request)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"}
        )
    return user


async def get_current_session(request: Request) -> Optional[Session]:
    """Get current session."""
    return getattr(request.state, 'session', None)


async def get_tenant_context(request: Request):
    """Get tenant context."""
    return getattr(request.state, 'tenant_context', None)


def require_permission(permission: PermissionType, resource_param: Optional[str] = None):
    """Dependency factory to require specific permission."""
    async def permission_dependency(
        user: User = Depends(require_authentication),
        request: Request = None,
        rbac_manager: RBACManager = Depends(lambda: get_rbac_manager())
    ):
        # Get resource ID from request if specified
        resource_id = None
        if resource_param and request:
            resource_id = request.path_params.get(resource_param)
        
        # Check permission
        if not await rbac_manager.check_permission(user, permission, resource_id):
            raise HTTPException(
                status_code=403,
                detail=f"Permission denied: {permission.value}"
            )
        
        return user
    
    return permission_dependency


def require_tenant_access(tenant_param: str = "tenant_id"):
    """Dependency factory to require tenant access."""
    async def tenant_dependency(
        user: User = Depends(require_authentication),
        request: Request = None,
        tenant_manager: TenantManager = Depends(lambda: get_tenant_manager())
    ):
        if not request:
            raise HTTPException(status_code=400, detail="Invalid request")
        
        tenant_id = request.path_params.get(tenant_param)
        if not tenant_id:
            tenant_id = user.tenant_id
        
        if not tenant_manager.validate_tenant_access(user, tenant_id):
            raise HTTPException(
                status_code=403,
                detail="Access denied to tenant resources"
            )
        
        return user
    
    return tenant_dependency


# Global managers (initialize these at startup)
_auth_manager: Optional[AuthManager] = None
_rbac_manager: Optional[RBACManager] = None
_tenant_manager: Optional[TenantManager] = None
_admin_interface: Optional[AdminInterface] = None


def get_auth_manager() -> AuthManager:
    if _auth_manager is None:
        raise RuntimeError("Authentication manager not initialized")
    return _auth_manager


def get_rbac_manager() -> RBACManager:
    if _rbac_manager is None:
        raise RuntimeError("RBAC manager not initialized")
    return _rbac_manager


def get_tenant_manager() -> TenantManager:
    if _tenant_manager is None:
        raise RuntimeError("Tenant manager not initialized")
    return _tenant_manager


def get_admin_interface() -> AdminInterface:
    if _admin_interface is None:
        raise RuntimeError("Admin interface not initialized")
    return _admin_interface


async def initialize_auth_system():
    """Initialize the authentication system."""
    global _auth_manager, _rbac_manager, _tenant_manager, _admin_interface
    
    # Create configuration
    config = AuthConfig()
    
    # Initialize managers
    _auth_manager = AuthManager(config)
    _rbac_manager = RBACManager()
    _tenant_manager = TenantManager()
    
    # Initialize all managers
    await _auth_manager.initialize()
    await _rbac_manager.initialize()
    await _tenant_manager.initialize()
    
    # Create admin interface
    _admin_interface = AdminInterface(
        _auth_manager,
        _rbac_manager,
        _tenant_manager,
        _auth_manager.api_key_manager
    )
    
    logger.info("Authentication system initialized")


def setup_auth_routes(app: FastAPI):
    """Set up authentication routes."""
    
    @app.post("/auth/login")
    async def login(credentials: Dict[str, Any]):
        """Authenticate user with credentials."""
        try:
            auth_manager = get_auth_manager()
            
            # Add request metadata
            credentials.update({
                'ip_address': '127.0.0.1',  # Would get from request
                'user_agent': 'FastAPI Client'
            })
            
            user, session = await auth_manager.authenticate(credentials)
            
            # Create JWT token
            jwt_token = await auth_manager.create_jwt_token(user, session)
            
            return {
                'access_token': jwt_token,
                'token_type': 'bearer',
                'expires_in': auth_manager.config.jwt_expiry_hours * 3600,
                'user': {
                    'id': user.id,
                    'email': user.email,
                    'full_name': user.full_name,
                    'tenant_id': user.tenant_id,
                    'roles': [role.name for role in user.roles]
                }
            }
            
        except Exception as e:
            raise HTTPException(status_code=401, detail=str(e))
    
    @app.post("/auth/logout")
    async def logout(session: Session = Depends(get_current_session)):
        """Logout user."""
        if session:
            auth_manager = get_auth_manager()
            await auth_manager.logout(session.token)
        
        return {'message': 'Logged out successfully'}
    
    @app.get("/auth/me")
    async def get_current_user_info(user: User = Depends(require_authentication)):
        """Get current user information."""
        rbac_manager = get_rbac_manager()
        permissions = await rbac_manager.get_user_permissions(user)
        
        return {
            'user': {
                'id': user.id,
                'email': user.email,
                'username': user.username,
                'full_name': user.full_name,
                'status': user.status.value,
                'tenant_id': user.tenant_id,
                'roles': [
                    {
                        'id': role.id,
                        'name': role.name,
                        'description': role.description
                    }
                    for role in user.roles
                ],
                'permissions': permissions,
                'last_login_at': user.last_login_at.isoformat() if user.last_login_at else None
            }
        }
    
    @app.get("/auth/oauth/{provider_id}/login")
    async def oauth_login(provider_id: str, redirect_uri: str, tenant_id: Optional[str] = None):
        """Initiate OAuth login."""
        try:
            auth_manager = get_auth_manager()
            oauth_provider = auth_manager.oauth_provider
            
            auth_url = oauth_provider.get_authorization_url(
                provider_id,
                redirect_uri,
                tenant_id
            )
            
            return {'authorization_url': auth_url}
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    @app.post("/auth/oauth/{provider_id}/callback")
    async def oauth_callback(provider_id: str, callback_data: Dict[str, Any]):
        """Handle OAuth callback."""
        try:
            auth_manager = get_auth_manager()
            
            # Add provider ID to callback data
            callback_data['provider_id'] = provider_id
            
            user, session = await auth_manager.authenticate(
                callback_data,
                auth_type=auth_manager.models.AuthProvider.OAUTH
            )
            
            # Create JWT token
            jwt_token = await auth_manager.create_jwt_token(user, session)
            
            return {
                'access_token': jwt_token,
                'token_type': 'bearer',
                'user': {
                    'id': user.id,
                    'email': user.email,
                    'full_name': user.full_name,
                    'tenant_id': user.tenant_id
                }
            }
            
        except Exception as e:
            raise HTTPException(status_code=401, detail=str(e))


def setup_admin_routes(app: FastAPI):
    """Set up admin interface routes."""
    
    @app.get("/admin/users")
    async def list_users(
        user: User = Depends(require_permission(PermissionType.READ_USER)),
        tenant_id: Optional[str] = None,
        page: int = 1,
        page_size: int = 50
    ):
        """List users."""
        admin = get_admin_interface()
        return await admin.list_users(user, tenant_id, page=page, page_size=page_size)
    
    @app.get("/admin/users/{user_id}")
    async def get_user(
        user_id: str,
        user: User = Depends(require_permission(PermissionType.READ_USER))
    ):
        """Get user details."""
        admin = get_admin_interface()
        return await admin.get_user_details(user, user_id)
    
    @app.post("/admin/users")
    async def create_user(
        user_data: Dict[str, Any],
        user: User = Depends(require_permission(PermissionType.CREATE_USER))
    ):
        """Create a new user."""
        admin = get_admin_interface()
        return await admin.create_user(user, user_data)
    
    @app.put("/admin/users/{user_id}")
    async def update_user(
        user_id: str,
        updates: Dict[str, Any],
        user: User = Depends(require_permission(PermissionType.UPDATE_USER))
    ):
        """Update user."""
        admin = get_admin_interface()
        return await admin.update_user(user, user_id, updates)
    
    @app.delete("/admin/users/{user_id}")
    async def delete_user(
        user_id: str,
        user: User = Depends(require_permission(PermissionType.DELETE_USER))
    ):
        """Delete user."""
        admin = get_admin_interface()
        return await admin.delete_user(user, user_id)
    
    @app.get("/admin/roles")
    async def list_roles(
        user: User = Depends(require_permission(PermissionType.MANAGE_ROLES))
    ):
        """List roles."""
        admin = get_admin_interface()
        return await admin.list_roles(user)
    
    @app.post("/admin/roles")
    async def create_role(
        role_data: Dict[str, Any],
        user: User = Depends(require_permission(PermissionType.MANAGE_ROLES))
    ):
        """Create a new role."""
        admin = get_admin_interface()
        return await admin.create_role(user, role_data)
    
    @app.get("/admin/tenants")
    async def list_tenants(
        user: User = Depends(require_permission(PermissionType.VIEW_TENANT))
    ):
        """List tenants."""
        admin = get_admin_interface()
        return await admin.list_tenants(user)
    
    @app.get("/admin/api-keys")
    async def list_api_keys(
        user: User = Depends(require_authentication)
    ):
        """List API keys."""
        admin = get_admin_interface()
        return await admin.list_api_keys(user)
    
    @app.post("/admin/api-keys")
    async def create_api_key(
        key_data: Dict[str, Any],
        user: User = Depends(require_permission(PermissionType.CREATE_API_KEY))
    ):
        """Create a new API key."""
        admin = get_admin_interface()
        return await admin.create_api_key(user, key_data)
    
    @app.delete("/admin/api-keys/{key_id}")
    async def revoke_api_key(
        key_id: str,
        user: User = Depends(require_authentication)
    ):
        """Revoke API key."""
        admin = get_admin_interface()
        return await admin.revoke_api_key(user, key_id)
    
    @app.get("/admin/stats")
    async def get_system_stats(
        user: User = Depends(require_permission(PermissionType.SYSTEM_ADMIN))
    ):
        """Get system statistics."""
        admin = get_admin_interface()
        return await admin.get_system_stats(user)


async def cleanup_auth_system():
    """Clean up authentication system resources."""
    global _auth_manager, _rbac_manager, _tenant_manager
    
    if _auth_manager:
        await _auth_manager.cleanup()
    if _rbac_manager:
        await _rbac_manager.cleanup()
    if _tenant_manager:
        await _tenant_manager.cleanup()


def create_authenticated_app() -> FastAPI:
    """Create FastAPI app with authentication enabled."""
    app = FastAPI(
        title="Voice Agent Platform",
        description="Enterprise Voice Agent Platform with Authentication",
        version="1.0.0"
    )
    
    # Initialize auth system on startup
    @app.on_event("startup")
    async def startup():
        await initialize_auth_system()
    
    @app.on_event("shutdown")
    async def shutdown():
        await cleanup_auth_system()
    
    # Add authentication middleware
    app.add_middleware(AuthenticationMiddleware, auth_manager=lambda: get_auth_manager())
    
    # Set up routes
    setup_auth_routes(app)
    setup_admin_routes(app)
    
    return app