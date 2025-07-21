"""
Enterprise Authentication and Authorization System

Provides comprehensive authentication infrastructure for voice agent deployments
including OAuth/SAML SSO, RBAC, API key management, and multi-tenant support.
"""

from .models import User, Role, Permission, Tenant, ApiKey
from .auth_manager import AuthManager
from .session_manager import SessionManager
from .rbac_manager import RBACManager
from .oauth_provider import OAuthProvider
from .saml_provider import SAMLProvider
from .api_key_manager import ApiKeyManager
from .tenant_manager import TenantManager
from .audit_logger import AuthAuditLogger

__all__ = [
    'User',
    'Role',
    'Permission',
    'Tenant',
    'ApiKey',
    'AuthManager',
    'SessionManager',
    'RBACManager',
    'OAuthProvider',
    'SAMLProvider',
    'ApiKeyManager',
    'TenantManager',
    'AuthAuditLogger'
]

__version__ = "1.0.0"