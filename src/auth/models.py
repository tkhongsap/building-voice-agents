"""
Authentication and Authorization Models

Defines core data models for the enterprise authentication system.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any, Set
import uuid
import hashlib
import secrets


class AuthProvider(Enum):
    """Authentication provider types."""
    LOCAL = "local"
    OAUTH = "oauth"
    SAML = "saml"
    API_KEY = "api_key"
    LDAP = "ldap"
    CUSTOM = "custom"


class UserStatus(Enum):
    """User account status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING = "pending"
    LOCKED = "locked"


class PermissionType(Enum):
    """Permission types for RBAC."""
    # Voice Agent Operations
    CREATE_AGENT = "voice_agent.create"
    READ_AGENT = "voice_agent.read"
    UPDATE_AGENT = "voice_agent.update"
    DELETE_AGENT = "voice_agent.delete"
    EXECUTE_AGENT = "voice_agent.execute"
    
    # Session Management
    CREATE_SESSION = "session.create"
    READ_SESSION = "session.read"
    UPDATE_SESSION = "session.update"
    DELETE_SESSION = "session.delete"
    MANAGE_ALL_SESSIONS = "session.manage_all"
    
    # Recording and Monitoring
    ACCESS_RECORDINGS = "recording.access"
    DELETE_RECORDINGS = "recording.delete"
    EXPORT_RECORDINGS = "recording.export"
    VIEW_ANALYTICS = "analytics.view"
    
    # Configuration Management
    MANAGE_CONFIG = "config.manage"
    VIEW_CONFIG = "config.view"
    
    # User Management
    CREATE_USER = "user.create"
    READ_USER = "user.read"
    UPDATE_USER = "user.update"
    DELETE_USER = "user.delete"
    MANAGE_ROLES = "user.manage_roles"
    
    # API Key Management
    CREATE_API_KEY = "api_key.create"
    REVOKE_API_KEY = "api_key.revoke"
    MANAGE_ALL_API_KEYS = "api_key.manage_all"
    
    # Tenant Management
    MANAGE_TENANT = "tenant.manage"
    VIEW_TENANT = "tenant.view"
    
    # System Administration
    SYSTEM_ADMIN = "system.admin"
    VIEW_AUDIT_LOGS = "audit.view"
    MANAGE_SECURITY = "security.manage"


@dataclass
class Permission:
    """Represents a permission in the system."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: PermissionType = PermissionType.READ_AGENT
    resource: Optional[str] = None  # Specific resource ID if applicable
    conditions: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def matches(self, requested_permission: str, resource_id: Optional[str] = None) -> bool:
        """Check if this permission matches the requested permission."""
        if self.type.value != requested_permission:
            return False
        
        # If permission is for a specific resource, check if it matches
        if self.resource and resource_id and self.resource != resource_id:
            return False
        
        # Check conditions (e.g., time-based, IP-based restrictions)
        # This can be extended based on requirements
        return True


@dataclass
class Role:
    """Represents a role in the RBAC system."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    permissions: List[Permission] = field(default_factory=list)
    is_system_role: bool = False  # System roles cannot be modified
    tenant_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def has_permission(self, permission_type: PermissionType, resource_id: Optional[str] = None) -> bool:
        """Check if role has a specific permission."""
        return any(p.matches(permission_type.value, resource_id) for p in self.permissions)
    
    def add_permission(self, permission: Permission):
        """Add a permission to the role."""
        if not self.is_system_role:
            self.permissions.append(permission)
            self.updated_at = datetime.utcnow()
    
    def remove_permission(self, permission_id: str):
        """Remove a permission from the role."""
        if not self.is_system_role:
            self.permissions = [p for p in self.permissions if p.id != permission_id]
            self.updated_at = datetime.utcnow()


@dataclass
class Tenant:
    """Represents a tenant in the multi-tenant system."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    slug: str = ""  # URL-friendly identifier
    status: str = "active"
    settings: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Limits and quotas
    max_users: int = 100
    max_agents: int = 10
    max_sessions_per_day: int = 1000
    max_storage_gb: int = 100
    
    # Feature flags
    features: Set[str] = field(default_factory=set)
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    
    def is_active(self) -> bool:
        """Check if tenant is active."""
        if self.status != "active":
            return False
        if self.expires_at and datetime.utcnow() > self.expires_at:
            return False
        return True
    
    def has_feature(self, feature: str) -> bool:
        """Check if tenant has access to a feature."""
        return feature in self.features


@dataclass
class User:
    """Represents a user in the system."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    email: str = ""
    username: str = ""
    full_name: str = ""
    status: UserStatus = UserStatus.ACTIVE
    
    # Authentication
    auth_provider: AuthProvider = AuthProvider.LOCAL
    auth_provider_id: Optional[str] = None  # External ID for OAuth/SAML
    password_hash: Optional[str] = None  # For local auth
    
    # Multi-tenant support
    tenant_id: Optional[str] = None
    
    # RBAC
    roles: List[Role] = field(default_factory=list)
    direct_permissions: List[Permission] = field(default_factory=list)
    
    # Security
    mfa_enabled: bool = False
    mfa_secret: Optional[str] = None
    failed_login_attempts: int = 0
    last_login_at: Optional[datetime] = None
    last_login_ip: Optional[str] = None
    
    # Profile
    profile: Dict[str, Any] = field(default_factory=dict)
    preferences: Dict[str, Any] = field(default_factory=dict)
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    password_changed_at: Optional[datetime] = None
    
    def set_password(self, password: str):
        """Set user password (hashed)."""
        # Use a strong hashing algorithm (bcrypt/scrypt/argon2)
        # This is a simplified example
        salt = secrets.token_hex(32)
        self.password_hash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000
        ).hex() + ':' + salt
        self.password_changed_at = datetime.utcnow()
    
    def verify_password(self, password: str) -> bool:
        """Verify user password."""
        if not self.password_hash:
            return False
        
        stored_hash, salt = self.password_hash.split(':')
        password_hash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000
        ).hex()
        
        return password_hash == stored_hash
    
    def has_permission(self, permission_type: PermissionType, resource_id: Optional[str] = None) -> bool:
        """Check if user has a specific permission."""
        # Check direct permissions
        for perm in self.direct_permissions:
            if perm.matches(permission_type.value, resource_id):
                return True
        
        # Check role permissions
        for role in self.roles:
            if role.has_permission(permission_type, resource_id):
                return True
        
        return False
    
    def add_role(self, role: Role):
        """Add a role to the user."""
        if role not in self.roles:
            self.roles.append(role)
            self.updated_at = datetime.utcnow()
    
    def remove_role(self, role_id: str):
        """Remove a role from the user."""
        self.roles = [r for r in self.roles if r.id != role_id]
        self.updated_at = datetime.utcnow()
    
    def is_active(self) -> bool:
        """Check if user account is active."""
        return self.status == UserStatus.ACTIVE


@dataclass
class ApiKey:
    """Represents an API key for authentication."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    key: str = field(default_factory=lambda: secrets.token_urlsafe(32))
    name: str = ""
    description: str = ""
    
    # Owner
    user_id: Optional[str] = None
    tenant_id: Optional[str] = None
    
    # Permissions and scopes
    scopes: List[str] = field(default_factory=list)
    permissions: List[Permission] = field(default_factory=list)
    
    # Restrictions
    allowed_ips: List[str] = field(default_factory=list)
    allowed_origins: List[str] = field(default_factory=list)
    rate_limit: Optional[int] = None  # Requests per minute
    
    # Status
    is_active: bool = True
    last_used_at: Optional[datetime] = None
    usage_count: int = 0
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    revoked_at: Optional[datetime] = None
    
    def is_valid(self) -> bool:
        """Check if API key is valid."""
        if not self.is_active:
            return False
        if self.revoked_at:
            return False
        if self.expires_at and datetime.utcnow() > self.expires_at:
            return False
        return True
    
    def has_scope(self, scope: str) -> bool:
        """Check if API key has a specific scope."""
        return scope in self.scopes
    
    def has_permission(self, permission_type: PermissionType, resource_id: Optional[str] = None) -> bool:
        """Check if API key has a specific permission."""
        for perm in self.permissions:
            if perm.matches(permission_type.value, resource_id):
                return True
        return False
    
    def record_usage(self, ip_address: Optional[str] = None):
        """Record API key usage."""
        self.last_used_at = datetime.utcnow()
        self.usage_count += 1


@dataclass
class Session:
    """Represents an authenticated session."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    token: str = field(default_factory=lambda: secrets.token_urlsafe(64))
    
    # User and tenant
    user_id: str = ""
    tenant_id: Optional[str] = None
    
    # Session details
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    device_id: Optional[str] = None
    
    # Security
    is_mfa_verified: bool = False
    security_level: str = "standard"  # standard, elevated, privileged
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_activity_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: datetime = field(default_factory=lambda: datetime.utcnow() + timedelta(hours=24))
    
    # Session data
    data: Dict[str, Any] = field(default_factory=dict)
    
    def is_valid(self) -> bool:
        """Check if session is still valid."""
        return datetime.utcnow() < self.expires_at
    
    def refresh(self, extend_by_hours: int = 1):
        """Refresh session expiry."""
        self.last_activity_at = datetime.utcnow()
        self.expires_at = datetime.utcnow() + timedelta(hours=extend_by_hours)


# Predefined system roles
SYSTEM_ROLES = {
    "super_admin": Role(
        id="role_super_admin",
        name="Super Administrator",
        description="Full system access",
        permissions=[Permission(type=PermissionType.SYSTEM_ADMIN)],
        is_system_role=True
    ),
    "tenant_admin": Role(
        id="role_tenant_admin",
        name="Tenant Administrator",
        description="Full access within tenant",
        permissions=[
            Permission(type=PermissionType.MANAGE_TENANT),
            Permission(type=PermissionType.CREATE_USER),
            Permission(type=PermissionType.UPDATE_USER),
            Permission(type=PermissionType.DELETE_USER),
            Permission(type=PermissionType.MANAGE_ROLES),
            Permission(type=PermissionType.MANAGE_CONFIG),
            Permission(type=PermissionType.VIEW_AUDIT_LOGS)
        ],
        is_system_role=True
    ),
    "agent_manager": Role(
        id="role_agent_manager",
        name="Voice Agent Manager",
        description="Manage voice agents and configurations",
        permissions=[
            Permission(type=PermissionType.CREATE_AGENT),
            Permission(type=PermissionType.READ_AGENT),
            Permission(type=PermissionType.UPDATE_AGENT),
            Permission(type=PermissionType.DELETE_AGENT),
            Permission(type=PermissionType.MANAGE_CONFIG)
        ],
        is_system_role=True
    ),
    "agent_operator": Role(
        id="role_agent_operator",
        name="Voice Agent Operator",
        description="Execute voice agents and manage sessions",
        permissions=[
            Permission(type=PermissionType.READ_AGENT),
            Permission(type=PermissionType.EXECUTE_AGENT),
            Permission(type=PermissionType.CREATE_SESSION),
            Permission(type=PermissionType.READ_SESSION),
            Permission(type=PermissionType.UPDATE_SESSION)
        ],
        is_system_role=True
    ),
    "analyst": Role(
        id="role_analyst",
        name="Analyst",
        description="View analytics and recordings",
        permissions=[
            Permission(type=PermissionType.VIEW_ANALYTICS),
            Permission(type=PermissionType.ACCESS_RECORDINGS),
            Permission(type=PermissionType.READ_SESSION)
        ],
        is_system_role=True
    ),
    "viewer": Role(
        id="role_viewer",
        name="Viewer",
        description="Read-only access",
        permissions=[
            Permission(type=PermissionType.READ_AGENT),
            Permission(type=PermissionType.READ_SESSION),
            Permission(type=PermissionType.VIEW_CONFIG)
        ],
        is_system_role=True
    )
}


from datetime import timedelta