"""
Multi-Tenant Support Manager

Manages tenants, tenant isolation, and tenant-specific configurations.
"""

import asyncio
import logging
from typing import Optional, Dict, Any, List, Set
from datetime import datetime, timedelta
import re

from .models import Tenant, User, Role
from .exceptions import TenantInactiveError, AuthorizationError

logger = logging.getLogger(__name__)


class TenantQuotaManager:
    """Manages tenant quotas and usage tracking."""
    
    def __init__(self):
        # Usage tracking (in production, use a database)
        self._usage_data: Dict[str, Dict[str, Any]] = {}
    
    def get_usage(self, tenant_id: str) -> Dict[str, Any]:
        """Get current usage data for tenant."""
        if tenant_id not in self._usage_data:
            self._usage_data[tenant_id] = {
                'users_count': 0,
                'agents_count': 0,
                'sessions_today': 0,
                'storage_used_gb': 0.0,
                'api_calls_today': 0,
                'last_reset': datetime.utcnow().date()
            }
        
        usage = self._usage_data[tenant_id]
        
        # Reset daily counters if needed
        if usage['last_reset'] != datetime.utcnow().date():
            usage['sessions_today'] = 0
            usage['api_calls_today'] = 0
            usage['last_reset'] = datetime.utcnow().date()
        
        return usage
    
    def check_quota(self, tenant: Tenant, resource: str, additional: int = 1) -> bool:
        """Check if tenant can use additional resources."""
        usage = self.get_usage(tenant.id)
        
        if resource == 'users':
            return usage['users_count'] + additional <= tenant.max_users
        elif resource == 'agents':
            return usage['agents_count'] + additional <= tenant.max_agents
        elif resource == 'sessions':
            return usage['sessions_today'] + additional <= tenant.max_sessions_per_day
        elif resource == 'storage':
            return usage['storage_used_gb'] + additional <= tenant.max_storage_gb
        
        return True
    
    def update_usage(self, tenant_id: str, resource: str, delta: int = 1):
        """Update usage for a resource."""
        usage = self.get_usage(tenant_id)
        
        if resource == 'users':
            usage['users_count'] = max(0, usage['users_count'] + delta)
        elif resource == 'agents':
            usage['agents_count'] = max(0, usage['agents_count'] + delta)
        elif resource == 'sessions':
            usage['sessions_today'] = max(0, usage['sessions_today'] + delta)
        elif resource == 'storage':
            usage['storage_used_gb'] = max(0.0, usage['storage_used_gb'] + delta)
        elif resource == 'api_calls':
            usage['api_calls_today'] = max(0, usage['api_calls_today'] + delta)


class TenantIsolation:
    """Handles tenant data isolation and security."""
    
    @staticmethod
    def validate_tenant_access(user: User, tenant_id: str) -> bool:
        """Validate that user can access tenant resources."""
        # Super admin can access any tenant
        if any(role.name == "Super Administrator" for role in user.roles):
            return True
        
        # User must belong to the tenant
        return user.tenant_id == tenant_id
    
    @staticmethod
    def filter_by_tenant(data: List[Any], user: User, tenant_field: str = 'tenant_id') -> List[Any]:
        """Filter data based on user's tenant access."""
        # Super admin sees all data
        if any(role.name == "Super Administrator" for role in user.roles):
            return data
        
        # Filter by user's tenant
        return [
            item for item in data
            if getattr(item, tenant_field, None) == user.tenant_id
        ]
    
    @staticmethod
    def generate_tenant_slug(name: str) -> str:
        """Generate URL-friendly tenant slug."""
        # Convert to lowercase and replace spaces/special chars
        slug = re.sub(r'[^a-z0-9]+', '-', name.lower().strip())
        slug = slug.strip('-')
        
        # Ensure it's not empty
        if not slug:
            slug = f"tenant-{secrets.token_hex(4)}"
        
        return slug


class TenantManager:
    """Manages multi-tenant functionality."""
    
    def __init__(self):
        # Tenant storage (in production, use a database)
        self._tenants: Dict[str, Tenant] = {}
        self._tenants_by_slug: Dict[str, Tenant] = {}
        
        # Managers
        self.quota_manager = TenantQuotaManager()
        
        # Default tenant for single-tenant mode
        self._default_tenant: Optional[Tenant] = None
    
    async def initialize(self):
        """Initialize tenant manager."""
        logger.info("Initializing tenant manager")
        
        # Create default tenant for single-tenant deployments
        await self._create_default_tenant()
        
        logger.info("Tenant manager initialized")
    
    async def _create_default_tenant(self):
        """Create default tenant for single-tenant mode."""
        default_tenant = Tenant(
            id="default",
            name="Default Organization",
            slug="default",
            status="active",
            max_users=1000,
            max_agents=100,
            max_sessions_per_day=10000,
            max_storage_gb=1000,
            features={
                "voice_agents",
                "session_recording",
                "analytics",
                "api_access",
                "custom_models"
            }
        )
        
        self._tenants[default_tenant.id] = default_tenant
        self._tenants_by_slug[default_tenant.slug] = default_tenant
        self._default_tenant = default_tenant
        
        logger.info("Created default tenant")
    
    async def create_tenant(
        self,
        name: str,
        slug: Optional[str] = None,
        settings: Optional[Dict[str, Any]] = None,
        limits: Optional[Dict[str, Any]] = None,
        features: Optional[Set[str]] = None,
        expires_in_days: Optional[int] = None
    ) -> Tenant:
        """
        Create a new tenant.
        
        Args:
            name: Tenant name
            slug: URL-friendly identifier (auto-generated if not provided)
            settings: Tenant-specific settings
            limits: Resource limits
            features: Enabled features
            expires_in_days: Optional expiry
            
        Returns:
            Created Tenant object
        """
        try:
            # Generate slug if not provided
            if not slug:
                slug = TenantIsolation.generate_tenant_slug(name)
            
            # Ensure slug is unique
            original_slug = slug
            counter = 1
            while slug in self._tenants_by_slug:
                slug = f"{original_slug}-{counter}"
                counter += 1
            
            # Create tenant
            tenant = Tenant(
                name=name,
                slug=slug,
                settings=settings or {},
                features=features or {
                    "voice_agents",
                    "session_recording",
                    "analytics"
                }
            )
            
            # Apply limits
            if limits:
                tenant.max_users = limits.get('max_users', tenant.max_users)
                tenant.max_agents = limits.get('max_agents', tenant.max_agents)
                tenant.max_sessions_per_day = limits.get('max_sessions_per_day', tenant.max_sessions_per_day)
                tenant.max_storage_gb = limits.get('max_storage_gb', tenant.max_storage_gb)
            
            # Set expiry
            if expires_in_days:
                tenant.expires_at = datetime.utcnow() + timedelta(days=expires_in_days)
            
            # Store tenant
            self._tenants[tenant.id] = tenant
            self._tenants_by_slug[slug] = tenant
            
            logger.info(f"Created tenant: {name} ({tenant.id})")
            return tenant
            
        except Exception as e:
            logger.error(f"Failed to create tenant: {e}")
            raise
    
    async def get_tenant(self, tenant_id: str) -> Optional[Tenant]:
        """Get tenant by ID."""
        return self._tenants.get(tenant_id)
    
    async def get_tenant_by_slug(self, slug: str) -> Optional[Tenant]:
        """Get tenant by slug."""
        return self._tenants_by_slug.get(slug)
    
    async def get_default_tenant(self) -> Optional[Tenant]:
        """Get default tenant."""
        return self._default_tenant
    
    async def update_tenant(
        self,
        tenant_id: str,
        updates: Dict[str, Any]
    ) -> Optional[Tenant]:
        """Update tenant information."""
        tenant = self._tenants.get(tenant_id)
        if not tenant:
            return None
        
        # Update allowed fields
        allowed_fields = [
            'name', 'status', 'settings', 'metadata',
            'max_users', 'max_agents', 'max_sessions_per_day', 'max_storage_gb'
        ]
        
        for field in allowed_fields:
            if field in updates:
                setattr(tenant, field, updates[field])
        
        # Handle slug update
        if 'slug' in updates and updates['slug'] != tenant.slug:
            new_slug = updates['slug']
            
            # Check if new slug is available
            if new_slug in self._tenants_by_slug:
                raise ValueError(f"Slug '{new_slug}' is already in use")
            
            # Update slug index
            del self._tenants_by_slug[tenant.slug]
            tenant.slug = new_slug
            self._tenants_by_slug[new_slug] = tenant
        
        # Handle features update
        if 'features' in updates:
            tenant.features = set(updates['features'])
        
        tenant.updated_at = datetime.utcnow()
        
        logger.info(f"Updated tenant: {tenant.name}")
        return tenant
    
    async def delete_tenant(self, tenant_id: str) -> bool:
        """Delete a tenant."""
        tenant = self._tenants.get(tenant_id)
        if not tenant:
            return False
        
        # Cannot delete default tenant
        if tenant_id == "default":
            raise AuthorizationError("Cannot delete default tenant")
        
        # Remove from indices
        del self._tenants[tenant_id]
        del self._tenants_by_slug[tenant.slug]
        
        logger.info(f"Deleted tenant: {tenant.name}")
        return True
    
    async def list_tenants(
        self,
        status: Optional[str] = None,
        include_expired: bool = False
    ) -> List[Tenant]:
        """List tenants with optional filters."""
        tenants = []
        
        for tenant in self._tenants.values():
            # Filter by status
            if status and tenant.status != status:
                continue
            
            # Filter expired
            if not include_expired and not tenant.is_active():
                continue
            
            tenants.append(tenant)
        
        return sorted(tenants, key=lambda t: t.created_at)
    
    async def check_tenant_quota(
        self,
        tenant_id: str,
        resource: str,
        additional: int = 1
    ) -> bool:
        """Check if tenant has quota for additional resources."""
        tenant = await self.get_tenant(tenant_id)
        if not tenant:
            return False
        
        if not tenant.is_active():
            return False
        
        return self.quota_manager.check_quota(tenant, resource, additional)
    
    async def consume_quota(
        self,
        tenant_id: str,
        resource: str,
        amount: int = 1
    ):
        """Consume tenant quota for a resource."""
        if not await self.check_tenant_quota(tenant_id, resource, amount):
            tenant = await self.get_tenant(tenant_id)
            if tenant:
                raise AuthorizationError(f"Tenant '{tenant.name}' quota exceeded for {resource}")
            else:
                raise AuthorizationError(f"Tenant not found or quota exceeded for {resource}")
        
        self.quota_manager.update_usage(tenant_id, resource, amount)
    
    async def release_quota(
        self,
        tenant_id: str,
        resource: str,
        amount: int = 1
    ):
        """Release tenant quota for a resource."""
        self.quota_manager.update_usage(tenant_id, resource, -amount)
    
    async def get_tenant_usage(self, tenant_id: str) -> Dict[str, Any]:
        """Get tenant resource usage."""
        tenant = await self.get_tenant(tenant_id)
        if not tenant:
            return {}
        
        usage = self.quota_manager.get_usage(tenant_id)
        
        return {
            'tenant_id': tenant_id,
            'tenant_name': tenant.name,
            'usage': usage,
            'limits': {
                'max_users': tenant.max_users,
                'max_agents': tenant.max_agents,
                'max_sessions_per_day': tenant.max_sessions_per_day,
                'max_storage_gb': tenant.max_storage_gb
            },
            'utilization': {
                'users': (usage['users_count'] / tenant.max_users * 100) if tenant.max_users > 0 else 0,
                'agents': (usage['agents_count'] / tenant.max_agents * 100) if tenant.max_agents > 0 else 0,
                'sessions': (usage['sessions_today'] / tenant.max_sessions_per_day * 100) if tenant.max_sessions_per_day > 0 else 0,
                'storage': (usage['storage_used_gb'] / tenant.max_storage_gb * 100) if tenant.max_storage_gb > 0 else 0
            }
        }
    
    def validate_tenant_access(self, user: User, tenant_id: str) -> bool:
        """Validate that user can access tenant resources."""
        return TenantIsolation.validate_tenant_access(user, tenant_id)
    
    def require_tenant_access(self, user: User, tenant_id: str):
        """Require user to have access to tenant, raise error if not."""
        if not self.validate_tenant_access(user, tenant_id):
            raise AuthorizationError(f"User does not have access to tenant {tenant_id}")
    
    def filter_by_tenant(self, data: List[Any], user: User, tenant_field: str = 'tenant_id') -> List[Any]:
        """Filter data based on user's tenant access."""
        return TenantIsolation.filter_by_tenant(data, user, tenant_field)
    
    async def setup_tenant_defaults(self, tenant_id: str):
        """Set up default resources for a new tenant."""
        tenant = await self.get_tenant(tenant_id)
        if not tenant:
            return
        
        # This would typically create default roles, configurations, etc.
        # For now, just log the setup
        logger.info(f"Setting up defaults for tenant: {tenant.name}")
    
    async def get_tenant_stats(self) -> Dict[str, Any]:
        """Get overall tenant statistics."""
        stats = {
            'total_tenants': len(self._tenants),
            'active_tenants': sum(1 for t in self._tenants.values() if t.is_active()),
            'expired_tenants': sum(1 for t in self._tenants.values() if not t.is_active()),
            'tenants_by_status': {}
        }
        
        # Count by status
        for tenant in self._tenants.values():
            status = tenant.status
            stats['tenants_by_status'][status] = stats['tenants_by_status'].get(status, 0) + 1
        
        return stats
    
    async def cleanup_expired_tenants(self):
        """Clean up expired tenants (mark as inactive)."""
        now = datetime.utcnow()
        expired_count = 0
        
        for tenant in self._tenants.values():
            if tenant.expires_at and now > tenant.expires_at and tenant.status == "active":
                tenant.status = "expired"
                tenant.updated_at = now
                expired_count += 1
                logger.info(f"Marked tenant as expired: {tenant.name}")
        
        if expired_count > 0:
            logger.info(f"Cleaned up {expired_count} expired tenants")
    
    async def cleanup(self):
        """Clean up resources."""
        await self.cleanup_expired_tenants()


# Tenant context for request processing
class TenantContext:
    """Context for tenant-aware request processing."""
    
    def __init__(self, tenant_id: str, tenant: Tenant):
        self.tenant_id = tenant_id
        self.tenant = tenant
    
    def has_feature(self, feature: str) -> bool:
        """Check if tenant has a specific feature enabled."""
        return self.tenant.has_feature(feature)
    
    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get tenant-specific setting."""
        return self.tenant.settings.get(key, default)


# Middleware for tenant resolution
class TenantMiddleware:
    """Middleware to resolve tenant from request."""
    
    def __init__(self, tenant_manager: TenantManager):
        self.tenant_manager = tenant_manager
    
    async def resolve_tenant(
        self,
        subdomain: Optional[str] = None,
        tenant_header: Optional[str] = None,
        tenant_param: Optional[str] = None,
        user_tenant_id: Optional[str] = None
    ) -> Optional[TenantContext]:
        """
        Resolve tenant from various sources.
        
        Args:
            subdomain: Subdomain from URL
            tenant_header: X-Tenant-ID header
            tenant_param: tenant query parameter
            user_tenant_id: Tenant ID from authenticated user
            
        Returns:
            TenantContext if resolved, None otherwise
        """
        tenant = None
        
        # Try subdomain first
        if subdomain:
            tenant = await self.tenant_manager.get_tenant_by_slug(subdomain)
        
        # Try header
        if not tenant and tenant_header:
            tenant = await self.tenant_manager.get_tenant(tenant_header)
        
        # Try parameter
        if not tenant and tenant_param:
            tenant = await self.tenant_manager.get_tenant(tenant_param)
            if not tenant:
                tenant = await self.tenant_manager.get_tenant_by_slug(tenant_param)
        
        # Try user's tenant
        if not tenant and user_tenant_id:
            tenant = await self.tenant_manager.get_tenant(user_tenant_id)
        
        # Use default tenant as fallback
        if not tenant:
            tenant = await self.tenant_manager.get_default_tenant()
        
        if tenant and tenant.is_active():
            return TenantContext(tenant.id, tenant)
        
        return None


import secrets