"""
Administrative Interface for User and Role Management

Provides web-based administration interface for managing users, roles, and authentication.
"""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
from dataclasses import asdict

from .auth_manager import AuthManager
from .rbac_manager import RBACManager
from .tenant_manager import TenantManager
from .api_key_manager import ApiKeyManager
from .models import User, Role, Tenant, Permission, PermissionType, UserStatus, AuthProvider
from .exceptions import AuthorizationError, PermissionDeniedError

logger = logging.getLogger(__name__)


class AdminInterface:
    """Administrative interface for authentication system management."""
    
    def __init__(
        self,
        auth_manager: AuthManager,
        rbac_manager: RBACManager,
        tenant_manager: TenantManager,
        api_key_manager: ApiKeyManager
    ):
        self.auth_manager = auth_manager
        self.rbac_manager = rbac_manager
        self.tenant_manager = tenant_manager
        self.api_key_manager = api_key_manager
    
    # User Management Methods
    
    async def list_users(
        self,
        requesting_user: User,
        tenant_id: Optional[str] = None,
        status: Optional[UserStatus] = None,
        page: int = 1,
        page_size: int = 50
    ) -> Dict[str, Any]:
        """List users with pagination and filtering."""
        # Check permissions
        await self.rbac_manager.require_permission(
            requesting_user,
            PermissionType.READ_USER,
            context={'tenant_id': tenant_id}
        )
        
        # Get all users (in production, this would be a database query)
        all_users = list(self.auth_manager._users.values())
        
        # Filter by tenant access
        if not any(role.name == "Super Administrator" for role in requesting_user.roles):
            all_users = [u for u in all_users if u.tenant_id == requesting_user.tenant_id]
        elif tenant_id:
            all_users = [u for u in all_users if u.tenant_id == tenant_id]
        
        # Filter by status
        if status:
            all_users = [u for u in all_users if u.status == status]
        
        # Sort by creation date
        all_users.sort(key=lambda u: u.created_at, reverse=True)
        
        # Paginate
        total = len(all_users)
        start = (page - 1) * page_size
        end = start + page_size
        users = all_users[start:end]
        
        # Convert to dict format (remove sensitive fields)
        user_data = []
        for user in users:
            user_dict = {
                'id': user.id,
                'email': user.email,
                'username': user.username,
                'full_name': user.full_name,
                'status': user.status.value,
                'auth_provider': user.auth_provider.value,
                'tenant_id': user.tenant_id,
                'roles': [{'id': r.id, 'name': r.name} for r in user.roles],
                'created_at': user.created_at.isoformat(),
                'last_login_at': user.last_login_at.isoformat() if user.last_login_at else None
            }
            user_data.append(user_dict)
        
        return {
            'users': user_data,
            'pagination': {
                'page': page,
                'page_size': page_size,
                'total': total,
                'total_pages': (total + page_size - 1) // page_size
            }
        }
    
    async def get_user_details(
        self,
        requesting_user: User,
        user_id: str
    ) -> Dict[str, Any]:
        """Get detailed user information."""
        # Check permissions
        await self.rbac_manager.require_permission(
            requesting_user,
            PermissionType.READ_USER
        )
        
        user = await self.auth_manager.get_user(user_id)
        if not user:
            raise ValueError("User not found")
        
        # Check tenant access
        if not self.tenant_manager.validate_tenant_access(requesting_user, user.tenant_id):
            raise PermissionDeniedError("Access denied to user from different tenant")
        
        # Get user sessions
        sessions = await self.auth_manager.session_manager.get_user_sessions(user_id)
        
        # Get user permissions
        permissions = await self.rbac_manager.get_user_permissions(user)
        
        return {
            'user': {
                'id': user.id,
                'email': user.email,
                'username': user.username,
                'full_name': user.full_name,
                'status': user.status.value,
                'auth_provider': user.auth_provider.value,
                'tenant_id': user.tenant_id,
                'mfa_enabled': user.mfa_enabled,
                'failed_login_attempts': user.failed_login_attempts,
                'created_at': user.created_at.isoformat(),
                'updated_at': user.updated_at.isoformat(),
                'last_login_at': user.last_login_at.isoformat() if user.last_login_at else None,
                'password_changed_at': user.password_changed_at.isoformat() if user.password_changed_at else None
            },
            'roles': [
                {
                    'id': role.id,
                    'name': role.name,
                    'description': role.description,
                    'is_system_role': role.is_system_role
                }
                for role in user.roles
            ],
            'permissions': permissions,
            'active_sessions': len([s for s in sessions if s.is_valid()]),
            'total_sessions': len(sessions)
        }
    
    async def create_user(
        self,
        requesting_user: User,
        user_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a new user."""
        # Check permissions
        await self.rbac_manager.require_permission(
            requesting_user,
            PermissionType.CREATE_USER
        )
        
        # Validate tenant access
        tenant_id = user_data.get('tenant_id', requesting_user.tenant_id)
        if not self.tenant_manager.validate_tenant_access(requesting_user, tenant_id):
            raise PermissionDeniedError("Cannot create user in different tenant")
        
        # Check tenant quota
        await self.tenant_manager.consume_quota(tenant_id, 'users')
        
        try:
            # Create user
            user = await self.auth_manager.create_user(
                email=user_data['email'],
                username=user_data.get('username', user_data['email']),
                full_name=user_data.get('full_name', ''),
                password=user_data.get('password'),
                tenant_id=tenant_id,
                roles=user_data.get('roles', []),
                auth_provider=AuthProvider(user_data.get('auth_provider', 'local'))
            )
            
            return {
                'user': {
                    'id': user.id,
                    'email': user.email,
                    'username': user.username,
                    'full_name': user.full_name,
                    'status': user.status.value,
                    'tenant_id': user.tenant_id,
                    'created_at': user.created_at.isoformat()
                }
            }
        except Exception as e:
            # Release quota on error
            await self.tenant_manager.release_quota(tenant_id, 'users')
            raise
    
    async def update_user(
        self,
        requesting_user: User,
        user_id: str,
        updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update user information."""
        # Check permissions
        await self.rbac_manager.require_permission(
            requesting_user,
            PermissionType.UPDATE_USER
        )
        
        user = await self.auth_manager.get_user(user_id)
        if not user:
            raise ValueError("User not found")
        
        # Check tenant access
        if not self.tenant_manager.validate_tenant_access(requesting_user, user.tenant_id):
            raise PermissionDeniedError("Cannot update user from different tenant")
        
        # Update user
        updated_user = await self.auth_manager.update_user(user_id, updates)
        
        # Handle role updates
        if 'roles' in updates:
            # Clear existing roles
            user.roles = []
            
            # Add new roles
            for role_name in updates['roles']:
                role = await self.rbac_manager.get_role_by_name(role_name)
                if role:
                    await self.rbac_manager.assign_role_to_user(user, role.id)
        
        return {
            'user': {
                'id': updated_user.id,
                'email': updated_user.email,
                'username': updated_user.username,
                'full_name': updated_user.full_name,
                'status': updated_user.status.value,
                'updated_at': updated_user.updated_at.isoformat()
            }
        }
    
    async def delete_user(
        self,
        requesting_user: User,
        user_id: str
    ) -> Dict[str, str]:
        """Delete a user."""
        # Check permissions
        await self.rbac_manager.require_permission(
            requesting_user,
            PermissionType.DELETE_USER
        )
        
        user = await self.auth_manager.get_user(user_id)
        if not user:
            raise ValueError("User not found")
        
        # Check tenant access
        if not self.tenant_manager.validate_tenant_access(requesting_user, user.tenant_id):
            raise PermissionDeniedError("Cannot delete user from different tenant")
        
        # Cannot delete self
        if user_id == requesting_user.id:
            raise AuthorizationError("Cannot delete your own account")
        
        # Delete user
        await self.auth_manager.delete_user(user_id)
        
        # Release quota
        if user.tenant_id:
            await self.tenant_manager.release_quota(user.tenant_id, 'users')
        
        return {'message': 'User deleted successfully'}
    
    # Role Management Methods
    
    async def list_roles(
        self,
        requesting_user: User,
        tenant_id: Optional[str] = None,
        include_system_roles: bool = True
    ) -> Dict[str, Any]:
        """List roles."""
        # Check permissions
        await self.rbac_manager.require_permission(
            requesting_user,
            PermissionType.MANAGE_ROLES
        )
        
        # Get roles
        roles = await self.rbac_manager.list_roles(
            tenant_id=tenant_id or requesting_user.tenant_id,
            include_system_roles=include_system_roles
        )
        
        role_data = []
        for role in roles:
            role_dict = {
                'id': role.id,
                'name': role.name,
                'description': role.description,
                'is_system_role': role.is_system_role,
                'tenant_id': role.tenant_id,
                'permissions': [p.type.value for p in role.permissions],
                'created_at': role.created_at.isoformat(),
                'updated_at': role.updated_at.isoformat()
            }
            role_data.append(role_dict)
        
        return {'roles': role_data}
    
    async def create_role(
        self,
        requesting_user: User,
        role_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a new role."""
        # Check permissions
        await self.rbac_manager.require_permission(
            requesting_user,
            PermissionType.MANAGE_ROLES
        )
        
        # Convert permission strings to PermissionType
        permissions = []
        for perm_str in role_data.get('permissions', []):
            try:
                permissions.append(PermissionType(perm_str))
            except ValueError:
                logger.warning(f"Invalid permission type: {perm_str}")
        
        # Create role
        role = await self.rbac_manager.create_role(
            name=role_data['name'],
            description=role_data.get('description', ''),
            permissions=permissions,
            tenant_id=role_data.get('tenant_id', requesting_user.tenant_id)
        )
        
        return {
            'role': {
                'id': role.id,
                'name': role.name,
                'description': role.description,
                'permissions': [p.type.value for p in role.permissions],
                'created_at': role.created_at.isoformat()
            }
        }
    
    async def update_role(
        self,
        requesting_user: User,
        role_id: str,
        updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update role information."""
        # Check permissions
        await self.rbac_manager.require_permission(
            requesting_user,
            PermissionType.MANAGE_ROLES
        )
        
        # Convert permission strings if provided
        if 'permissions' in updates:
            permissions = []
            for perm_str in updates['permissions']:
                try:
                    permissions.append(PermissionType(perm_str))
                except ValueError:
                    logger.warning(f"Invalid permission type: {perm_str}")
            updates['permissions'] = permissions
        
        # Update role
        role = await self.rbac_manager.update_role(role_id, updates)
        if not role:
            raise ValueError("Role not found")
        
        return {
            'role': {
                'id': role.id,
                'name': role.name,
                'description': role.description,
                'permissions': [p.type.value for p in role.permissions],
                'updated_at': role.updated_at.isoformat()
            }
        }
    
    async def delete_role(
        self,
        requesting_user: User,
        role_id: str
    ) -> Dict[str, str]:
        """Delete a role."""
        # Check permissions
        await self.rbac_manager.require_permission(
            requesting_user,
            PermissionType.MANAGE_ROLES
        )
        
        # Delete role
        success = await self.rbac_manager.delete_role(role_id)
        if not success:
            raise ValueError("Role not found or cannot be deleted")
        
        return {'message': 'Role deleted successfully'}
    
    # Tenant Management Methods
    
    async def list_tenants(
        self,
        requesting_user: User,
        status: Optional[str] = None
    ) -> Dict[str, Any]:
        """List tenants."""
        # Check permissions
        await self.rbac_manager.require_permission(
            requesting_user,
            PermissionType.VIEW_TENANT
        )
        
        # Super admin can see all tenants
        if any(role.name == "Super Administrator" for role in requesting_user.roles):
            tenants = await self.tenant_manager.list_tenants(status=status)
        else:
            # Regular users can only see their own tenant
            tenant = await self.tenant_manager.get_tenant(requesting_user.tenant_id)
            tenants = [tenant] if tenant else []
        
        tenant_data = []
        for tenant in tenants:
            usage = await self.tenant_manager.get_tenant_usage(tenant.id)
            
            tenant_dict = {
                'id': tenant.id,
                'name': tenant.name,
                'slug': tenant.slug,
                'status': tenant.status,
                'features': list(tenant.features),
                'created_at': tenant.created_at.isoformat(),
                'expires_at': tenant.expires_at.isoformat() if tenant.expires_at else None,
                'limits': {
                    'max_users': tenant.max_users,
                    'max_agents': tenant.max_agents,
                    'max_sessions_per_day': tenant.max_sessions_per_day,
                    'max_storage_gb': tenant.max_storage_gb
                },
                'usage': usage.get('usage', {}),
                'utilization': usage.get('utilization', {})
            }
            tenant_data.append(tenant_dict)
        
        return {'tenants': tenant_data}
    
    async def create_tenant(
        self,
        requesting_user: User,
        tenant_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a new tenant."""
        # Check permissions (only super admin can create tenants)
        await self.rbac_manager.require_permission(
            requesting_user,
            PermissionType.MANAGE_TENANT
        )
        
        # Create tenant
        tenant = await self.tenant_manager.create_tenant(
            name=tenant_data['name'],
            slug=tenant_data.get('slug'),
            settings=tenant_data.get('settings', {}),
            limits=tenant_data.get('limits', {}),
            features=set(tenant_data.get('features', [])),
            expires_in_days=tenant_data.get('expires_in_days')
        )
        
        # Set up tenant defaults
        await self.tenant_manager.setup_tenant_defaults(tenant.id)
        
        return {
            'tenant': {
                'id': tenant.id,
                'name': tenant.name,
                'slug': tenant.slug,
                'status': tenant.status,
                'created_at': tenant.created_at.isoformat()
            }
        }
    
    # API Key Management Methods
    
    async def list_api_keys(
        self,
        requesting_user: User,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """List API keys."""
        # Check permissions
        if user_id and user_id != requesting_user.id:
            await self.rbac_manager.require_permission(
                requesting_user,
                PermissionType.MANAGE_ALL_API_KEYS
            )
        else:
            user_id = requesting_user.id
        
        # Get API keys
        api_keys = await self.api_key_manager.list_api_keys(
            user_id=user_id,
            tenant_id=requesting_user.tenant_id
        )
        
        key_data = []
        for api_key in api_keys:
            key_dict = {
                'id': api_key.id,
                'name': api_key.name,
                'description': api_key.description,
                'scopes': api_key.scopes,
                'is_active': api_key.is_active,
                'usage_count': api_key.usage_count,
                'last_used_at': api_key.last_used_at.isoformat() if api_key.last_used_at else None,
                'created_at': api_key.created_at.isoformat(),
                'expires_at': api_key.expires_at.isoformat() if api_key.expires_at else None
            }
            key_data.append(key_dict)
        
        return {'api_keys': key_data}
    
    async def create_api_key(
        self,
        requesting_user: User,
        key_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a new API key."""
        # Check permissions
        await self.rbac_manager.require_permission(
            requesting_user,
            PermissionType.CREATE_API_KEY
        )
        
        # Convert permission strings
        permissions = []
        for perm_str in key_data.get('permissions', []):
            try:
                permissions.append(PermissionType(perm_str))
            except ValueError:
                logger.warning(f"Invalid permission type: {perm_str}")
        
        # Create API key
        api_key, key_string = await self.api_key_manager.create_api_key(
            name=key_data['name'],
            description=key_data.get('description', ''),
            user_id=requesting_user.id,
            tenant_id=requesting_user.tenant_id,
            scopes=key_data.get('scopes', []),
            permissions=permissions,
            allowed_ips=key_data.get('allowed_ips', []),
            allowed_origins=key_data.get('allowed_origins', []),
            rate_limit=key_data.get('rate_limit'),
            expires_in_days=key_data.get('expires_in_days')
        )
        
        return {
            'api_key': {
                'id': api_key.id,
                'name': api_key.name,
                'key': key_string,  # Only returned once
                'created_at': api_key.created_at.isoformat(),
                'expires_at': api_key.expires_at.isoformat() if api_key.expires_at else None
            },
            'warning': 'Store this API key securely. It will not be shown again.'
        }
    
    async def revoke_api_key(
        self,
        requesting_user: User,
        key_id: str,
        reason: str = ""
    ) -> Dict[str, str]:
        """Revoke an API key."""
        # Check permissions
        api_key = await self.api_key_manager.get_api_key(key_id)
        if not api_key:
            raise ValueError("API key not found")
        
        if api_key.user_id != requesting_user.id:
            await self.rbac_manager.require_permission(
                requesting_user,
                PermissionType.REVOKE_API_KEY
            )
        
        # Revoke key
        await self.api_key_manager.revoke_api_key(key_id, reason)
        
        return {'message': 'API key revoked successfully'}
    
    # System Information Methods
    
    async def get_system_stats(
        self,
        requesting_user: User
    ) -> Dict[str, Any]:
        """Get system statistics."""
        # Check permissions
        await self.rbac_manager.require_permission(
            requesting_user,
            PermissionType.SYSTEM_ADMIN
        )
        
        # Get various statistics
        tenant_stats = await self.tenant_manager.get_tenant_stats()
        api_key_stats = self.api_key_manager.get_metrics()
        session_stats = self.auth_manager.session_manager.get_session_statistics()
        
        return {
            'tenants': tenant_stats,
            'api_keys': api_key_stats,
            'sessions': session_stats,
            'timestamp': datetime.utcnow().isoformat()
        }