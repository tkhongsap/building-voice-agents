"""
Role-Based Access Control (RBAC) Manager

Manages roles, permissions, and access control policies.
"""

import asyncio
import logging
from typing import Optional, Dict, Any, List, Set, Tuple
from datetime import datetime
import re

from .models import User, Role, Permission, PermissionType, SYSTEM_ROLES
from .exceptions import AuthorizationError, PermissionDeniedError

logger = logging.getLogger(__name__)


class AccessPolicy:
    """Represents an access control policy."""
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.rules: List[Dict[str, Any]] = []
    
    def add_rule(
        self,
        effect: str,  # "allow" or "deny"
        permissions: List[str],
        resources: Optional[List[str]] = None,
        conditions: Optional[Dict[str, Any]] = None
    ):
        """Add a rule to the policy."""
        self.rules.append({
            'effect': effect,
            'permissions': permissions,
            'resources': resources or ['*'],
            'conditions': conditions or {}
        })
    
    def evaluate(
        self,
        permission: str,
        resource: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[bool]:
        """Evaluate policy for a permission request."""
        for rule in self.rules:
            # Check if permission matches
            if not self._matches_permission(permission, rule['permissions']):
                continue
            
            # Check if resource matches
            if not self._matches_resource(resource, rule['resources']):
                continue
            
            # Check conditions
            if not self._evaluate_conditions(rule['conditions'], context):
                continue
            
            # Rule matches, return effect
            return rule['effect'] == 'allow'
        
        # No matching rule
        return None
    
    def _matches_permission(self, permission: str, patterns: List[str]) -> bool:
        """Check if permission matches any pattern."""
        for pattern in patterns:
            if pattern == '*':
                return True
            if pattern == permission:
                return True
            # Support wildcards like "voice_agent.*"
            if '*' in pattern:
                regex = pattern.replace('.', r'\.').replace('*', '.*')
                if re.match(f"^{regex}$", permission):
                    return True
        return False
    
    def _matches_resource(self, resource: Optional[str], patterns: List[str]) -> bool:
        """Check if resource matches any pattern."""
        if not resource:
            return '*' in patterns
        
        for pattern in patterns:
            if pattern == '*':
                return True
            if pattern == resource:
                return True
            # Support wildcards
            if '*' in pattern:
                regex = pattern.replace('*', '.*')
                if re.match(f"^{regex}$", resource):
                    return True
        return False
    
    def _evaluate_conditions(self, conditions: Dict[str, Any], context: Optional[Dict[str, Any]]) -> bool:
        """Evaluate policy conditions."""
        if not conditions:
            return True
        
        if not context:
            return False
        
        # Example conditions: time-based, IP-based, etc.
        for key, value in conditions.items():
            if key == 'ip_range' and 'ip_address' in context:
                # Check IP range (simplified)
                if not self._check_ip_range(context['ip_address'], value):
                    return False
            
            elif key == 'time_range' and 'current_time' in context:
                # Check time range
                if not self._check_time_range(context['current_time'], value):
                    return False
            
            elif key == 'mfa_required' and value:
                # Check MFA status
                if not context.get('mfa_verified', False):
                    return False
        
        return True
    
    def _check_ip_range(self, ip: str, allowed_ranges: List[str]) -> bool:
        """Check if IP is in allowed ranges (simplified)."""
        # In production, use proper IP range checking
        return any(ip.startswith(range.split('/')[0]) for range in allowed_ranges)
    
    def _check_time_range(self, current_time: datetime, time_range: Dict[str, str]) -> bool:
        """Check if current time is within allowed range."""
        # Simplified time range checking
        return True


class RBACManager:
    """Manages roles, permissions, and access control."""
    
    def __init__(self):
        # Role storage (in production, use a database)
        self._roles: Dict[str, Role] = {}
        self._roles_by_name: Dict[str, Role] = {}
        
        # Policy storage
        self._policies: Dict[str, AccessPolicy] = {}
        
        # Permission cache for performance
        self._permission_cache: Dict[str, Dict[str, bool]] = {}
        
        # Initialize with system roles
        self._initialize_system_roles()
    
    def _initialize_system_roles(self):
        """Initialize predefined system roles."""
        for role_id, role in SYSTEM_ROLES.items():
            self._roles[role.id] = role
            self._roles_by_name[role.name] = role
    
    async def initialize(self):
        """Initialize RBAC manager."""
        logger.info("Initializing RBAC manager")
        
        # Initialize default policies
        await self._initialize_default_policies()
        
        logger.info("RBAC manager initialized")
    
    async def _initialize_default_policies(self):
        """Initialize default access policies."""
        # Admin policy
        admin_policy = AccessPolicy("admin_policy", "Full system access")
        admin_policy.add_rule("allow", ["*"], ["*"])
        self._policies["admin_policy"] = admin_policy
        
        # Agent operator policy
        operator_policy = AccessPolicy("operator_policy", "Voice agent operation")
        operator_policy.add_rule(
            "allow",
            [
                "voice_agent.read",
                "voice_agent.execute",
                "session.create",
                "session.read",
                "session.update"
            ],
            ["*"]
        )
        operator_policy.add_rule(
            "deny",
            ["voice_agent.delete", "voice_agent.create"],
            ["*"]
        )
        self._policies["operator_policy"] = operator_policy
        
        # Viewer policy
        viewer_policy = AccessPolicy("viewer_policy", "Read-only access")
        viewer_policy.add_rule(
            "allow",
            ["*.read", "*.view"],
            ["*"]
        )
        viewer_policy.add_rule(
            "deny",
            ["*.create", "*.update", "*.delete", "*.execute"],
            ["*"]
        )
        self._policies["viewer_policy"] = viewer_policy
    
    async def check_permission(
        self,
        user: User,
        permission_type: PermissionType,
        resource_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Check if user has a specific permission.
        
        Args:
            user: User to check
            permission_type: Permission type to check
            resource_id: Optional resource ID
            context: Optional context for policy evaluation
            
        Returns:
            True if permission is granted, False otherwise
        """
        # Super admin bypass
        if any(role.name == "Super Administrator" for role in user.roles):
            return True
        
        # Check cache
        cache_key = f"{user.id}:{permission_type.value}:{resource_id or '*'}"
        if cache_key in self._permission_cache:
            return self._permission_cache[cache_key]
        
        # Check user's direct permissions
        if user.has_permission(permission_type, resource_id):
            self._permission_cache[cache_key] = True
            return True
        
        # Check role permissions
        for role in user.roles:
            if role.has_permission(permission_type, resource_id):
                self._permission_cache[cache_key] = True
                return True
        
        # Check policies
        for role in user.roles:
            policy_name = f"{role.name.lower().replace(' ', '_')}_policy"
            if policy_name in self._policies:
                policy = self._policies[policy_name]
                result = policy.evaluate(permission_type.value, resource_id, context)
                if result is not None:
                    self._permission_cache[cache_key] = result
                    return result
        
        # Default deny
        self._permission_cache[cache_key] = False
        return False
    
    async def require_permission(
        self,
        user: User,
        permission_type: PermissionType,
        resource_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Require user to have a specific permission, raise error if not.
        
        Args:
            user: User to check
            permission_type: Permission type to check
            resource_id: Optional resource ID
            context: Optional context for policy evaluation
            
        Raises:
            PermissionDeniedError: If permission is not granted
        """
        if not await self.check_permission(user, permission_type, resource_id, context):
            raise PermissionDeniedError(
                f"User {user.username} lacks permission {permission_type.value}"
                + (f" for resource {resource_id}" if resource_id else "")
            )
    
    async def get_user_permissions(self, user: User) -> List[str]:
        """Get all permissions for a user."""
        permissions = set()
        
        # Add direct permissions
        for perm in user.direct_permissions:
            permissions.add(perm.type.value)
        
        # Add role permissions
        for role in user.roles:
            for perm in role.permissions:
                permissions.add(perm.type.value)
        
        return list(permissions)
    
    async def get_user_resources(
        self,
        user: User,
        permission_type: PermissionType
    ) -> List[str]:
        """Get all resources a user can access with a specific permission."""
        resources = []
        
        # Check direct permissions
        for perm in user.direct_permissions:
            if perm.type == permission_type:
                resources.append(perm.resource or '*')
        
        # Check role permissions
        for role in user.roles:
            for perm in role.permissions:
                if perm.type == permission_type:
                    resources.append(perm.resource or '*')
        
        # Remove duplicates and sort
        return sorted(list(set(resources)))
    
    async def create_role(
        self,
        name: str,
        description: str,
        permissions: List[PermissionType],
        tenant_id: Optional[str] = None
    ) -> Role:
        """Create a new role."""
        # Check if role name already exists
        if name in self._roles_by_name:
            raise ValueError(f"Role '{name}' already exists")
        
        # Create role
        role = Role(
            name=name,
            description=description,
            tenant_id=tenant_id
        )
        
        # Add permissions
        for perm_type in permissions:
            permission = Permission(type=perm_type)
            role.add_permission(permission)
        
        # Store role
        self._roles[role.id] = role
        self._roles_by_name[name] = role
        
        # Clear permission cache
        self._permission_cache.clear()
        
        logger.info(f"Created role: {name}")
        return role
    
    async def update_role(
        self,
        role_id: str,
        updates: Dict[str, Any]
    ) -> Optional[Role]:
        """Update a role."""
        role = self._roles.get(role_id)
        if not role:
            return None
        
        if role.is_system_role:
            raise AuthorizationError("Cannot modify system roles")
        
        # Update allowed fields
        if 'name' in updates and updates['name'] != role.name:
            # Update name index
            del self._roles_by_name[role.name]
            role.name = updates['name']
            self._roles_by_name[role.name] = role
        
        if 'description' in updates:
            role.description = updates['description']
        
        if 'permissions' in updates:
            # Replace permissions
            role.permissions = []
            for perm_type in updates['permissions']:
                if isinstance(perm_type, str):
                    perm_type = PermissionType(perm_type)
                permission = Permission(type=perm_type)
                role.add_permission(permission)
        
        role.updated_at = datetime.utcnow()
        
        # Clear permission cache
        self._permission_cache.clear()
        
        return role
    
    async def delete_role(self, role_id: str) -> bool:
        """Delete a role."""
        role = self._roles.get(role_id)
        if not role:
            return False
        
        if role.is_system_role:
            raise AuthorizationError("Cannot delete system roles")
        
        # Remove from storage
        del self._roles[role_id]
        del self._roles_by_name[role.name]
        
        # Clear permission cache
        self._permission_cache.clear()
        
        logger.info(f"Deleted role: {role.name}")
        return True
    
    async def get_role(self, role_id: str) -> Optional[Role]:
        """Get role by ID."""
        return self._roles.get(role_id)
    
    async def get_role_by_name(self, name: str) -> Optional[Role]:
        """Get role by name."""
        return self._roles_by_name.get(name)
    
    async def list_roles(
        self,
        tenant_id: Optional[str] = None,
        include_system_roles: bool = True
    ) -> List[Role]:
        """List all roles."""
        roles = []
        
        for role in self._roles.values():
            # Filter by tenant
            if tenant_id and role.tenant_id != tenant_id:
                continue
            
            # Filter system roles
            if not include_system_roles and role.is_system_role:
                continue
            
            roles.append(role)
        
        return sorted(roles, key=lambda r: r.name)
    
    async def assign_role_to_user(self, user: User, role_id: str):
        """Assign a role to a user."""
        role = self._roles.get(role_id)
        if not role:
            raise ValueError(f"Role {role_id} not found")
        
        # Check tenant compatibility
        if role.tenant_id and user.tenant_id != role.tenant_id:
            raise AuthorizationError("Cannot assign role from different tenant")
        
        user.add_role(role)
        
        # Clear permission cache for user
        self._clear_user_cache(user.id)
    
    async def remove_role_from_user(self, user: User, role_id: str):
        """Remove a role from a user."""
        user.remove_role(role_id)
        
        # Clear permission cache for user
        self._clear_user_cache(user.id)
    
    async def create_policy(self, policy: AccessPolicy):
        """Create a new access policy."""
        self._policies[policy.name] = policy
        
        # Clear permission cache
        self._permission_cache.clear()
    
    async def get_policy(self, name: str) -> Optional[AccessPolicy]:
        """Get policy by name."""
        return self._policies.get(name)
    
    async def list_policies(self) -> List[AccessPolicy]:
        """List all policies."""
        return list(self._policies.values())
    
    def _clear_user_cache(self, user_id: str):
        """Clear permission cache for a specific user."""
        keys_to_remove = [
            key for key in self._permission_cache.keys()
            if key.startswith(f"{user_id}:")
        ]
        for key in keys_to_remove:
            del self._permission_cache[key]
    
    async def get_permission_matrix(
        self,
        users: List[User],
        permissions: List[PermissionType]
    ) -> Dict[str, Dict[str, bool]]:
        """
        Generate a permission matrix for users and permissions.
        
        Returns:
            Dict mapping user_id -> permission -> has_permission
        """
        matrix = {}
        
        for user in users:
            user_perms = {}
            for perm_type in permissions:
                user_perms[perm_type.value] = await self.check_permission(
                    user, perm_type
                )
            matrix[user.id] = user_perms
        
        return matrix
    
    async def cleanup(self):
        """Clean up resources."""
        self._permission_cache.clear()


# Permission decorators for easy use
def require_permission(permission_type: PermissionType, resource_id_param: Optional[str] = None):
    """
    Decorator to require permission for a function.
    
    Usage:
        @require_permission(PermissionType.CREATE_AGENT)
        async def create_agent(user: User, agent_data: dict):
            ...
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Extract user from arguments
            user = None
            for arg in args:
                if isinstance(arg, User):
                    user = arg
                    break
            
            if not user and 'user' in kwargs:
                user = kwargs['user']
            
            if not user:
                raise AuthorizationError("No user provided for permission check")
            
            # Get resource ID if specified
            resource_id = None
            if resource_id_param:
                resource_id = kwargs.get(resource_id_param)
            
            # Check permission
            from .rbac_manager import RBACManager
            rbac = RBACManager()
            await rbac.require_permission(user, permission_type, resource_id)
            
            # Call original function
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator