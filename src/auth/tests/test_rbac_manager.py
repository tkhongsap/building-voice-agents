"""
Tests for RBACManager

Tests for Role-Based Access Control functionality.
"""

import pytest
import asyncio

from ..rbac_manager import RBACManager
from ..models import User, Role, Permission, PermissionType, SYSTEM_ROLES
from ..exceptions import AuthorizationError, PermissionDeniedError


@pytest.fixture
async def rbac_manager():
    """Create RBACManager for testing."""
    manager = RBACManager()
    await manager.initialize()
    
    yield manager
    
    await manager.cleanup()


@pytest.fixture
def test_user():
    """Create test user."""
    return User(
        id="user1",
        email="test@example.com",
        username="testuser",
        full_name="Test User",
        tenant_id="tenant1"
    )


class TestRBACManager:
    """Test cases for RBACManager."""
    
    @pytest.mark.asyncio
    async def test_system_roles_initialized(self, rbac_manager):
        """Test that system roles are properly initialized."""
        # Check that system roles exist
        admin_role = await rbac_manager.get_role_by_name("Super Administrator")
        assert admin_role is not None
        assert admin_role.is_system_role is True
        
        viewer_role = await rbac_manager.get_role_by_name("Viewer")
        assert viewer_role is not None
        assert viewer_role.is_system_role is True
        
        # Check permissions
        assert len(admin_role.permissions) > 0
        assert admin_role.has_permission(PermissionType.SYSTEM_ADMIN)
    
    @pytest.mark.asyncio
    async def test_create_custom_role(self, rbac_manager):
        """Test creating a custom role."""
        permissions = [PermissionType.CREATE_AGENT, PermissionType.READ_AGENT]
        
        role = await rbac_manager.create_role(
            name="Agent Creator",
            description="Can create and read agents",
            permissions=permissions,
            tenant_id="tenant1"
        )
        
        assert role.name == "Agent Creator"
        assert role.description == "Can create and read agents"
        assert role.tenant_id == "tenant1"
        assert role.is_system_role is False
        assert len(role.permissions) == 2
        assert role.has_permission(PermissionType.CREATE_AGENT)
        assert role.has_permission(PermissionType.READ_AGENT)
        assert not role.has_permission(PermissionType.DELETE_AGENT)
    
    @pytest.mark.asyncio
    async def test_duplicate_role_name(self, rbac_manager):
        """Test that duplicate role names are rejected."""
        permissions = [PermissionType.READ_AGENT]
        
        await rbac_manager.create_role(
            name="Test Role",
            description="First role",
            permissions=permissions
        )
        
        with pytest.raises(ValueError, match="Role 'Test Role' already exists"):
            await rbac_manager.create_role(
                name="Test Role",
                description="Second role",
                permissions=permissions
            )
    
    @pytest.mark.asyncio
    async def test_update_role(self, rbac_manager):
        """Test updating a role."""
        # Create role
        role = await rbac_manager.create_role(
            name="Original Role",
            description="Original description",
            permissions=[PermissionType.READ_AGENT]
        )
        
        # Update role
        updates = {
            'name': 'Updated Role',
            'description': 'Updated description',
            'permissions': [PermissionType.READ_AGENT, PermissionType.CREATE_AGENT]
        }
        
        updated_role = await rbac_manager.update_role(role.id, updates)
        
        assert updated_role.name == 'Updated Role'
        assert updated_role.description == 'Updated description'
        assert len(updated_role.permissions) == 2
        assert updated_role.has_permission(PermissionType.CREATE_AGENT)
    
    @pytest.mark.asyncio
    async def test_update_system_role_fails(self, rbac_manager):
        """Test that system roles cannot be updated."""
        admin_role = await rbac_manager.get_role_by_name("Super Administrator")
        
        with pytest.raises(AuthorizationError, match="Cannot modify system roles"):
            await rbac_manager.update_role(admin_role.id, {'name': 'Modified Admin'})
    
    @pytest.mark.asyncio
    async def test_delete_role(self, rbac_manager):
        """Test deleting a role."""
        # Create role
        role = await rbac_manager.create_role(
            name="Deletable Role",
            description="This role will be deleted",
            permissions=[PermissionType.READ_AGENT]
        )
        
        role_id = role.id
        
        # Delete role
        success = await rbac_manager.delete_role(role_id)
        assert success is True
        
        # Role should not exist
        deleted_role = await rbac_manager.get_role(role_id)
        assert deleted_role is None
    
    @pytest.mark.asyncio
    async def test_delete_system_role_fails(self, rbac_manager):
        """Test that system roles cannot be deleted."""
        admin_role = await rbac_manager.get_role_by_name("Super Administrator")
        
        with pytest.raises(AuthorizationError, match="Cannot delete system roles"):
            await rbac_manager.delete_role(admin_role.id)
    
    @pytest.mark.asyncio
    async def test_assign_role_to_user(self, rbac_manager, test_user):
        """Test assigning a role to a user."""
        # Create role
        role = await rbac_manager.create_role(
            name="Test Assignment Role",
            description="For testing role assignment",
            permissions=[PermissionType.READ_AGENT, PermissionType.CREATE_AGENT]
        )
        
        # Assign role
        await rbac_manager.assign_role_to_user(test_user, role.id)
        
        # Check assignment
        assert len(test_user.roles) == 1
        assert test_user.roles[0].id == role.id
        assert test_user.has_permission(PermissionType.READ_AGENT)
        assert test_user.has_permission(PermissionType.CREATE_AGENT)
    
    @pytest.mark.asyncio
    async def test_remove_role_from_user(self, rbac_manager, test_user):
        """Test removing a role from a user."""
        # Create and assign role
        role = await rbac_manager.create_role(
            name="Removable Role",
            description="This role will be removed",
            permissions=[PermissionType.READ_AGENT]
        )
        
        await rbac_manager.assign_role_to_user(test_user, role.id)
        assert len(test_user.roles) == 1
        
        # Remove role
        await rbac_manager.remove_role_from_user(test_user, role.id)
        
        # Check removal
        assert len(test_user.roles) == 0
        assert not test_user.has_permission(PermissionType.READ_AGENT)
    
    @pytest.mark.asyncio
    async def test_check_permission_success(self, rbac_manager, test_user):
        """Test successful permission check."""
        # Assign admin role (has all permissions)
        admin_role = await rbac_manager.get_role_by_name("Super Administrator")
        test_user.add_role(admin_role)
        
        # Check permission
        has_permission = await rbac_manager.check_permission(
            test_user,
            PermissionType.CREATE_AGENT
        )
        
        assert has_permission is True
    
    @pytest.mark.asyncio
    async def test_check_permission_failure(self, rbac_manager, test_user):
        """Test failed permission check."""
        # Assign viewer role (read-only)
        viewer_role = await rbac_manager.get_role_by_name("Viewer")
        test_user.add_role(viewer_role)
        
        # Check for write permission (should fail)
        has_permission = await rbac_manager.check_permission(
            test_user,
            PermissionType.CREATE_AGENT
        )
        
        assert has_permission is False
    
    @pytest.mark.asyncio
    async def test_require_permission_success(self, rbac_manager, test_user):
        """Test require permission with success."""
        # Assign role with required permission
        role = await rbac_manager.create_role(
            name="Agent Reader",
            description="Can read agents",
            permissions=[PermissionType.READ_AGENT]
        )
        test_user.add_role(role)
        
        # Should not raise exception
        await rbac_manager.require_permission(test_user, PermissionType.READ_AGENT)
    
    @pytest.mark.asyncio
    async def test_require_permission_failure(self, rbac_manager, test_user):
        """Test require permission with failure."""
        # User has no roles/permissions
        
        # Should raise PermissionDeniedError
        with pytest.raises(PermissionDeniedError):
            await rbac_manager.require_permission(test_user, PermissionType.CREATE_AGENT)
    
    @pytest.mark.asyncio
    async def test_get_user_permissions(self, rbac_manager, test_user):
        """Test getting all user permissions."""
        # Create role with specific permissions
        role = await rbac_manager.create_role(
            name="Multi Permission Role",
            description="Has multiple permissions",
            permissions=[
                PermissionType.READ_AGENT,
                PermissionType.CREATE_AGENT,
                PermissionType.READ_SESSION
            ]
        )
        test_user.add_role(role)
        
        # Add direct permission
        direct_permission = Permission(type=PermissionType.VIEW_ANALYTICS)
        test_user.direct_permissions.append(direct_permission)
        
        # Get all permissions
        permissions = await rbac_manager.get_user_permissions(test_user)
        
        # Should include both role and direct permissions
        assert PermissionType.READ_AGENT.value in permissions
        assert PermissionType.CREATE_AGENT.value in permissions
        assert PermissionType.READ_SESSION.value in permissions
        assert PermissionType.VIEW_ANALYTICS.value in permissions
    
    @pytest.mark.asyncio
    async def test_list_roles(self, rbac_manager):
        """Test listing roles."""
        # Create some custom roles
        await rbac_manager.create_role(
            name="Custom Role 1",
            description="First custom role",
            permissions=[PermissionType.READ_AGENT],
            tenant_id="tenant1"
        )
        
        await rbac_manager.create_role(
            name="Custom Role 2",
            description="Second custom role",
            permissions=[PermissionType.CREATE_AGENT],
            tenant_id="tenant2"
        )
        
        # List all roles
        all_roles = await rbac_manager.list_roles(include_system_roles=True)
        assert len(all_roles) >= 2  # At least our custom roles + system roles
        
        # List roles for specific tenant
        tenant1_roles = await rbac_manager.list_roles(
            tenant_id="tenant1",
            include_system_roles=False
        )
        assert len(tenant1_roles) == 1
        assert tenant1_roles[0].name == "Custom Role 1"
    
    @pytest.mark.asyncio
    async def test_permission_caching(self, rbac_manager, test_user):
        """Test that permission checks are cached for performance."""
        # Assign role
        role = await rbac_manager.create_role(
            name="Cacheable Role",
            description="For testing caching",
            permissions=[PermissionType.READ_AGENT]
        )
        test_user.add_role(role)
        
        # First check should populate cache
        result1 = await rbac_manager.check_permission(test_user, PermissionType.READ_AGENT)
        assert result1 is True
        
        # Second check should use cache (verify by checking cache directly)
        cache_key = f"{test_user.id}:{PermissionType.READ_AGENT.value}:*"
        assert cache_key in rbac_manager._permission_cache
        assert rbac_manager._permission_cache[cache_key] is True
        
        # Second check
        result2 = await rbac_manager.check_permission(test_user, PermissionType.READ_AGENT)
        assert result2 is True
    
    @pytest.mark.asyncio
    async def test_super_admin_bypass(self, rbac_manager, test_user):
        """Test that super admin bypasses all permission checks."""
        # Assign super admin role
        admin_role = await rbac_manager.get_role_by_name("Super Administrator")
        test_user.add_role(admin_role)
        
        # Should have all permissions
        for permission_type in PermissionType:
            has_permission = await rbac_manager.check_permission(test_user, permission_type)
            assert has_permission is True


@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()