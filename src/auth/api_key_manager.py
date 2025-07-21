"""
API Key Management System

Manages API keys for programmatic access to the voice agent platform.
"""

import asyncio
import logging
import secrets
import hashlib
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timedelta
import ipaddress

from .models import ApiKey, User, Permission, PermissionType, AuthProvider
from .exceptions import ApiKeyError, InvalidApiKeyError, RateLimitExceededError

logger = logging.getLogger(__name__)


class ApiKeyManager:
    """Manages API keys for authentication and authorization."""
    
    def __init__(self, config: Any):
        self.config = config
        
        # API key storage (in production, use a database)
        self._api_keys: Dict[str, ApiKey] = {}
        self._key_hash_index: Dict[str, str] = {}  # hash -> key_id mapping
        
        # Rate limiting
        self._rate_limit_tracker: Dict[str, List[datetime]] = {}
        
        # Metrics
        self._metrics: Dict[str, Any] = {
            'total_keys_created': 0,
            'total_authentications': 0,
            'failed_authentications': 0
        }
    
    async def initialize(self):
        """Initialize API key manager."""
        logger.info("Initializing API key manager")
        # Any initialization logic here
        logger.info("API key manager initialized")
    
    def _hash_api_key(self, key: str) -> str:
        """Hash API key for secure storage."""
        return hashlib.sha256(key.encode()).hexdigest()
    
    async def create_api_key(
        self,
        name: str,
        description: str = "",
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        scopes: Optional[List[str]] = None,
        permissions: Optional[List[PermissionType]] = None,
        allowed_ips: Optional[List[str]] = None,
        allowed_origins: Optional[List[str]] = None,
        rate_limit: Optional[int] = None,
        expires_in_days: Optional[int] = None
    ) -> Tuple[ApiKey, str]:
        """
        Create a new API key.
        
        Returns:
            Tuple of (ApiKey object, actual key string)
            The actual key is only returned once and should be securely stored by the user
        """
        # Generate secure key
        key_string = f"vap_{secrets.token_urlsafe(48)}"  # vap = voice agent platform
        
        # Create API key object
        api_key = ApiKey(
            key=key_string,  # Will be replaced with hash
            name=name,
            description=description,
            user_id=user_id,
            tenant_id=tenant_id,
            scopes=scopes or [],
            allowed_ips=allowed_ips or [],
            allowed_origins=allowed_origins or [],
            rate_limit=rate_limit or self.config.rate_limit_requests_per_minute
        )
        
        # Add permissions
        if permissions:
            for perm_type in permissions:
                permission = Permission(type=perm_type)
                api_key.permissions.append(permission)
        
        # Set expiry
        if expires_in_days:
            api_key.expires_at = datetime.utcnow() + timedelta(days=expires_in_days)
        
        # Store key hash instead of actual key
        key_hash = self._hash_api_key(key_string)
        api_key.key = key_hash
        
        # Store in indices
        self._api_keys[api_key.id] = api_key
        self._key_hash_index[key_hash] = api_key.id
        
        # Update metrics
        self._metrics['total_keys_created'] += 1
        
        logger.info(f"Created API key: {name} (ID: {api_key.id})")
        
        # Return both the object and the actual key
        return api_key, key_string
    
    async def authenticate(self, api_key_string: str) -> Optional[User]:
        """
        Authenticate using API key.
        
        Args:
            api_key_string: The API key string
            
        Returns:
            User object if authentication successful, None otherwise
        """
        try:
            # Hash the provided key
            key_hash = self._hash_api_key(api_key_string)
            
            # Look up key
            if key_hash not in self._key_hash_index:
                self._metrics['failed_authentications'] += 1
                raise InvalidApiKeyError("Invalid API key")
            
            key_id = self._key_hash_index[key_hash]
            api_key = self._api_keys.get(key_id)
            
            if not api_key:
                self._metrics['failed_authentications'] += 1
                raise InvalidApiKeyError("Invalid API key")
            
            # Validate key
            if not api_key.is_valid():
                self._metrics['failed_authentications'] += 1
                raise InvalidApiKeyError("API key is expired or revoked")
            
            # Record usage
            api_key.record_usage()
            self._metrics['total_authentications'] += 1
            
            # Create user object for API key
            user = User(
                id=f"api_key_{api_key.id}",
                email=f"api_key_{api_key.id}@system",
                username=f"api_key_{api_key.name}",
                full_name=f"API Key: {api_key.name}",
                auth_provider=AuthProvider.API_KEY,
                auth_provider_id=api_key.id,
                tenant_id=api_key.tenant_id
            )
            
            # Copy permissions from API key
            user.direct_permissions = api_key.permissions.copy()
            
            # Add API key info to user profile
            user.profile['api_key_id'] = api_key.id
            user.profile['api_key_scopes'] = api_key.scopes
            
            return user
            
        except Exception as e:
            logger.error(f"API key authentication failed: {e}")
            raise
    
    async def validate_api_key(
        self,
        api_key_string: str,
        required_scope: Optional[str] = None,
        required_permission: Optional[PermissionType] = None,
        ip_address: Optional[str] = None,
        origin: Optional[str] = None
    ) -> ApiKey:
        """
        Validate API key and check requirements.
        
        Args:
            api_key_string: The API key string
            required_scope: Optional required scope
            required_permission: Optional required permission
            ip_address: Client IP address for validation
            origin: Request origin for validation
            
        Returns:
            ApiKey object if valid
            
        Raises:
            InvalidApiKeyError: If key is invalid
            ApiKeyError: If requirements not met
        """
        # Hash and look up key
        key_hash = self._hash_api_key(api_key_string)
        
        if key_hash not in self._key_hash_index:
            raise InvalidApiKeyError("Invalid API key")
        
        key_id = self._key_hash_index[key_hash]
        api_key = self._api_keys.get(key_id)
        
        if not api_key or not api_key.is_valid():
            raise InvalidApiKeyError("Invalid or expired API key")
        
        # Check rate limit
        if not await self._check_rate_limit(api_key):
            raise RateLimitExceededError("API key rate limit exceeded")
        
        # Check IP restrictions
        if ip_address and api_key.allowed_ips:
            if not self._check_ip_allowed(ip_address, api_key.allowed_ips):
                raise ApiKeyError(f"IP address {ip_address} not allowed")
        
        # Check origin restrictions
        if origin and api_key.allowed_origins:
            if not any(origin.startswith(allowed) for allowed in api_key.allowed_origins):
                raise ApiKeyError(f"Origin {origin} not allowed")
        
        # Check scope
        if required_scope and not api_key.has_scope(required_scope):
            raise ApiKeyError(f"API key lacks required scope: {required_scope}")
        
        # Check permission
        if required_permission and not api_key.has_permission(required_permission):
            raise ApiKeyError(f"API key lacks required permission: {required_permission.value}")
        
        # Record usage
        api_key.record_usage(ip_address)
        
        return api_key
    
    def _check_ip_allowed(self, ip_address: str, allowed_ips: List[str]) -> bool:
        """Check if IP address is in allowed list."""
        try:
            client_ip = ipaddress.ip_address(ip_address)
            
            for allowed in allowed_ips:
                if '/' in allowed:
                    # CIDR notation
                    network = ipaddress.ip_network(allowed, strict=False)
                    if client_ip in network:
                        return True
                else:
                    # Single IP
                    if str(client_ip) == allowed:
                        return True
            
            return False
            
        except ValueError:
            logger.error(f"Invalid IP address: {ip_address}")
            return False
    
    async def _check_rate_limit(self, api_key: ApiKey) -> bool:
        """Check rate limiting for API key."""
        if not api_key.rate_limit:
            return True
        
        key_id = api_key.id
        now = datetime.utcnow()
        window_start = now - timedelta(minutes=1)
        
        # Get or create request history
        if key_id not in self._rate_limit_tracker:
            self._rate_limit_tracker[key_id] = []
        
        # Remove old requests
        self._rate_limit_tracker[key_id] = [
            ts for ts in self._rate_limit_tracker[key_id]
            if ts > window_start
        ]
        
        # Check limit
        if len(self._rate_limit_tracker[key_id]) >= api_key.rate_limit:
            return False
        
        # Add current request
        self._rate_limit_tracker[key_id].append(now)
        return True
    
    async def get_api_key(self, key_id: str) -> Optional[ApiKey]:
        """Get API key by ID."""
        return self._api_keys.get(key_id)
    
    async def list_api_keys(
        self,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        include_revoked: bool = False
    ) -> List[ApiKey]:
        """List API keys with optional filters."""
        keys = []
        
        for api_key in self._api_keys.values():
            # Filter by user
            if user_id and api_key.user_id != user_id:
                continue
            
            # Filter by tenant
            if tenant_id and api_key.tenant_id != tenant_id:
                continue
            
            # Filter revoked
            if not include_revoked and api_key.revoked_at:
                continue
            
            keys.append(api_key)
        
        return sorted(keys, key=lambda k: k.created_at, reverse=True)
    
    async def update_api_key(
        self,
        key_id: str,
        updates: Dict[str, Any]
    ) -> Optional[ApiKey]:
        """Update API key properties."""
        api_key = self._api_keys.get(key_id)
        if not api_key:
            return None
        
        # Update allowed fields
        allowed_fields = [
            'name', 'description', 'scopes', 'allowed_ips',
            'allowed_origins', 'rate_limit'
        ]
        
        for field in allowed_fields:
            if field in updates:
                setattr(api_key, field, updates[field])
        
        # Update permissions if provided
        if 'permissions' in updates:
            api_key.permissions = []
            for perm_type in updates['permissions']:
                if isinstance(perm_type, str):
                    perm_type = PermissionType(perm_type)
                permission = Permission(type=perm_type)
                api_key.permissions.append(permission)
        
        logger.info(f"Updated API key: {api_key.name}")
        return api_key
    
    async def revoke_api_key(self, key_id: str, reason: str = "") -> bool:
        """Revoke an API key."""
        api_key = self._api_keys.get(key_id)
        if not api_key:
            return False
        
        api_key.is_active = False
        api_key.revoked_at = datetime.utcnow()
        
        logger.info(f"Revoked API key: {api_key.name} (reason: {reason})")
        return True
    
    async def rotate_api_key(self, key_id: str) -> Tuple[ApiKey, str]:
        """
        Rotate an API key (revoke old, create new).
        
        Returns:
            Tuple of (new ApiKey object, new key string)
        """
        old_key = self._api_keys.get(key_id)
        if not old_key:
            raise ApiKeyError(f"API key {key_id} not found")
        
        # Revoke old key
        await self.revoke_api_key(key_id, "Key rotation")
        
        # Create new key with same properties
        new_key, new_key_string = await self.create_api_key(
            name=f"{old_key.name} (rotated)",
            description=old_key.description,
            user_id=old_key.user_id,
            tenant_id=old_key.tenant_id,
            scopes=old_key.scopes,
            permissions=[p.type for p in old_key.permissions],
            allowed_ips=old_key.allowed_ips,
            allowed_origins=old_key.allowed_origins,
            rate_limit=old_key.rate_limit
        )
        
        logger.info(f"Rotated API key: {old_key.name} -> {new_key.name}")
        return new_key, new_key_string
    
    async def cleanup_expired_keys(self):
        """Remove expired API keys."""
        now = datetime.utcnow()
        expired_keys = []
        
        for key_id, api_key in self._api_keys.items():
            if api_key.expires_at and api_key.expires_at < now:
                expired_keys.append(key_id)
        
        for key_id in expired_keys:
            api_key = self._api_keys[key_id]
            # Remove from indices
            del self._api_keys[key_id]
            del self._key_hash_index[api_key.key]
            logger.info(f"Cleaned up expired API key: {api_key.name}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get API key usage metrics."""
        active_keys = sum(
            1 for key in self._api_keys.values()
            if key.is_valid()
        )
        
        return {
            **self._metrics,
            'active_keys': active_keys,
            'total_keys': len(self._api_keys)
        }
    
    async def cleanup(self):
        """Clean up resources."""
        self._rate_limit_tracker.clear()


# Middleware for API key authentication
class ApiKeyAuthMiddleware:
    """Middleware for API key authentication in web frameworks."""
    
    def __init__(self, api_key_manager: ApiKeyManager):
        self.api_key_manager = api_key_manager
    
    async def authenticate_request(
        self,
        headers: Dict[str, str],
        query_params: Dict[str, str],
        ip_address: Optional[str] = None,
        origin: Optional[str] = None
    ) -> Optional[User]:
        """
        Authenticate request using API key from headers or query params.
        
        Args:
            headers: Request headers
            query_params: Query parameters
            ip_address: Client IP address
            origin: Request origin
            
        Returns:
            User object if authenticated, None otherwise
        """
        # Check header first
        api_key = headers.get(
            self.api_key_manager.config.api_key_header,
            headers.get('Authorization', '').replace('Bearer ', '')
        )
        
        # Check query parameter
        if not api_key:
            api_key = query_params.get(
                self.api_key_manager.config.api_key_query_param
            )
        
        if not api_key:
            return None
        
        try:
            # Validate key
            validated_key = await self.api_key_manager.validate_api_key(
                api_key,
                ip_address=ip_address,
                origin=origin
            )
            
            # Authenticate and get user
            return await self.api_key_manager.authenticate(api_key)
            
        except (InvalidApiKeyError, ApiKeyError) as e:
            logger.warning(f"API key authentication failed: {e}")
            return None