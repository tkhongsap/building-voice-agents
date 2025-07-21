"""
OAuth 2.0 Provider Implementation

Supports multiple OAuth providers for enterprise SSO integration.
"""

import asyncio
import logging
import secrets
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import httpx
import jwt
from urllib.parse import urlencode, parse_qs, urlparse

from .models import User, AuthProvider
from .exceptions import OAuthError, AuthenticationError

logger = logging.getLogger(__name__)


@dataclass
class OAuthProviderConfig:
    """Configuration for an OAuth provider."""
    name: str
    client_id: str
    client_secret: str
    authorize_url: str
    token_url: str
    userinfo_url: str
    scope: List[str] = field(default_factory=lambda: ["openid", "profile", "email"])
    
    # Optional endpoints
    jwks_url: Optional[str] = None
    revoke_url: Optional[str] = None
    
    # Mapping configuration
    user_id_field: str = "sub"
    email_field: str = "email"
    name_field: str = "name"
    
    # Additional settings
    use_pkce: bool = True
    verify_ssl: bool = True
    timeout: int = 30


class OAuthState:
    """Manages OAuth state for CSRF protection."""
    def __init__(self):
        self._states: Dict[str, Dict[str, Any]] = {}
        self._cleanup_interval = 300  # 5 minutes
    
    def create_state(self, redirect_uri: str, tenant_id: Optional[str] = None) -> str:
        """Create a new state token."""
        state = secrets.token_urlsafe(32)
        self._states[state] = {
            'redirect_uri': redirect_uri,
            'tenant_id': tenant_id,
            'created_at': time.time(),
            'code_verifier': secrets.token_urlsafe(64) if True else None  # PKCE
        }
        return state
    
    def verify_state(self, state: str) -> Optional[Dict[str, Any]]:
        """Verify and consume a state token."""
        if state not in self._states:
            return None
        
        state_data = self._states.pop(state)
        
        # Check expiry (10 minutes)
        if time.time() - state_data['created_at'] > 600:
            return None
        
        return state_data
    
    def cleanup_expired(self):
        """Remove expired state tokens."""
        current_time = time.time()
        expired = [
            state for state, data in self._states.items()
            if current_time - data['created_at'] > 600
        ]
        for state in expired:
            del self._states[state]


class OAuthProvider:
    """Handles OAuth 2.0 authentication flow."""
    
    def __init__(self, config: Any):
        self.config = config
        self.providers: Dict[str, OAuthProviderConfig] = {}
        self.state_manager = OAuthState()
        self._http_client: Optional[httpx.AsyncClient] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Cache for provider metadata
        self._provider_metadata: Dict[str, Dict[str, Any]] = {}
        self._jwks_cache: Dict[str, Dict[str, Any]] = {}
    
    async def initialize(self):
        """Initialize OAuth provider."""
        logger.info("Initializing OAuth provider")
        
        # Create HTTP client
        self._http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0),
            follow_redirects=True
        )
        
        # Load provider configurations from config
        if hasattr(self.config, 'oauth_providers'):
            for provider_id, provider_config in self.config.oauth_providers.items():
                await self.register_provider(provider_id, provider_config)
        
        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info("OAuth provider initialized")
    
    async def register_provider(self, provider_id: str, config: Dict[str, Any]):
        """Register an OAuth provider."""
        try:
            provider = OAuthProviderConfig(
                name=config.get('name', provider_id),
                client_id=config['client_id'],
                client_secret=config['client_secret'],
                authorize_url=config['authorize_url'],
                token_url=config['token_url'],
                userinfo_url=config['userinfo_url'],
                scope=config.get('scope', ['openid', 'profile', 'email']),
                jwks_url=config.get('jwks_url'),
                revoke_url=config.get('revoke_url'),
                user_id_field=config.get('user_id_field', 'sub'),
                email_field=config.get('email_field', 'email'),
                name_field=config.get('name_field', 'name'),
                use_pkce=config.get('use_pkce', True),
                verify_ssl=config.get('verify_ssl', True)
            )
            
            self.providers[provider_id] = provider
            
            # Try to fetch provider metadata if available
            await self._fetch_provider_metadata(provider_id, provider)
            
            logger.info(f"Registered OAuth provider: {provider_id}")
            
        except Exception as e:
            logger.error(f"Failed to register OAuth provider {provider_id}: {e}")
            raise OAuthError(f"Failed to register provider: {e}")
    
    async def _fetch_provider_metadata(self, provider_id: str, provider: OAuthProviderConfig):
        """Fetch OpenID Connect discovery metadata if available."""
        try:
            # Try to construct discovery URL
            parsed_url = urlparse(provider.authorize_url)
            base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
            discovery_url = f"{base_url}/.well-known/openid-configuration"
            
            response = await self._http_client.get(
                discovery_url,
                timeout=10.0
            )
            
            if response.status_code == 200:
                metadata = response.json()
                self._provider_metadata[provider_id] = metadata
                
                # Update provider config with discovered endpoints
                if 'jwks_uri' in metadata and not provider.jwks_url:
                    provider.jwks_url = metadata['jwks_uri']
                if 'revocation_endpoint' in metadata and not provider.revoke_url:
                    provider.revoke_url = metadata['revocation_endpoint']
                
                logger.info(f"Fetched metadata for provider {provider_id}")
        except Exception:
            # Metadata discovery is optional
            pass
    
    def get_authorization_url(
        self,
        provider_id: str,
        redirect_uri: str,
        tenant_id: Optional[str] = None,
        additional_params: Optional[Dict[str, str]] = None
    ) -> str:
        """Generate authorization URL for OAuth flow."""
        if provider_id not in self.providers:
            raise OAuthError(f"Unknown OAuth provider: {provider_id}")
        
        provider = self.providers[provider_id]
        
        # Create state for CSRF protection
        state = self.state_manager.create_state(redirect_uri, tenant_id)
        state_data = self.state_manager._states[state]
        
        # Build authorization URL
        params = {
            'client_id': provider.client_id,
            'redirect_uri': redirect_uri,
            'response_type': 'code',
            'scope': ' '.join(provider.scope),
            'state': state
        }
        
        # Add PKCE challenge if enabled
        if provider.use_pkce and state_data['code_verifier']:
            code_challenge = self._generate_code_challenge(state_data['code_verifier'])
            params['code_challenge'] = code_challenge
            params['code_challenge_method'] = 'S256'
        
        # Add additional parameters
        if additional_params:
            params.update(additional_params)
        
        # Add tenant hint if available
        if tenant_id:
            params['tenant_id'] = tenant_id
        
        return f"{provider.authorize_url}?{urlencode(params)}"
    
    def _generate_code_challenge(self, verifier: str) -> str:
        """Generate PKCE code challenge from verifier."""
        import hashlib
        import base64
        
        digest = hashlib.sha256(verifier.encode('utf-8')).digest()
        return base64.urlsafe_b64encode(digest).decode('utf-8').rstrip('=')
    
    async def authenticate(self, auth_data: Dict[str, Any], tenant_id: Optional[str] = None) -> Optional[User]:
        """
        Authenticate user with OAuth provider.
        
        Args:
            auth_data: Contains 'provider_id', 'code', 'state', and 'redirect_uri'
            tenant_id: Optional tenant ID
            
        Returns:
            Authenticated User object
        """
        try:
            provider_id = auth_data.get('provider_id')
            code = auth_data.get('code')
            state = auth_data.get('state')
            redirect_uri = auth_data.get('redirect_uri')
            
            if not all([provider_id, code, state, redirect_uri]):
                raise OAuthError("Missing required OAuth parameters")
            
            if provider_id not in self.providers:
                raise OAuthError(f"Unknown OAuth provider: {provider_id}")
            
            provider = self.providers[provider_id]
            
            # Verify state
            state_data = self.state_manager.verify_state(state)
            if not state_data:
                raise OAuthError("Invalid or expired state")
            
            # Verify redirect URI matches
            if state_data['redirect_uri'] != redirect_uri:
                raise OAuthError("Redirect URI mismatch")
            
            # Exchange code for token
            token_data = await self._exchange_code_for_token(
                provider,
                code,
                redirect_uri,
                state_data.get('code_verifier')
            )
            
            # Get user info
            user_info = await self._get_user_info(provider, token_data['access_token'])
            
            # Create or update user
            user = await self._create_or_update_user(
                provider_id,
                provider,
                user_info,
                token_data,
                tenant_id or state_data.get('tenant_id')
            )
            
            return user
            
        except Exception as e:
            logger.error(f"OAuth authentication failed: {e}")
            raise
    
    async def _exchange_code_for_token(
        self,
        provider: OAuthProviderConfig,
        code: str,
        redirect_uri: str,
        code_verifier: Optional[str] = None
    ) -> Dict[str, Any]:
        """Exchange authorization code for access token."""
        try:
            data = {
                'grant_type': 'authorization_code',
                'code': code,
                'redirect_uri': redirect_uri,
                'client_id': provider.client_id,
                'client_secret': provider.client_secret
            }
            
            # Add PKCE verifier if used
            if provider.use_pkce and code_verifier:
                data['code_verifier'] = code_verifier
            
            response = await self._http_client.post(
                provider.token_url,
                data=data,
                headers={'Accept': 'application/json'},
                timeout=provider.timeout
            )
            
            if response.status_code != 200:
                error_data = response.json() if response.headers.get('content-type', '').startswith('application/json') else {}
                raise OAuthError(f"Token exchange failed: {error_data.get('error_description', response.text)}")
            
            token_data = response.json()
            
            # Validate token if ID token is present
            if 'id_token' in token_data:
                await self._validate_id_token(provider, token_data['id_token'])
            
            return token_data
            
        except httpx.RequestError as e:
            raise OAuthError(f"Token exchange request failed: {e}")
    
    async def _get_user_info(self, provider: OAuthProviderConfig, access_token: str) -> Dict[str, Any]:
        """Get user information from OAuth provider."""
        try:
            response = await self._http_client.get(
                provider.userinfo_url,
                headers={
                    'Authorization': f'Bearer {access_token}',
                    'Accept': 'application/json'
                },
                timeout=provider.timeout
            )
            
            if response.status_code != 200:
                raise OAuthError(f"Failed to get user info: {response.status_code}")
            
            return response.json()
            
        except httpx.RequestError as e:
            raise OAuthError(f"User info request failed: {e}")
    
    async def _validate_id_token(self, provider: OAuthProviderConfig, id_token: str):
        """Validate OpenID Connect ID token."""
        try:
            # Decode header to get key ID
            header = jwt.get_unverified_header(id_token)
            
            # Get signing key
            if provider.jwks_url:
                jwks = await self._get_jwks(provider)
                # Find the key with matching kid
                key = next((k for k in jwks.get('keys', []) if k.get('kid') == header.get('kid')), None)
                if not key:
                    raise OAuthError("No matching key found in JWKS")
                
                # Convert JWK to PEM (simplified - use proper library in production)
                # This is a placeholder - implement proper JWK to PEM conversion
                public_key = None
            else:
                # Use client secret for HS256 signed tokens
                public_key = provider.client_secret
            
            # Verify token
            payload = jwt.decode(
                id_token,
                public_key or provider.client_secret,
                algorithms=['RS256', 'HS256'],
                audience=provider.client_id,
                options={"verify_exp": True}
            )
            
            # Additional validations
            if payload.get('iss') not in [provider.authorize_url, self._provider_metadata.get(provider.name, {}).get('issuer')]:
                raise OAuthError("Invalid token issuer")
            
        except jwt.InvalidTokenError as e:
            raise OAuthError(f"Invalid ID token: {e}")
    
    async def _get_jwks(self, provider: OAuthProviderConfig) -> Dict[str, Any]:
        """Get JSON Web Key Set from provider."""
        if provider.name in self._jwks_cache:
            return self._jwks_cache[provider.name]
        
        try:
            response = await self._http_client.get(
                provider.jwks_url,
                timeout=provider.timeout
            )
            
            if response.status_code != 200:
                raise OAuthError("Failed to fetch JWKS")
            
            jwks = response.json()
            self._jwks_cache[provider.name] = jwks
            
            return jwks
            
        except httpx.RequestError as e:
            raise OAuthError(f"JWKS request failed: {e}")
    
    async def _create_or_update_user(
        self,
        provider_id: str,
        provider: OAuthProviderConfig,
        user_info: Dict[str, Any],
        token_data: Dict[str, Any],
        tenant_id: Optional[str] = None
    ) -> User:
        """Create or update user from OAuth user info."""
        # Extract user data based on provider mapping
        external_id = str(user_info.get(provider.user_id_field))
        email = user_info.get(provider.email_field)
        name = user_info.get(provider.name_field, email)
        
        if not external_id:
            raise OAuthError("No user ID found in OAuth response")
        
        # Create user object
        user = User(
            email=email or f"{external_id}@{provider_id}",
            username=email or external_id,
            full_name=name,
            auth_provider=AuthProvider.OAUTH,
            auth_provider_id=f"{provider_id}:{external_id}",
            tenant_id=tenant_id
        )
        
        # Store OAuth tokens in user profile
        user.profile['oauth_tokens'] = {
            'access_token': token_data.get('access_token'),
            'refresh_token': token_data.get('refresh_token'),
            'expires_at': datetime.utcnow() + timedelta(seconds=token_data.get('expires_in', 3600))
        }
        
        # Store additional user info
        user.profile['oauth_user_info'] = user_info
        
        return user
    
    async def refresh_token(self, provider_id: str, refresh_token: str) -> Dict[str, Any]:
        """Refresh OAuth access token."""
        if provider_id not in self.providers:
            raise OAuthError(f"Unknown OAuth provider: {provider_id}")
        
        provider = self.providers[provider_id]
        
        try:
            data = {
                'grant_type': 'refresh_token',
                'refresh_token': refresh_token,
                'client_id': provider.client_id,
                'client_secret': provider.client_secret
            }
            
            response = await self._http_client.post(
                provider.token_url,
                data=data,
                headers={'Accept': 'application/json'},
                timeout=provider.timeout
            )
            
            if response.status_code != 200:
                raise OAuthError("Token refresh failed")
            
            return response.json()
            
        except httpx.RequestError as e:
            raise OAuthError(f"Token refresh request failed: {e}")
    
    async def revoke_token(self, provider_id: str, token: str, token_type: str = 'access_token'):
        """Revoke OAuth token."""
        if provider_id not in self.providers:
            raise OAuthError(f"Unknown OAuth provider: {provider_id}")
        
        provider = self.providers[provider_id]
        
        if not provider.revoke_url:
            logger.warning(f"Provider {provider_id} does not support token revocation")
            return
        
        try:
            data = {
                'token': token,
                'token_type_hint': token_type,
                'client_id': provider.client_id,
                'client_secret': provider.client_secret
            }
            
            response = await self._http_client.post(
                provider.revoke_url,
                data=data,
                timeout=provider.timeout
            )
            
            if response.status_code not in [200, 204]:
                logger.warning(f"Token revocation failed: {response.status_code}")
            
        except httpx.RequestError as e:
            logger.error(f"Token revocation request failed: {e}")
    
    async def _cleanup_loop(self):
        """Periodic cleanup of expired state tokens."""
        while True:
            try:
                await asyncio.sleep(300)  # 5 minutes
                self.state_manager.cleanup_expired()
                
                # Clear old JWKS cache
                self._jwks_cache.clear()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"OAuth cleanup error: {e}")
    
    async def cleanup(self):
        """Clean up resources."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        if self._http_client:
            await self._http_client.aclose()


# Predefined OAuth provider configurations
OAUTH_PROVIDERS = {
    "google": {
        "name": "Google",
        "authorize_url": "https://accounts.google.com/o/oauth2/v2/auth",
        "token_url": "https://oauth2.googleapis.com/token",
        "userinfo_url": "https://www.googleapis.com/oauth2/v3/userinfo",
        "jwks_url": "https://www.googleapis.com/oauth2/v3/certs",
        "revoke_url": "https://oauth2.googleapis.com/revoke",
        "scope": ["openid", "profile", "email"]
    },
    "microsoft": {
        "name": "Microsoft",
        "authorize_url": "https://login.microsoftonline.com/common/oauth2/v2.0/authorize",
        "token_url": "https://login.microsoftonline.com/common/oauth2/v2.0/token",
        "userinfo_url": "https://graph.microsoft.com/v1.0/me",
        "jwks_url": "https://login.microsoftonline.com/common/discovery/v2.0/keys",
        "scope": ["openid", "profile", "email", "User.Read"]
    },
    "github": {
        "name": "GitHub",
        "authorize_url": "https://github.com/login/oauth/authorize",
        "token_url": "https://github.com/login/oauth/access_token",
        "userinfo_url": "https://api.github.com/user",
        "scope": ["user:email"],
        "email_field": "email",
        "user_id_field": "id",
        "name_field": "name"
    },
    "okta": {
        "name": "Okta",
        "authorize_url": "{okta_domain}/oauth2/v1/authorize",
        "token_url": "{okta_domain}/oauth2/v1/token",
        "userinfo_url": "{okta_domain}/oauth2/v1/userinfo",
        "jwks_url": "{okta_domain}/oauth2/v1/keys",
        "revoke_url": "{okta_domain}/oauth2/v1/revoke",
        "scope": ["openid", "profile", "email"]
    }
}