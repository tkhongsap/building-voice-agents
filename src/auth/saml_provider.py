"""
SAML 2.0 Provider Implementation

Supports SAML-based authentication for enterprise SSO integration.
"""

import asyncio
import logging
import base64
import xml.etree.ElementTree as ET
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import secrets
from urllib.parse import urlencode

from .models import User, AuthProvider
from .exceptions import SAMLError, AuthenticationError

logger = logging.getLogger(__name__)


@dataclass
class SAMLProviderConfig:
    """Configuration for a SAML provider."""
    name: str
    entity_id: str
    sso_url: str
    slo_url: Optional[str] = None
    x509_cert: Optional[str] = None
    
    # Attribute mapping
    attribute_mapping: Dict[str, str] = field(default_factory=lambda: {
        'email': 'http://schemas.xmlsoap.org/ws/2005/05/identity/claims/emailaddress',
        'name': 'http://schemas.xmlsoap.org/ws/2005/05/identity/claims/name',
        'first_name': 'http://schemas.xmlsoap.org/ws/2005/05/identity/claims/givenname',
        'last_name': 'http://schemas.xmlsoap.org/ws/2005/05/identity/claims/surname',
        'user_id': 'http://schemas.xmlsoap.org/ws/2005/05/identity/claims/nameidentifier'
    })
    
    # Settings
    want_assertions_signed: bool = True
    want_assertions_encrypted: bool = False
    allow_unsolicited_responses: bool = False
    force_authn: bool = False
    
    # Service Provider settings
    sp_entity_id: str = ""
    sp_acs_url: str = ""
    sp_x509_cert: Optional[str] = None
    sp_private_key: Optional[str] = None


class SAMLRequest:
    """Represents a SAML authentication request."""
    def __init__(self, provider_config: SAMLProviderConfig):
        self.provider_config = provider_config
        self.id = f"_{secrets.token_hex(20)}"
        self.issue_instant = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    
    def generate_authn_request(self, relay_state: Optional[str] = None) -> str:
        """Generate SAML AuthnRequest XML."""
        # Create AuthnRequest element
        authn_request = f"""<samlp:AuthnRequest
            xmlns:samlp="urn:oasis:names:tc:SAML:2.0:protocol"
            xmlns:saml="urn:oasis:names:tc:SAML:2.0:assertion"
            ID="{self.id}"
            Version="2.0"
            IssueInstant="{self.issue_instant}"
            Destination="{self.provider_config.sso_url}"
            AssertionConsumerServiceURL="{self.provider_config.sp_acs_url}"
            ProtocolBinding="urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST">
            <saml:Issuer>{self.provider_config.sp_entity_id}</saml:Issuer>
            <samlp:NameIDPolicy
                Format="urn:oasis:names:tc:SAML:1.1:nameid-format:emailAddress"
                AllowCreate="true"/>
        </samlp:AuthnRequest>"""
        
        # Base64 encode the request
        encoded_request = base64.b64encode(authn_request.encode('utf-8')).decode('utf-8')
        
        # Build redirect URL
        params = {'SAMLRequest': encoded_request}
        if relay_state:
            params['RelayState'] = relay_state
        
        return f"{self.provider_config.sso_url}?{urlencode(params)}"


class SAMLResponse:
    """Represents and validates a SAML response."""
    def __init__(self, saml_response: str, provider_config: SAMLProviderConfig):
        self.provider_config = provider_config
        self.raw_response = saml_response
        self.decoded_response = base64.b64decode(saml_response).decode('utf-8')
        self.root = ET.fromstring(self.decoded_response)
        
        # Define namespaces
        self.namespaces = {
            'samlp': 'urn:oasis:names:tc:SAML:2.0:protocol',
            'saml': 'urn:oasis:names:tc:SAML:2.0:assertion',
            'ds': 'http://www.w3.org/2000/09/xmldsig#'
        }
    
    def validate(self) -> bool:
        """Validate SAML response."""
        try:
            # Check status
            status = self.root.find('.//samlp:Status/samlp:StatusCode', self.namespaces)
            if status is None or status.get('Value') != 'urn:oasis:names:tc:SAML:2.0:status:Success':
                raise SAMLError("SAML response indicates failure")
            
            # Validate conditions
            assertion = self.root.find('.//saml:Assertion', self.namespaces)
            if assertion is None:
                raise SAMLError("No assertion found in SAML response")
            
            conditions = assertion.find('.//saml:Conditions', self.namespaces)
            if conditions is not None:
                # Check NotBefore and NotOnOrAfter
                not_before = conditions.get('NotBefore')
                not_on_or_after = conditions.get('NotOnOrAfter')
                
                current_time = datetime.utcnow()
                
                if not_before:
                    not_before_time = datetime.strptime(not_before, "%Y-%m-%dT%H:%M:%SZ")
                    if current_time < not_before_time:
                        raise SAMLError("SAML assertion not yet valid")
                
                if not_on_or_after:
                    not_on_or_after_time = datetime.strptime(not_on_or_after, "%Y-%m-%dT%H:%M:%SZ")
                    if current_time >= not_on_or_after_time:
                        raise SAMLError("SAML assertion has expired")
            
            # Validate signature if required
            if self.provider_config.want_assertions_signed:
                # This is a simplified check - in production, use proper XML signature validation
                signature = assertion.find('.//ds:Signature', self.namespaces)
                if signature is None:
                    raise SAMLError("SAML assertion is not signed")
            
            return True
            
        except Exception as e:
            logger.error(f"SAML validation failed: {e}")
            raise
    
    def get_attributes(self) -> Dict[str, Any]:
        """Extract attributes from SAML response."""
        attributes = {}
        
        assertion = self.root.find('.//saml:Assertion', self.namespaces)
        if assertion is None:
            return attributes
        
        # Get NameID
        name_id = assertion.find('.//saml:Subject/saml:NameID', self.namespaces)
        if name_id is not None:
            attributes['name_id'] = name_id.text
            attributes['name_id_format'] = name_id.get('Format', '')
        
        # Get attributes
        attribute_statement = assertion.find('.//saml:AttributeStatement', self.namespaces)
        if attribute_statement is not None:
            for attribute in attribute_statement.findall('.//saml:Attribute', self.namespaces):
                attr_name = attribute.get('Name')
                attr_values = []
                
                for value in attribute.findall('.//saml:AttributeValue', self.namespaces):
                    if value.text:
                        attr_values.append(value.text)
                
                if attr_values:
                    attributes[attr_name] = attr_values[0] if len(attr_values) == 1 else attr_values
        
        return attributes


class SAMLProvider:
    """Handles SAML 2.0 authentication flow."""
    
    def __init__(self, config: Any):
        self.config = config
        self.providers: Dict[str, SAMLProviderConfig] = {}
        self._request_cache: Dict[str, Dict[str, Any]] = {}
    
    async def initialize(self):
        """Initialize SAML provider."""
        logger.info("Initializing SAML provider")
        
        # Load provider configurations from config
        if hasattr(self.config, 'saml_providers'):
            for provider_id, provider_config in self.config.saml_providers.items():
                await self.register_provider(provider_id, provider_config)
        
        logger.info("SAML provider initialized")
    
    async def register_provider(self, provider_id: str, config: Dict[str, Any]):
        """Register a SAML provider."""
        try:
            provider = SAMLProviderConfig(
                name=config.get('name', provider_id),
                entity_id=config['entity_id'],
                sso_url=config['sso_url'],
                slo_url=config.get('slo_url'),
                x509_cert=config.get('x509_cert'),
                attribute_mapping=config.get('attribute_mapping', {}),
                want_assertions_signed=config.get('want_assertions_signed', True),
                want_assertions_encrypted=config.get('want_assertions_encrypted', False),
                allow_unsolicited_responses=config.get('allow_unsolicited_responses', False),
                force_authn=config.get('force_authn', False),
                sp_entity_id=config.get('sp_entity_id', 'voice-agent-platform'),
                sp_acs_url=config.get('sp_acs_url', ''),
                sp_x509_cert=config.get('sp_x509_cert'),
                sp_private_key=config.get('sp_private_key')
            )
            
            self.providers[provider_id] = provider
            logger.info(f"Registered SAML provider: {provider_id}")
            
        except Exception as e:
            logger.error(f"Failed to register SAML provider {provider_id}: {e}")
            raise SAMLError(f"Failed to register provider: {e}")
    
    def get_login_url(
        self,
        provider_id: str,
        relay_state: Optional[str] = None,
        tenant_id: Optional[str] = None
    ) -> str:
        """Generate SAML login URL."""
        if provider_id not in self.providers:
            raise SAMLError(f"Unknown SAML provider: {provider_id}")
        
        provider = self.providers[provider_id]
        
        # Create SAML request
        saml_request = SAMLRequest(provider)
        
        # Store request info for validation
        request_id = saml_request.id
        self._request_cache[request_id] = {
            'provider_id': provider_id,
            'relay_state': relay_state,
            'tenant_id': tenant_id,
            'created_at': datetime.utcnow()
        }
        
        # Add tenant info to relay state
        if tenant_id and relay_state:
            relay_state = f"{relay_state}|tenant:{tenant_id}"
        elif tenant_id:
            relay_state = f"tenant:{tenant_id}"
        
        return saml_request.generate_authn_request(relay_state)
    
    async def authenticate(self, auth_data: Dict[str, Any], tenant_id: Optional[str] = None) -> Optional[User]:
        """
        Authenticate user with SAML response.
        
        Args:
            auth_data: Contains 'SAMLResponse' and optionally 'RelayState'
            tenant_id: Optional tenant ID
            
        Returns:
            Authenticated User object
        """
        try:
            saml_response_data = auth_data.get('SAMLResponse')
            relay_state = auth_data.get('RelayState', '')
            
            if not saml_response_data:
                raise SAMLError("Missing SAML response")
            
            # Extract tenant from relay state if present
            if 'tenant:' in relay_state:
                parts = relay_state.split('|')
                for part in parts:
                    if part.startswith('tenant:'):
                        tenant_id = part.replace('tenant:', '')
            
            # Find provider based on response
            # In production, match based on Issuer in response
            provider_id = self._identify_provider(saml_response_data)
            if not provider_id:
                raise SAMLError("Could not identify SAML provider")
            
            provider = self.providers[provider_id]
            
            # Parse and validate response
            saml_response = SAMLResponse(saml_response_data, provider)
            if not saml_response.validate():
                raise SAMLError("SAML response validation failed")
            
            # Extract attributes
            attributes = saml_response.get_attributes()
            
            # Create or update user
            user = await self._create_or_update_user(
                provider_id,
                provider,
                attributes,
                tenant_id
            )
            
            # Clean up old request cache entries
            self._cleanup_request_cache()
            
            return user
            
        except Exception as e:
            logger.error(f"SAML authentication failed: {e}")
            raise
    
    def _identify_provider(self, saml_response_data: str) -> Optional[str]:
        """Identify provider from SAML response."""
        try:
            decoded = base64.b64decode(saml_response_data).decode('utf-8')
            root = ET.fromstring(decoded)
            
            # Look for Issuer
            issuer = root.find('.//{urn:oasis:names:tc:SAML:2.0:assertion}Issuer')
            if issuer is not None and issuer.text:
                # Match issuer against registered providers
                for provider_id, provider in self.providers.items():
                    if provider.entity_id == issuer.text:
                        return provider_id
            
            # If no match, return first provider (for testing)
            return list(self.providers.keys())[0] if self.providers else None
            
        except Exception as e:
            logger.error(f"Failed to identify SAML provider: {e}")
            return None
    
    async def _create_or_update_user(
        self,
        provider_id: str,
        provider: SAMLProviderConfig,
        attributes: Dict[str, Any],
        tenant_id: Optional[str] = None
    ) -> User:
        """Create or update user from SAML attributes."""
        # Map attributes using provider's mapping
        email = None
        name = None
        user_id = attributes.get('name_id')
        
        # Try to extract attributes using mapping
        for local_attr, saml_attr in provider.attribute_mapping.items():
            if saml_attr in attributes:
                if local_attr == 'email':
                    email = attributes[saml_attr]
                elif local_attr == 'name':
                    name = attributes[saml_attr]
                elif local_attr == 'user_id' and not user_id:
                    user_id = attributes[saml_attr]
        
        # Fallback to common attribute names
        if not email:
            email = attributes.get('email') or attributes.get('mail')
        if not name:
            name = attributes.get('name') or attributes.get('displayName')
        
        if not user_id:
            raise SAMLError("No user identifier found in SAML response")
        
        # Create user object
        user = User(
            email=email or f"{user_id}@{provider_id}",
            username=email or user_id,
            full_name=name or email or user_id,
            auth_provider=AuthProvider.SAML,
            auth_provider_id=f"{provider_id}:{user_id}",
            tenant_id=tenant_id
        )
        
        # Store SAML attributes in profile
        user.profile['saml_attributes'] = attributes
        user.profile['saml_session_index'] = attributes.get('session_index')
        
        return user
    
    def generate_logout_request(
        self,
        provider_id: str,
        name_id: str,
        session_index: Optional[str] = None
    ) -> str:
        """Generate SAML logout request."""
        if provider_id not in self.providers:
            raise SAMLError(f"Unknown SAML provider: {provider_id}")
        
        provider = self.providers[provider_id]
        
        if not provider.slo_url:
            raise SAMLError("Provider does not support single logout")
        
        # Create LogoutRequest
        logout_id = f"_{secrets.token_hex(20)}"
        issue_instant = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        
        logout_request = f"""<samlp:LogoutRequest
            xmlns:samlp="urn:oasis:names:tc:SAML:2.0:protocol"
            xmlns:saml="urn:oasis:names:tc:SAML:2.0:assertion"
            ID="{logout_id}"
            Version="2.0"
            IssueInstant="{issue_instant}"
            Destination="{provider.slo_url}">
            <saml:Issuer>{provider.sp_entity_id}</saml:Issuer>
            <saml:NameID Format="urn:oasis:names:tc:SAML:1.1:nameid-format:emailAddress">{name_id}</saml:NameID>"""
        
        if session_index:
            logout_request += f"<samlp:SessionIndex>{session_index}</samlp:SessionIndex>"
        
        logout_request += "</samlp:LogoutRequest>"
        
        # Base64 encode
        encoded_request = base64.b64encode(logout_request.encode('utf-8')).decode('utf-8')
        
        return f"{provider.slo_url}?SAMLRequest={encoded_request}"
    
    def _cleanup_request_cache(self):
        """Remove old entries from request cache."""
        cutoff_time = datetime.utcnow() - timedelta(minutes=10)
        
        expired_keys = [
            key for key, data in self._request_cache.items()
            if data['created_at'] < cutoff_time
        ]
        
        for key in expired_keys:
            del self._request_cache[key]
    
    def get_metadata(self, provider_id: Optional[str] = None) -> str:
        """Generate SP metadata for SAML configuration."""
        # Generate service provider metadata
        # This is a simplified version - extend as needed
        
        sp_entity_id = "voice-agent-platform"
        acs_url = f"{self.config.base_url}/auth/saml/acs"
        
        metadata = f"""<?xml version="1.0" encoding="UTF-8"?>
<EntityDescriptor
    xmlns="urn:oasis:names:tc:SAML:2.0:metadata"
    entityID="{sp_entity_id}">
    <SPSSODescriptor
        AuthnRequestsSigned="false"
        WantAssertionsSigned="true"
        protocolSupportEnumeration="urn:oasis:names:tc:SAML:2.0:protocol">
        <NameIDFormat>urn:oasis:names:tc:SAML:1.1:nameid-format:emailAddress</NameIDFormat>
        <AssertionConsumerService
            Binding="urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST"
            Location="{acs_url}"
            index="1"/>
    </SPSSODescriptor>
</EntityDescriptor>"""
        
        return metadata
    
    async def cleanup(self):
        """Clean up resources."""
        self._request_cache.clear()


# Common SAML provider configurations
SAML_PROVIDERS = {
    "okta": {
        "name": "Okta",
        "attribute_mapping": {
            "email": "http://schemas.xmlsoap.org/ws/2005/05/identity/claims/emailaddress",
            "name": "http://schemas.xmlsoap.org/ws/2005/05/identity/claims/name",
            "first_name": "http://schemas.xmlsoap.org/ws/2005/05/identity/claims/givenname",
            "last_name": "http://schemas.xmlsoap.org/ws/2005/05/identity/claims/surname"
        }
    },
    "azure_ad": {
        "name": "Azure AD",
        "attribute_mapping": {
            "email": "http://schemas.xmlsoap.org/ws/2005/05/identity/claims/emailaddress",
            "name": "http://schemas.microsoft.com/identity/claims/displayname",
            "user_id": "http://schemas.microsoft.com/identity/claims/objectidentifier"
        }
    },
    "ping_identity": {
        "name": "PingIdentity",
        "attribute_mapping": {
            "email": "email",
            "name": "displayName",
            "user_id": "uid"
        }
    }
}