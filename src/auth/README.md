# Enterprise Authentication & Authorization System

A comprehensive authentication and authorization system for the Voice Agent Platform, providing enterprise-grade security features including OAuth/SAML SSO, RBAC, API key management, and multi-tenant support.

## Features

### 🔐 Authentication Providers
- **Local Authentication**: Traditional username/password with secure hashing
- **OAuth 2.0 SSO**: Support for Google, Microsoft, GitHub, Okta, and custom providers
- **SAML SSO**: Enterprise SAML 2.0 integration for identity providers
- **API Key Authentication**: Programmatic access with scoped permissions
- **Multi-factor Authentication**: TOTP-based MFA support

### 🛡️ Security Features
- **Role-Based Access Control (RBAC)**: Fine-grained permission system
- **Session Management**: Secure session handling with security monitoring
- **Rate Limiting**: Prevent brute force and abuse
- **Account Lockout**: Automatic protection against failed login attempts  
- **Audit Logging**: Comprehensive security event logging
- **IP Restrictions**: Allow/deny lists for enhanced security

### 🏢 Multi-Tenant Support
- **Tenant Isolation**: Complete data separation between tenants
- **Resource Quotas**: Configurable limits per tenant
- **Feature Flags**: Enable/disable features per tenant
- **Custom Branding**: Tenant-specific configurations

### 📊 Administration
- **Web Admin Interface**: User and role management
- **API Management**: Create, revoke, and monitor API keys
- **Security Dashboard**: Real-time security monitoring
- **Audit Reports**: Compliance and security reporting

## Quick Start

### Installation

```bash
# Install required dependencies
pip install fastapi uvicorn python-jose[cryptography] passlib[bcrypt] httpx
```

### Basic Setup

```python
from src.auth import (
    AuthManager, AuthConfig, 
    create_authenticated_app,
    initialize_auth_system
)

# Create FastAPI app with authentication
app = create_authenticated_app()

# Or manually initialize
async def startup():
    await initialize_auth_system()
    
app.add_event_handler("startup", startup)
```

### Create Your First User

```python
from src.auth import get_auth_manager

auth_manager = get_auth_manager()

# Create admin user
admin_user = await auth_manager.create_user(
    email="admin@company.com",
    username="admin",
    full_name="System Administrator",
    password="SecurePassword123!",
    roles=["Super Administrator"]
)
```

## Architecture Overview

### Core Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   AuthManager   │────│   RBACManager   │────│  TenantManager  │
│                 │    │                 │    │                 │
│ • Authentication│    │ • Roles & Perms │    │ • Multi-tenant  │
│ • User Management│    │ • Access Control│    │ • Quotas        │
│ • Provider      │    │ • Policies      │    │ • Isolation     │
│   Coordination  │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
         ┌─────────────────┐     │     ┌─────────────────┐
         │ SessionManager  │─────┼─────│  ApiKeyManager  │
         │                 │     │     │                 │
         │ • Session State │     │     │ • API Keys      │
         │ • Security      │     │     │ • Rate Limiting │
         │ • Monitoring    │     │     │ • Scopes        │
         └─────────────────┘     │     └─────────────────┘
                                 │
                    ┌─────────────────┐
                    │  AuditLogger    │
                    │                 │
                    │ • Event Logging │
                    │ • Security Mon. │
                    │ • Compliance    │
                    └─────────────────┘
```

### Authentication Flow

```mermaid
sequenceDiagram
    participant Client
    participant API
    participant AuthManager
    participant Provider
    participant Database

    Client->>API: Login Request
    API->>AuthManager: authenticate(credentials)
    AuthManager->>Provider: validate_credentials()
    Provider->>Database: check_user()
    Database-->>Provider: user_data
    Provider-->>AuthManager: user_validated
    AuthManager->>SessionManager: create_session()
    SessionManager-->>AuthManager: session
    AuthManager->>AuditLogger: log_event(login_success)
    AuthManager-->>API: user + session
    API-->>Client: JWT Token + User Info
```

## Configuration

### Environment Variables

```bash
# Authentication
AUTH_JWT_SECRET=your-super-secret-jwt-key
AUTH_JWT_EXPIRY_HOURS=24
AUTH_SESSION_TIMEOUT_MINUTES=30
AUTH_MAX_FAILED_ATTEMPTS=5

# OAuth Configuration
OAUTH_GOOGLE_CLIENT_ID=your-google-client-id
OAUTH_GOOGLE_CLIENT_SECRET=your-google-client-secret

# SAML Configuration  
SAML_ENTITY_ID=voice-agent-platform
SAML_SSO_URL=https://your-idp.com/sso
SAML_X509_CERT=path/to/certificate.pem

# Database
DATABASE_URL=postgresql://user:pass@localhost/voiceagent

# Redis (for sessions)
REDIS_URL=redis://localhost:6379
```

### Python Configuration

```python
from src.auth import AuthConfig

config = AuthConfig()
config.jwt_expiry_hours = 24
config.max_failed_attempts = 5
config.require_mfa_for_admin = True
config.password_min_length = 12

# OAuth providers
config.oauth_providers = {
    'google': {
        'client_id': 'your-client-id',
        'client_secret': 'your-client-secret',
        'authorize_url': 'https://accounts.google.com/o/oauth2/v2/auth',
        'token_url': 'https://oauth2.googleapis.com/token',
        'userinfo_url': 'https://www.googleapis.com/oauth2/v3/userinfo',
        'scope': ['openid', 'profile', 'email']
    }
}

# SAML providers
config.saml_providers = {
    'corporate_sso': {
        'entity_id': 'corporate-identity-provider',
        'sso_url': 'https://sso.company.com/saml/login',
        'x509_cert': 'path/to/cert.pem',
        'attribute_mapping': {
            'email': 'http://schemas.xmlsoap.org/ws/2005/05/identity/claims/emailaddress',
            'name': 'http://schemas.xmlsoap.org/ws/2005/05/identity/claims/name'
        }
    }
}
```

## Usage Examples

### User Authentication

```python
from src.auth import get_auth_manager
from src.auth.models import AuthProvider

auth_manager = get_auth_manager()

# Local authentication
user, session = await auth_manager.authenticate({
    'email': 'user@company.com',
    'password': 'userpassword',
    'ip_address': '192.168.1.100',
    'user_agent': 'Mozilla/5.0...'
})

# OAuth authentication
oauth_url = auth_manager.oauth_provider.get_authorization_url(
    'google',
    redirect_uri='https://yourapp.com/auth/callback',
    tenant_id='company_tenant'
)

# Handle OAuth callback
user, session = await auth_manager.authenticate({
    'provider_id': 'google',
    'code': 'oauth_code_from_callback',
    'state': 'csrf_state_token',
    'redirect_uri': 'https://yourapp.com/auth/callback'
}, auth_type=AuthProvider.OAUTH)
```

### Role-Based Access Control

```python
from src.auth import get_rbac_manager
from src.auth.models import PermissionType

rbac_manager = get_rbac_manager()

# Create custom role
role = await rbac_manager.create_role(
    name="Voice Agent Operator",
    description="Can manage voice agents and sessions",
    permissions=[
        PermissionType.CREATE_AGENT,
        PermissionType.READ_AGENT,
        PermissionType.UPDATE_AGENT,
        PermissionType.CREATE_SESSION,
        PermissionType.READ_SESSION
    ]
)

# Assign role to user
await rbac_manager.assign_role_to_user(user, role.id)

# Check permissions
if await rbac_manager.check_permission(user, PermissionType.CREATE_AGENT):
    # User can create agents
    pass

# Require permission (raises exception if denied)
await rbac_manager.require_permission(user, PermissionType.DELETE_AGENT)
```

### API Key Management

```python
from src.auth import get_auth_manager
from src.auth.models import PermissionType

auth_manager = get_auth_manager()
api_key_manager = auth_manager.api_key_manager

# Create API key
api_key, key_string = await api_key_manager.create_api_key(
    name="Integration API Key",
    description="For third-party integration",
    user_id=user.id,
    tenant_id=user.tenant_id,
    scopes=['voice_agents', 'sessions'],
    permissions=[PermissionType.CREATE_AGENT, PermissionType.READ_AGENT],
    allowed_ips=['192.168.1.0/24'],
    rate_limit=1000,  # requests per minute
    expires_in_days=365
)

# Use API key for authentication
api_user = await api_key_manager.authenticate(key_string)
```

### Multi-Tenant Setup

```python
from src.auth import get_tenant_manager

tenant_manager = get_tenant_manager()

# Create tenant
tenant = await tenant_manager.create_tenant(
    name="Acme Corporation",
    slug="acme-corp",
    limits={
        'max_users': 100,
        'max_agents': 50,
        'max_sessions_per_day': 10000,
        'max_storage_gb': 500
    },
    features={'voice_agents', 'session_recording', 'analytics', 'api_access'},
    expires_in_days=365
)

# Check quota before resource creation
if await tenant_manager.check_tenant_quota(tenant.id, 'users'):
    # Create user
    await tenant_manager.consume_quota(tenant.id, 'users')
    user = await auth_manager.create_user(...)
```

### FastAPI Integration

```python
from fastapi import FastAPI, Depends
from src.auth.fastapi_integration import (
    require_authentication,
    require_permission,
    get_current_user
)
from src.auth.models import PermissionType, User

app = FastAPI()

@app.get("/protected")
async def protected_endpoint(
    user: User = Depends(require_authentication)
):
    return {"message": f"Hello {user.full_name}!"}

@app.post("/agents")
async def create_agent(
    agent_data: dict,
    user: User = Depends(require_permission(PermissionType.CREATE_AGENT))
):
    # User has CREATE_AGENT permission
    return {"message": "Agent created"}

@app.get("/admin/users")
async def list_users(
    user: User = Depends(require_permission(PermissionType.READ_USER))
):
    # Admin endpoint - requires READ_USER permission
    admin = get_admin_interface()
    return await admin.list_users(user)
```

### Audit Logging

```python
from src.auth import get_auth_manager

auth_manager = get_auth_manager()
audit_logger = auth_manager.audit_logger

# Manual event logging
await audit_logger.log_security_event(
    "suspicious_activity",
    user_id=user.id,
    ip_address="192.168.1.100",
    details={"reason": "Multiple failed API calls"}
)

# Query audit events
events = await audit_logger.query_events(
    start_time=datetime.now() - timedelta(days=7),
    user_id=user.id,
    event_types=[AuditEventType.LOGIN_FAILURE],
    limit=100
)

# Generate security report
report = await audit_logger.generate_security_report(
    start_time=datetime.now() - timedelta(days=30),
    end_time=datetime.now(),
    tenant_id=user.tenant_id
)
```

## Security Best Practices

### Password Policy

```python
# Strong password requirements (enforced by default)
# - Minimum 12 characters
# - Must contain uppercase, lowercase, numbers, and symbols
# - Cannot be common passwords
# - Cannot reuse last 5 passwords

config.password_min_length = 12
config.password_require_complexity = True
config.password_history_count = 5
```

### Session Security

```python
# Secure session configuration
config.session_timeout_minutes = 30
config.max_sessions_per_user = 5
config.allow_concurrent_sessions = True

# Session security monitoring
# - Detects IP address changes
# - Monitors for suspicious activity
# - Automatic session invalidation on security violations
```

### API Key Security

```python
# API key best practices
api_key, key_string = await api_key_manager.create_api_key(
    name="Production Integration",
    allowed_ips=['10.0.0.0/24'],  # Restrict to specific networks
    allowed_origins=['https://api.company.com'],  # CORS restrictions
    rate_limit=500,  # Prevent abuse
    expires_in_days=90,  # Regular rotation
    scopes=['read_only']  # Minimal required permissions
)
```

### Tenant Isolation

```python
# Ensure complete tenant data isolation
@app.get("/agents/{agent_id}")
async def get_agent(
    agent_id: str,
    user: User = Depends(require_authentication),
    tenant_context = Depends(get_tenant_context)
):
    # Automatically filtered by user's tenant
    agents = tenant_manager.filter_by_tenant([agent], user)
    if not agents:
        raise HTTPException(404, "Agent not found")
    return agents[0]
```

## Production Deployment

### Database Setup

```sql
-- PostgreSQL schema for production
CREATE DATABASE voiceagent_auth;

-- Users table
CREATE TABLE auth_users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    username VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255),
    tenant_id UUID,
    status VARCHAR(20) DEFAULT 'active',
    created_at TIMESTAMP DEFAULT NOW()
);

-- Sessions table
CREATE TABLE auth_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    token VARCHAR(255) UNIQUE NOT NULL,
    user_id UUID REFERENCES auth_users(id),
    expires_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Audit events table
CREATE TABLE auth_audit_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_type VARCHAR(100) NOT NULL,
    user_id UUID,
    tenant_id UUID,
    ip_address INET,
    details JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX idx_auth_users_email ON auth_users(email);
CREATE INDEX idx_auth_users_tenant ON auth_users(tenant_id);
CREATE INDEX idx_auth_sessions_token ON auth_sessions(token);
CREATE INDEX idx_auth_sessions_user ON auth_sessions(user_id);
CREATE INDEX idx_audit_events_type ON auth_audit_events(event_type);
CREATE INDEX idx_audit_events_user ON auth_audit_events(user_id);
CREATE INDEX idx_audit_events_created ON auth_audit_events(created_at);
```

### Redis Configuration

```bash
# Redis for session storage
redis-server --port 6379 --save 900 1 --save 300 10 --save 60 10000

# Redis configuration for production
maxmemory 2gb
maxmemory-policy allkeys-lru
tcp-keepalive 300
timeout 0
```

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY backend/ ./backend/

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Environment Configuration

```yaml
# docker-compose.yml
version: '3.8'
services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@postgres:5432/voiceagent
      - REDIS_URL=redis://redis:6379
      - AUTH_JWT_SECRET=${JWT_SECRET}
    depends_on:
      - postgres
      - redis

  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: voiceagent
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
```

## Monitoring and Alerts

### Security Monitoring

```python
# Set up security event monitoring
from src.auth.audit_logger import SecurityEventAnalyzer

# Automatic security analysis
analyzer = SecurityEventAnalyzer(audit_logger)

# Custom alert handlers
async def security_alert_handler(event):
    if event.severity == AuditSeverity.CRITICAL:
        # Send to security team
        await send_slack_alert(f"CRITICAL: {event.event_type}")
        await send_email_alert(event)

audit_logger.add_event_processor(security_alert_handler)
```

### Metrics and Dashboards

```python
# Prometheus metrics
from prometheus_client import Counter, Histogram, Gauge

auth_requests_total = Counter('auth_requests_total', 'Total authentication requests', ['method', 'status'])
auth_duration = Histogram('auth_duration_seconds', 'Authentication duration')
active_sessions = Gauge('auth_active_sessions', 'Number of active sessions')
```

## Troubleshooting

### Common Issues

1. **JWT Token Expired**
   ```python
   # Solution: Implement token refresh
   try:
       payload = await auth_manager.validate_jwt_token(token)
   except SessionExpiredError:
       # Redirect to login or refresh token
       pass
   ```

2. **Rate Limiting Triggered**
   ```python
   # Check rate limit status
   if not await auth_manager._check_rate_limit(user_id):
       # Wait or implement backoff
       pass
   ```

3. **OAuth Callback Errors**
   ```python
   # Validate OAuth state parameter
   state_data = oauth_provider.state_manager.verify_state(state)
   if not state_data:
       raise ValueError("Invalid OAuth state")
   ```

### Debugging

```python
# Enable debug logging
import logging
logging.getLogger('src.auth').setLevel(logging.DEBUG)

# Check audit logs
events = await audit_logger.query_events(
    user_id=user.id,
    event_types=[AuditEventType.LOGIN_FAILURE],
    limit=10
)
```

## API Reference

See the individual module documentation:

- [AuthManager API](./auth_manager.py)
- [RBACManager API](./rbac_manager.py)
- [SessionManager API](./session_manager.py)
- [TenantManager API](./tenant_manager.py)
- [ApiKeyManager API](./api_key_manager.py)
- [AuditLogger API](./audit_logger.py)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run tests
pytest src/auth/tests/

# Run with coverage
pytest --cov=src.auth src/auth/tests/
```

## License

This authentication system is part of the Voice Agent Platform and follows the same license terms.

## Support

For enterprise support and custom integrations, contact the development team.