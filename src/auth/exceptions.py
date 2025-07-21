"""
Authentication and Authorization Exceptions

Custom exceptions for the authentication system.
"""


class AuthException(Exception):
    """Base exception for authentication system."""
    pass


class AuthenticationError(AuthException):
    """General authentication error."""
    pass


class AuthorizationError(AuthException):
    """Authorization/permission denied error."""
    pass


class InvalidCredentialsError(AuthenticationError):
    """Invalid username/password or credentials."""
    pass


class AccountLockedError(AuthenticationError):
    """Account is locked due to security reasons."""
    pass


class AccountInactiveError(AuthenticationError):
    """Account is inactive or suspended."""
    pass


class SessionExpiredError(AuthenticationError):
    """Session has expired."""
    pass


class InvalidTokenError(AuthenticationError):
    """Invalid authentication token."""
    pass


class TenantInactiveError(AuthenticationError):
    """Tenant is inactive or expired."""
    pass


class PermissionDeniedError(AuthorizationError):
    """User lacks required permissions."""
    pass


class RateLimitExceededError(AuthException):
    """Rate limit has been exceeded."""
    pass


class MFARequiredError(AuthenticationError):
    """Multi-factor authentication is required."""
    pass


class InvalidMFACodeError(AuthenticationError):
    """Invalid MFA code provided."""
    pass


class PasswordPolicyError(AuthException):
    """Password does not meet policy requirements."""
    pass


class ApiKeyError(AuthException):
    """API key related error."""
    pass


class InvalidApiKeyError(ApiKeyError):
    """Invalid or expired API key."""
    pass


class OAuthError(AuthException):
    """OAuth authentication error."""
    pass


class SAMLError(AuthException):
    """SAML authentication error."""
    pass


class ConfigurationError(AuthException):
    """Authentication configuration error."""
    pass