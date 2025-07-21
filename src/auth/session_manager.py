"""
Session Management System

Manages authenticated sessions with security features and monitoring.
"""

import asyncio
import logging
import secrets
import time
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import json
import hashlib

from .models import Session, User
from .exceptions import SessionExpiredError, InvalidTokenError

logger = logging.getLogger(__name__)


class SessionStore:
    """Abstract session storage interface."""
    
    async def get(self, session_token: str) -> Optional[Session]:
        raise NotImplementedError
    
    async def set(self, session_token: str, session: Session, ttl: Optional[int] = None):
        raise NotImplementedError
    
    async def delete(self, session_token: str):
        raise NotImplementedError
    
    async def list_user_sessions(self, user_id: str) -> List[Session]:
        raise NotImplementedError
    
    async def cleanup_expired(self):
        raise NotImplementedError


class MemorySessionStore(SessionStore):
    """In-memory session storage (for development/testing)."""
    
    def __init__(self):
        self._sessions: Dict[str, Session] = {}
        self._user_sessions: Dict[str, List[str]] = {}  # user_id -> session_tokens
    
    async def get(self, session_token: str) -> Optional[Session]:
        session = self._sessions.get(session_token)
        if session and session.is_valid():
            return session
        elif session:
            # Remove expired session
            await self.delete(session_token)
        return None
    
    async def set(self, session_token: str, session: Session, ttl: Optional[int] = None):
        self._sessions[session_token] = session
        
        # Update user sessions index
        if session.user_id not in self._user_sessions:
            self._user_sessions[session.user_id] = []
        
        if session_token not in self._user_sessions[session.user_id]:
            self._user_sessions[session.user_id].append(session_token)
    
    async def delete(self, session_token: str):
        session = self._sessions.get(session_token)
        if session:
            # Remove from main storage
            del self._sessions[session_token]
            
            # Remove from user sessions index
            if session.user_id in self._user_sessions:
                try:
                    self._user_sessions[session.user_id].remove(session_token)
                    if not self._user_sessions[session.user_id]:
                        del self._user_sessions[session.user_id]
                except ValueError:
                    pass
    
    async def list_user_sessions(self, user_id: str) -> List[Session]:
        session_tokens = self._user_sessions.get(user_id, [])
        sessions = []
        
        for token in session_tokens[:]:  # Copy to avoid modification during iteration
            session = await self.get(token)
            if session:
                sessions.append(session)
            else:
                # Clean up stale reference
                try:
                    self._user_sessions[user_id].remove(token)
                except ValueError:
                    pass
        
        return sessions
    
    async def cleanup_expired(self):
        current_time = datetime.utcnow()
        expired_tokens = []
        
        for token, session in self._sessions.items():
            if current_time >= session.expires_at:
                expired_tokens.append(token)
        
        for token in expired_tokens:
            await self.delete(token)


class SessionSecurity:
    """Security features for session management."""
    
    def __init__(self):
        self.failed_attempts: Dict[str, List[datetime]] = {}
        self.suspicious_activity: Dict[str, List[Dict[str, Any]]] = {}
    
    def check_session_security(
        self,
        session: Session,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> Dict[str, Any]:
        """Check session for security issues."""
        issues = []
        risk_score = 0
        
        # Check for IP address changes
        if ip_address and session.ip_address and ip_address != session.ip_address:
            issues.append("ip_address_changed")
            risk_score += 30
        
        # Check for user agent changes
        if (user_agent and session.user_agent and 
            user_agent != session.user_agent):
            issues.append("user_agent_changed")
            risk_score += 20
        
        # Check session age
        session_age = datetime.utcnow() - session.created_at
        if session_age > timedelta(hours=24):
            issues.append("long_session")
            risk_score += 10
        
        # Check for concurrent sessions
        # This would require access to session store
        
        return {
            'issues': issues,
            'risk_score': risk_score,
            'requires_reauth': risk_score > 50
        }
    
    def record_suspicious_activity(
        self,
        session_id: str,
        activity_type: str,
        details: Dict[str, Any]
    ):
        """Record suspicious activity for a session."""
        if session_id not in self.suspicious_activity:
            self.suspicious_activity[session_id] = []
        
        self.suspicious_activity[session_id].append({
            'type': activity_type,
            'timestamp': datetime.utcnow(),
            'details': details
        })
        
        # Keep only last 10 activities
        self.suspicious_activity[session_id] = \
            self.suspicious_activity[session_id][-10:]


class SessionManager:
    """Manages user sessions with security and monitoring."""
    
    def __init__(self, config: Any):
        self.config = config
        
        # Session storage
        self._session_store = MemorySessionStore()  # Use Redis in production
        
        # Security monitoring
        self._security = SessionSecurity()
        
        # Session configuration
        self.default_timeout = timedelta(minutes=getattr(config, 'session_timeout_minutes', 30))
        self.max_sessions_per_user = getattr(config, 'max_sessions_per_user', 5)
        self.allow_concurrent_sessions = getattr(config, 'allow_concurrent_sessions', True)
        
        # Background cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
    
    async def initialize(self):
        """Initialize session manager."""
        logger.info("Initializing session manager")
        
        # Start background cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info("Session manager initialized")
    
    async def create_session(
        self,
        user_id: str,
        tenant_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        device_id: Optional[str] = None,
        security_level: str = "standard",
        session_duration: Optional[timedelta] = None
    ) -> Session:
        """
        Create a new authenticated session.
        
        Args:
            user_id: User ID
            tenant_id: Tenant ID
            ip_address: Client IP address
            user_agent: Client user agent
            device_id: Device identifier
            security_level: Security level (standard, elevated, privileged)
            session_duration: Custom session duration
            
        Returns:
            New Session object
        """
        try:
            # Check concurrent session limit
            if not self.allow_concurrent_sessions:
                await self._invalidate_user_sessions(user_id, "new_session_created")
            else:
                existing_sessions = await self._session_store.list_user_sessions(user_id)
                if len(existing_sessions) >= self.max_sessions_per_user:
                    # Remove oldest session
                    oldest_session = min(existing_sessions, key=lambda s: s.created_at)
                    await self.invalidate_session(oldest_session.token)
            
            # Generate secure session token
            token = self._generate_session_token()
            
            # Create session
            session = Session(
                token=token,
                user_id=user_id,
                tenant_id=tenant_id,
                ip_address=ip_address,
                user_agent=user_agent,
                device_id=device_id,
                security_level=security_level
            )
            
            # Set expiry
            duration = session_duration or self.default_timeout
            session.expires_at = datetime.utcnow() + duration
            
            # Store session
            await self._session_store.set(token, session)
            
            logger.info(f"Created session for user {user_id}: {session.id}")
            return session
            
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            raise
    
    def _generate_session_token(self) -> str:
        """Generate a secure session token."""
        # Generate random token
        token = secrets.token_urlsafe(48)
        
        # Add timestamp and hash for additional entropy
        timestamp = str(int(time.time()))
        combined = f"{token}:{timestamp}"
        hash_digest = hashlib.sha256(combined.encode()).hexdigest()[:16]
        
        return f"sess_{token}_{hash_digest}"
    
    async def get_session(self, session_token: str) -> Optional[Session]:
        """Get session by token."""
        try:
            return await self._session_store.get(session_token)
        except Exception as e:
            logger.error(f"Failed to get session: {e}")
            return None
    
    async def validate_session(
        self,
        session_token: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        required_security_level: Optional[str] = None
    ) -> Session:
        """
        Validate session and check security requirements.
        
        Args:
            session_token: Session token
            ip_address: Current IP address for security check
            user_agent: Current user agent for security check
            required_security_level: Required security level
            
        Returns:
            Valid Session object
            
        Raises:
            SessionExpiredError: If session is invalid or expired
            InvalidTokenError: If security requirements not met
        """
        session = await self.get_session(session_token)
        if not session:
            raise SessionExpiredError("Session not found or expired")
        
        # Security checks
        security_check = self._security.check_session_security(
            session, ip_address, user_agent
        )
        
        if security_check['requires_reauth']:
            await self.invalidate_session(session_token)
            raise InvalidTokenError("Session requires re-authentication due to security concerns")
        
        # Check security level
        if required_security_level:
            level_order = {'standard': 1, 'elevated': 2, 'privileged': 3}
            required_level = level_order.get(required_security_level, 1)
            session_level = level_order.get(session.security_level, 1)
            
            if session_level < required_level:
                raise InvalidTokenError(f"Session security level insufficient: {session.security_level} < {required_security_level}")
        
        # Record suspicious activity if any
        if security_check['issues']:
            self._security.record_suspicious_activity(
                session.id,
                'security_check',
                {
                    'issues': security_check['issues'],
                    'ip_address': ip_address,
                    'user_agent': user_agent
                }
            )
        
        return session
    
    async def refresh_session(
        self,
        session_token: str,
        extend_by: Optional[timedelta] = None
    ) -> Session:
        """
        Refresh session expiry.
        
        Args:
            session_token: Session token
            extend_by: Time to extend session (default: configured timeout)
            
        Returns:
            Updated Session object
        """
        session = await self.get_session(session_token)
        if not session:
            raise SessionExpiredError("Session not found")
        
        # Extend session
        extension = extend_by or self.default_timeout
        session.refresh(extension.total_seconds() / 3600)  # Convert to hours
        
        # Update in storage
        await self._session_store.set(session_token, session)
        
        return session
    
    async def invalidate_session(self, session_token: str, reason: str = ""):
        """Invalidate a specific session."""
        try:
            session = await self.get_session(session_token)
            if session:
                await self._session_store.delete(session_token)
                logger.info(f"Invalidated session {session.id} (reason: {reason})")
        except Exception as e:
            logger.error(f"Failed to invalidate session: {e}")
    
    async def invalidate_user_sessions(
        self,
        user_id: str,
        except_session: Optional[str] = None,
        reason: str = ""
    ):
        """Invalidate all sessions for a user."""
        await self._invalidate_user_sessions(user_id, reason, except_session)
    
    async def _invalidate_user_sessions(
        self,
        user_id: str,
        reason: str = "",
        except_session: Optional[str] = None
    ):
        """Internal method to invalidate user sessions."""
        try:
            sessions = await self._session_store.list_user_sessions(user_id)
            
            for session in sessions:
                if except_session and session.token == except_session:
                    continue
                
                await self._session_store.delete(session.token)
                logger.info(f"Invalidated session {session.id} for user {user_id} (reason: {reason})")
        
        except Exception as e:
            logger.error(f"Failed to invalidate user sessions: {e}")
    
    async def get_user_sessions(self, user_id: str) -> List[Session]:
        """Get all active sessions for a user."""
        try:
            return await self._session_store.list_user_sessions(user_id)
        except Exception as e:
            logger.error(f"Failed to get user sessions: {e}")
            return []
    
    async def update_session_data(
        self,
        session_token: str,
        data: Dict[str, Any]
    ) -> Optional[Session]:
        """Update session data."""
        session = await self.get_session(session_token)
        if not session:
            return None
        
        # Update data
        session.data.update(data)
        session.last_activity_at = datetime.utcnow()
        
        # Store updated session
        await self._session_store.set(session_token, session)
        
        return session
    
    async def elevate_session_security(
        self,
        session_token: str,
        new_level: str,
        mfa_verified: bool = False
    ) -> Optional[Session]:
        """
        Elevate session security level.
        
        Args:
            session_token: Session token
            new_level: New security level (elevated, privileged)
            mfa_verified: Whether MFA was verified
            
        Returns:
            Updated Session object
        """
        session = await self.get_session(session_token)
        if not session:
            return None
        
        level_order = {'standard': 1, 'elevated': 2, 'privileged': 3}
        current_level = level_order.get(session.security_level, 1)
        target_level = level_order.get(new_level, 1)
        
        # Can only elevate, not downgrade
        if target_level <= current_level:
            return session
        
        # For privileged access, MFA is required
        if new_level == 'privileged' and not mfa_verified:
            raise InvalidTokenError("MFA verification required for privileged access")
        
        # Update security level
        session.security_level = new_level
        session.is_mfa_verified = mfa_verified
        session.last_activity_at = datetime.utcnow()
        
        # Store updated session
        await self._session_store.set(session_token, session)
        
        logger.info(f"Elevated session {session.id} security to {new_level}")
        return session
    
    def get_session_statistics(self) -> Dict[str, Any]:
        """Get session statistics."""
        # This would be more comprehensive with proper storage
        return {
            'total_sessions': len(self._session_store._sessions),
            'active_sessions': len([
                s for s in self._session_store._sessions.values()
                if s.is_valid()
            ])
        }
    
    async def _cleanup_loop(self):
        """Background task to clean up expired sessions."""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                await self._session_store.cleanup_expired()
                
                # Clean up security data
                cutoff_time = datetime.utcnow() - timedelta(hours=24)
                
                # Clean up failed attempts
                for key in list(self._security.failed_attempts.keys()):
                    self._security.failed_attempts[key] = [
                        ts for ts in self._security.failed_attempts[key]
                        if ts > cutoff_time
                    ]
                    if not self._security.failed_attempts[key]:
                        del self._security.failed_attempts[key]
                
                # Clean up old suspicious activity
                for session_id in list(self._security.suspicious_activity.keys()):
                    activities = self._security.suspicious_activity[session_id]
                    recent_activities = [
                        activity for activity in activities
                        if activity['timestamp'] > cutoff_time
                    ]
                    
                    if recent_activities:
                        self._security.suspicious_activity[session_id] = recent_activities
                    else:
                        del self._security.suspicious_activity[session_id]
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Session cleanup error: {e}")
    
    async def cleanup(self):
        """Clean up resources."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        await self._session_store.cleanup_expired()


# Session middleware for web frameworks
class SessionMiddleware:
    """Middleware to handle session validation in web requests."""
    
    def __init__(self, session_manager: SessionManager):
        self.session_manager = session_manager
    
    async def process_request(
        self,
        headers: Dict[str, str],
        cookies: Dict[str, str],
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> Optional[Session]:
        """
        Process request and validate session.
        
        Returns:
            Valid Session object if found, None otherwise
        """
        # Try to get session token from various sources
        session_token = None
        
        # Check Authorization header
        auth_header = headers.get('Authorization', '')
        if auth_header.startswith('Bearer '):
            session_token = auth_header[7:]
        
        # Check session cookie
        if not session_token:
            session_token = cookies.get('session_token')
        
        # Check custom header
        if not session_token:
            session_token = headers.get('X-Session-Token')
        
        if not session_token:
            return None
        
        try:
            return await self.session_manager.validate_session(
                session_token,
                ip_address=ip_address,
                user_agent=user_agent
            )
        except (SessionExpiredError, InvalidTokenError) as e:
            logger.debug(f"Session validation failed: {e}")
            return None