"""
Comprehensive Audit Logging System

Logs all authentication and authorization events for security monitoring and compliance.
"""

import asyncio
import logging
import json
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import uuid

logger = logging.getLogger(__name__)


class AuditEventType(Enum):
    """Types of audit events."""
    # Authentication events
    LOGIN_SUCCESS = "auth.login.success"
    LOGIN_FAILURE = "auth.login.failure"
    LOGOUT = "auth.logout"
    PASSWORD_CHANGE = "auth.password.change"
    PASSWORD_RESET = "auth.password.reset"
    MFA_ENABLED = "auth.mfa.enabled"
    MFA_DISABLED = "auth.mfa.disabled"
    MFA_CHALLENGE = "auth.mfa.challenge"
    
    # Session events
    SESSION_CREATED = "session.created"
    SESSION_EXPIRED = "session.expired"
    SESSION_INVALIDATED = "session.invalidated"
    SESSION_ELEVATED = "session.elevated"
    
    # Authorization events
    PERMISSION_GRANTED = "authz.permission.granted"
    PERMISSION_DENIED = "authz.permission.denied"
    ROLE_ASSIGNED = "authz.role.assigned"
    ROLE_REMOVED = "authz.role.removed"
    
    # User management events
    USER_CREATED = "user.created"
    USER_UPDATED = "user.updated"
    USER_DELETED = "user.deleted"
    USER_SUSPENDED = "user.suspended"
    USER_ACTIVATED = "user.activated"
    USER_LOCKED = "user.locked"
    
    # Role and permission events
    ROLE_CREATED = "role.created"
    ROLE_UPDATED = "role.updated"
    ROLE_DELETED = "role.deleted"
    PERMISSION_CREATED = "permission.created"
    PERMISSION_UPDATED = "permission.updated"
    PERMISSION_DELETED = "permission.deleted"
    
    # API key events
    API_KEY_CREATED = "api_key.created"
    API_KEY_USED = "api_key.used"
    API_KEY_REVOKED = "api_key.revoked"
    API_KEY_EXPIRED = "api_key.expired"
    
    # Tenant events
    TENANT_CREATED = "tenant.created"
    TENANT_UPDATED = "tenant.updated"
    TENANT_DELETED = "tenant.deleted"
    TENANT_SUSPENDED = "tenant.suspended"
    TENANT_QUOTA_EXCEEDED = "tenant.quota.exceeded"
    
    # Security events
    SECURITY_VIOLATION = "security.violation"
    SUSPICIOUS_ACTIVITY = "security.suspicious"
    BRUTE_FORCE_ATTEMPT = "security.brute_force"
    IP_BLOCKED = "security.ip_blocked"
    
    # System events
    SYSTEM_CONFIG_CHANGED = "system.config.changed"
    SYSTEM_STARTUP = "system.startup"
    SYSTEM_SHUTDOWN = "system.shutdown"
    BACKUP_CREATED = "system.backup.created"
    BACKUP_RESTORED = "system.backup.restored"


class AuditSeverity(Enum):
    """Audit event severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    """Represents an audit event."""
    id: str
    event_type: AuditEventType
    timestamp: datetime
    severity: AuditSeverity
    
    # Actor information
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    api_key_id: Optional[str] = None
    
    # Context information
    tenant_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    
    # Event details
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    action: Optional[str] = None
    details: Dict[str, Any] = None
    
    # Result information
    success: bool = True
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    
    # Additional metadata
    correlation_id: Optional[str] = None
    trace_id: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['event_type'] = self.event_type.value
        data['severity'] = self.severity.value
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)


class AuditStorage:
    """Abstract interface for audit storage."""
    
    async def store_event(self, event: AuditEvent):
        """Store an audit event."""
        raise NotImplementedError
    
    async def query_events(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        event_types: Optional[List[AuditEventType]] = None,
        severity: Optional[AuditSeverity] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[AuditEvent]:
        """Query audit events with filters."""
        raise NotImplementedError
    
    async def get_event(self, event_id: str) -> Optional[AuditEvent]:
        """Get a specific audit event."""
        raise NotImplementedError
    
    async def cleanup_old_events(self, older_than: datetime):
        """Clean up events older than specified date."""
        raise NotImplementedError


class MemoryAuditStorage(AuditStorage):
    """In-memory audit storage (for development/testing)."""
    
    def __init__(self, max_events: int = 10000):
        self.max_events = max_events
        self._events: List[AuditEvent] = []
        self._events_by_id: Dict[str, AuditEvent] = {}
    
    async def store_event(self, event: AuditEvent):
        """Store an audit event in memory."""
        self._events.append(event)
        self._events_by_id[event.id] = event
        
        # Remove old events if we exceed max
        if len(self._events) > self.max_events:
            old_event = self._events.pop(0)
            del self._events_by_id[old_event.id]
    
    async def query_events(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        event_types: Optional[List[AuditEventType]] = None,
        severity: Optional[AuditSeverity] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[AuditEvent]:
        """Query events with filters."""
        filtered_events = []
        
        for event in self._events:
            # Apply filters
            if start_time and event.timestamp < start_time:
                continue
            if end_time and event.timestamp > end_time:
                continue
            if user_id and event.user_id != user_id:
                continue
            if tenant_id and event.tenant_id != tenant_id:
                continue
            if event_types and event.event_type not in event_types:
                continue
            if severity and event.severity != severity:
                continue
            
            filtered_events.append(event)
        
        # Sort by timestamp (newest first)
        filtered_events.sort(key=lambda e: e.timestamp, reverse=True)
        
        # Apply pagination
        return filtered_events[offset:offset + limit]
    
    async def get_event(self, event_id: str) -> Optional[AuditEvent]:
        """Get a specific event."""
        return self._events_by_id.get(event_id)
    
    async def cleanup_old_events(self, older_than: datetime):
        """Remove events older than specified date."""
        old_events = [e for e in self._events if e.timestamp < older_than]
        
        for event in old_events:
            self._events.remove(event)
            del self._events_by_id[event.id]
        
        logger.info(f"Cleaned up {len(old_events)} old audit events")


class FileAuditStorage(AuditStorage):
    """File-based audit storage for persistent logging."""
    
    def __init__(self, log_file: str = "audit.log", rotate_size: int = 10 * 1024 * 1024):
        self.log_file = log_file
        self.rotate_size = rotate_size
        
        # Set up file logger
        self.file_logger = logging.getLogger("audit")
        self.file_logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        for handler in self.file_logger.handlers[:]:
            self.file_logger.removeHandler(handler)
        
        # Add rotating file handler
        from logging.handlers import RotatingFileHandler
        handler = RotatingFileHandler(
            log_file,
            maxBytes=rotate_size,
            backupCount=5
        )
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.file_logger.addHandler(handler)
    
    async def store_event(self, event: AuditEvent):
        """Store event to file."""
        self.file_logger.info(event.to_json())
    
    async def query_events(self, **kwargs) -> List[AuditEvent]:
        """File storage doesn't support querying (use external log analysis tools)."""
        raise NotImplementedError("File storage doesn't support querying")
    
    async def get_event(self, event_id: str) -> Optional[AuditEvent]:
        """File storage doesn't support individual event retrieval."""
        raise NotImplementedError("File storage doesn't support individual event retrieval")
    
    async def cleanup_old_events(self, older_than: datetime):
        """File cleanup is handled by log rotation."""
        pass


class AuthAuditLogger:
    """Main audit logger for authentication events."""
    
    def __init__(self, storage: Optional[AuditStorage] = None):
        self.storage = storage or MemoryAuditStorage()
        
        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Event processors
        self._event_processors: List[callable] = []
        
        # Metrics
        self._metrics = {
            'total_events': 0,
            'events_by_type': {},
            'events_by_severity': {},
            'security_events': 0
        }
    
    async def initialize(self):
        """Initialize audit logger."""
        logger.info("Initializing audit logger")
        
        # Start background tasks
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        # Log system startup
        await self.log_system_event(AuditEventType.SYSTEM_STARTUP)
        
        logger.info("Audit logger initialized")
    
    def add_event_processor(self, processor: callable):
        """Add event processor for real-time analysis."""
        self._event_processors.append(processor)
    
    async def log_event(self, event: AuditEvent):
        """Log an audit event."""
        try:
            # Store event
            await self.storage.store_event(event)
            
            # Update metrics
            self._update_metrics(event)
            
            # Process event with registered processors
            for processor in self._event_processors:
                try:
                    if asyncio.iscoroutinefunction(processor):
                        await processor(event)
                    else:
                        processor(event)
                except Exception as e:
                    logger.error(f"Error in event processor: {e}")
            
            # Log security events at higher level
            if event.severity in [AuditSeverity.HIGH, AuditSeverity.CRITICAL]:
                logger.warning(f"Security event: {event.event_type.value} - {event.details}")
            
        except Exception as e:
            logger.error(f"Failed to log audit event: {e}")
    
    def _update_metrics(self, event: AuditEvent):
        """Update audit metrics."""
        self._metrics['total_events'] += 1
        
        event_type = event.event_type.value
        self._metrics['events_by_type'][event_type] = \
            self._metrics['events_by_type'].get(event_type, 0) + 1
        
        severity = event.severity.value
        self._metrics['events_by_severity'][severity] = \
            self._metrics['events_by_severity'].get(severity, 0) + 1
        
        if 'security' in event_type:
            self._metrics['security_events'] += 1
    
    async def log_authentication(
        self,
        event_type: AuditEventType = AuditEventType.LOGIN_SUCCESS,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        auth_type: Optional[str] = None,
        success: bool = True,
        error: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        session_id: Optional[str] = None
    ):
        """Log authentication event."""
        severity = AuditSeverity.LOW if success else AuditSeverity.MEDIUM
        
        event = AuditEvent(
            id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=datetime.utcnow(),
            severity=severity,
            user_id=user_id,
            session_id=session_id,
            tenant_id=tenant_id,
            ip_address=ip_address,
            user_agent=user_agent,
            success=success,
            error_message=error,
            details={
                'auth_type': auth_type
            }
        )
        
        await self.log_event(event)
    
    async def log_authorization(
        self,
        event_type: AuditEventType = AuditEventType.PERMISSION_GRANTED,
        user_id: Optional[str] = None,
        permission: Optional[str] = None,
        resource_id: Optional[str] = None,
        success: bool = True,
        tenant_id: Optional[str] = None
    ):
        """Log authorization event."""
        severity = AuditSeverity.LOW if success else AuditSeverity.MEDIUM
        
        event = AuditEvent(
            id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=datetime.utcnow(),
            severity=severity,
            user_id=user_id,
            tenant_id=tenant_id,
            resource_id=resource_id,
            success=success,
            details={
                'permission': permission
            }
        )
        
        await self.log_event(event)
    
    async def log_user_management(
        self,
        action: str,
        user_id: str,
        tenant_id: Optional[str] = None,
        actor_user_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Log user management event."""
        event_type_map = {
            'create_user': AuditEventType.USER_CREATED,
            'update_user': AuditEventType.USER_UPDATED,
            'delete_user': AuditEventType.USER_DELETED,
            'suspend_user': AuditEventType.USER_SUSPENDED,
            'activate_user': AuditEventType.USER_ACTIVATED
        }
        
        event_type = event_type_map.get(action, AuditEventType.USER_UPDATED)
        
        event = AuditEvent(
            id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=datetime.utcnow(),
            severity=AuditSeverity.MEDIUM,
            user_id=actor_user_id,
            tenant_id=tenant_id,
            resource_type="user",
            resource_id=user_id,
            action=action,
            details=details or {}
        )
        
        await self.log_event(event)
    
    async def log_api_key_event(
        self,
        event_type: AuditEventType,
        api_key_id: str,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Log API key event."""
        severity = AuditSeverity.MEDIUM if event_type == AuditEventType.API_KEY_CREATED else AuditSeverity.LOW
        
        event = AuditEvent(
            id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=datetime.utcnow(),
            severity=severity,
            user_id=user_id,
            api_key_id=api_key_id,
            tenant_id=tenant_id,
            ip_address=ip_address,
            resource_type="api_key",
            resource_id=api_key_id,
            details=details or {}
        )
        
        await self.log_event(event)
    
    async def log_security_event(
        self,
        event_type: str,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        severity: AuditSeverity = AuditSeverity.HIGH,
        details: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Log security-related event."""
        event = AuditEvent(
            id=str(uuid.uuid4()),
            event_type=AuditEventType.SECURITY_VIOLATION,
            timestamp=datetime.utcnow(),
            severity=severity,
            user_id=user_id,
            tenant_id=tenant_id,
            ip_address=ip_address,
            details={
                'event_type': event_type,
                **(details or {}),
                **kwargs
            }
        )
        
        await self.log_event(event)
    
    async def log_system_event(
        self,
        event_type: AuditEventType,
        details: Optional[Dict[str, Any]] = None
    ):
        """Log system event."""
        event = AuditEvent(
            id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=datetime.utcnow(),
            severity=AuditSeverity.LOW,
            details=details or {}
        )
        
        await self.log_event(event)
    
    async def query_events(self, **filters) -> List[AuditEvent]:
        """Query audit events."""
        return await self.storage.query_events(**filters)
    
    async def get_event(self, event_id: str) -> Optional[AuditEvent]:
        """Get specific event."""
        return await self.storage.get_event(event_id)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get audit metrics."""
        return self._metrics.copy()
    
    async def generate_security_report(
        self,
        start_time: datetime,
        end_time: datetime,
        tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate security report for time period."""
        # Get security events
        security_events = await self.query_events(
            start_time=start_time,
            end_time=end_time,
            tenant_id=tenant_id,
            event_types=[
                AuditEventType.LOGIN_FAILURE,
                AuditEventType.PERMISSION_DENIED,
                AuditEventType.SECURITY_VIOLATION,
                AuditEventType.SUSPICIOUS_ACTIVITY
            ]
        )
        
        # Analyze events
        failed_logins = [e for e in security_events if e.event_type == AuditEventType.LOGIN_FAILURE]
        permission_denials = [e for e in security_events if e.event_type == AuditEventType.PERMISSION_DENIED]
        security_violations = [e for e in security_events if e.event_type == AuditEventType.SECURITY_VIOLATION]
        
        # Count by IP
        ip_counts = {}
        for event in failed_logins:
            if event.ip_address:
                ip_counts[event.ip_address] = ip_counts.get(event.ip_address, 0) + 1
        
        return {
            'period': {
                'start': start_time.isoformat(),
                'end': end_time.isoformat()
            },
            'summary': {
                'total_security_events': len(security_events),
                'failed_logins': len(failed_logins),
                'permission_denials': len(permission_denials),
                'security_violations': len(security_violations)
            },
            'top_failed_ips': sorted(ip_counts.items(), key=lambda x: x[1], reverse=True)[:10],
            'events': [e.to_dict() for e in security_events[:100]]  # Latest 100 events
        }
    
    async def _cleanup_loop(self):
        """Background task to clean up old audit events."""
        while True:
            try:
                await asyncio.sleep(24 * 3600)  # Daily cleanup
                
                # Clean up events older than 1 year
                cutoff_time = datetime.utcnow() - timedelta(days=365)
                await self.storage.cleanup_old_events(cutoff_time)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Audit cleanup error: {e}")
    
    async def cleanup(self):
        """Clean up resources."""
        # Log system shutdown
        await self.log_system_event(AuditEventType.SYSTEM_SHUTDOWN)
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass


# Security event analyzers
class SecurityEventAnalyzer:
    """Analyzes audit events for security patterns."""
    
    def __init__(self, audit_logger: AuthAuditLogger):
        self.audit_logger = audit_logger
        self.suspicious_patterns = {}
        
        # Register as event processor
        audit_logger.add_event_processor(self.analyze_event)
    
    async def analyze_event(self, event: AuditEvent):
        """Analyze event for security patterns."""
        if event.event_type == AuditEventType.LOGIN_FAILURE:
            await self._check_brute_force(event)
        elif event.event_type == AuditEventType.PERMISSION_DENIED:
            await self._check_privilege_escalation(event)
    
    async def _check_brute_force(self, event: AuditEvent):
        """Check for brute force attempts."""
        if not event.ip_address:
            return
        
        # Count failed attempts from this IP
        recent_failures = await self.audit_logger.query_events(
            start_time=datetime.utcnow() - timedelta(minutes=15),
            event_types=[AuditEventType.LOGIN_FAILURE],
            limit=100
        )
        
        ip_failures = [e for e in recent_failures if e.ip_address == event.ip_address]
        
        if len(ip_failures) >= 5:
            await self.audit_logger.log_security_event(
                "brute_force_detected",
                ip_address=event.ip_address,
                severity=AuditSeverity.HIGH,
                details={
                    'failed_attempts': len(ip_failures),
                    'time_window': '15_minutes'
                }
            )
    
    async def _check_privilege_escalation(self, event: AuditEvent):
        """Check for potential privilege escalation attempts."""
        if not event.user_id:
            return
        
        # Count permission denials for this user
        recent_denials = await self.audit_logger.query_events(
            start_time=datetime.utcnow() - timedelta(hours=1),
            user_id=event.user_id,
            event_types=[AuditEventType.PERMISSION_DENIED],
            limit=50
        )
        
        if len(recent_denials) >= 10:
            await self.audit_logger.log_security_event(
                "privilege_escalation_attempt",
                user_id=event.user_id,
                severity=AuditSeverity.MEDIUM,
                details={
                    'denied_attempts': len(recent_denials),
                    'time_window': '1_hour'
                }
            )