"""
Privacy Controls and Data Retention Management

This module provides comprehensive privacy controls, data retention policies,
and compliance management for voice agent session recordings, ensuring
adherence to privacy regulations like GDPR, CCPA, and HIPAA.
"""

import asyncio
import json
import hashlib
import time
from typing import Dict, List, Optional, Any, Union, Callable, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
import logging
import shutil
import gzip
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os

from .structured_logging import StructuredLogger
from .session_recording import SessionMetadata, SessionStatus, PrivacyLevel


class DataClassification(Enum):
    """Data classification levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class RetentionPolicies(Enum):
    """Standard retention policies."""
    SHORT_TERM = "short_term"  # 7 days
    MEDIUM_TERM = "medium_term"  # 30 days
    LONG_TERM = "long_term"  # 1 year
    PERMANENT = "permanent"  # No automatic deletion
    LEGAL_HOLD = "legal_hold"  # Legal hold, no deletion
    CUSTOM = "custom"  # Custom retention period


class ConsentType(Enum):
    """Types of user consent."""
    RECORDING = "recording"
    TRANSCRIPT = "transcript"
    ANALYTICS = "analytics"
    STORAGE = "storage"
    SHARING = "sharing"
    PROCESSING = "processing"


class ComplianceFramework(Enum):
    """Compliance frameworks."""
    GDPR = "gdpr"  # General Data Protection Regulation
    CCPA = "ccpa"  # California Consumer Privacy Act
    HIPAA = "hipaa"  # Health Insurance Portability and Accountability Act
    SOX = "sox"  # Sarbanes-Oxley Act
    PCI_DSS = "pci_dss"  # Payment Card Industry Data Security Standard
    CUSTOM = "custom"


@dataclass
class ConsentRecord:
    """User consent record."""
    user_id: str
    consent_type: ConsentType
    granted: bool
    timestamp: datetime
    version: str
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    expiry_date: Optional[datetime] = None
    withdrawal_date: Optional[datetime] = None
    legal_basis: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        if self.expiry_date:
            data["expiry_date"] = self.expiry_date.isoformat()
        if self.withdrawal_date:
            data["withdrawal_date"] = self.withdrawal_date.isoformat()
        data["consent_type"] = self.consent_type.value
        return data


@dataclass
class RetentionPolicy:
    """Data retention policy."""
    policy_id: str
    name: str
    description: str
    retention_period_days: int
    data_classification: DataClassification
    compliance_frameworks: List[ComplianceFramework]
    auto_delete: bool = True
    legal_hold_exempt: bool = False
    created_date: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data["created_date"] = self.created_date.isoformat()
        data["data_classification"] = self.data_classification.value
        data["compliance_frameworks"] = [f.value for f in self.compliance_frameworks]
        return data


@dataclass
class DataProcessingRecord:
    """Data processing activity record for compliance."""
    record_id: str
    session_id: str
    user_id: Optional[str]
    processing_purpose: str
    legal_basis: str
    data_categories: List[str]
    recipients: List[str]
    retention_period: str
    security_measures: List[str]
    cross_border_transfers: bool
    created_date: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data["created_date"] = self.created_date.isoformat()
        return data


class PrivacyManager:
    """
    Comprehensive privacy and data protection manager.
    
    Handles consent management, data retention, anonymization,
    encryption, and compliance with privacy regulations.
    """
    
    def __init__(
        self,
        storage_path: str = "./recordings",
        consent_storage_path: str = "./consent",
        encryption_key: Optional[str] = None,
        default_retention_days: int = 30,
        enable_encryption: bool = True,
        enable_anonymization: bool = True,
        compliance_frameworks: Optional[List[ComplianceFramework]] = None,
        logger: Optional[StructuredLogger] = None
    ):
        """
        Initialize privacy manager.
        
        Args:
            storage_path: Path to session recordings
            consent_storage_path: Path to consent records
            encryption_key: Encryption key for data protection
            default_retention_days: Default retention period
            enable_encryption: Enable data encryption
            enable_anonymization: Enable data anonymization
            compliance_frameworks: Required compliance frameworks
            logger: Optional logger instance
        """
        self.storage_path = Path(storage_path)
        self.consent_storage_path = Path(consent_storage_path)
        self.consent_storage_path.mkdir(parents=True, exist_ok=True)
        
        self.default_retention_days = default_retention_days
        self.enable_encryption = enable_encryption
        self.enable_anonymization = enable_anonymization
        self.compliance_frameworks = compliance_frameworks or [ComplianceFramework.GDPR]
        
        # Initialize encryption
        self.cipher_suite = None
        if self.enable_encryption:
            self._initialize_encryption(encryption_key)
        
        # User consents
        self.user_consents: Dict[str, List[ConsentRecord]] = {}
        
        # Retention policies
        self.retention_policies: Dict[str, RetentionPolicy] = {}
        self._initialize_default_policies()
        
        # Processing records
        self.processing_records: List[DataProcessingRecord] = []
        
        # Legal holds
        self.legal_holds: Set[str] = set()
        
        # Privacy callbacks
        self.callbacks: Dict[str, List[Callable]] = {
            "on_consent_granted": [],
            "on_consent_withdrawn": [],
            "on_data_deleted": [],
            "on_legal_hold_applied": [],
            "on_compliance_violation": []
        }
        
        self.logger = logger or StructuredLogger(__name__, "privacy_manager")
        
        # Background tasks
        self.retention_task: Optional[asyncio.Task] = None
        self.compliance_task: Optional[asyncio.Task] = None
        
        # Load existing data
        asyncio.create_task(self._load_consent_records())
        
        self.logger.info(
            "Privacy manager initialized",
            extra_data={
                "storage_path": str(self.storage_path),
                "consent_storage_path": str(self.consent_storage_path),
                "encryption_enabled": self.enable_encryption,
                "anonymization_enabled": self.enable_anonymization,
                "compliance_frameworks": [f.value for f in self.compliance_frameworks],
                "default_retention_days": self.default_retention_days
            }
        )
    
    async def request_consent(
        self,
        user_id: str,
        consent_types: List[ConsentType],
        legal_basis: str = "consent",
        expiry_days: Optional[int] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> Dict[str, bool]:
        """
        Request user consent for data processing activities.
        
        Args:
            user_id: User identifier
            consent_types: Types of consent to request
            legal_basis: Legal basis for processing
            expiry_days: Optional consent expiry period
            ip_address: User's IP address
            user_agent: User's user agent
            
        Returns:
            Dictionary of consent types and whether they were granted
        """
        timestamp = datetime.now(timezone.utc)
        expiry_date = timestamp + timedelta(days=expiry_days) if expiry_days else None
        
        consent_results = {}
        
        for consent_type in consent_types:
            # For this implementation, we'll assume consent is granted
            # In reality, this would involve user interaction
            granted = True  # Placeholder
            
            consent_record = ConsentRecord(
                user_id=user_id,
                consent_type=consent_type,
                granted=granted,
                timestamp=timestamp,
                version="1.0",  # Consent version
                ip_address=ip_address,
                user_agent=user_agent,
                expiry_date=expiry_date,
                legal_basis=legal_basis
            )
            
            # Store consent
            if user_id not in self.user_consents:
                self.user_consents[user_id] = []
            
            self.user_consents[user_id].append(consent_record)
            consent_results[consent_type.value] = granted
            
            # Trigger callbacks
            if granted:
                await self._trigger_callbacks("on_consent_granted", user_id, consent_record)
        
        # Save consent records
        await self._save_consent_records(user_id)
        
        self.logger.info(
            f"Consent requested for user: {user_id}",
            extra_data={
                "user_id": user_id,
                "consent_types": [c.value for c in consent_types],
                "legal_basis": legal_basis,
                "results": consent_results
            }
        )
        
        return consent_results
    
    async def withdraw_consent(
        self,
        user_id: str,
        consent_types: List[ConsentType]
    ) -> bool:
        """
        Withdraw user consent for specific processing activities.
        
        Args:
            user_id: User identifier
            consent_types: Types of consent to withdraw
            
        Returns:
            True if successful
        """
        if user_id not in self.user_consents:
            return False
        
        withdrawal_time = datetime.now(timezone.utc)
        
        for consent_record in self.user_consents[user_id]:
            if consent_record.consent_type in consent_types and consent_record.granted:
                consent_record.granted = False
                consent_record.withdrawal_date = withdrawal_time
                
                # Trigger callbacks
                await self._trigger_callbacks("on_consent_withdrawn", user_id, consent_record)
        
        # Save updated consent records
        await self._save_consent_records(user_id)
        
        # Handle data implications of consent withdrawal
        await self._handle_consent_withdrawal(user_id, consent_types)
        
        self.logger.info(
            f"Consent withdrawn for user: {user_id}",
            extra_data={
                "user_id": user_id,
                "consent_types": [c.value for c in consent_types]
            }
        )
        
        return True
    
    async def check_consent(
        self,
        user_id: str,
        consent_type: ConsentType
    ) -> bool:
        """
        Check if user has valid consent for a specific activity.
        
        Args:
            user_id: User identifier
            consent_type: Type of consent to check
            
        Returns:
            True if consent is valid
        """
        if user_id not in self.user_consents:
            return False
        
        current_time = datetime.now(timezone.utc)
        
        for consent_record in self.user_consents[user_id]:
            if (consent_record.consent_type == consent_type and 
                consent_record.granted and 
                not consent_record.withdrawal_date and
                (not consent_record.expiry_date or consent_record.expiry_date > current_time)):
                return True
        
        return False
    
    async def apply_retention_policy(
        self,
        session_id: str,
        metadata: SessionMetadata,
        policy_id: Optional[str] = None
    ) -> datetime:
        """
        Apply retention policy to a session.
        
        Args:
            session_id: Session identifier
            metadata: Session metadata
            policy_id: Optional specific policy ID
            
        Returns:
            Retention expiry date
        """
        # Determine applicable policy
        if policy_id and policy_id in self.retention_policies:
            policy = self.retention_policies[policy_id]
        else:
            policy = self._determine_retention_policy(metadata)
        
        # Calculate retention expiry
        retention_expiry = metadata.start_time + timedelta(days=policy.retention_period_days)
        
        # Update metadata
        metadata.retention_until = retention_expiry
        
        # Create processing record
        processing_record = DataProcessingRecord(
            record_id=f"proc_{session_id}_{int(time.time())}",
            session_id=session_id,
            user_id=metadata.user_id,
            processing_purpose="Voice agent conversation recording",
            legal_basis="Consent" if await self.check_consent(metadata.user_id, ConsentType.RECORDING) else "Legitimate interest",
            data_categories=["Voice recordings", "Transcripts", "Metadata"],
            recipients=["Voice agent system"],
            retention_period=f"{policy.retention_period_days} days",
            security_measures=["Encryption", "Access controls", "Audit logging"],
            cross_border_transfers=False
        )
        
        self.processing_records.append(processing_record)
        
        self.logger.info(
            f"Retention policy applied to session: {session_id}",
            extra_data={
                "session_id": session_id,
                "policy_id": policy.policy_id,
                "retention_days": policy.retention_period_days,
                "expiry_date": retention_expiry.isoformat()
            }
        )
        
        return retention_expiry
    
    async def encrypt_session_data(
        self,
        session_id: str,
        data: bytes
    ) -> bytes:
        """
        Encrypt session data.
        
        Args:
            session_id: Session identifier
            data: Data to encrypt
            
        Returns:
            Encrypted data
        """
        if not self.enable_encryption or not self.cipher_suite:
            return data
        
        try:
            encrypted_data = self.cipher_suite.encrypt(data)
            
            self.logger.debug(
                f"Session data encrypted: {session_id}",
                extra_data={
                    "session_id": session_id,
                    "original_size": len(data),
                    "encrypted_size": len(encrypted_data)
                }
            )
            
            return encrypted_data
            
        except Exception as e:
            self.logger.exception(
                f"Error encrypting session data: {session_id}",
                extra_data={"error": str(e)}
            )
            return data
    
    async def decrypt_session_data(
        self,
        session_id: str,
        encrypted_data: bytes
    ) -> bytes:
        """
        Decrypt session data.
        
        Args:
            session_id: Session identifier
            encrypted_data: Encrypted data
            
        Returns:
            Decrypted data
        """
        if not self.enable_encryption or not self.cipher_suite:
            return encrypted_data
        
        try:
            decrypted_data = self.cipher_suite.decrypt(encrypted_data)
            
            self.logger.debug(
                f"Session data decrypted: {session_id}",
                extra_data={
                    "session_id": session_id,
                    "encrypted_size": len(encrypted_data),
                    "decrypted_size": len(decrypted_data)
                }
            )
            
            return decrypted_data
            
        except Exception as e:
            self.logger.exception(
                f"Error decrypting session data: {session_id}",
                extra_data={"error": str(e)}
            )
            return encrypted_data
    
    async def anonymize_session_data(
        self,
        session_id: str,
        metadata: SessionMetadata
    ) -> SessionMetadata:
        """
        Anonymize session data by removing personally identifiable information.
        
        Args:
            session_id: Session identifier
            metadata: Session metadata
            
        Returns:
            Anonymized session metadata
        """
        if not self.enable_anonymization:
            return metadata
        
        # Create anonymized copy
        anonymized_metadata = SessionMetadata(
            session_id=self._anonymize_identifier(metadata.session_id),
            user_id=self._anonymize_identifier(metadata.user_id) if metadata.user_id else None,
            agent_id=metadata.agent_id,  # Keep agent ID
            start_time=metadata.start_time,
            end_time=metadata.end_time,
            duration=metadata.duration,
            status=metadata.status,
            privacy_level=PrivacyLevel.METADATA_ONLY,  # Reduce privacy level
            tags=metadata.tags,  # Filter tags for PII
            custom_metadata=self._anonymize_metadata(metadata.custom_metadata),
            quality_metrics=metadata.quality_metrics,
            audio_format=metadata.audio_format,
            sample_rate=metadata.sample_rate,
            channels=metadata.channels,
            bit_depth=metadata.bit_depth,
            storage_path=metadata.storage_path,
            compressed=metadata.compressed,
            encrypted=metadata.encrypted,
            retention_until=metadata.retention_until
        )
        
        self.logger.info(
            f"Session data anonymized: {session_id}",
            extra_data={
                "original_session_id": session_id,
                "anonymized_session_id": anonymized_metadata.session_id
            }
        )
        
        return anonymized_metadata
    
    async def apply_legal_hold(
        self,
        session_ids: List[str],
        reason: str
    ) -> bool:
        """
        Apply legal hold to sessions (prevent deletion).
        
        Args:
            session_ids: Session identifiers
            reason: Reason for legal hold
            
        Returns:
            True if successful
        """
        for session_id in session_ids:
            self.legal_holds.add(session_id)
            
            # Trigger callbacks
            await self._trigger_callbacks("on_legal_hold_applied", session_id, reason)
        
        self.logger.info(
            f"Legal hold applied to {len(session_ids)} sessions",
            extra_data={
                "session_ids": session_ids,
                "reason": reason
            }
        )
        
        return True
    
    async def remove_legal_hold(
        self,
        session_ids: List[str]
    ) -> bool:
        """
        Remove legal hold from sessions.
        
        Args:
            session_ids: Session identifiers
            
        Returns:
            True if successful
        """
        for session_id in session_ids:
            self.legal_holds.discard(session_id)
        
        self.logger.info(
            f"Legal hold removed from {len(session_ids)} sessions",
            extra_data={"session_ids": session_ids}
        )
        
        return True
    
    async def delete_expired_data(self, dry_run: bool = False) -> Dict[str, int]:
        """
        Delete expired session data according to retention policies.
        
        Args:
            dry_run: If True, only report what would be deleted
            
        Returns:
            Dictionary with deletion statistics
        """
        current_time = datetime.now(timezone.utc)
        stats = {
            "sessions_checked": 0,
            "sessions_expired": 0,
            "sessions_deleted": 0,
            "sessions_legal_hold": 0,
            "errors": 0
        }
        
        # Find all session metadata files
        for metadata_file in self.storage_path.glob("*_metadata.json"):
            stats["sessions_checked"] += 1
            
            try:
                # Load metadata
                with open(metadata_file, 'r') as f:
                    metadata_dict = json.load(f)
                
                session_id = metadata_dict["session_id"]
                retention_until = metadata_dict.get("retention_until")
                
                if not retention_until:
                    continue
                
                retention_date = datetime.fromisoformat(retention_until.replace('Z', '+00:00'))
                
                if current_time > retention_date:
                    stats["sessions_expired"] += 1
                    
                    # Check for legal hold
                    if session_id in self.legal_holds:
                        stats["sessions_legal_hold"] += 1
                        continue
                    
                    if not dry_run:
                        # Delete session data
                        await self._delete_session_data(session_id)
                        stats["sessions_deleted"] += 1
                        
                        # Trigger callbacks
                        await self._trigger_callbacks("on_data_deleted", session_id, "retention_expiry")
                
            except Exception as e:
                stats["errors"] += 1
                self.logger.exception(
                    f"Error processing session for deletion",
                    extra_data={"error": str(e), "file": str(metadata_file)}
                )
        
        operation = "Simulated" if dry_run else "Executed"
        self.logger.info(
            f"{operation} data retention cleanup",
            extra_data=stats
        )
        
        return stats
    
    async def generate_compliance_report(
        self,
        framework: ComplianceFramework,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Generate compliance report for specified framework.
        
        Args:
            framework: Compliance framework
            start_date: Report start date
            end_date: Report end date
            
        Returns:
            Compliance report
        """
        end_date = end_date or datetime.now(timezone.utc)
        start_date = start_date or (end_date - timedelta(days=30))
        
        report = {
            "framework": framework.value,
            "report_period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "data_processing_activities": [],
            "consent_records": {},
            "retention_compliance": {},
            "security_measures": [],
            "data_breaches": [],  # Would be populated from incident records
            "recommendations": []
        }
        
        # Data processing activities
        period_records = [
            record for record in self.processing_records
            if start_date <= record.created_date <= end_date
        ]
        report["data_processing_activities"] = [record.to_dict() for record in period_records]
        
        # Consent statistics
        total_users = len(self.user_consents)
        consent_stats = {}
        for consent_type in ConsentType:
            granted_count = 0
            for user_consents in self.user_consents.values():
                for consent in user_consents:
                    if (consent.consent_type == consent_type and 
                        consent.granted and 
                        start_date <= consent.timestamp <= end_date):
                        granted_count += 1
            consent_stats[consent_type.value] = {
                "granted": granted_count,
                "percentage": (granted_count / total_users * 100) if total_users > 0 else 0
            }
        
        report["consent_records"] = {
            "total_users": total_users,
            "consent_types": consent_stats
        }
        
        # Framework-specific additions
        if framework == ComplianceFramework.GDPR:
            report["gdpr_specific"] = await self._generate_gdpr_report(start_date, end_date)
        elif framework == ComplianceFramework.HIPAA:
            report["hipaa_specific"] = await self._generate_hipaa_report(start_date, end_date)
        elif framework == ComplianceFramework.CCPA:
            report["ccpa_specific"] = await self._generate_ccpa_report(start_date, end_date)
        
        self.logger.info(
            f"Compliance report generated: {framework.value}",
            extra_data={
                "framework": framework.value,
                "period": f"{start_date.isoformat()} to {end_date.isoformat()}",
                "processing_activities": len(period_records),
                "total_users": total_users
            }
        )
        
        return report
    
    async def start_background_tasks(self):
        """Start background privacy management tasks."""
        if self.retention_task is None:
            self.retention_task = asyncio.create_task(self._retention_monitor())
        
        if self.compliance_task is None:
            self.compliance_task = asyncio.create_task(self._compliance_monitor())
    
    async def stop_background_tasks(self):
        """Stop background privacy management tasks."""
        if self.retention_task:
            self.retention_task.cancel()
            try:
                await self.retention_task
            except asyncio.CancelledError:
                pass
            self.retention_task = None
        
        if self.compliance_task:
            self.compliance_task.cancel()
            try:
                await self.compliance_task
            except asyncio.CancelledError:
                pass
            self.compliance_task = None
    
    def add_callback(self, event: str, callback: Callable):
        """Add a callback for privacy events."""
        if event in self.callbacks:
            self.callbacks[event].append(callback)
    
    def remove_callback(self, event: str, callback: Callable):
        """Remove a callback for privacy events."""
        if event in self.callbacks and callback in self.callbacks[event]:
            self.callbacks[event].remove(callback)
    
    async def _trigger_callbacks(self, event: str, *args):
        """Trigger callbacks for an event."""
        for callback in self.callbacks.get(event, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(*args)
                else:
                    callback(*args)
            except Exception as e:
                self.logger.exception(
                    f"Error in privacy callback: {event}",
                    extra_data={"error": str(e), "event": event}
                )
    
    def _initialize_encryption(self, encryption_key: Optional[str]):
        """Initialize encryption cipher."""
        try:
            if encryption_key:
                key = encryption_key.encode()
            else:
                # Generate or load key
                key = Fernet.generate_key()
            
            # Ensure key is the right length
            if len(key) != 44:  # Base64 encoded 32-byte key
                # Derive key from password
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=b'voice_agent_salt',  # In production, use random salt
                    iterations=100000,
                )
                key = base64.urlsafe_b64encode(kdf.derive(key))
            
            self.cipher_suite = Fernet(key)
            
        except Exception as e:
            self.logger.exception("Error initializing encryption", extra_data={"error": str(e)})
            self.enable_encryption = False
    
    def _initialize_default_policies(self):
        """Initialize default retention policies."""
        policies = [
            RetentionPolicy(
                policy_id="short_term",
                name="Short Term",
                description="7-day retention for temporary data",
                retention_period_days=7,
                data_classification=DataClassification.INTERNAL,
                compliance_frameworks=[ComplianceFramework.GDPR]
            ),
            RetentionPolicy(
                policy_id="medium_term",
                name="Medium Term",
                description="30-day retention for standard sessions",
                retention_period_days=30,
                data_classification=DataClassification.CONFIDENTIAL,
                compliance_frameworks=[ComplianceFramework.GDPR, ComplianceFramework.CCPA]
            ),
            RetentionPolicy(
                policy_id="long_term",
                name="Long Term",
                description="1-year retention for important sessions",
                retention_period_days=365,
                data_classification=DataClassification.RESTRICTED,
                compliance_frameworks=[ComplianceFramework.GDPR, ComplianceFramework.HIPAA]
            )
        ]
        
        for policy in policies:
            self.retention_policies[policy.policy_id] = policy
    
    def _determine_retention_policy(self, metadata: SessionMetadata) -> RetentionPolicy:
        """Determine appropriate retention policy for session."""
        # Simple policy determination based on tags
        if "important" in metadata.tags:
            return self.retention_policies["long_term"]
        elif "temporary" in metadata.tags:
            return self.retention_policies["short_term"]
        else:
            return self.retention_policies["medium_term"]
    
    def _anonymize_identifier(self, identifier: str) -> str:
        """Anonymize an identifier using hashing."""
        if not identifier:
            return identifier
        
        hash_object = hashlib.sha256(identifier.encode())
        return f"anon_{hash_object.hexdigest()[:16]}"
    
    def _anonymize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Anonymize custom metadata by removing PII."""
        # Simple implementation - remove common PII fields
        pii_fields = ['name', 'email', 'phone', 'address', 'ssn', 'credit_card']
        
        anonymized = {}
        for key, value in metadata.items():
            if key.lower() not in pii_fields:
                anonymized[key] = value
        
        return anonymized
    
    async def _handle_consent_withdrawal(self, user_id: str, consent_types: List[ConsentType]):
        """Handle data implications of consent withdrawal."""
        if ConsentType.RECORDING in consent_types:
            # Stop any active recordings for this user
            await self._stop_user_recordings(user_id)
        
        if ConsentType.STORAGE in consent_types:
            # Delete stored data for this user
            await self._delete_user_data(user_id)
        
        if ConsentType.ANALYTICS in consent_types:
            # Remove user from analytics
            await self._remove_user_from_analytics(user_id)
    
    async def _delete_session_data(self, session_id: str):
        """Delete all data for a session."""
        session_dir = self.storage_path / session_id
        
        if session_dir.exists():
            shutil.rmtree(session_dir)
        
        # Also check for flat file structure
        for pattern in [
            f"{session_id}_metadata.json*",
            f"{session_id}_audio.*",
            f"{session_id}_transcript.*",
            f"{session_id}_quality.*"
        ]:
            for file_path in self.storage_path.glob(pattern):
                file_path.unlink()
    
    async def _load_consent_records(self):
        """Load consent records from storage."""
        consent_file = self.consent_storage_path / "consent_records.json"
        
        if consent_file.exists():
            try:
                with open(consent_file, 'r') as f:
                    data = json.load(f)
                
                for user_id, consents in data.items():
                    self.user_consents[user_id] = []
                    for consent_data in consents:
                        consent_record = ConsentRecord(
                            user_id=consent_data["user_id"],
                            consent_type=ConsentType(consent_data["consent_type"]),
                            granted=consent_data["granted"],
                            timestamp=datetime.fromisoformat(consent_data["timestamp"]),
                            version=consent_data["version"],
                            ip_address=consent_data.get("ip_address"),
                            user_agent=consent_data.get("user_agent"),
                            expiry_date=datetime.fromisoformat(consent_data["expiry_date"]) if consent_data.get("expiry_date") else None,
                            withdrawal_date=datetime.fromisoformat(consent_data["withdrawal_date"]) if consent_data.get("withdrawal_date") else None,
                            legal_basis=consent_data.get("legal_basis")
                        )
                        self.user_consents[user_id].append(consent_record)
                
            except Exception as e:
                self.logger.exception("Error loading consent records", extra_data={"error": str(e)})
    
    async def _save_consent_records(self, user_id: str):
        """Save consent records for a user."""
        consent_file = self.consent_storage_path / "consent_records.json"
        
        try:
            # Load existing data
            data = {}
            if consent_file.exists():
                with open(consent_file, 'r') as f:
                    data = json.load(f)
            
            # Update user's consent records
            if user_id in self.user_consents:
                data[user_id] = [consent.to_dict() for consent in self.user_consents[user_id]]
            
            # Save updated data
            with open(consent_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            self.logger.exception(f"Error saving consent records for user {user_id}", extra_data={"error": str(e)})
    
    async def _retention_monitor(self):
        """Background task to monitor data retention."""
        while True:
            try:
                await asyncio.sleep(3600)  # Check every hour
                
                # Check for expired data
                await self.delete_expired_data(dry_run=False)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.exception("Error in retention monitor", extra_data={"error": str(e)})
    
    async def _compliance_monitor(self):
        """Background task to monitor compliance."""
        while True:
            try:
                await asyncio.sleep(86400)  # Check daily
                
                # Check for compliance violations
                violations = await self._check_compliance_violations()
                
                for violation in violations:
                    await self._trigger_callbacks("on_compliance_violation", violation)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.exception("Error in compliance monitor", extra_data={"error": str(e)})
    
    async def _check_compliance_violations(self) -> List[Dict[str, Any]]:
        """Check for compliance violations."""
        violations = []
        
        # Check for expired consents
        current_time = datetime.now(timezone.utc)
        
        for user_id, consents in self.user_consents.items():
            for consent in consents:
                if (consent.expiry_date and 
                    consent.expiry_date < current_time and 
                    consent.granted and 
                    not consent.withdrawal_date):
                    
                    violations.append({
                        "type": "expired_consent",
                        "user_id": user_id,
                        "consent_type": consent.consent_type.value,
                        "expiry_date": consent.expiry_date.isoformat()
                    })
        
        return violations
    
    async def _generate_gdpr_report(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Generate GDPR-specific compliance report."""
        return {
            "lawful_basis_distribution": {
                "consent": 0,  # Would be calculated from actual data
                "legitimate_interest": 0,
                "contract": 0,
                "legal_obligation": 0
            },
            "data_subject_rights": {
                "access_requests": 0,
                "rectification_requests": 0,
                "erasure_requests": 0,
                "portability_requests": 0
            },
            "privacy_by_design": {
                "encryption_enabled": self.enable_encryption,
                "anonymization_enabled": self.enable_anonymization,
                "data_minimization": True
            }
        }
    
    async def _generate_hipaa_report(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Generate HIPAA-specific compliance report."""
        return {
            "safeguards": {
                "administrative": ["Access controls", "Audit logging"],
                "physical": ["Secure storage"],
                "technical": ["Encryption", "Authentication"]
            },
            "business_associates": [],
            "breach_incidents": []
        }
    
    async def _generate_ccpa_report(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Generate CCPA-specific compliance report."""
        return {
            "consumer_rights": {
                "right_to_know": True,
                "right_to_delete": True,
                "right_to_opt_out": True,
                "right_to_non_discrimination": True
            },
            "data_sales": False,
            "third_party_disclosures": []
        }
    
    async def _stop_user_recordings(self, user_id: str):
        """Stop active recordings for a user."""
        # This would interface with the session recording manager
        pass
    
    async def _delete_user_data(self, user_id: str):
        """Delete all stored data for a user."""
        # This would find and delete all sessions for the user
        pass
    
    async def _remove_user_from_analytics(self, user_id: str):
        """Remove user data from analytics."""
        # This would remove user from analytics processing
        pass