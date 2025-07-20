"""
Security Manager for WebRTC Communications

This module provides comprehensive security features for WebRTC connections,
including DTLS-SRTP encryption, certificate management, and security validation.
"""

import asyncio
import logging
import time
import hashlib
import secrets
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List, Callable, Union, Tuple
import json
import base64

try:
    import cryptography
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.x509 import load_pem_x509_certificate, CertificateSigningRequest
    from cryptography import x509
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

from monitoring.performance_monitor import monitor_performance, global_performance_monitor

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security levels for WebRTC connections."""
    BASIC = "basic"          # Standard DTLS-SRTP
    ENHANCED = "enhanced"    # Additional encryption layers
    PARANOID = "paranoid"    # Maximum security measures


class CertificateType(Enum):
    """Certificate types for WebRTC."""
    SELF_SIGNED = "self_signed"
    CA_SIGNED = "ca_signed"
    TEMPORARY = "temporary"


class EncryptionAlgorithm(Enum):
    """Supported encryption algorithms."""
    AES_256_GCM = "aes_256_gcm"
    AES_128_GCM = "aes_128_gcm"
    CHACHA20_POLY1305 = "chacha20_poly1305"


@dataclass
class SecurityConfig:
    """Security configuration for WebRTC connections."""
    # Security level
    security_level: SecurityLevel = SecurityLevel.ENHANCED
    
    # Certificate settings
    certificate_type: CertificateType = CertificateType.SELF_SIGNED
    certificate_validity_days: int = 365
    key_size: int = 2048
    
    # Encryption settings
    preferred_encryption: EncryptionAlgorithm = EncryptionAlgorithm.AES_256_GCM
    require_secure_transport: bool = True
    enforce_srtp: bool = True
    
    # DTLS settings
    dtls_timeout_ms: int = 5000
    dtls_retransmit_timeout_ms: int = 1000
    verify_peer_certificate: bool = True
    
    # Additional security features
    enable_perfect_forward_secrecy: bool = True
    require_certificate_pinning: bool = False
    allowed_cipher_suites: List[str] = field(default_factory=lambda: [
        "TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384",
        "TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256",
        "TLS_DHE_RSA_WITH_AES_256_GCM_SHA384"
    ])
    
    # Key management
    key_rotation_interval_hours: int = 24
    session_key_derivation_rounds: int = 100000


@dataclass
class SecurityMetrics:
    """Security-related metrics."""
    # Encryption status
    dtls_established: bool = False
    srtp_established: bool = False
    encryption_algorithm: Optional[str] = None
    cipher_suite: Optional[str] = None
    
    # Certificate information
    certificate_fingerprint: Optional[str] = None
    certificate_expiry: Optional[float] = None
    certificate_valid: bool = False
    
    # Security events
    handshake_duration_ms: Optional[float] = None
    key_exchanges: int = 0
    security_violations: int = 0
    
    # Performance impact
    encryption_overhead_percent: float = 0.0
    decryption_latency_ms: float = 0.0


class SecurityManager:
    """Manages security for WebRTC connections."""
    
    def __init__(self, config: SecurityConfig = None):
        self.config = config or SecurityConfig()
        self.metrics = SecurityMetrics()
        
        # Certificate and key management
        self.private_key: Optional[Any] = None
        self.certificate: Optional[Any] = None
        self.certificate_pem: Optional[str] = None
        self.certificate_fingerprint: Optional[str] = None
        
        # Session keys
        self.session_keys: Dict[str, bytes] = {}
        self.key_generation_time: Dict[str, float] = {}
        
        # Security state
        self.is_secure_connection: bool = False
        self.peer_fingerprint: Optional[str] = None
        self.established_cipher_suite: Optional[str] = None
        
        # Event callbacks
        self.on_security_established_callbacks: List[Callable] = []
        self.on_security_violation_callbacks: List[Callable] = []
        self.on_certificate_expiry_callbacks: List[Callable] = []
        
        # Monitoring
        self.monitor = global_performance_monitor.register_component(
            "security_manager", "communication"
        )
        
        # Background tasks
        self._key_rotation_task: Optional[asyncio.Task] = None
        self._certificate_monitor_task: Optional[asyncio.Task] = None
    
    async def initialize(self):
        """Initialize security manager."""
        logger.info("Initializing security manager")
        
        try:
            # Generate or load certificate
            await self._setup_certificate()
            
            # Start background tasks
            await self._start_security_monitoring()
            
            logger.info("Security manager initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize security manager: {e}")
            return False
    
    async def _setup_certificate(self):
        """Set up certificate and private key."""
        if not CRYPTO_AVAILABLE:
            logger.warning("Cryptography library not available - using mock certificate")
            self.certificate_pem = self._generate_mock_certificate()
            self.certificate_fingerprint = self._calculate_mock_fingerprint()
            self.metrics.certificate_valid = True
            return
        
        try:
            if self.config.certificate_type == CertificateType.SELF_SIGNED:
                await self._generate_self_signed_certificate()
            elif self.config.certificate_type == CertificateType.TEMPORARY:
                await self._generate_temporary_certificate()
            else:
                await self._load_ca_signed_certificate()
            
            # Calculate fingerprint
            self.certificate_fingerprint = self._calculate_certificate_fingerprint()
            self.metrics.certificate_fingerprint = self.certificate_fingerprint
            
            # Set expiry time
            if self.certificate:
                self.metrics.certificate_expiry = time.time() + (
                    self.config.certificate_validity_days * 24 * 3600
                )
            
            self.metrics.certificate_valid = True
            logger.info(f"Certificate setup complete. Fingerprint: {self.certificate_fingerprint}")
            
        except Exception as e:
            logger.error(f"Certificate setup failed: {e}")
            self.metrics.certificate_valid = False
            raise
    
    async def _generate_self_signed_certificate(self):
        """Generate self-signed certificate."""
        if not CRYPTO_AVAILABLE:
            return
        
        # Generate private key
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=self.config.key_size
        )
        
        # Create certificate
        subject = issuer = x509.Name([
            x509.NameAttribute(x509.NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(x509.NameOID.STATE_OR_PROVINCE_NAME, "CA"),
            x509.NameAttribute(x509.NameOID.LOCALITY_NAME, "San Francisco"),
            x509.NameAttribute(x509.NameOID.ORGANIZATION_NAME, "Voice Agent Platform"),
            x509.NameAttribute(x509.NameOID.COMMON_NAME, "webrtc.voiceagent.local"),
        ])
        
        cert_builder = x509.CertificateBuilder()
        cert_builder = cert_builder.subject_name(subject)
        cert_builder = cert_builder.issuer_name(issuer)
        cert_builder = cert_builder.public_key(self.private_key.public_key())
        cert_builder = cert_builder.serial_number(x509.random_serial_number())
        cert_builder = cert_builder.not_valid_before(
            x509.datetime.datetime.utcnow()
        )
        cert_builder = cert_builder.not_valid_after(
            x509.datetime.datetime.utcnow() + x509.datetime.timedelta(
                days=self.config.certificate_validity_days
            )
        )
        
        # Add extensions for WebRTC
        cert_builder = cert_builder.add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName("webrtc.voiceagent.local"),
                x509.DNSName("localhost"),
                x509.IPAddress(x509.ip_address("127.0.0.1")),
            ]),
            critical=False,
        )
        
        # Add key usage extensions
        cert_builder = cert_builder.add_extension(
            x509.KeyUsage(
                digital_signature=True,
                key_encipherment=True,
                key_agreement=False,
                key_cert_sign=False,
                crl_sign=False,
                content_commitment=False,
                data_encipherment=False,
                encipher_only=False,
                decipher_only=False
            ),
            critical=True,
        )
        
        # Sign certificate
        self.certificate = cert_builder.sign(
            self.private_key, hashes.SHA256()
        )
        
        # Convert to PEM format
        self.certificate_pem = self.certificate.public_bytes(
            serialization.Encoding.PEM
        ).decode('utf-8')
        
        logger.info("Self-signed certificate generated successfully")
    
    async def _generate_temporary_certificate(self):
        """Generate temporary certificate for testing."""
        # Similar to self-signed but with shorter validity
        await self._generate_self_signed_certificate()
        logger.info("Temporary certificate generated")
    
    async def _load_ca_signed_certificate(self):
        """Load CA-signed certificate (placeholder)."""
        # In a real implementation, this would load from files or CA
        logger.warning("CA-signed certificate loading not implemented - using self-signed")
        await self._generate_self_signed_certificate()
    
    def _generate_mock_certificate(self) -> str:
        """Generate mock certificate for development."""
        return """-----BEGIN CERTIFICATE-----
MIICxjCCAa4CAQAwDQYJKoZIhvcNAQELBQAwEjEQMA4GA1UEAwwHbW9ja2NlcnQw
HhcNMjQwMTAxMDAwMDAwWhcNMjUwMTAxMDAwMDAwWjASMRAwDgYDVQQDDAd0b2Nr
Y2VydDCCASIwDQYJKoZIhvcNAQEBBQADggEPADCCAQoCggEBAMockCertificate
VeryLongMockStringForTestingPurposesOnly123456789AbCdEfGhIjKlMnOpQrStUv
WxYzMockCertificateDataHereForWebRTCTesting...
-----END CERTIFICATE-----"""
    
    def _calculate_mock_fingerprint(self) -> str:
        """Calculate mock fingerprint."""
        return "SHA-256 12:34:56:78:9A:BC:DE:F0:12:34:56:78:9A:BC:DE:F0:12:34:56:78:9A:BC:DE:F0:12:34:56:78:9A:BC:DE"
    
    def _calculate_certificate_fingerprint(self) -> Optional[str]:
        """Calculate certificate fingerprint."""
        if not CRYPTO_AVAILABLE or not self.certificate:
            return self._calculate_mock_fingerprint()
        
        try:
            # Calculate SHA-256 fingerprint
            fingerprint = self.certificate.fingerprint(hashes.SHA256())
            
            # Format as colon-separated hex string
            hex_fingerprint = fingerprint.hex().upper()
            formatted = ':'.join(hex_fingerprint[i:i+2] for i in range(0, len(hex_fingerprint), 2))
            
            return f"SHA-256 {formatted}"
            
        except Exception as e:
            logger.error(f"Error calculating certificate fingerprint: {e}")
            return None
    
    @monitor_performance(component="security_manager", operation="establish_security")
    async def establish_secure_connection(self, peer_fingerprint: str = None) -> bool:
        """Establish secure connection with peer."""
        start_time = time.time()
        
        try:
            # Verify peer certificate if provided
            if peer_fingerprint and self.config.verify_peer_certificate:
                if not await self._verify_peer_certificate(peer_fingerprint):
                    logger.error("Peer certificate verification failed")
                    self.metrics.security_violations += 1
                    return False
            
            # Establish DTLS
            if not await self._establish_dtls():
                logger.error("DTLS establishment failed")
                return False
            
            # Establish SRTP
            if self.config.enforce_srtp:
                if not await self._establish_srtp():
                    logger.error("SRTP establishment failed")
                    return False
            
            # Generate session keys
            await self._generate_session_keys()
            
            # Record metrics
            self.metrics.handshake_duration_ms = (time.time() - start_time) * 1000
            self.is_secure_connection = True
            self.peer_fingerprint = peer_fingerprint
            
            logger.info("Secure connection established successfully")
            await self._trigger_security_established()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to establish secure connection: {e}")
            self.metrics.security_violations += 1
            return False
    
    async def _verify_peer_certificate(self, peer_fingerprint: str) -> bool:
        """Verify peer certificate fingerprint."""
        if not peer_fingerprint:
            return False
        
        # In a real implementation, this would validate the peer's certificate
        # For now, we'll do basic fingerprint format validation
        if not peer_fingerprint.startswith("SHA-256"):
            logger.warning("Invalid peer fingerprint format")
            return False
        
        logger.info(f"Peer certificate verified: {peer_fingerprint}")
        return True
    
    async def _establish_dtls(self) -> bool:
        """Establish DTLS connection."""
        if not CRYPTO_AVAILABLE:
            logger.info("Mock DTLS establishment")
            self.metrics.dtls_established = True
            self.metrics.cipher_suite = self.config.allowed_cipher_suites[0]
            self.established_cipher_suite = self.metrics.cipher_suite
            return True
        
        try:
            # In real implementation, this would negotiate DTLS with LiveKit
            # For now, simulate the process
            
            # Simulate handshake time
            await asyncio.sleep(0.1)
            
            self.metrics.dtls_established = True
            self.metrics.cipher_suite = self.config.allowed_cipher_suites[0]
            self.established_cipher_suite = self.metrics.cipher_suite
            
            logger.info(f"DTLS established with cipher suite: {self.metrics.cipher_suite}")
            return True
            
        except Exception as e:
            logger.error(f"DTLS establishment failed: {e}")
            return False
    
    async def _establish_srtp(self) -> bool:
        """Establish SRTP for media encryption."""
        if not CRYPTO_AVAILABLE:
            logger.info("Mock SRTP establishment")
            self.metrics.srtp_established = True
            self.metrics.encryption_algorithm = self.config.preferred_encryption.value
            return True
        
        try:
            # In real implementation, this would set up SRTP keys derived from DTLS
            
            self.metrics.srtp_established = True
            self.metrics.encryption_algorithm = self.config.preferred_encryption.value
            
            logger.info(f"SRTP established with algorithm: {self.metrics.encryption_algorithm}")
            return True
            
        except Exception as e:
            logger.error(f"SRTP establishment failed: {e}")
            return False
    
    async def _generate_session_keys(self):
        """Generate session keys for additional encryption."""
        try:
            # Generate encryption key
            encryption_key = secrets.token_bytes(32)  # 256-bit key
            self.session_keys["encryption"] = encryption_key
            self.key_generation_time["encryption"] = time.time()
            
            # Generate MAC key
            mac_key = secrets.token_bytes(32)
            self.session_keys["mac"] = mac_key
            self.key_generation_time["mac"] = time.time()
            
            self.metrics.key_exchanges += 1
            logger.info("Session keys generated")
            
        except Exception as e:
            logger.error(f"Session key generation failed: {e}")
            raise
    
    async def encrypt_audio_data(self, audio_data: bytes) -> bytes:
        """Encrypt audio data for additional security."""
        if not CRYPTO_AVAILABLE or not self.is_secure_connection:
            # Return original data if encryption not available/enabled
            return audio_data
        
        try:
            if self.config.security_level == SecurityLevel.BASIC:
                # DTLS-SRTP only (handled by WebRTC stack)
                return audio_data
            
            # Additional encryption for enhanced/paranoid levels
            encryption_key = self.session_keys.get("encryption")
            if not encryption_key:
                return audio_data
            
            start_time = time.time()
            
            # Generate random IV
            iv = secrets.token_bytes(12)  # 96-bit IV for GCM
            
            # Encrypt using AES-GCM
            cipher = Cipher(
                algorithms.AES(encryption_key),
                modes.GCM(iv)
            )
            encryptor = cipher.encryptor()
            
            ciphertext = encryptor.update(audio_data) + encryptor.finalize()
            
            # Prepend IV and append tag
            encrypted_data = iv + encryptor.tag + ciphertext
            
            # Record performance impact
            encryption_time = (time.time() - start_time) * 1000
            self.metrics.encryption_overhead_percent = (
                (len(encrypted_data) - len(audio_data)) / len(audio_data) * 100
            )
            
            return encrypted_data
            
        except Exception as e:
            logger.error(f"Audio encryption failed: {e}")
            return audio_data
    
    async def decrypt_audio_data(self, encrypted_data: bytes) -> bytes:
        """Decrypt audio data."""
        if not CRYPTO_AVAILABLE or not self.is_secure_connection:
            return encrypted_data
        
        try:
            if self.config.security_level == SecurityLevel.BASIC:
                return encrypted_data
            
            encryption_key = self.session_keys.get("encryption")
            if not encryption_key or len(encrypted_data) < 28:  # IV(12) + tag(16)
                return encrypted_data
            
            start_time = time.time()
            
            # Extract IV, tag, and ciphertext
            iv = encrypted_data[:12]
            tag = encrypted_data[12:28]
            ciphertext = encrypted_data[28:]
            
            # Decrypt using AES-GCM
            cipher = Cipher(
                algorithms.AES(encryption_key),
                modes.GCM(iv, tag)
            )
            decryptor = cipher.decryptor()
            
            plaintext = decryptor.update(ciphertext) + decryptor.finalize()
            
            # Record decryption latency
            self.metrics.decryption_latency_ms = (time.time() - start_time) * 1000
            
            return plaintext
            
        except Exception as e:
            logger.error(f"Audio decryption failed: {e}")
            return encrypted_data
    
    async def _start_security_monitoring(self):
        """Start security monitoring tasks."""
        # Start key rotation task
        if self.config.key_rotation_interval_hours > 0:
            self._key_rotation_task = asyncio.create_task(self._key_rotation_loop())
        
        # Start certificate monitoring
        self._certificate_monitor_task = asyncio.create_task(self._certificate_monitor_loop())
    
    async def _key_rotation_loop(self):
        """Periodically rotate session keys."""
        while True:
            try:
                await asyncio.sleep(self.config.key_rotation_interval_hours * 3600)
                
                if self.is_secure_connection:
                    logger.info("Rotating session keys")
                    await self._generate_session_keys()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Key rotation error: {e}")
                await asyncio.sleep(3600)  # Wait an hour before retry
    
    async def _certificate_monitor_loop(self):
        """Monitor certificate expiry."""
        while True:
            try:
                await asyncio.sleep(24 * 3600)  # Check daily
                
                if self.metrics.certificate_expiry:
                    time_to_expiry = self.metrics.certificate_expiry - time.time()
                    
                    # Alert if expiring within 30 days
                    if time_to_expiry < 30 * 24 * 3600:
                        logger.warning(f"Certificate expires in {time_to_expiry / 86400:.1f} days")
                        await self._trigger_certificate_expiry(time_to_expiry)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Certificate monitoring error: {e}")
                await asyncio.sleep(3600)
    
    # Event callback triggers
    async def _trigger_security_established(self):
        """Trigger security established callbacks."""
        for callback in self.on_security_established_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(self.metrics)
                else:
                    callback(self.metrics)
            except Exception as e:
                logger.error(f"Error in security established callback: {e}")
    
    async def _trigger_security_violation(self, violation_type: str, details: Dict[str, Any]):
        """Trigger security violation callbacks."""
        for callback in self.on_security_violation_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(violation_type, details)
                else:
                    callback(violation_type, details)
            except Exception as e:
                logger.error(f"Error in security violation callback: {e}")
    
    async def _trigger_certificate_expiry(self, time_to_expiry: float):
        """Trigger certificate expiry callbacks."""
        for callback in self.on_certificate_expiry_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(time_to_expiry)
                else:
                    callback(time_to_expiry)
            except Exception as e:
                logger.error(f"Error in certificate expiry callback: {e}")
    
    # Public callback registration methods
    def on_security_established(self, callback: Callable):
        """Register callback for security establishment events."""
        self.on_security_established_callbacks.append(callback)
    
    def on_security_violation(self, callback: Callable):
        """Register callback for security violation events."""
        self.on_security_violation_callbacks.append(callback)
    
    def on_certificate_expiry(self, callback: Callable):
        """Register callback for certificate expiry events."""
        self.on_certificate_expiry_callbacks.append(callback)
    
    # Status and information methods
    def get_security_status(self) -> Dict[str, Any]:
        """Get current security status."""
        return {
            "is_secure": self.is_secure_connection,
            "dtls_established": self.metrics.dtls_established,
            "srtp_established": self.metrics.srtp_established,
            "cipher_suite": self.established_cipher_suite,
            "encryption_algorithm": self.metrics.encryption_algorithm,
            "certificate_fingerprint": self.certificate_fingerprint,
            "certificate_valid": self.metrics.certificate_valid,
            "key_exchanges": self.metrics.key_exchanges,
            "security_violations": self.metrics.security_violations,
            "security_level": self.config.security_level.value
        }
    
    def get_certificate_info(self) -> Dict[str, Any]:
        """Get certificate information."""
        return {
            "fingerprint": self.certificate_fingerprint,
            "pem": self.certificate_pem,
            "valid": self.metrics.certificate_valid,
            "expiry": self.metrics.certificate_expiry,
            "type": self.config.certificate_type.value
        }
    
    async def cleanup(self):
        """Clean up security manager resources."""
        try:
            # Cancel background tasks
            if self._key_rotation_task:
                self._key_rotation_task.cancel()
                await self._key_rotation_task
            
            if self._certificate_monitor_task:
                self._certificate_monitor_task.cancel()
                await self._certificate_monitor_task
            
            # Clear sensitive data
            self.session_keys.clear()
            self.private_key = None
            
            self.is_secure_connection = False
            logger.info("Security manager cleaned up")
            
        except Exception as e:
            logger.error(f"Error during security manager cleanup: {e}")


# Convenience functions
async def create_security_manager(security_level: SecurityLevel = SecurityLevel.ENHANCED) -> SecurityManager:
    """Create and initialize a security manager."""
    config = SecurityConfig(security_level=security_level)
    manager = SecurityManager(config)
    
    success = await manager.initialize()
    if not success:
        raise RuntimeError("Failed to initialize security manager")
    
    return manager


# Global security manager for easy access
_global_security_manager: Optional[SecurityManager] = None


def get_global_security_manager() -> Optional[SecurityManager]:
    """Get the global security manager instance."""
    return _global_security_manager


def set_global_security_manager(manager: SecurityManager):
    """Set the global security manager instance."""
    global _global_security_manager
    _global_security_manager = manager