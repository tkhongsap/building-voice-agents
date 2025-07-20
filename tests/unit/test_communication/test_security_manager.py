"""
Unit tests for Security Manager.

Tests security management, DTLS-SRTP encryption, certificate handling,
key rotation, and security policy enforcement.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import sys
import os
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from communication.security_manager import (
    SecurityManager,
    SecurityConfig,
    SecurityLevel,
    CertificateType,
    EncryptionAlgorithm,
    SecurityPolicy,
    CertificateInfo,
    KeyRotationManager,
    SecurityEvent,
    SecurityEventType,
    DTLSContext,
    SRTPContext
)


class TestSecurityConfig:
    """Test security configuration."""
    
    def test_default_config(self):
        """Test default security configuration."""
        config = SecurityConfig()
        
        assert config.security_level == SecurityLevel.STANDARD
        assert config.certificate_type == CertificateType.SELF_SIGNED
        assert config.certificate_validity_days == 365
        assert config.key_size == 2048
        assert config.preferred_encryption == EncryptionAlgorithm.AES_256_GCM
        assert config.require_secure_transport == True
        assert config.enforce_srtp == True
        assert config.dtls_timeout_ms == 5000
        assert config.verify_peer_certificate == True
        assert config.key_rotation_interval_hours == 24
    
    def test_enhanced_security_config(self):
        """Test enhanced security configuration."""
        config = SecurityConfig(
            security_level=SecurityLevel.ENHANCED,
            certificate_type=CertificateType.CA_SIGNED,
            key_size=4096,
            preferred_encryption=EncryptionAlgorithm.AES_256_GCM,
            key_rotation_interval_hours=12
        )
        
        assert config.security_level == SecurityLevel.ENHANCED
        assert config.certificate_type == CertificateType.CA_SIGNED
        assert config.key_size == 4096
        assert config.preferred_encryption == EncryptionAlgorithm.AES_256_GCM
        assert config.key_rotation_interval_hours == 12


class TestSecurityLevel:
    """Test security level enum."""
    
    def test_security_levels(self):
        """Test security level values."""
        assert SecurityLevel.MINIMAL.value == "minimal"
        assert SecurityLevel.STANDARD.value == "standard"
        assert SecurityLevel.ENHANCED.value == "enhanced"
        assert SecurityLevel.MAXIMUM.value == "maximum"
    
    def test_security_level_ordering(self):
        """Test security level ordering."""
        levels = [SecurityLevel.MINIMAL, SecurityLevel.STANDARD, 
                 SecurityLevel.ENHANCED, SecurityLevel.MAXIMUM]
        
        # Test that levels are properly ordered
        for i in range(len(levels) - 1):
            assert levels[i].get_numeric_value() < levels[i + 1].get_numeric_value()


class TestCertificateInfo:
    """Test certificate information handling."""
    
    def test_certificate_info_creation(self):
        """Test certificate info creation."""
        valid_until = datetime.now() + timedelta(days=365)
        
        cert_info = CertificateInfo(
            certificate_type=CertificateType.SELF_SIGNED,
            subject="CN=test.example.com",
            issuer="CN=test.example.com",
            serial_number="123456789",
            valid_from=datetime.now(),
            valid_until=valid_until,
            fingerprint="aa:bb:cc:dd:ee:ff",
            key_size=2048
        )
        
        assert cert_info.certificate_type == CertificateType.SELF_SIGNED
        assert cert_info.subject == "CN=test.example.com"
        assert cert_info.fingerprint == "aa:bb:cc:dd:ee:ff"
        assert cert_info.key_size == 2048
    
    def test_certificate_expiry_check(self):
        """Test certificate expiry checking."""
        # Valid certificate
        future_date = datetime.now() + timedelta(days=30)
        cert_info = CertificateInfo(
            certificate_type=CertificateType.SELF_SIGNED,
            valid_until=future_date
        )
        assert cert_info.is_expired() == False
        assert cert_info.days_until_expiry() > 25
        
        # Expired certificate
        past_date = datetime.now() - timedelta(days=1)
        cert_info_expired = CertificateInfo(
            certificate_type=CertificateType.SELF_SIGNED,
            valid_until=past_date
        )
        assert cert_info_expired.is_expired() == True
        assert cert_info_expired.days_until_expiry() < 0


class TestSecurityEvent:
    """Test security event handling."""
    
    def test_security_event_creation(self):
        """Test security event creation."""
        event = SecurityEvent(
            event_type=SecurityEventType.CERTIFICATE_EXPIRED,
            severity=Mock(value="high"),
            message="Certificate has expired",
            component="certificate_manager",
            metadata={"certificate_fingerprint": "aa:bb:cc:dd:ee:ff"}
        )
        
        assert event.event_type == SecurityEventType.CERTIFICATE_EXPIRED
        assert event.severity.value == "high"
        assert event.message == "Certificate has expired"
        assert event.component == "certificate_manager"
        assert "certificate_fingerprint" in event.metadata
    
    def test_event_types(self):
        """Test security event types."""
        assert SecurityEventType.SECURITY_ESTABLISHED.value == "security_established"
        assert SecurityEventType.SECURITY_VIOLATION.value == "security_violation"
        assert SecurityEventType.CERTIFICATE_EXPIRED.value == "certificate_expired"
        assert SecurityEventType.KEY_ROTATION.value == "key_rotation"
        assert SecurityEventType.DTLS_HANDSHAKE_FAILED.value == "dtls_handshake_failed"


class TestKeyRotationManager:
    """Test key rotation manager."""
    
    @pytest.fixture
    def rotation_manager(self, mock_security_config):
        """Create key rotation manager for testing."""
        with patch('communication.security_manager.CRYPTO_AVAILABLE', False):
            manager = KeyRotationManager(mock_security_config)
            return manager
    
    def test_rotation_manager_initialization(self, rotation_manager):
        """Test key rotation manager initialization."""
        assert rotation_manager.config is not None
        assert rotation_manager.current_key_generation == 0
        assert rotation_manager.last_rotation_time is None
        assert rotation_manager.rotation_in_progress == False
    
    @pytest.mark.asyncio
    async def test_key_rotation_needed(self, rotation_manager):
        """Test key rotation need detection."""
        # Initially should need rotation
        assert rotation_manager.is_rotation_needed() == True
        
        # After setting recent rotation time
        rotation_manager.last_rotation_time = time.time()
        assert rotation_manager.is_rotation_needed() == False
        
        # After enough time has passed
        rotation_manager.last_rotation_time = time.time() - (25 * 3600)  # 25 hours ago
        assert rotation_manager.is_rotation_needed() == True
    
    @pytest.mark.asyncio
    async def test_key_rotation_execution(self, rotation_manager):
        """Test key rotation execution."""
        initial_generation = rotation_manager.current_key_generation
        
        # Perform rotation
        result = await rotation_manager.rotate_keys()
        
        # Should succeed with mock implementation
        assert result == True
        assert rotation_manager.current_key_generation > initial_generation
        assert rotation_manager.last_rotation_time is not None
        assert rotation_manager.rotation_in_progress == False
    
    @pytest.mark.asyncio
    async def test_concurrent_rotation_prevention(self, rotation_manager):
        """Test prevention of concurrent rotations."""
        rotation_manager.rotation_in_progress = True
        
        # Should not start new rotation
        result = await rotation_manager.rotate_keys()
        assert result == False


class TestDTLSContext:
    """Test DTLS context management."""
    
    @pytest.fixture
    def dtls_context(self, mock_security_config):
        """Create DTLS context for testing."""
        with patch('communication.security_manager.CRYPTO_AVAILABLE', False):
            context = DTLSContext(mock_security_config)
            return context
    
    def test_dtls_context_initialization(self, dtls_context):
        """Test DTLS context initialization."""
        assert dtls_context.config is not None
        assert dtls_context.is_initialized == False
        assert dtls_context.certificate_info is None
    
    @pytest.mark.asyncio
    async def test_dtls_context_setup(self, dtls_context):
        """Test DTLS context setup."""
        result = await dtls_context.initialize()
        
        # Should succeed with mock implementation
        assert result == True
        assert dtls_context.is_initialized == True
    
    @pytest.mark.asyncio
    async def test_dtls_handshake(self, dtls_context):
        """Test DTLS handshake process."""
        await dtls_context.initialize()
        
        # Mock peer context
        mock_peer = Mock()
        mock_peer.get_peer_certificate = Mock(return_value=Mock())
        
        # Perform handshake
        result = await dtls_context.perform_handshake(mock_peer)
        
        # Should succeed with mock implementation
        assert result == True
    
    def test_certificate_verification(self, dtls_context):
        """Test certificate verification."""
        # Mock certificate
        mock_cert = Mock()
        mock_cert.fingerprint = Mock(return_value=b'test_fingerprint')
        
        # Should verify successfully with mock
        result = dtls_context.verify_peer_certificate(mock_cert)
        assert isinstance(result, bool)


class TestSRTPContext:
    """Test SRTP context management."""
    
    @pytest.fixture
    def srtp_context(self, mock_security_config):
        """Create SRTP context for testing."""
        with patch('communication.security_manager.CRYPTO_AVAILABLE', False):
            context = SRTPContext(mock_security_config)
            return context
    
    def test_srtp_context_initialization(self, srtp_context):
        """Test SRTP context initialization."""
        assert srtp_context.config is not None
        assert srtp_context.is_active == False
        assert srtp_context.encryption_key is None
        assert srtp_context.authentication_key is None
    
    @pytest.mark.asyncio
    async def test_srtp_key_derivation(self, srtp_context):
        """Test SRTP key derivation."""
        master_key = b'test_master_key_12345678901234567890'
        master_salt = b'test_salt_123456'
        
        result = await srtp_context.derive_keys(master_key, master_salt)
        
        # Should succeed with mock implementation
        assert result == True
        assert srtp_context.encryption_key is not None
        assert srtp_context.authentication_key is not None
    
    @pytest.mark.asyncio
    async def test_srtp_encryption(self, srtp_context):
        """Test SRTP packet encryption."""
        # Setup keys first
        await srtp_context.derive_keys(b'test_key' * 8, b'test_salt' * 2)
        
        # Test packet
        rtp_packet = b'test_rtp_packet_data'
        
        # Encrypt packet
        encrypted = await srtp_context.encrypt_packet(rtp_packet)
        
        # Should return encrypted data (mock implementation)
        assert encrypted is not None
        assert isinstance(encrypted, bytes)
    
    @pytest.mark.asyncio
    async def test_srtp_decryption(self, srtp_context):
        """Test SRTP packet decryption."""
        # Setup keys first
        await srtp_context.derive_keys(b'test_key' * 8, b'test_salt' * 2)
        
        # Test encrypted packet
        encrypted_packet = b'encrypted_test_packet'
        
        # Decrypt packet
        decrypted = await srtp_context.decrypt_packet(encrypted_packet)
        
        # Should return decrypted data (mock implementation)
        assert decrypted is not None
        assert isinstance(decrypted, bytes)


class TestSecurityManager:
    """Test security manager functionality."""
    
    @pytest.fixture
    def security_manager(self, mock_security_config):
        """Create security manager for testing."""
        with patch('communication.security_manager.CRYPTO_AVAILABLE', False):
            manager = SecurityManager(mock_security_config)
            return manager
    
    @pytest.fixture
    def security_manager_with_mocks(self, mock_security_config):
        """Create security manager with mocked dependencies."""
        with patch('communication.security_manager.CRYPTO_AVAILABLE', True), \
             patch('communication.security_manager.x509') as mock_x509, \
             patch('communication.security_manager.rsa') as mock_rsa:
            
            # Mock certificate generation
            mock_cert = Mock()
            mock_cert.fingerprint = Mock(return_value=b'mock_fingerprint')
            mock_cert.public_bytes = Mock(return_value=b'mock_cert_pem')
            mock_x509.CertificateBuilder.return_value.subject_name.return_value.issuer_name.return_value.build.return_value = mock_cert
            
            # Mock key generation
            mock_key = Mock()
            mock_rsa.generate_private_key.return_value = mock_key
            
            manager = SecurityManager(mock_security_config)
            return manager
    
    def test_security_manager_initialization(self, security_manager):
        """Test security manager initialization."""
        assert security_manager.config is not None
        assert security_manager.is_initialized == False
        assert security_manager.dtls_context is not None
        assert security_manager.srtp_context is not None
        assert security_manager.key_rotation_manager is not None
    
    @pytest.mark.asyncio
    async def test_security_manager_initialization_flow(self, security_manager):
        """Test security manager initialization flow."""
        result = await security_manager.initialize()
        
        # Should succeed with mock implementation
        assert result == True
        assert security_manager.is_initialized == True
    
    @pytest.mark.asyncio
    async def test_secure_connection_establishment(self, security_manager):
        """Test secure connection establishment."""
        await security_manager.initialize()
        
        # Mock connection context
        mock_connection = Mock()
        mock_connection.get_peer_info = Mock(return_value={"address": "192.168.1.100"})
        
        result = await security_manager.establish_secure_connection(mock_connection)
        
        # Should succeed with mock implementation
        assert result == True
    
    def test_security_policy_enforcement(self, security_manager):
        """Test security policy enforcement."""
        # Test minimum security level enforcement
        policy = SecurityPolicy(
            min_security_level=SecurityLevel.STANDARD,
            require_encryption=True,
            require_authentication=True
        )
        
        # Check if connection meets policy
        connection_info = {
            "security_level": SecurityLevel.ENHANCED,
            "encrypted": True,
            "authenticated": True
        }
        
        result = security_manager.enforce_policy(policy, connection_info)
        assert result == True
        
        # Test policy violation
        weak_connection = {
            "security_level": SecurityLevel.MINIMAL,
            "encrypted": False,
            "authenticated": False
        }
        
        result = security_manager.enforce_policy(policy, weak_connection)
        assert result == False
    
    def test_callback_registration(self, security_manager):
        """Test security event callback registration."""
        events_received = []
        
        def security_callback(event):
            events_received.append(event)
        
        security_manager.on_security_established(security_callback)
        security_manager.on_security_violation(security_callback)
        security_manager.on_certificate_expiry(security_callback)
        
        assert len(security_manager.security_established_callbacks) == 1
        assert len(security_manager.security_violation_callbacks) == 1
        assert len(security_manager.certificate_expiry_callbacks) == 1
    
    @pytest.mark.asyncio
    async def test_security_event_handling(self, security_manager):
        """Test security event handling."""
        events_received = []
        
        def event_handler(event):
            events_received.append(event)
        
        security_manager.on_security_violation(event_handler)
        
        # Trigger security violation
        violation_event = SecurityEvent(
            event_type=SecurityEventType.SECURITY_VIOLATION,
            severity=Mock(value="high"),
            message="Unauthorized access attempt"
        )
        
        await security_manager._trigger_security_violation(violation_event)
        
        assert len(events_received) == 1
        assert events_received[0] == violation_event
    
    @pytest.mark.asyncio
    async def test_certificate_management(self, security_manager_with_mocks):
        """Test certificate management."""
        await security_manager_with_mocks.initialize()
        
        # Should have certificate info after initialization
        cert_info = security_manager_with_mocks.get_certificate_info()
        assert cert_info is not None
        
        # Test certificate renewal
        result = await security_manager_with_mocks.renew_certificate()
        assert result == True
    
    @pytest.mark.asyncio
    async def test_key_rotation_integration(self, security_manager):
        """Test key rotation integration."""
        await security_manager.initialize()
        
        # Trigger key rotation
        result = await security_manager.rotate_encryption_keys()
        
        # Should succeed with mock implementation
        assert result == True
    
    @pytest.mark.asyncio
    async def test_cleanup(self, security_manager):
        """Test security manager cleanup."""
        await security_manager.initialize()
        
        # Cleanup
        await security_manager.cleanup()
        
        # Should reset state
        assert security_manager.is_initialized == False


class TestSecurityIntegration:
    """Test security integration scenarios."""
    
    @pytest.fixture
    def integration_manager(self, mock_security_config):
        """Create security manager for integration testing."""
        with patch('communication.security_manager.CRYPTO_AVAILABLE', False):
            manager = SecurityManager(mock_security_config)
            return manager
    
    @pytest.mark.asyncio
    async def test_full_security_flow(self, integration_manager):
        """Test complete security establishment flow."""
        # Initialize security
        assert await integration_manager.initialize() == True
        
        # Establish secure connection
        mock_peer = Mock()
        assert await integration_manager.establish_secure_connection(mock_peer) == True
        
        # Verify security is active
        assert integration_manager.is_security_active() == True
        
        # Cleanup
        await integration_manager.cleanup()
        assert integration_manager.is_security_active() == False
    
    @pytest.mark.asyncio
    async def test_security_level_upgrade(self, integration_manager):
        """Test security level upgrade."""
        await integration_manager.initialize()
        
        # Start with standard security
        initial_level = integration_manager.get_current_security_level()
        
        # Upgrade to enhanced security
        result = await integration_manager.upgrade_security_level(SecurityLevel.ENHANCED)
        
        # Should succeed with mock implementation
        assert result == True
        
        # Security level should be upgraded
        new_level = integration_manager.get_current_security_level()
        assert new_level.get_numeric_value() >= initial_level.get_numeric_value()
    
    @pytest.mark.asyncio
    async def test_security_failure_recovery(self, integration_manager):
        """Test recovery from security failures."""
        await integration_manager.initialize()
        
        # Simulate security failure
        await integration_manager._handle_security_failure("Test failure")
        
        # Should attempt recovery
        result = await integration_manager.recover_from_failure()
        assert result == True


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.fixture
    def error_manager(self, mock_security_config):
        """Create security manager for error testing."""
        with patch('communication.security_manager.CRYPTO_AVAILABLE', False):
            manager = SecurityManager(mock_security_config)
            return manager
    
    @pytest.mark.asyncio
    async def test_initialization_failure_handling(self, error_manager):
        """Test initialization failure handling."""
        # Mock initialization failure
        with patch.object(error_manager.dtls_context, 'initialize', side_effect=Exception("Init failed")):
            result = await error_manager.initialize()
            
            assert result == False
            assert error_manager.is_initialized == False
    
    @pytest.mark.asyncio
    async def test_connection_failure_handling(self, error_manager):
        """Test connection establishment failure handling."""
        await error_manager.initialize()
        
        # Mock connection failure
        with patch.object(error_manager, '_perform_dtls_handshake', side_effect=Exception("Handshake failed")):
            mock_connection = Mock()
            result = await error_manager.establish_secure_connection(mock_connection)
            
            assert result == False
    
    @pytest.mark.asyncio
    async def test_certificate_error_handling(self, error_manager):
        """Test certificate error handling."""
        # Mock certificate expiry
        expired_cert = CertificateInfo(
            certificate_type=CertificateType.SELF_SIGNED,
            valid_until=datetime.now() - timedelta(days=1)
        )
        
        error_manager.dtls_context.certificate_info = expired_cert
        
        # Should detect expiry and handle gracefully
        assert error_manager._check_certificate_validity() == False
    
    def test_invalid_configuration_handling(self):
        """Test handling of invalid security configuration."""
        # Test with invalid key size
        config = SecurityConfig(key_size=512)  # Too small
        
        with patch('communication.security_manager.CRYPTO_AVAILABLE', False):
            manager = SecurityManager(config)
            
            # Should handle invalid config gracefully
            assert manager is not None
    
    @pytest.mark.asyncio
    async def test_callback_error_handling(self, error_manager):
        """Test error handling in security callbacks."""
        def failing_callback(event):
            raise Exception("Callback failed")
        
        error_manager.on_security_violation(failing_callback)
        
        # Should not crash when callback fails
        violation_event = SecurityEvent(
            event_type=SecurityEventType.SECURITY_VIOLATION,
            severity=Mock(value="medium"),
            message="Test violation"
        )
        
        await error_manager._trigger_security_violation(violation_event)
        # Test continues if no exception raised


class TestPerformance:
    """Test performance characteristics."""
    
    @pytest.fixture
    def perf_manager(self, mock_security_config):
        """Create security manager for performance testing."""
        with patch('communication.security_manager.CRYPTO_AVAILABLE', False):
            manager = SecurityManager(mock_security_config)
            return manager
    
    @pytest.mark.asyncio
    async def test_initialization_performance(self, perf_manager):
        """Test initialization performance."""
        start_time = time.time()
        await perf_manager.initialize()
        init_time = time.time() - start_time
        
        # Should initialize quickly
        assert init_time < 1.0  # Less than 1 second
    
    @pytest.mark.asyncio
    async def test_encryption_performance(self, perf_manager):
        """Test encryption performance."""
        await perf_manager.initialize()
        
        # Test data
        test_data = b'test_data' * 1000  # 9KB of data
        
        start_time = time.time()
        encrypted = await perf_manager.encrypt_data(test_data)
        encryption_time = time.time() - start_time
        
        # Should encrypt quickly
        assert encryption_time < 0.1  # Less than 100ms
        assert encrypted is not None


# Integration test markers
pytestmark = pytest.mark.unit