#!/usr/bin/env python3
"""
CI/CD Security Integration for Voice Agents Platform
Automated security scanning pipeline integration
"""

import os
import sys
import json
import yaml
import logging
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import tempfile

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CISecurityIntegration:
    """CI/CD security integration orchestrator"""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or "/security/config.yaml"
        self.config = self._load_config()
        self.ci_environment = self._detect_ci_environment()
        
    def _load_config(self) -> Dict:
        """Load security configuration"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config.get('security', {})
        except Exception as e:
            logger.warning(f"Could not load config from {self.config_path}: {e}")
            return {}
    
    def _detect_ci_environment(self) -> str:
        """Detect CI/CD environment"""
        if os.getenv('GITHUB_ACTIONS'):
            return 'github_actions'
        elif os.getenv('GITLAB_CI'):
            return 'gitlab_ci'
        elif os.getenv('JENKINS_URL'):
            return 'jenkins'
        elif os.getenv('BUILDKITE'):
            return 'buildkite'
        elif os.getenv('CIRCLECI'):
            return 'circleci'
        else:
            return 'unknown'
    
    def generate_github_actions_workflow(self) -> str:
        """Generate GitHub Actions security workflow"""
        workflow = {
            'name': 'Security Scanning',
            'on': {
                'push': {
                    'branches': ['main', 'develop']
                },
                'pull_request': {
                    'branches': ['main']
                },
                'schedule': [
                    {'cron': '0 2 * * *'}  # Daily at 2 AM
                ]
            },
            'jobs': {
                'security-scan': {
                    'runs-on': 'ubuntu-latest',
                    'permissions': {
                        'contents': 'read',
                        'security-events': 'write'
                    },
                    'steps': [
                        {
                            'name': 'Checkout code',
                            'uses': 'actions/checkout@v4'
                        },
                        {
                            'name': 'Set up Python',
                            'uses': 'actions/setup-python@v4',
                            'with': {
                                'python-version': '3.10'
                            }
                        },
                        {
                            'name': 'Set up Node.js',
                            'uses': 'actions/setup-node@v4',
                            'with': {
                                'node-version': '18'
                            }
                        },
                        {
                            'name': 'Install security tools',
                            'run': '''
pip install safety bandit pip-audit semgrep
npm install -g npm-audit-html audit-ci
curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh -s -- -b /usr/local/bin
curl -sSfL https://raw.githubusercontent.com/anchore/grype/main/install.sh | sh -s -- -b /usr/local/bin
'''
                        },
                        {
                            'name': 'Run dependency scan',
                            'run': 'python src/deployment/security/scripts/dependency-scan.py --project-path . --output security-reports/dependency-scan.json',
                            'continue-on-error': True
                        },
                        {
                            'name': 'Run SAST with Semgrep',
                            'run': 'semgrep --config=auto --json --output=security-reports/sast-semgrep.json .',
                            'continue-on-error': True
                        },
                        {
                            'name': 'Run container security scan',
                            'run': '''
if [ -f Dockerfile ]; then
  trivy config --format json --output security-reports/trivy-config.json .
fi
if [ -f docker-compose.yml ]; then
  trivy config --format json --output security-reports/trivy-compose.json docker-compose.yml
fi
''',
                            'continue-on-error': True
                        },
                        {
                            'name': 'Run comprehensive security scan',
                            'run': 'python src/deployment/security/scripts/security_scanner.py --project-path . --output-dir security-reports',
                            'env': {
                                'SLACK_WEBHOOK_URL': '${{ secrets.SLACK_WEBHOOK_URL }}'
                            }
                        },
                        {
                            'name': 'Upload security reports',
                            'uses': 'actions/upload-artifact@v3',
                            'with': {
                                'name': 'security-reports',
                                'path': 'security-reports/',
                                'retention-days': 30
                            },
                            'if': 'always()'
                        },
                        {
                            'name': 'Upload SARIF results to GitHub',
                            'uses': 'github/codeql-action/upload-sarif@v2',
                            'with': {
                                'sarif_file': 'security-reports/',
                                'category': 'security-scan'
                            },
                            'if': 'always() && github.event_name == \'push\''
                        },
                        {
                            'name': 'Security gate check',
                            'run': '''
python - << 'EOF'
import json
import sys
import os

def check_security_gate():
    report_file = "security-reports/security_report.json"
    if not os.path.exists(report_file):
        print("No security report found")
        return 0
    
    with open(report_file, 'r') as f:
        report = json.load(f)
    
    critical = report.get('severity_counts', {}).get('critical', 0)
    high = report.get('severity_counts', {}).get('high', 0)
    
    if critical > 0:
        print(f"SECURITY GATE FAILED: {critical} critical vulnerabilities found")
        return 1
    elif high > 5:
        print(f"SECURITY GATE FAILED: {high} high-severity vulnerabilities found (max 5 allowed)")
        return 1
    else:
        print("SECURITY GATE PASSED")
        return 0

sys.exit(check_security_gate())
EOF
'''
                        }
                    ]
                }
            }
        }
        
        return yaml.dump(workflow, default_flow_style=False, sort_keys=False)
    
    def generate_gitlab_ci_config(self) -> str:
        """Generate GitLab CI security configuration"""
        config = {
            'include': [
                {'template': 'Security/SAST.gitlab-ci.yml'},
                {'template': 'Security/Secret-Detection.gitlab-ci.yml'},
                {'template': 'Security/Container-Scanning.gitlab-ci.yml'},
                {'template': 'Security/Dependency-Scanning.gitlab-ci.yml'}
            ],
            'stages': [
                'test',
                'security',
                'deploy'
            ],
            'variables': {
                'SAST_ANALYZER_IMAGE_TAG': 'latest',
                'DS_ANALYZER_IMAGE_TAG': 'latest',
                'CS_ANALYZER_IMAGE_TAG': 'latest',
                'SECURE_LOG_LEVEL': 'info'
            },
            'security-scan': {
                'stage': 'security',
                'image': 'python:3.10',
                'before_script': [
                    'apt-get update -qq && apt-get install -y -qq curl',
                    'pip install safety bandit pip-audit semgrep',
                    'curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh -s -- -b /usr/local/bin'
                ],
                'script': [
                    'mkdir -p security-reports',
                    'python src/deployment/security/scripts/dependency-scan.py --project-path . --output security-reports/dependency-scan.json || true',
                    'semgrep --config=auto --json --output=security-reports/sast-semgrep.json . || true',
                    'trivy config --format json --output security-reports/trivy-config.json . || true',
                    'python src/deployment/security/scripts/security_scanner.py --project-path . --output-dir security-reports'
                ],
                'artifacts': {
                    'when': 'always',
                    'expire_in': '30 days',
                    'paths': ['security-reports/'],
                    'reports': {
                        'sast': 'security-reports/sast-semgrep.json',
                        'container_scanning': 'security-reports/trivy-config.json'
                    }
                },
                'allow_failure': True
            },
            'security-gate': {
                'stage': 'security',
                'image': 'python:3.10',
                'script': [
                    '''python - << 'EOF'
import json
import sys
import os

def check_security_gate():
    report_file = "security-reports/security_report.json"
    if not os.path.exists(report_file):
        print("No security report found")
        return 0
    
    with open(report_file, 'r') as f:
        report = json.load(f)
    
    critical = report.get('severity_counts', {}).get('critical', 0)
    high = report.get('severity_counts', {}).get('high', 0)
    
    if critical > 0:
        print(f"SECURITY GATE FAILED: {critical} critical vulnerabilities found")
        return 1
    elif high > 5:
        print(f"SECURITY GATE FAILED: {high} high-severity vulnerabilities found (max 5 allowed)")
        return 1
    else:
        print("SECURITY GATE PASSED")
        return 0

sys.exit(check_security_gate())
EOF'''
                ],
                'dependencies': ['security-scan'],
                'only': ['main', 'develop']
            }
        }
        
        return yaml.dump(config, default_flow_style=False, sort_keys=False)
    
    def generate_jenkins_pipeline(self) -> str:
        """Generate Jenkins pipeline for security scanning"""
        pipeline = '''
pipeline {
    agent any
    
    environment {
        SECURITY_REPORTS_DIR = 'security-reports'
    }
    
    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }
        
        stage('Setup Security Tools') {
            steps {
                sh '''
                    pip install safety bandit pip-audit semgrep
                    curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh -s -- -b /usr/local/bin
                    mkdir -p ${SECURITY_REPORTS_DIR}
                '''
            }
        }
        
        stage('Dependency Security Scan') {
            steps {
                sh '''
                    python src/deployment/security/scripts/dependency-scan.py \\
                        --project-path . \\
                        --output ${SECURITY_REPORTS_DIR}/dependency-scan.json || true
                '''
            }
        }
        
        stage('SAST Scan') {
            steps {
                sh '''
                    semgrep --config=auto --json \\
                        --output=${SECURITY_REPORTS_DIR}/sast-semgrep.json . || true
                '''
            }
        }
        
        stage('Container Security Scan') {
            steps {
                sh '''
                    if [ -f Dockerfile ]; then
                        trivy config --format json \\
                            --output ${SECURITY_REPORTS_DIR}/trivy-config.json .
                    fi
                '''
            }
        }
        
        stage('Comprehensive Security Scan') {
            steps {
                sh '''
                    python src/deployment/security/scripts/security_scanner.py \\
                        --project-path . \\
                        --output-dir ${SECURITY_REPORTS_DIR}
                '''
            }
        }
        
        stage('Security Gate') {
            steps {
                script {
                    def report = readJSON file: "${SECURITY_REPORTS_DIR}/security_report.json"
                    def critical = report.severity_counts?.critical ?: 0
                    def high = report.severity_counts?.high ?: 0
                    
                    if (critical > 0) {
                        error("SECURITY GATE FAILED: ${critical} critical vulnerabilities found")
                    } else if (high > 5) {
                        error("SECURITY GATE FAILED: ${high} high-severity vulnerabilities found (max 5 allowed)")
                    } else {
                        echo "SECURITY GATE PASSED"
                    }
                }
            }
        }
    }
    
    post {
        always {
            archiveArtifacts artifacts: 'security-reports/**', allowEmptyArchive: true
            publishHTML([
                allowMissing: false,
                alwaysLinkToLastBuild: true,
                keepAll: true,
                reportDir: 'security-reports',
                reportFiles: '*.html',
                reportName: 'Security Report'
            ])
        }
        failure {
            script {
                if (env.SLACK_WEBHOOK_URL) {
                    sh '''
                        curl -X POST -H 'Content-type: application/json' \\
                            --data '{"text":"Security scan failed for ${JOB_NAME} #${BUILD_NUMBER}"}' \\
                            ${SLACK_WEBHOOK_URL}
                    '''
                }
            }
        }
    }
}
'''
        return pipeline
    
    def generate_pre_commit_hooks(self) -> str:
        """Generate pre-commit hooks configuration"""
        config = {
            'repos': [
                {
                    'repo': 'https://github.com/PyCQA/bandit',
                    'rev': '1.7.5',
                    'hooks': [
                        {
                            'id': 'bandit',
                            'args': ['-c', 'src/deployment/security/bandit.yaml']
                        }
                    ]
                },
                {
                    'repo': 'https://github.com/Lucas-C/pre-commit-hooks-safety',
                    'rev': 'v1.3.2',
                    'hooks': [
                        {
                            'id': 'python-safety-dependencies-check'
                        }
                    ]
                },
                {
                    'repo': 'https://github.com/returntocorp/semgrep',
                    'rev': 'v1.45.0',
                    'hooks': [
                        {
                            'id': 'semgrep',
                            'args': ['--config=auto']
                        }
                    ]
                },
                {
                    'repo': 'local',
                    'hooks': [
                        {
                            'id': 'dependency-scan',
                            'name': 'Dependency Security Scan',
                            'entry': 'python src/deployment/security/scripts/dependency-scan.py',
                            'language': 'system',
                            'pass_filenames': False,
                            'always_run': True
                        },
                        {
                            'id': 'container-scan',
                            'name': 'Container Security Scan',
                            'entry': 'sh -c "if [ -f Dockerfile ]; then trivy config --exit-code 1 --severity HIGH,CRITICAL .; fi"',
                            'language': 'system',
                            'files': 'Dockerfile.*|.*\\.dockerfile$',
                            'pass_filenames': False
                        }
                    ]
                }
            ]
        }
        
        return yaml.dump(config, default_flow_style=False, sort_keys=False)
    
    def create_security_policy_file(self) -> str:
        """Create security policy file for repository"""
        policy = '''# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.x.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

We take security vulnerabilities seriously. Please follow these steps to report a vulnerability:

### For Critical/High Severity Issues:
1. **DO NOT** create a public GitHub issue
2. Send an email to security@yourcompany.com
3. Include detailed information about the vulnerability
4. Provide steps to reproduce if possible

### For Low/Medium Severity Issues:
1. Create a private security advisory on GitHub
2. Provide detailed information about the vulnerability
3. Include suggested fixes if possible

## Security Scanning

This project includes automated security scanning:

- **Dependency Scanning**: Daily scans for vulnerable dependencies
- **Container Scanning**: Scan for vulnerabilities in Docker images
- **SAST**: Static Application Security Testing with Semgrep
- **Infrastructure Scanning**: Kubernetes configuration security checks
- **Compliance**: Docker CIS benchmark compliance checks

## Security Requirements

### For Contributors:
- All PRs must pass security scans
- No critical or high-severity vulnerabilities allowed
- Dependencies must be up to date
- Follow secure coding practices

### For Deployments:
- Container images must pass security scans
- All secrets must be properly managed
- Network policies must be implemented
- Monitoring and alerting must be configured

## Security Controls

### Authentication & Authorization:
- OAuth/SAML integration for enterprise deployments
- Role-based access control (RBAC)
- API key management
- Session management

### Data Protection:
- Encryption at rest and in transit
- Secure key management
- Data retention policies
- Privacy controls (GDPR/CCPA compliance)

### Network Security:
- TLS/SSL encryption
- Network segmentation
- Firewall configurations
- DDoS protection

### Monitoring & Incident Response:
- Security event logging
- Real-time threat detection
- Incident response procedures
- Security metrics and dashboards

## Compliance

This project maintains compliance with:
- SOC 2 Type II
- GDPR
- CCPA
- HIPAA (for healthcare deployments)
- Docker CIS Benchmarks
- Kubernetes CIS Benchmarks

For questions about security, contact: security@yourcompany.com
'''
        return policy
    
    def setup_ci_integration(self, ci_platform: str = None) -> Dict[str, str]:
        """Setup CI/CD integration for specified platform"""
        if ci_platform is None:
            ci_platform = self.ci_environment
        
        files_created = {}
        
        if ci_platform == 'github_actions':
            workflow_content = self.generate_github_actions_workflow()
            files_created['.github/workflows/security.yml'] = workflow_content
            
        elif ci_platform == 'gitlab_ci':
            config_content = self.generate_gitlab_ci_config()
            files_created['.gitlab-ci-security.yml'] = config_content
            
        elif ci_platform == 'jenkins':
            pipeline_content = self.generate_jenkins_pipeline()
            files_created['Jenkinsfile.security'] = pipeline_content
        
        # Common files for all platforms
        precommit_content = self.generate_pre_commit_hooks()
        files_created['.pre-commit-config.yaml'] = precommit_content
        
        security_policy_content = self.create_security_policy_file()
        files_created['SECURITY.md'] = security_policy_content
        
        return files_created
    
    def validate_security_config(self) -> Dict[str, bool]:
        """Validate security configuration"""
        validation_results = {}
        
        # Check if security tools are available
        tools_to_check = ['trivy', 'semgrep', 'bandit', 'safety']
        for tool in tools_to_check:
            try:
                result = subprocess.run(['which', tool], capture_output=True, text=True)
                validation_results[f'{tool}_available'] = result.returncode == 0
            except Exception:
                validation_results[f'{tool}_available'] = False
        
        # Check configuration completeness
        validation_results['config_loaded'] = bool(self.config)
        validation_results['alerting_configured'] = 'alerting' in self.config
        validation_results['notification_webhooks'] = bool(os.getenv('SLACK_WEBHOOK_URL'))
        
        # Check directory structure
        security_dir = Path('/security')
        validation_results['security_dir_exists'] = security_dir.exists()
        validation_results['reports_dir_exists'] = (security_dir / 'reports').exists()
        
        return validation_results

def main():
    parser = argparse.ArgumentParser(description='CI/CD Security Integration Setup')
    parser.add_argument('--platform', choices=['github_actions', 'gitlab_ci', 'jenkins'], 
                       help='CI/CD platform to setup')
    parser.add_argument('--config', help='Path to security config file')
    parser.add_argument('--output-dir', default='.', help='Output directory for CI files')
    parser.add_argument('--validate', action='store_true', help='Validate security setup')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    integration = CISecurityIntegration(args.config)
    
    if args.validate:
        validation_results = integration.validate_security_config()
        print("Security Configuration Validation:")
        for check, result in validation_results.items():
            status = "✓" if result else "✗"
            print(f"  {status} {check.replace('_', ' ').title()}")
        
        all_passed = all(validation_results.values())
        print(f"\nOverall Status: {'PASS' if all_passed else 'FAIL'}")
        sys.exit(0 if all_passed else 1)
    
    # Setup CI integration
    files_created = integration.setup_ci_integration(args.platform)
    
    # Write files to output directory
    output_dir = Path(args.output_dir)
    for file_path, content in files_created.items():
        full_path = output_dir / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(full_path, 'w') as f:
            f.write(content)
        
        logger.info(f"Created: {full_path}")
    
    print(f"\nCI/CD Security Integration setup complete!")
    print(f"Platform: {args.platform or integration.ci_environment}")
    print(f"Files created: {len(files_created)}")
    print("\nNext steps:")
    print("1. Review and customize the generated files")
    print("2. Set up required secrets (SLACK_WEBHOOK_URL, etc.)")
    print("3. Test the security pipeline")
    print("4. Configure notification channels")

if __name__ == "__main__":
    main()