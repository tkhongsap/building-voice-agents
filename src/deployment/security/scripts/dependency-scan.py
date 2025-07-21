#!/usr/bin/env python3
"""
Dependency Security Scanner for Voice Agents Platform
Comprehensive security scanning for Python and Node.js dependencies
"""

import os
import sys
import json
import subprocess
import logging
import argparse
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import concurrent.futures
import tempfile

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DependencyScanner:
    """Main dependency security scanner class"""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or "/security/config.yaml"
        self.config = self._load_config()
        self.results = {
            'scan_timestamp': datetime.utcnow().isoformat(),
            'python': {},
            'nodejs': {},
            'summary': {
                'total_vulnerabilities': 0,
                'critical': 0,
                'high': 0,
                'medium': 0,
                'low': 0,
                'info': 0
            }
        }
        
    def _load_config(self) -> Dict:
        """Load security configuration"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config.get('security', {}).get('dependency_scanning', {})
        except Exception as e:
            logger.warning(f"Could not load config from {self.config_path}: {e}")
            return self._default_config()
    
    def _default_config(self) -> Dict:
        """Default configuration if config file not found"""
        return {
            'enabled': True,
            'python': {
                'safety': {'enabled': True},
                'bandit': {'enabled': True},
                'pip_audit': {'enabled': True}
            },
            'nodejs': {
                'npm_audit': {'enabled': True},
                'yarn_audit': {'enabled': True}
            }
        }
    
    def _run_command(self, command: List[str], cwd: str = None, timeout: int = 300) -> Tuple[int, str, str]:
        """Run shell command and return exit code, stdout, stderr"""
        try:
            result = subprocess.run(
                command,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            logger.error(f"Command timed out: {' '.join(command)}")
            return 1, "", "Command timed out"
        except Exception as e:
            logger.error(f"Error running command {' '.join(command)}: {e}")
            return 1, "", str(e)
    
    def scan_python_safety(self, project_path: str) -> Dict:
        """Scan Python dependencies with Safety"""
        logger.info("Running Safety scan for Python dependencies")
        
        safety_config = self.config.get('python', {}).get('safety', {})
        if not safety_config.get('enabled', True):
            return {'skipped': True, 'reason': 'Safety scanning disabled'}
        
        # Create temporary requirements file if needed
        requirements_files = []
        for req_file in ['requirements.txt', 'backend/requirements.txt', 'requirements-dev.txt']:
            req_path = os.path.join(project_path, req_file)
            if os.path.exists(req_path):
                requirements_files.append(req_path)
        
        if not requirements_files:
            return {'error': 'No requirements.txt files found'}
        
        results = []
        for req_file in requirements_files:
            command = ['safety', 'check', '--json', '-r', req_file]
            
            # Add ignore file if exists
            ignore_file = safety_config.get('ignore_files', ['.safety-ignore'])[0]
            ignore_path = os.path.join(project_path, 'src/deployment/security', ignore_file)
            if os.path.exists(ignore_path):
                command.extend(['--ignore-unpinned'])
            
            exit_code, stdout, stderr = self._run_command(command, cwd=project_path)
            
            try:
                if stdout:
                    result = json.loads(stdout)
                    results.extend(result)
            except json.JSONDecodeError:
                logger.warning(f"Could not parse Safety output for {req_file}")
        
        # Process results
        vulnerabilities = []
        for vuln in results:
            severity = self._map_safety_severity(vuln.get('vulnerability_id', ''))
            vulnerabilities.append({
                'id': vuln.get('vulnerability_id'),
                'package': vuln.get('package_name'),
                'installed_version': vuln.get('installed_version'),
                'vulnerable_spec': vuln.get('vulnerable_spec'),
                'severity': severity,
                'description': vuln.get('advisory'),
                'cve': vuln.get('cve'),
                'source': 'safety'
            })
            self.results['summary'][severity] += 1
            self.results['summary']['total_vulnerabilities'] += 1
        
        return {
            'tool': 'safety',
            'vulnerabilities': vulnerabilities,
            'total_count': len(vulnerabilities)
        }
    
    def scan_python_bandit(self, project_path: str) -> Dict:
        """Scan Python code with Bandit"""
        logger.info("Running Bandit scan for Python code security")
        
        bandit_config = self.config.get('python', {}).get('bandit', {})
        if not bandit_config.get('enabled', True):
            return {'skipped': True, 'reason': 'Bandit scanning disabled'}
        
        # Check for Python source directories
        python_dirs = []
        for dir_name in ['src', 'backend', '.']:
            dir_path = os.path.join(project_path, dir_name)
            if os.path.exists(dir_path) and any(f.endswith('.py') for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))):
                python_dirs.append(dir_name)
        
        if not python_dirs:
            return {'error': 'No Python source directories found'}
        
        command = ['bandit', '-f', 'json', '-r'] + python_dirs
        
        # Add config file if exists
        config_file = os.path.join(project_path, 'src/deployment/security/bandit.yaml')
        if os.path.exists(config_file):
            command.extend(['-c', config_file])
        
        exit_code, stdout, stderr = self._run_command(command, cwd=project_path)
        
        try:
            if stdout:
                result = json.loads(stdout)
                vulnerabilities = []
                
                for issue in result.get('results', []):
                    severity = issue.get('issue_severity', 'MEDIUM').lower()
                    vulnerabilities.append({
                        'id': issue.get('test_id'),
                        'file': issue.get('filename'),
                        'line': issue.get('line_number'),
                        'severity': severity,
                        'confidence': issue.get('issue_confidence', 'MEDIUM').lower(),
                        'description': issue.get('issue_text'),
                        'code': issue.get('code'),
                        'source': 'bandit'
                    })
                    self.results['summary'][severity] += 1
                    self.results['summary']['total_vulnerabilities'] += 1
                
                return {
                    'tool': 'bandit',
                    'vulnerabilities': vulnerabilities,
                    'total_count': len(vulnerabilities),
                    'metrics': result.get('metrics', {})
                }
            else:
                return {'error': 'No output from Bandit'}
        except json.JSONDecodeError as e:
            logger.error(f"Could not parse Bandit output: {e}")
            return {'error': f'Failed to parse Bandit output: {e}'}
    
    def scan_python_pip_audit(self, project_path: str) -> Dict:
        """Scan Python dependencies with pip-audit"""
        logger.info("Running pip-audit scan for Python dependencies")
        
        pip_audit_config = self.config.get('python', {}).get('pip_audit', {})
        if not pip_audit_config.get('enabled', True):
            return {'skipped': True, 'reason': 'pip-audit scanning disabled'}
        
        command = ['pip-audit', '--format=json', '--progress-spinner=off']
        
        # Scan requirements files
        for req_file in ['requirements.txt', 'backend/requirements.txt']:
            req_path = os.path.join(project_path, req_file)
            if os.path.exists(req_path):
                command.extend(['-r', req_path])
                break
        
        exit_code, stdout, stderr = self._run_command(command, cwd=project_path)
        
        try:
            if stdout:
                result = json.loads(stdout)
                vulnerabilities = []
                
                for vuln in result.get('vulnerabilities', []):
                    severity = self._map_pip_audit_severity(vuln)
                    vulnerabilities.append({
                        'id': vuln.get('id'),
                        'package': vuln.get('package'),
                        'installed_version': vuln.get('installed_version'),
                        'fixed_versions': vuln.get('fixed_versions', []),
                        'severity': severity,
                        'description': vuln.get('description'),
                        'aliases': vuln.get('aliases', []),
                        'source': 'pip-audit'
                    })
                    self.results['summary'][severity] += 1
                    self.results['summary']['total_vulnerabilities'] += 1
                
                return {
                    'tool': 'pip-audit',
                    'vulnerabilities': vulnerabilities,
                    'total_count': len(vulnerabilities)
                }
            else:
                return {'info': 'No vulnerabilities found by pip-audit'}
        except json.JSONDecodeError as e:
            logger.error(f"Could not parse pip-audit output: {e}")
            return {'error': f'Failed to parse pip-audit output: {e}'}
    
    def scan_nodejs_npm_audit(self, project_path: str) -> Dict:
        """Scan Node.js dependencies with npm audit"""
        logger.info("Running npm audit scan for Node.js dependencies")
        
        npm_config = self.config.get('nodejs', {}).get('npm_audit', {})
        if not npm_config.get('enabled', True):
            return {'skipped': True, 'reason': 'npm audit scanning disabled'}
        
        # Check for package.json
        package_json_dirs = []
        for dir_name in ['frontend', '.']:
            package_path = os.path.join(project_path, dir_name, 'package.json')
            if os.path.exists(package_path):
                package_json_dirs.append(os.path.join(project_path, dir_name))
        
        if not package_json_dirs:
            return {'error': 'No package.json files found'}
        
        all_vulnerabilities = []
        for npm_dir in package_json_dirs:
            command = ['npm', 'audit', '--json']
            if npm_config.get('production_only', True):
                command.append('--production')
            
            exit_code, stdout, stderr = self._run_command(command, cwd=npm_dir)
            
            try:
                if stdout:
                    result = json.loads(stdout)
                    vulnerabilities = []
                    
                    for vuln_id, vuln_data in result.get('vulnerabilities', {}).items():
                        severity = vuln_data.get('severity', 'low')
                        vulnerabilities.append({
                            'id': vuln_id,
                            'package': vuln_data.get('name'),
                            'severity': severity,
                            'description': vuln_data.get('title'),
                            'vulnerable_versions': vuln_data.get('range'),
                            'patched_versions': vuln_data.get('fixAvailable', {}),
                            'cwe': vuln_data.get('cwe', []),
                            'source': 'npm-audit',
                            'directory': npm_dir
                        })
                        self.results['summary'][severity] += 1
                        self.results['summary']['total_vulnerabilities'] += 1
                    
                    all_vulnerabilities.extend(vulnerabilities)
            except json.JSONDecodeError as e:
                logger.warning(f"Could not parse npm audit output for {npm_dir}: {e}")
        
        return {
            'tool': 'npm-audit',
            'vulnerabilities': all_vulnerabilities,
            'total_count': len(all_vulnerabilities)
        }
    
    def scan_nodejs_yarn_audit(self, project_path: str) -> Dict:
        """Scan Node.js dependencies with yarn audit"""
        logger.info("Running yarn audit scan for Node.js dependencies")
        
        yarn_config = self.config.get('nodejs', {}).get('yarn_audit', {})
        if not yarn_config.get('enabled', True):
            return {'skipped': True, 'reason': 'yarn audit scanning disabled'}
        
        # Check for yarn.lock files
        yarn_dirs = []
        for dir_name in ['frontend', '.']:
            yarn_lock_path = os.path.join(project_path, dir_name, 'yarn.lock')
            if os.path.exists(yarn_lock_path):
                yarn_dirs.append(os.path.join(project_path, dir_name))
        
        if not yarn_dirs:
            return {'info': 'No yarn.lock files found, skipping yarn audit'}
        
        all_vulnerabilities = []
        for yarn_dir in yarn_dirs:
            command = ['yarn', 'audit', '--json']
            level = yarn_config.get('level', 'moderate')
            command.extend(['--level', level])
            
            exit_code, stdout, stderr = self._run_command(command, cwd=yarn_dir)
            
            # Parse line-delimited JSON output
            vulnerabilities = []
            if stdout:
                for line in stdout.strip().split('\n'):
                    try:
                        data = json.loads(line)
                        if data.get('type') == 'auditAdvisory':
                            advisory = data.get('data', {}).get('advisory', {})
                            severity = advisory.get('severity', 'low')
                            vulnerabilities.append({
                                'id': advisory.get('id'),
                                'package': advisory.get('module_name'),
                                'severity': severity,
                                'description': advisory.get('title'),
                                'vulnerable_versions': advisory.get('vulnerable_versions'),
                                'patched_versions': advisory.get('patched_versions'),
                                'cwe': advisory.get('cwe'),
                                'source': 'yarn-audit',
                                'directory': yarn_dir
                            })
                            self.results['summary'][severity] += 1
                            self.results['summary']['total_vulnerabilities'] += 1
                    except json.JSONDecodeError:
                        continue
            
            all_vulnerabilities.extend(vulnerabilities)
        
        return {
            'tool': 'yarn-audit',
            'vulnerabilities': all_vulnerabilities,
            'total_count': len(all_vulnerabilities)
        }
    
    def _map_safety_severity(self, vuln_id: str) -> str:
        """Map Safety vulnerability to severity level"""
        # Safety doesn't provide severity, so we use a basic mapping
        # In a real implementation, you'd use CVE databases or Safety's commercial API
        return 'medium'  # Default severity
    
    def _map_pip_audit_severity(self, vuln: Dict) -> str:
        """Map pip-audit vulnerability to severity level"""
        # pip-audit provides severity information in some cases
        return vuln.get('severity', 'medium').lower()
    
    def scan_all(self, project_path: str) -> Dict:
        """Run all enabled dependency scans"""
        logger.info(f"Starting dependency security scan for {project_path}")
        
        if not self.config.get('enabled', True):
            logger.info("Dependency scanning is disabled")
            return {'skipped': True, 'reason': 'Dependency scanning disabled'}
        
        # Python scans
        if self.config.get('python', {}).get('safety', {}).get('enabled', True):
            self.results['python']['safety'] = self.scan_python_safety(project_path)
        
        if self.config.get('python', {}).get('bandit', {}).get('enabled', True):
            self.results['python']['bandit'] = self.scan_python_bandit(project_path)
        
        if self.config.get('python', {}).get('pip_audit', {}).get('enabled', True):
            self.results['python']['pip_audit'] = self.scan_python_pip_audit(project_path)
        
        # Node.js scans
        if self.config.get('nodejs', {}).get('npm_audit', {}).get('enabled', True):
            self.results['nodejs']['npm_audit'] = self.scan_nodejs_npm_audit(project_path)
        
        if self.config.get('nodejs', {}).get('yarn_audit', {}).get('enabled', True):
            self.results['nodejs']['yarn_audit'] = self.scan_nodejs_yarn_audit(project_path)
        
        self.results['scan_completed'] = datetime.utcnow().isoformat()
        return self.results
    
    def save_results(self, output_path: str):
        """Save scan results to file"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Results saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Dependency Security Scanner')
    parser.add_argument('--project-path', default='.', help='Path to project to scan')
    parser.add_argument('--config', help='Path to security config file')
    parser.add_argument('--output', default='/security/reports/dependency-scan.json', 
                       help='Output file for results')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    scanner = DependencyScanner(args.config)
    results = scanner.scan_all(args.project_path)
    scanner.save_results(args.output)
    
    # Print summary
    summary = results.get('summary', {})
    print(f"\nDependency Security Scan Complete")
    print(f"Total vulnerabilities found: {summary.get('total_vulnerabilities', 0)}")
    print(f"  Critical: {summary.get('critical', 0)}")
    print(f"  High: {summary.get('high', 0)}")
    print(f"  Medium: {summary.get('medium', 0)}")
    print(f"  Low: {summary.get('low', 0)}")
    print(f"  Info: {summary.get('info', 0)}")
    
    # Exit with error code if high severity vulnerabilities found
    if summary.get('critical', 0) > 0 or summary.get('high', 0) > 0:
        sys.exit(1)

if __name__ == "__main__":
    main()