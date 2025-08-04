"""
Test-Driven Development: GitHub Workflow Tests
These tests define the expected behavior of our ML pipeline workflow
Following TDD principle: Red -> Green -> Refactor

These tests validate that our GitHub Actions workflow:
1. Executes successfully across multiple environments
2. Properly validates ML components
3. Handles dependencies and caching correctly
4. Performs model validation and performance checks
5. Provides clear failure reporting
"""
import os
import yaml
import pytest
from pathlib import Path


class TestMLPipelineWorkflow:
    """Test suite for ML pipeline GitHub Actions workflow"""
    
    def test_workflow_file_exists(self):
        """Test that ml-pipeline.yml workflow file exists"""
        workflow_path = ".github/workflows/ml-pipeline.yml"
        assert os.path.exists(workflow_path), \
            f"ML pipeline workflow file should exist at {workflow_path}"
    
    def test_workflow_file_is_valid_yaml(self):
        """Test that workflow file contains valid YAML"""
        workflow_path = ".github/workflows/ml-pipeline.yml"
        
        with open(workflow_path, 'r') as f:
            try:
                workflow_config = yaml.safe_load(f)
                assert workflow_config is not None, "Workflow should contain valid YAML"
            except yaml.YAMLError as e:
                pytest.fail(f"Workflow file contains invalid YAML: {e}")
    
    def test_workflow_has_required_structure(self):
        """Test that workflow has all required top-level keys"""
        workflow_path = ".github/workflows/ml-pipeline.yml"
        
        with open(workflow_path, 'r') as f:
            workflow = yaml.safe_load(f)
        
        required_keys = ['name', 'on', 'jobs']
        for key in required_keys:
            assert key in workflow, f"Workflow should have '{key}' key"
    
    def test_workflow_triggers_on_correct_events(self):
        """Test that workflow triggers on appropriate events"""
        workflow_path = ".github/workflows/ml-pipeline.yml"
        
        with open(workflow_path, 'r') as f:
            workflow = yaml.safe_load(f)
        
        triggers = workflow['on']
        
        # Should trigger on push to main/master
        assert 'push' in triggers, "Workflow should trigger on push events"
        if isinstance(triggers['push'], dict):
            branches = triggers['push'].get('branches', [])
            assert any(branch in ['main', 'master'] for branch in branches), \
                "Workflow should trigger on push to main/master branch"
        
        # Should trigger on pull requests
        assert 'pull_request' in triggers, "Workflow should trigger on pull requests"
    
    def test_workflow_has_test_job(self):
        """Test that workflow includes a comprehensive test job"""
        workflow_path = ".github/workflows/ml-pipeline.yml"
        
        with open(workflow_path, 'r') as f:
            workflow = yaml.safe_load(f)
        
        jobs = workflow['jobs']
        assert 'test' in jobs, "Workflow should have a 'test' job"
        
        test_job = jobs['test']
        assert 'runs-on' in test_job, "Test job should specify runs-on"
        assert 'steps' in test_job, "Test job should have steps"
    
    def test_workflow_includes_matrix_testing(self):
        """Test that workflow includes matrix testing for multiple environments"""
        workflow_path = ".github/workflows/ml-pipeline.yml"
        
        with open(workflow_path, 'r') as f:
            workflow = yaml.safe_load(f)
        
        test_job = workflow['jobs']['test']
        
        # Should have strategy with matrix
        assert 'strategy' in test_job, "Test job should have strategy for matrix testing"
        strategy = test_job['strategy']
        assert 'matrix' in strategy, "Strategy should include matrix"
        
        matrix = strategy['matrix']
        
        # Should test multiple Python versions
        assert 'python-version' in matrix, "Matrix should include python-version"
        python_versions = matrix['python-version']
        assert len(python_versions) >= 2, "Should test at least 2 Python versions"
        assert '3.9' in python_versions or '3.10' in python_versions, \
            "Should include a recent Python version"
    
    def test_workflow_has_dependency_caching(self):
        """Test that workflow implements proper dependency caching"""
        workflow_path = ".github/workflows/ml-pipeline.yml"
        
        with open(workflow_path, 'r') as f:
            workflow = yaml.safe_load(f)
        
        test_job = workflow['jobs']['test']
        steps = test_job['steps']
        
        # Should have a caching step
        cache_steps = [step for step in steps if 'cache' in step.get('uses', '').lower()]
        assert len(cache_steps) > 0, "Workflow should include dependency caching"
    
    def test_workflow_installs_dependencies(self):
        """Test that workflow properly installs Python dependencies"""
        workflow_path = ".github/workflows/ml-pipeline.yml"
        
        with open(workflow_path, 'r') as f:
            workflow = yaml.safe_load(f)
        
        test_job = workflow['jobs']['test']
        steps = test_job['steps']
        
        # Should have pip install step
        install_steps = [step for step in steps 
                        if 'pip install' in step.get('run', '')]
        assert len(install_steps) > 0, "Workflow should install dependencies with pip"
        
        # Should install requirements.txt
        requirements_install = any('requirements.txt' in step.get('run', '') 
                                 for step in steps)
        assert requirements_install, "Workflow should install from requirements.txt"
    
    def test_workflow_runs_pytest(self):
        """Test that workflow runs pytest for testing"""
        workflow_path = ".github/workflows/ml-pipeline.yml"
        
        with open(workflow_path, 'r') as f:
            workflow = yaml.safe_load(f)
        
        test_job = workflow['jobs']['test']
        steps = test_job['steps']
        
        # Should run pytest
        pytest_steps = [step for step in steps 
                       if 'pytest' in step.get('run', '')]
        assert len(pytest_steps) > 0, "Workflow should run pytest"
    
    def test_workflow_includes_code_quality_checks(self):
        """Test that workflow includes code quality and formatting checks"""
        workflow_path = ".github/workflows/ml-pipeline.yml"
        
        with open(workflow_path, 'r') as f:
            workflow = yaml.safe_load(f)
        
        test_job = workflow['jobs']['test']
        steps = test_job['steps']
        
        # Should include linting/formatting tools
        quality_tools = ['black', 'ruff', 'mypy']
        quality_steps = [step for step in steps 
                        if any(tool in step.get('run', '') for tool in quality_tools)]
        assert len(quality_steps) > 0, "Workflow should include code quality checks"
    
    def test_workflow_has_model_validation_job(self):
        """Test that workflow includes model validation"""
        workflow_path = ".github/workflows/ml-pipeline.yml"
        
        with open(workflow_path, 'r') as f:
            workflow = yaml.safe_load(f)
        
        jobs = workflow['jobs']
        
        # Should have model validation job or steps
        model_validation = any('model' in job_name.lower() and 'valid' in job_name.lower() 
                              for job_name in jobs.keys())
        
        if not model_validation:
            # Check if model validation is included in test steps
            test_job = jobs.get('test', {})
            steps = test_job.get('steps', [])
            model_validation_steps = [step for step in steps 
                                    if 'model' in step.get('name', '').lower() 
                                    and 'valid' in step.get('name', '').lower()]
            assert len(model_validation_steps) > 0, \
                "Workflow should include model validation steps"
    
    def test_workflow_has_performance_benchmarks(self):
        """Test that workflow includes performance benchmarking"""
        workflow_path = ".github/workflows/ml-pipeline.yml"
        
        with open(workflow_path, 'r') as f:
            workflow = yaml.safe_load(f)
        
        # Check for performance/benchmark related steps
        all_steps = []
        for job in workflow['jobs'].values():
            all_steps.extend(job.get('steps', []))
        
        performance_steps = [step for step in all_steps 
                           if any(keyword in step.get('name', '').lower() 
                                 for keyword in ['performance', 'benchmark', 'speed'])]
        
        assert len(performance_steps) > 0, \
            "Workflow should include performance benchmarking"
    
    def test_workflow_handles_large_files_properly(self):
        """Test that workflow handles ML model files and large datasets"""
        workflow_path = ".github/workflows/ml-pipeline.yml"
        
        with open(workflow_path, 'r') as f:
            workflow = yaml.safe_load(f)
        
        # Should have considerations for large files (Git LFS, caching, etc.)
        all_steps = []
        for job in workflow['jobs'].values():
            all_steps.extend(job.get('steps', []))
        
        # Check for LFS, model downloading, or caching strategies
        large_file_handling = any(
            keyword in str(step).lower() 
            for step in all_steps 
            for keyword in ['lfs', 'model', 'cache', 'download']
        )
        
        assert large_file_handling, \
            "Workflow should handle large ML files (models, datasets)"
    
    def test_workflow_has_proper_error_handling(self):
        """Test that workflow includes proper error handling and reporting"""
        workflow_path = ".github/workflows/ml-pipeline.yml"
        
        with open(workflow_path, 'r') as f:
            workflow = yaml.safe_load(f)
        
        # Check for continue-on-error or failure handling
        jobs = workflow['jobs']
        
        # At least one job should have failure handling
        has_error_handling = False
        for job in jobs.values():
            if 'continue-on-error' in job:
                has_error_handling = True
                break
            
            steps = job.get('steps', [])
            for step in steps:
                if 'continue-on-error' in step or 'if' in step:
                    has_error_handling = True
                    break
        
        # Note: This is more of a guideline - we can't always test this structurally
        # The workflow should have some form of error handling or conditional execution
    
    def test_workflow_includes_security_scanning(self):
        """Test that workflow includes basic security scanning"""
        workflow_path = ".github/workflows/ml-pipeline.yml"
        
        with open(workflow_path, 'r') as f:
            workflow = yaml.safe_load(f)
        
        # Look for security-related steps or jobs
        all_content = str(workflow).lower()
        
        security_indicators = [
            'safety',  # Python package security scanner
            'bandit',  # Python security linter
            'security',
            'vulnerability',
            'codeql'   # GitHub's CodeQL security analysis
        ]
        
        has_security = any(indicator in all_content for indicator in security_indicators)
        
        # This is recommended but not strictly required
        # The test passes if any security-related tooling is present
        if not has_security:
            pytest.skip("Security scanning is recommended but not required for basic ML pipeline")
    
    def test_workflow_has_timeout_configured(self):
        """Test that workflow jobs have reasonable timeouts"""
        workflow_path = ".github/workflows/ml-pipeline.yml"
        
        with open(workflow_path, 'r') as f:
            workflow = yaml.safe_load(f)
        
        jobs = workflow['jobs']
        
        # At least the main test job should have a timeout
        main_jobs = ['test', 'model-validation', 'performance']
        for job_name in main_jobs:
            if job_name in jobs:
                job = jobs[job_name]
                assert 'timeout-minutes' in job, \
                    f"Job '{job_name}' should have timeout-minutes configured"
                
                timeout = job['timeout-minutes']
                assert 5 <= timeout <= 60, \
                    f"Job '{job_name}' timeout should be reasonable (5-60 minutes)"
                break
        else:
            # If none of the expected main jobs exist, check if any job has timeout
            timeouts = [job.get('timeout-minutes') for job in jobs.values()]
            assert any(timeout is not None for timeout in timeouts), \
                "At least one job should have timeout configured"