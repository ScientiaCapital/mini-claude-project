"""
Test-Driven Development: Project Setup Tests
These tests should be written FIRST, before creating any project structure
Following TDD principle: Red -> Green -> Refactor
"""
import os
import pytest


def test_project_directories_exist():
    """Test that all required directories exist"""
    required_dirs = [
        "src", 
        "tests", 
        "data", 
        "notebooks", 
        "resources", 
        "resources/repos"
    ]
    
    for dir_path in required_dirs:
        assert os.path.exists(dir_path), f"Directory {dir_path} does not exist"


def test_readme_exists():
    """Test that README.md exists with basic content"""
    assert os.path.exists("README.md"), "README.md does not exist"
    
    with open("README.md", "r") as f:
        content = f.read()
        assert "Mini-Claude" in content, "README.md should mention Mini-Claude"
        assert "## Installation" in content, "README.md should have Installation section"
        assert "## Quick Start" in content, "README.md should have Quick Start section"


def test_requirements_file_exists():
    """Test that requirements.txt exists with core dependencies"""
    assert os.path.exists("requirements.txt"), "requirements.txt does not exist"
    
    with open("requirements.txt", "r") as f:
        content = f.read().lower()
        
        # Core dependencies for our project
        required_packages = [
            "torch",
            "transformers",
            "gradio",
            "pytest",  # TDD requires pytest!
            "datasets",
            "accelerate",
            "peft",  # For LoRA
        ]
        
        for package in required_packages:
            assert package in content, f"requirements.txt should include {package}"


def test_gitignore_exists():
    """Test that .gitignore exists with proper Python entries"""
    assert os.path.exists(".gitignore"), ".gitignore does not exist"
    
    with open(".gitignore", "r") as f:
        content = f.read()
        
        # Essential Python gitignore entries (or their equivalents)
        required_patterns = [
            ("__pycache__", "__pycache__"),
            ("*.pyc", ["*.pyc", "*.py[cod]"]),  # *.py[cod] includes *.pyc
            ("venv/", "venv/"),
            ("*.egg-info", "*.egg-info"),
            (".pytest_cache", ".pytest_cache"),
            ("*.log", "*.log"),
            (".env", ".env"),
            ("models/", "models/"),  # Don't commit large model files
        ]
        
        for name, patterns in required_patterns:
            if isinstance(patterns, str):
                patterns = [patterns]
            
            found = any(pattern in content for pattern in patterns)
            assert found, f".gitignore should include {name} (or equivalent)"


def test_src_package_structure():
    """Test that src is a proper Python package"""
    assert os.path.exists("src/__init__.py"), "src should be a Python package with __init__.py"


def test_tests_package_structure():
    """Test that tests directory has proper structure"""
    assert os.path.exists("tests/__init__.py"), "tests should be a Python package"
    assert os.path.exists("tests/unit"), "tests/unit directory should exist"
    assert os.path.exists("tests/integration"), "tests/integration directory should exist"