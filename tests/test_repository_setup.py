"""
Test-Driven Development: Repository Setup Tests
These tests verify that all required learning repositories are cloned
and accessible for our Mini-Claude project.
"""
import os
import pytest
import sys


REQUIRED_REPOS = {
    "LLMs-from-scratch": {
        "url": "https://github.com/rasbt/LLMs-from-scratch.git",
        "key_files": ["README.md", "ch02", "ch03", "requirements.txt"],
        "description": "Core transformer education"
    },
    "course": {
        "url": "https://github.com/huggingface/course.git",
        "key_files": ["chapters", "requirements.txt"],
        "description": "Hugging Face transformers course"
    },
    "ai-gradio": {
        "url": "https://github.com/AK391/ai-gradio.git",
        "key_files": ["README.md", "ai_gradio"],
        "description": "Quick interface building"
    },
    "LLaMA-Factory": {
        "url": "https://github.com/hiyouga/LLaMA-Factory.git",
        "key_files": ["README.md", "src", "requirements.txt"],
        "description": "Training and fine-tuning framework"
    },
    "lobe-chat": {
        "url": "https://github.com/lobehub/lobe-chat.git",
        "key_files": ["README.md", "package.json"],
        "description": "Modern UI reference"
    },
}

ADDITIONAL_REPOS = {
    "picoGPT": {
        "url": "https://github.com/jaymody/picoGPT.git",
        "key_files": ["gpt2.py", "README.md"],
        "description": "Minimal GPT implementation"
    },
    "LoRA": {
        "url": "https://github.com/microsoft/LoRA.git",
        "key_files": ["README.md"],
        "description": "Official LoRA implementation"
    },
}


def test_resources_directory_exists():
    """Test that resources/repos directory exists"""
    assert os.path.exists("resources"), "resources directory does not exist"
    assert os.path.exists("resources/repos"), "resources/repos directory does not exist"


def test_all_required_repos_cloned():
    """Test that all required repositories are cloned"""
    for repo_name in REQUIRED_REPOS.keys():
        repo_path = f"resources/repos/{repo_name}"
        assert os.path.exists(repo_path), f"Repository {repo_name} not found at {repo_path}"
        assert os.path.exists(f"{repo_path}/.git"), f"Repository {repo_name} is not a git repository"


def test_additional_repos_cloned():
    """Test that additional valuable repositories are cloned"""
    for repo_name in ADDITIONAL_REPOS.keys():
        repo_path = f"resources/repos/{repo_name}"
        assert os.path.exists(repo_path), f"Repository {repo_name} not found at {repo_path}"
        assert os.path.exists(f"{repo_path}/.git"), f"Repository {repo_name} is not a git repository"


def test_repo_has_expected_files():
    """Test that cloned repos have expected key files"""
    all_repos = {**REQUIRED_REPOS, **ADDITIONAL_REPOS}
    
    for repo_name, repo_info in all_repos.items():
        repo_path = f"resources/repos/{repo_name}"
        
        if os.path.exists(repo_path):
            for expected_file in repo_info["key_files"]:
                file_path = os.path.join(repo_path, expected_file)
                assert os.path.exists(file_path), \
                    f"Expected file/directory '{expected_file}' not found in {repo_name}"


def test_can_import_from_picoGPT():
    """Test that we can import from the picoGPT repository"""
    repo_path = "resources/repos/picoGPT"
    
    if os.path.exists(repo_path):
        # Add to Python path temporarily
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)
        
        try:
            import gpt2
            assert hasattr(gpt2, 'gpt2'), "gpt2 module should have gpt2 function"
        except ImportError as e:
            pytest.fail(f"Cannot import gpt2 from picoGPT repo: {e}")
        finally:
            # Clean up sys.path
            if repo_path in sys.path:
                sys.path.remove(repo_path)


def test_repo_guide_exists():
    """Test that repository guide documentation exists"""
    guide_path = "resources/REPO_GUIDE.md"
    assert os.path.exists(guide_path), "REPO_GUIDE.md should exist in resources/"
    
    with open(guide_path, "r") as f:
        content = f.read()
        
        # Check that all repos are documented
        for repo_name in REQUIRED_REPOS.keys():
            assert repo_name in content, f"{repo_name} should be documented in REPO_GUIDE.md"


def test_learning_resources_accessible():
    """Test that key learning files are accessible from cloned repos"""
    critical_learning_files = [
        "resources/repos/LLMs-from-scratch/ch02/01_main-chapter-code/ch02.ipynb",
        "resources/repos/course/chapters/en/chapter1/1.mdx",
        "resources/repos/picoGPT/gpt2.py",
    ]
    
    # Only test files if their parent repo exists
    for file_path in critical_learning_files:
        repo_name = file_path.split("/")[2]  # Extract repo name from path
        repo_path = f"resources/repos/{repo_name}"
        
        if os.path.exists(repo_path):
            assert os.path.exists(file_path), \
                f"Critical learning resource not found: {file_path}"