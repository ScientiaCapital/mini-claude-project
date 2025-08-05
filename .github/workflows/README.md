# ML Pipeline GitHub Actions Workflow

## Overview

This directory contains the GitHub Actions workflow for automated testing and validation of our ML components. The workflow follows Test-Driven Development (TDD) principles and provides comprehensive CI/CD for machine learning projects.

## Workflow: `ml-pipeline.yml`

### Architecture

The workflow consists of 4 main jobs that run in sequence:

1. **`test`** - Core testing with matrix strategy
2. **`model-validation`** - ML-specific validation and performance tests
3. **`integration-test`** - End-to-end integration testing
4. **`pipeline-summary`** - Results summary and reporting

### Features

#### 🔄 Matrix Testing
- **Python versions**: 3.9, 3.10, 3.11
- **Operating systems**: Ubuntu (latest & 20.04), Windows, macOS
- **Fail-fast disabled**: Continue testing other combinations even if one fails

#### 📦 Advanced Caching
- **Pip dependencies**: Cached based on requirements.txt hash
- **HuggingFace models**: Cached to reduce download time
- **PyTorch models**: Cached for faster subsequent runs

#### 🧪 Comprehensive Testing
- **Unit tests**: Individual component testing
- **Integration tests**: Full system testing
- **Performance benchmarks**: Response time and memory usage
- **Model validation**: Loading, consistency, and functionality tests

#### 🔒 Security & Quality
- **Safety**: Python package vulnerability scanning
- **Bandit**: Security linting for Python code
- **Black**: Code formatting validation
- **Ruff**: Fast Python linter
- **MyPy**: Type checking (non-blocking)

#### 🎯 ML-Specific Features
- **Git LFS support**: For large model files
- **Model consistency tests**: Verify deterministic behavior
- **Memory efficiency checks**: Monitor resource usage
- **Response time benchmarks**: Ensure performance standards

#### ⚡ Performance Optimizations
- **Conditional execution**: Skip unnecessary steps
- **Parallel job execution**: Independent jobs run simultaneously
- **Smart caching**: Multi-level caching strategy
- **Timeout management**: Prevent hanging jobs

### Triggers

The workflow runs on:
- **Push to main/master**: Full validation on primary branches
- **Pull requests**: Validate changes before merging
- **Manual dispatch**: On-demand execution
- **Path filtering**: Only runs when relevant files change

### Environment Variables

```yaml
PYTHONPATH: ${{ github.workspace }}/src
PYTEST_TIMEOUT: 300  # 5 minutes timeout for tests
MODEL_CACHE_PATH: ~/.cache/huggingface
```

## Job Details

### 1. Test Job (`test`)

**Duration**: ~15-45 minutes  
**Purpose**: Core testing across multiple environments

**Key steps**:
- Environment setup with caching
- Dependency installation
- Security scanning
- Code quality checks
- Unit and integration tests
- Coverage reporting

### 2. Model Validation (`model-validation`)

**Duration**: ~10-30 minutes  
**Purpose**: ML-specific validation and performance testing

**Key steps**:
- Model loading validation
- Memory efficiency testing
- Consistency and determinism checks
- Basic functionality verification

### 3. Integration Test (`integration-test`)

**Duration**: ~5-20 minutes  
**Purpose**: End-to-end system validation

**Key steps**:
- Workflow validation tests
- Project structure verification
- Complete chatbot conversation flow testing

### 4. Pipeline Summary (`pipeline-summary`)

**Duration**: ~1-2 minutes  
**Purpose**: Aggregate results and provide clear feedback

**Features**:
- Job status summary
- Success/failure indicators
- Actionable recommendations
- GitHub Summary integration

## Usage

### Automatic Execution

The workflow runs automatically on:
```bash
# Push to main branch
git push origin main

# Create pull request
gh pr create --title "Feature" --body "Description"
```

### Manual Execution

```bash
# Trigger via GitHub CLI
gh workflow run "ML Pipeline - Testing & Model Validation"

# Or via GitHub web interface
# Go to Actions tab > Select workflow > Run workflow
```

### Local Testing

Validate your changes locally before pushing:

```bash
# Run validation script
python3 validate_workflow.py

# Run tests locally
pytest tests/ -v

# Check code quality
black --check src/ tests/
ruff check src/ tests/
```

## Monitoring & Debugging

### Artifacts

The workflow generates several artifacts:
- **Coverage reports**: XML format for external tools
- **Security reports**: JSON format from Safety and Bandit
- **Test results**: Per-environment test outcomes

### Logs

Each job provides detailed logs:
- **Setup logs**: Environment and dependency installation
- **Test logs**: Detailed test execution with timing
- **Performance logs**: Memory and speed benchmarks
- **Summary logs**: Aggregated results and recommendations

### Troubleshooting

Common issues and solutions:

1. **Timeout errors**: Increase `timeout-minutes` or optimize tests
2. **Cache misses**: Check cache key patterns and dependencies
3. **Model download failures**: Verify network connectivity and HuggingFace access
4. **Memory issues**: Review model size and available resources

## Configuration

### Customization

To modify the workflow:

1. **Add Python versions**: Update the matrix in the `test` job
2. **Add dependencies**: Modify the pip install steps
3. **Add tests**: Include new test commands in appropriate jobs
4. **Add security tools**: Extend the security scanning section

### Environment-Specific Settings

For different environments, you can:
- Set repository secrets for API keys
- Configure environment-specific variables
- Adjust timeout values based on resources
- Modify caching strategies

## Best Practices

1. **Keep tests fast**: Optimize test execution time
2. **Use appropriate timeouts**: Balance thoroughness with speed
3. **Cache effectively**: Leverage multi-level caching
4. **Monitor resources**: Track memory and time usage
5. **Document changes**: Update this README for modifications

## Integration

This workflow integrates with:
- **Codecov**: For coverage reporting
- **GitHub Security**: For vulnerability alerts
- **GitHub Summaries**: For rich reporting
- **External tools**: Via artifact uploads

## Maintenance

Regular maintenance tasks:
- Update action versions (checkout, setup-python, etc.)
- Review and update Python versions in matrix
- Update security tool versions
- Monitor performance trends
- Review and optimize caching strategies

---

*Generated by GitHub Actions ML Pipeline - Built with TDD principles*