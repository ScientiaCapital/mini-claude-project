#!/bin/bash
# Pre-Task Context Loading Hook for Claude Code
# Loads agent-specific context based on Task tool subagent_type

# Read JSON input from stdin
INPUT=$(cat)

# Extract agent type from Task parameters
AGENT_TYPE=$(echo "$INPUT" | jq -r '.params.subagent_type // empty')

if [ -z "$AGENT_TYPE" ]; then
    echo "[PRE-TASK] No agent type detected, loading general context..."
    exit 0
fi

echo "[PRE-TASK] Loading context for: $AGENT_TYPE"

# Function to load database context
load_db_context() {
    echo "[CONTEXT] Loading database schema and patterns..."
    echo "- Current Neon PostgreSQL connection"
    echo "- Schema: users, conversations, messages tables"
    echo "- Query optimization patterns"
    echo "- Connection pooling best practices"
}

# Function to load deployment context
load_deploy_context() {
    echo "[CONTEXT] Loading Vercel deployment configuration..."
    echo "- Environment variables: NEON_DATABASE_URL, GOOGLE_API_KEY, ELEVENLABS_API_KEY"
    echo "- Build configuration: Next.js 14"
    echo "- Deployment history and patterns"
    echo "- Performance metrics"
}

# Function to load security context
load_security_context() {
    echo "[CONTEXT] Loading security patterns and audit history..."
    echo "- API key management practices"
    echo "- Environment variable security"
    echo "- Previous vulnerability assessments"
    echo "- Security best practices"
}

# Function to load API integration context
load_api_context() {
    echo "[CONTEXT] Loading API integration patterns..."
    echo "- Google Gemini API configuration"
    echo "- ElevenLabs voice synthesis setup"
    echo "- Rate limiting and error handling"
    echo "- Integration best practices"
}

# Function to load performance context
load_performance_context() {
    echo "[CONTEXT] Loading performance optimization history..."
    echo "- Current bundle size metrics"
    echo "- Core Web Vitals scores"
    echo "- Optimization techniques applied"
    echo "- Performance bottlenecks identified"
}

# Function to load documentation context
load_docs_context() {
    echo "[CONTEXT] Loading documentation standards..."
    echo "- Current documentation state"
    echo "- Learning milestone progress"
    echo "- Documentation templates"
    echo "- Knowledge gaps identified"
}

# Load context based on agent type
case "$AGENT_TYPE" in
    "neon-database-architect")
        load_db_context
        ;;
    "vercel-deployment-specialist")
        load_deploy_context
        ;;
    "security-auditor-expert"|"security-audit-specialist")
        load_security_context
        ;;
    "api-integration-specialist")
        load_api_context
        ;;
    "nextjs-performance-optimizer"|"nextjs-speed-optimizer")
        load_performance_context
        ;;
    "project-docs-curator")
        load_docs_context
        ;;
    "fullstack-tdd-architect")
        echo "[CONTEXT] Loading full-stack TDD patterns..."
        echo "- TDD cycle: RED → GREEN → REFACTOR"
        echo "- Test coverage requirements (>95%)"
        echo "- TypeScript best practices"
        echo "- Current tech stack: Next.js + Neon + Google Gemini + ElevenLabs"
        ;;
    *)
        echo "[CONTEXT] Loading general context for: $AGENT_TYPE"
        ;;
esac

echo "[PRE-TASK] Context loaded successfully for $AGENT_TYPE"

# Always load shared architectural context
echo "[SHARED] Current Architecture:"
echo "- Frontend: Next.js 14 + TypeScript + Tailwind"
echo "- Database: Neon PostgreSQL (serverless)"
echo "- APIs: Google Gemini + ElevenLabs"
echo "- Deployment: Vercel"
echo "- Testing: Jest + TDD (>95% coverage)"

# Pass through the input unchanged
echo "$INPUT"