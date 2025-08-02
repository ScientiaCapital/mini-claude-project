#!/bin/bash
# Post-Task Knowledge Update Hook for Claude Code
# Saves agent-specific knowledge and patterns

# Read JSON input from stdin
INPUT=$(cat)

# Extract agent type and output
AGENT_TYPE=$(echo "$INPUT" | jq -r '.params.subagent_type // empty')
TASK_OUTPUT=$(echo "$INPUT" | jq -r '.output // empty')

if [ -z "$AGENT_TYPE" ]; then
    echo "[POST-TASK] No agent type detected, skipping knowledge update..."
    echo "$INPUT"
    exit 0
fi

echo "[POST-TASK] Saving knowledge for: $AGENT_TYPE"

# Create agent-specific knowledge directory
KNOWLEDGE_DIR="/Users/tmk/Documents/mini-claude-project/.claude/knowledge/agents/$AGENT_TYPE"
mkdir -p "$KNOWLEDGE_DIR"

# Generate timestamp for knowledge entry
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Function to save database patterns
save_db_knowledge() {
    echo "[KNOWLEDGE] Saving database patterns..."
    cat > "$KNOWLEDGE_DIR/pattern_$TIMESTAMP.json" << EOF
{
  "timestamp": "$TIMESTAMP",
  "type": "database_pattern",
  "agent": "$AGENT_TYPE",
  "patterns": {
    "query_optimizations": [],
    "schema_improvements": [],
    "connection_patterns": []
  },
  "output": "$TASK_OUTPUT"
}
EOF
}

# Function to save deployment patterns
save_deploy_knowledge() {
    echo "[KNOWLEDGE] Saving deployment patterns..."
    cat > "$KNOWLEDGE_DIR/deployment_$TIMESTAMP.json" << EOF
{
  "timestamp": "$TIMESTAMP",
  "type": "deployment_pattern",
  "agent": "$AGENT_TYPE",
  "patterns": {
    "build_configurations": [],
    "environment_setups": [],
    "optimization_techniques": []
  },
  "output": "$TASK_OUTPUT"
}
EOF
}

# Function to save security findings
save_security_knowledge() {
    echo "[KNOWLEDGE] Saving security audit results..."
    cat > "$KNOWLEDGE_DIR/security_$TIMESTAMP.json" << EOF
{
  "timestamp": "$TIMESTAMP",
  "type": "security_audit",
  "agent": "$AGENT_TYPE",
  "findings": {
    "vulnerabilities_found": [],
    "best_practices_applied": [],
    "recommendations": []
  },
  "output": "$TASK_OUTPUT"
}
EOF
}

# Function to save API patterns
save_api_knowledge() {
    echo "[KNOWLEDGE] Saving API integration patterns..."
    cat > "$KNOWLEDGE_DIR/api_pattern_$TIMESTAMP.json" << EOF
{
  "timestamp": "$TIMESTAMP",
  "type": "api_integration",
  "agent": "$AGENT_TYPE",
  "patterns": {
    "integration_approaches": [],
    "error_handling": [],
    "rate_limiting_strategies": []
  },
  "output": "$TASK_OUTPUT"
}
EOF
}

# Function to save performance improvements
save_performance_knowledge() {
    echo "[KNOWLEDGE] Saving performance improvements..."
    cat > "$KNOWLEDGE_DIR/performance_$TIMESTAMP.json" << EOF
{
  "timestamp": "$TIMESTAMP",
  "type": "performance_optimization",
  "agent": "$AGENT_TYPE",
  "improvements": {
    "metrics_before": {},
    "metrics_after": {},
    "techniques_applied": []
  },
  "output": "$TASK_OUTPUT"
}
EOF
}

# Function to save documentation updates
save_docs_knowledge() {
    echo "[KNOWLEDGE] Saving documentation updates..."
    cat > "$KNOWLEDGE_DIR/docs_update_$TIMESTAMP.json" << EOF
{
  "timestamp": "$TIMESTAMP",
  "type": "documentation_update",
  "agent": "$AGENT_TYPE",
  "updates": {
    "files_modified": [],
    "sections_updated": [],
    "knowledge_gaps_filled": []
  },
  "output": "$TASK_OUTPUT"
}
EOF
    
    # Auto-commit documentation if changes detected
    if echo "$TASK_OUTPUT" | grep -q "documentation.*updated\|docs.*updated"; then
        echo "[AUTO-COMMIT] Documentation changes detected, preparing commit..."
    fi
}

# Save knowledge based on agent type
case "$AGENT_TYPE" in
    "neon-database-architect")
        save_db_knowledge
        ;;
    "vercel-deployment-specialist")
        save_deploy_knowledge
        ;;
    "security-auditor-expert"|"security-audit-specialist")
        save_security_knowledge
        ;;
    "api-integration-specialist")
        save_api_knowledge
        ;;
    "nextjs-performance-optimizer"|"nextjs-speed-optimizer")
        save_performance_knowledge
        ;;
    "project-docs-curator")
        save_docs_knowledge
        ;;
    "fullstack-tdd-architect")
        echo "[KNOWLEDGE] Saving TDD patterns..."
        cat > "$KNOWLEDGE_DIR/tdd_pattern_$TIMESTAMP.json" << EOF
{
  "timestamp": "$TIMESTAMP",
  "type": "tdd_implementation",
  "agent": "$AGENT_TYPE",
  "patterns": {
    "tests_written": [],
    "implementation_approach": "",
    "refactoring_done": []
  },
  "output": "$TASK_OUTPUT"
}
EOF
        ;;
    *)
        echo "[KNOWLEDGE] Saving general knowledge for: $AGENT_TYPE"
        ;;
esac

echo "[POST-TASK] Knowledge saved successfully"

# Update shared architectural knowledge if significant changes
SHARED_DIR="/Users/tmk/Documents/mini-claude-project/.claude/knowledge/shared"
mkdir -p "$SHARED_DIR/architecture"

# Pass through the input unchanged
echo "$INPUT"