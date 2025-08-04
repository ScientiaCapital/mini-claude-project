# Guardrail System Documentation

## Overview

This directory contains the comprehensive guardrail system for the mini-claude project, designed to ensure development reliability, detect phantom work, and maintain system integrity.

## Quick Start

### For Developers

```bash
# From mini-claude-web directory
npm run verify-agent          # Verify current agent work
npm run agent-checkpoint      # Create a checkpoint
npm run monitor-agents        # View active sessions
npm run list-sessions         # List all sessions
npm run verify-all           # Run all verifications
```

### For Testing

```bash
# Run comprehensive system test
node .claude/guardrails/test-guardrail-system.js

# Test individual components
node .claude/guardrails/verify-agent.js list
node .claude/guardrails/reliability-tracker.js report
node .claude/guardrails/checkpoint-manager.js list
```

## System Components

### 1. Agent Verification (`verify-agent.js`)
Detects phantom work and verifies agent deliverables.

**Key Features:**
- Detects file creation claims without Write tool usage
- Validates claimed work against actual tool calls
- Runs tests and build verification
- Generates detailed verification reports

**Usage:**
```bash
node verify-agent.js verify <session-id>  # Verify specific session
node verify-agent.js list                 # List all sessions
```

### 2. Execution Monitor (`execution-monitor.js`)
Tracks agent sessions and tool usage in real-time.

**Key Features:**
- Automatic session tracking via hooks
- Tool call logging
- Session persistence
- Performance metrics

**Usage:**
```bash
node execution-monitor.js start <agent-type> <task>  # Start monitoring
node execution-monitor.js end                        # End current session
node execution-monitor.js report                     # Show all sessions
```

### 3. Reliability Tracker (`reliability-tracker.js`)
Scores agent reliability over time.

**Key Features:**
- Success/failure rate tracking
- Phantom work incident tracking
- Trend analysis
- Deployment recommendations

**Usage:**
```bash
node reliability-tracker.js report                    # Full report
node reliability-tracker.js agent <type>             # Agent-specific
node reliability-tracker.js should-deploy <type>     # Deployment advice
```

### 4. Checkpoint Manager (`checkpoint-manager.js`)
Creates recovery points for safe rollback.

**Key Features:**
- Git-based checkpoints
- Stash preservation
- Quick rollback capability
- Automatic cleanup

**Usage:**
```bash
node checkpoint-manager.js create <agent> [desc]     # Create checkpoint
node checkpoint-manager.js rollback <id>            # Rollback to checkpoint
node checkpoint-manager.js list                      # List checkpoints
```

## Hook System Integration

The guardrail system integrates with Claude Code hooks:

### Pre-Task Hook (`pre-task-context.sh`)
- Loads agent-specific context
- Provides relevant project information
- Sets up validation requirements

### Post-Task Hook (`post-agent-update.sh`)
- Saves agent learnings
- Updates knowledge base
- Records successful patterns

### Validation Hook (`validate-agent-work.sh`)
- Runs agent-specific validations
- Checks deployment health
- Verifies security compliance

## Phantom Work Detection

The system detects various phantom work patterns:

### Obvious Phantom Work
- Claims: "Created file X"
- Reality: No Write tool used
- Detection: 100% accuracy

### Subtle Phantom Work
- Claims: "Added helper functions"
- Reality: Only Read tools used
- Detection: Pattern matching

### Over-claiming
- Claims: 10 improvements made
- Reality: 2 tool calls executed
- Detection: Ratio analysis

## API Modules

### verify-agent-api.js
```javascript
const { verifyWork } = require('./verify-agent-api');

const result = await verifyWork({
  claimedActions: ['Created test.js'],
  actualToolCalls: [{ tool: 'Write', params: {...} }]
});
```

### reliability-tracker-api.js
```javascript
const tracker = require('./reliability-tracker-api');

tracker.recordSuccess('agent-type', 'operation');
tracker.recordFailure('agent-type', 'phantom_work');
const score = tracker.getAgentScore('agent-type');
```

## Dashboard

Access the health dashboard by opening `dashboard.html` in a browser.

**Features:**
- Real-time system status
- Hook execution metrics
- Phantom work detection stats
- Reliability scores
- Auto-refresh every 30 seconds

## Test Suite

Run the comprehensive test suite:
```bash
node test-guardrail-system.js
```

**Tests:**
- Hook system functionality
- Component integration
- Phantom work detection
- Recovery procedures
- Dashboard generation

## Data Files

- `reliability-data.json` - Agent reliability metrics
- `checkpoints.json` - Checkpoint history
- `logs/` - Session logs and verification results
- `dashboard-data.json` - Dashboard metrics

## Best Practices

### For Agent Development
1. Always use appropriate tools for claimed actions
2. Don't claim file creation without Write tools
3. Match claims to actual work performed
4. Run verification after significant changes

### For Managers
1. Review reliability reports weekly
2. Check phantom work alerts daily
3. Verify high-risk agent deployments
4. Monitor trend changes

### For System Maintenance
1. Logs rotate automatically after 7 days
2. Run checkpoint cleanup monthly
3. Archive reliability data quarterly
4. Update dashboard metrics as needed

## Troubleshooting

### Common Issues

**Hook not triggering:**
- Check `.claude/settings.json` configuration
- Verify hook scripts are executable
- Check Claude Code is using Task tool

**Phantom work not detected:**
- Ensure session is properly tracked
- Verify claims are recorded
- Check pattern matching rules

**Checkpoint rollback fails:**
- Verify git repository state
- Check for uncommitted changes
- Ensure proper permissions

## Support

For issues or questions:
1. Check test results in `guardrail-test-report.json`
2. Review logs in `logs/` directory
3. Run diagnostic tests
4. Check validation report

---

**Version**: 1.0.0  
**Last Updated**: January 3, 2025  
**Status**: Fully Operational