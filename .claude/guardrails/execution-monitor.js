#!/usr/bin/env node

/**
 * Agent Execution Monitor
 * 
 * Tracks all tool calls made by agents and compares against claimed deliverables
 * to detect phantom work (agents claiming to do work without executing tools).
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

class AgentExecutionMonitor {
    constructor() {
        this.projectRoot = this.findProjectRoot();
        this.logDir = path.join(this.projectRoot, '.claude', 'guardrails', 'logs');
        this.ensureLogDirectory();
        this.currentSession = null;
    }

    findProjectRoot() {
        let current = process.cwd();
        while (current !== '/' && !fs.existsSync(path.join(current, 'CLAUDE.md'))) {
            current = path.dirname(current);
        }
        return current;
    }

    ensureLogDirectory() {
        if (!fs.existsSync(this.logDir)) {
            fs.mkdirSync(this.logDir, { recursive: true });
        }
    }

    /**
     * Start monitoring an agent session
     */
    startSession(agentType, taskDescription) {
        const sessionId = `${agentType}_${Date.now()}`;
        
        this.currentSession = {
            sessionId,
            agentType,
            taskDescription,
            startTime: new Date().toISOString(),
            toolCalls: [],
            gitSnapshotBefore: this.createGitSnapshot('before'),
            fileStateBefore: this.captureFileState(),
            claims: [],
            verificationStatus: 'pending'
        };

        console.log(`🔍 [MONITOR] Started tracking session: ${sessionId}`);
        console.log(`📋 Agent: ${agentType}`);
        console.log(`📝 Task: ${taskDescription}`);
        
        return sessionId;
    }

    /**
     * Record a tool call made by the agent
     */
    recordToolCall(toolName, parameters, result, timestamp = null) {
        if (!this.currentSession) {
            console.warn('⚠️  [MONITOR] No active session - tool call not recorded');
            return;
        }

        const toolCall = {
            tool: toolName,
            parameters: this.sanitizeParameters(parameters),
            result: this.sanitizeResult(result),
            timestamp: timestamp || new Date().toISOString(),
            success: !result?.error
        };

        this.currentSession.toolCalls.push(toolCall);
        
        console.log(`🔧 [MONITOR] Tool call recorded: ${toolName}`);
        
        // Auto-save session data
        this.saveSession();
    }

    /**
     * Record a claim made by the agent
     */
    recordClaim(claimType, description, expectedFiles = []) {
        if (!this.currentSession) {
            console.warn('⚠️  [MONITOR] No active session - claim not recorded');
            return;
        }

        const claim = {
            type: claimType,
            description,
            expectedFiles,
            timestamp: new Date().toISOString()
        };

        this.currentSession.claims.push(claim);
        
        console.log(`📢 [MONITOR] Claim recorded: ${claimType} - ${description}`);
        
        this.saveSession();
    }

    /**
     * End monitoring and verify agent work
     */
    async endSession() {
        if (!this.currentSession) {
            console.warn('⚠️  [MONITOR] No active session to end');
            return null;
        }

        console.log(`🏁 [MONITOR] Ending session: ${this.currentSession.sessionId}`);

        // Capture end state
        this.currentSession.endTime = new Date().toISOString();
        this.currentSession.gitSnapshotAfter = this.createGitSnapshot('after');
        this.currentSession.fileStateAfter = this.captureFileState();

        // Run verification
        const verificationResults = await this.verifyAgentWork();
        this.currentSession.verification = verificationResults;
        this.currentSession.verificationStatus = verificationResults.overall.status;

        // Generate report
        const report = this.generateReport();
        
        // Save final session
        this.saveSession();
        
        // Clear current session
        const completedSession = this.currentSession;
        this.currentSession = null;

        return {
            sessionId: completedSession.sessionId,
            status: completedSession.verificationStatus,
            report: report,
            session: completedSession
        };
    }

    /**
     * Verify that agent claims match actual tool executions
     */
    async verifyAgentWork() {
        const verification = {
            overall: { status: 'unknown', score: 0 },
            toolCallVerification: this.verifyToolCalls(),
            fileVerification: this.verifyFileChanges(),
            claimVerification: this.verifyClaims(),
            phantomDetection: this.detectPhantomWork()
        };

        // Calculate overall score
        const scores = [
            verification.toolCallVerification.score,
            verification.fileVerification.score,
            verification.claimVerification.score
        ];
        
        verification.overall.score = scores.reduce((a, b) => a + b, 0) / scores.length;
        verification.overall.status = verification.overall.score >= 0.8 ? 'PASS' : 
                                     verification.overall.score >= 0.6 ? 'WARNING' : 'FAIL';

        return verification;
    }

    /**
     * Verify tool calls were actually made
     */
    verifyToolCalls() {
        const totalClaims = this.currentSession.claims.length;
        const toolCallsMade = this.currentSession.toolCalls.length;
        
        // Basic heuristic: agents should make tool calls if they claim to do work
        const hasFileCreationClaims = this.currentSession.claims.some(claim => 
            claim.type.includes('file') || claim.type.includes('create')
        );
        
        const hasWriteToolCalls = this.currentSession.toolCalls.some(call => 
            call.tool === 'Write' || call.tool === 'MultiEdit'
        );

        let score = 1.0;
        const issues = [];

        if (hasFileCreationClaims && !hasWriteToolCalls) {
            score = 0.0;
            issues.push('PHANTOM_WORK: Agent claimed file creation but no Write tool calls detected');
        }

        if (totalClaims > 0 && toolCallsMade === 0) {
            score = 0.0;
            issues.push('NO_TOOL_CALLS: Agent made claims but executed no tools');
        }

        return {
            score,
            issues,
            toolCallCount: toolCallsMade,
            claimCount: totalClaims,
            hasFileCreationClaims,
            hasWriteToolCalls
        };
    }

    /**
     * Verify file changes actually occurred
     */
    verifyFileChanges() {
        const expectedFiles = this.currentSession.claims
            .flatMap(claim => claim.expectedFiles || []);
        
        const actualChanges = this.getFileChanges();
        const existingFiles = expectedFiles.filter(file => fs.existsSync(file));
        
        const score = expectedFiles.length > 0 ? 
            existingFiles.length / expectedFiles.length : 1.0;

        const issues = [];
        if (score < 1.0) {
            const missingFiles = expectedFiles.filter(file => !fs.existsSync(file));
            issues.push(`MISSING_FILES: ${missingFiles.join(', ')}`);
        }

        return {
            score,
            issues,
            expectedFiles,
            existingFiles,
            actualChanges,
            missingFiles: expectedFiles.filter(file => !fs.existsSync(file))
        };
    }

    /**
     * Verify claims against actual execution
     */
    verifyClaims() {
        const claims = this.currentSession.claims;
        let verifiedClaims = 0;
        const issues = [];

        for (const claim of claims) {
            let verified = false;

            // Verify file creation claims
            if (claim.type.includes('file') && claim.expectedFiles) {
                verified = claim.expectedFiles.every(file => fs.existsSync(file));
                if (!verified) {
                    issues.push(`UNVERIFIED_CLAIM: ${claim.description}`);
                }
            } else {
                // For other claims, check if there were relevant tool calls
                verified = this.currentSession.toolCalls.length > 0;
            }

            if (verified) verifiedClaims++;
        }

        const score = claims.length > 0 ? verifiedClaims / claims.length : 1.0;

        return {
            score,
            issues,
            totalClaims: claims.length,
            verifiedClaims
        };
    }

    /**
     * Detect phantom work patterns
     */
    detectPhantomWork() {
        const phantoms = [];

        // Pattern 1: Claims without tool calls
        if (this.currentSession.claims.length > 0 && this.currentSession.toolCalls.length === 0) {
            phantoms.push({
                type: 'NO_EXECUTION',
                description: 'Agent made claims but executed no tools',
                severity: 'HIGH'
            });
        }

        // Pattern 2: File creation claims without Write tools
        const fileCreationClaims = this.currentSession.claims.filter(claim => 
            claim.type.includes('file') || claim.type.includes('create')
        );
        const writeToolCalls = this.currentSession.toolCalls.filter(call => 
            call.tool === 'Write' || call.tool === 'MultiEdit'
        );

        if (fileCreationClaims.length > 0 && writeToolCalls.length === 0) {
            phantoms.push({
                type: 'PHANTOM_FILE_CREATION',
                description: 'Agent claimed to create files but made no Write tool calls',
                severity: 'HIGH'
            });
        }

        // Pattern 3: Claims vs actual file changes mismatch
        const expectedFiles = this.currentSession.claims
            .flatMap(claim => claim.expectedFiles || []);
        const existingFiles = expectedFiles.filter(file => fs.existsSync(file));
        
        if (expectedFiles.length > 0 && existingFiles.length / expectedFiles.length < 0.5) {
            phantoms.push({
                type: 'FILE_EXPECTATION_MISMATCH',
                description: `Expected ${expectedFiles.length} files, found ${existingFiles.length}`,
                severity: 'MEDIUM'
            });
        }

        return {
            detected: phantoms.length > 0,
            phantoms,
            riskLevel: phantoms.some(p => p.severity === 'HIGH') ? 'HIGH' : 
                      phantoms.some(p => p.severity === 'MEDIUM') ? 'MEDIUM' : 'LOW'
        };
    }

    /**
     * Create git snapshot
     */
    createGitSnapshot(stage) {
        try {
            const hash = execSync('git rev-parse HEAD', { 
                cwd: this.projectRoot,
                encoding: 'utf8' 
            }).trim();
            
            const status = execSync('git status --porcelain', { 
                cwd: this.projectRoot,
                encoding: 'utf8' 
            });

            return {
                hash,
                status,
                timestamp: new Date().toISOString(),
                stage
            };
        } catch (error) {
            return {
                error: error.message,
                timestamp: new Date().toISOString(),
                stage
            };
        }
    }

    /**
     * Capture current file state
     */
    captureFileState() {
        try {
            const files = this.getProjectFiles();
            const state = {};
            
            for (const file of files.slice(0, 100)) { // Limit to first 100 files
                try {
                    const stats = fs.statSync(file);
                    state[file] = {
                        size: stats.size,
                        mtime: stats.mtime.toISOString()
                    };
                } catch (err) {
                    // File might have been deleted, skip
                }
            }
            
            return state;
        } catch (error) {
            return { error: error.message };
        }
    }

    /**
     * Get file changes between before/after states
     */
    getFileChanges() {
        if (!this.currentSession.fileStateBefore || !this.currentSession.fileStateAfter) {
            return [];
        }

        const changes = [];
        const beforeFiles = Object.keys(this.currentSession.fileStateBefore);
        const afterFiles = Object.keys(this.currentSession.fileStateAfter);

        // New files
        for (const file of afterFiles) {
            if (!beforeFiles.includes(file)) {
                changes.push({ type: 'CREATED', file });
            }
        }

        // Modified files
        for (const file of beforeFiles) {
            if (afterFiles.includes(file)) {
                const before = this.currentSession.fileStateBefore[file];
                const after = this.currentSession.fileStateAfter[file];
                
                if (before.mtime !== after.mtime || before.size !== after.size) {
                    changes.push({ type: 'MODIFIED', file });
                }
            }
        }

        // Deleted files
        for (const file of beforeFiles) {
            if (!afterFiles.includes(file)) {
                changes.push({ type: 'DELETED', file });
            }
        }

        return changes;
    }

    /**
     * Get all project files (excluding node_modules, .git, etc.)
     */
    getProjectFiles() {
        const excluded = [
            'node_modules', '.git', '.next', 'dist', 'build', 
            '.pytest_cache', '__pycache__', 'venv', '.venv'
        ];
        
        const files = [];
        
        function walkDir(dir) {
            try {
                const items = fs.readdirSync(dir);
                for (const item of items) {
                    if (excluded.some(ex => item.includes(ex))) continue;
                    
                    const fullPath = path.join(dir, item);
                    const stats = fs.statSync(fullPath);
                    
                    if (stats.isDirectory()) {
                        walkDir(fullPath);
                    } else {
                        files.push(fullPath);
                    }
                }
            } catch (err) {
                // Skip directories we can't read
            }
        }
        
        walkDir(this.projectRoot);
        return files;
    }

    /**
     * Generate comprehensive report
     */
    generateReport() {
        if (!this.currentSession.verification) {
            return 'No verification data available';
        }

        const v = this.currentSession.verification;
        
        return `
🔍 AGENT EXECUTION REPORT
========================

Session: ${this.currentSession.sessionId}
Agent: ${this.currentSession.agentType}
Task: ${this.currentSession.taskDescription}
Duration: ${this.getSessionDuration()}

OVERALL STATUS: ${v.overall.status} (Score: ${(v.overall.score * 100).toFixed(1)}%)

TOOL CALL ANALYSIS:
- Claims Made: ${v.toolCallVerification.claimCount}
- Tool Calls Executed: ${v.toolCallVerification.toolCallCount}
- File Creation Claims: ${v.toolCallVerification.hasFileCreationClaims ? 'YES' : 'NO'}
- Write Tool Usage: ${v.toolCallVerification.hasWriteToolCalls ? 'YES' : 'NO'}

FILE VERIFICATION:
- Expected Files: ${v.fileVerification.expectedFiles.length}
- Existing Files: ${v.fileVerification.existingFiles.length}
- Missing Files: ${v.fileVerification.missingFiles.length}

PHANTOM WORK DETECTION:
- Phantoms Detected: ${v.phantomDetection.detected ? 'YES' : 'NO'}
- Risk Level: ${v.phantomDetection.riskLevel}
${v.phantomDetection.phantoms.map(p => `- ${p.type}: ${p.description}`).join('\n')}

ISSUES FOUND:
${[...v.toolCallVerification.issues, ...v.fileVerification.issues, ...v.claimVerification.issues].join('\n')}

RECOMMENDATIONS:
${this.generateRecommendations(v).join('\n')}
`;
    }

    /**
     * Generate recommendations based on verification results
     */
    generateRecommendations(verification) {
        const recommendations = [];
        
        if (verification.phantomDetection.detected) {
            recommendations.push('❌ CRITICAL: This agent exhibited phantom work behavior - do not trust without verification');
        }
        
        if (verification.overall.score < 0.6) {
            recommendations.push('⚠️  WARNING: Low verification score - review agent implementation');
        }
        
        if (verification.fileVerification.missingFiles.length > 0) {
            recommendations.push('📁 ACTION: Verify missing files manually or re-run agent with proper tool usage');
        }
        
        if (verification.toolCallVerification.hasFileCreationClaims && !verification.toolCallVerification.hasWriteToolCalls) {
            recommendations.push('🚨 PHANTOM: Agent claimed file creation but made no Write tool calls - likely phantom work');
        }
        
        if (recommendations.length === 0) {
            recommendations.push('✅ GOOD: Agent work appears legitimate and properly executed');
        }
        
        return recommendations;
    }

    /**
     * Get session duration
     */
    getSessionDuration() {
        if (!this.currentSession.startTime) return 'Unknown';
        
        const start = new Date(this.currentSession.startTime);
        const end = this.currentSession.endTime ? 
            new Date(this.currentSession.endTime) : new Date();
        
        const durationMs = end - start;
        const minutes = Math.floor(durationMs / 60000);
        const seconds = Math.floor((durationMs % 60000) / 1000);
        
        return `${minutes}m ${seconds}s`;
    }

    /**
     * Save session data
     */
    saveSession() {
        if (!this.currentSession) return;
        
        const sessionFile = path.join(this.logDir, `${this.currentSession.sessionId}.json`);
        fs.writeFileSync(sessionFile, JSON.stringify(this.currentSession, null, 2));
    }

    /**
     * Sanitize parameters for logging
     */
    sanitizeParameters(params) {
        if (typeof params === 'string') return params.substring(0, 500);
        if (typeof params === 'object') {
            const sanitized = {};
            for (const [key, value] of Object.entries(params)) {
                if (typeof value === 'string') {
                    sanitized[key] = value.substring(0, 200);
                } else {
                    sanitized[key] = value;
                }
            }
            return sanitized;
        }
        return params;
    }

    /**
     * Sanitize result for logging
     */
    sanitizeResult(result) {
        if (typeof result === 'string') return result.substring(0, 500);
        if (typeof result === 'object' && result) {
            return {
                success: !result.error,
                error: result.error || null,
                truncated: true
            };
        }
        return result;
    }

    /**
     * Load session from file
     */
    static loadSession(sessionId, projectRoot = null) {
        const root = projectRoot || this.prototype.findProjectRoot();
        const logDir = path.join(root, '.claude', 'guardrails', 'logs');
        const sessionFile = path.join(logDir, `${sessionId}.json`);
        
        if (fs.existsSync(sessionFile)) {
            return JSON.parse(fs.readFileSync(sessionFile, 'utf8'));
        }
        return null;
    }

    /**
     * Get all session logs
     */
    static getAllSessions(projectRoot = null) {
        const root = projectRoot || this.prototype.findProjectRoot();
        const logDir = path.join(root, '.claude', 'guardrails', 'logs');
        
        if (!fs.existsSync(logDir)) return [];
        
        return fs.readdirSync(logDir)
            .filter(file => file.endsWith('.json'))
            .map(file => {
                try {
                    return JSON.parse(fs.readFileSync(path.join(logDir, file), 'utf8'));
                } catch (err) {
                    return null;
                }
            })
            .filter(Boolean);
    }
}

// CLI usage
if (require.main === module) {
    const monitor = new AgentExecutionMonitor();
    
    const command = process.argv[2];
    
    switch (command) {
        case 'start':
            const agentType = process.argv[3];
            const task = process.argv[4];
            if (!agentType || !task) {
                console.error('Usage: execution-monitor.js start <agent-type> <task-description>');
                process.exit(1);
            }
            const sessionId = monitor.startSession(agentType, task);
            console.log(`Session started: ${sessionId}`);
            break;
            
        case 'end':
            monitor.endSession().then(result => {
                if (result) {
                    console.log(`Session ended: ${result.sessionId}`);
                    console.log(`Status: ${result.status}`);
                    console.log('\n' + result.report);
                } else {
                    console.log('No active session to end');
                }
            });
            break;
            
        case 'report':
            const sessions = AgentExecutionMonitor.getAllSessions();
            console.log(`Found ${sessions.length} sessions`);
            sessions.forEach(session => {
                console.log(`${session.sessionId}: ${session.agentType} - ${session.verificationStatus || 'pending'}`);
            });
            break;
            
        default:
            console.log('Usage: execution-monitor.js [start|end|report]');
            console.log('  start <agent-type> <task> - Start monitoring session');
            console.log('  end                        - End current session and generate report');
            console.log('  report                     - Show all session reports');
    }
}

module.exports = AgentExecutionMonitor;