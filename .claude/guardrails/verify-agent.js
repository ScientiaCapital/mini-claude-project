#!/usr/bin/env node

/**
 * Agent Work Verification CLI
 * 
 * Command-line tool for managers to verify agent deliverables and detect phantom work.
 * Usage: node verify-agent.js [session-id] [options]
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');
const AgentExecutionMonitor = require('./execution-monitor');

class AgentVerifier {
    constructor() {
        this.projectRoot = this.findProjectRoot();
        this.guardrailsDir = path.join(this.projectRoot, '.claude', 'guardrails');
        this.logsDir = path.join(this.guardrailsDir, 'logs');
    }

    findProjectRoot() {
        let current = process.cwd();
        while (current !== '/' && !fs.existsSync(path.join(current, 'CLAUDE.md'))) {
            current = path.dirname(current);
        }
        return current;
    }

    /**
     * Verify a specific agent session
     */
    async verifySession(sessionId, options = {}) {
        console.log(`🔍 VERIFYING AGENT SESSION: ${sessionId}`);
        console.log('=' .repeat(60));

        // Load session data
        const session = AgentExecutionMonitor.loadSession(sessionId, this.projectRoot);
        if (!session) {
            throw new Error(`Session ${sessionId} not found`);
        }

        // Run comprehensive verification
        const verification = {
            sessionInfo: this.getSessionInfo(session),
            phantomWorkCheck: this.checkForPhantomWork(session),
            fileVerification: this.verifyFiles(session),
            testVerification: await this.runTests(session, options),
            buildVerification: await this.verifyBuild(session, options),
            gitVerification: this.verifyGitState(session),
            overall: { status: 'pending', score: 0, issues: [] }
        };

        // Calculate overall score and status
        this.calculateOverallScore(verification);

        // Generate detailed report
        const report = this.generateVerificationReport(verification, session);

        // Save verification results
        this.saveVerificationResults(sessionId, verification);

        return {
            sessionId,
            status: verification.overall.status,
            score: verification.overall.score,
            verification,
            report
        };
    }

    /**
     * Get basic session information
     */
    getSessionInfo(session) {
        return {
            agentType: session.agentType,
            taskDescription: session.taskDescription,
            duration: this.calculateDuration(session.startTime, session.endTime),
            toolCallsCount: session.toolCalls?.length || 0,
            claimsCount: session.claims?.length || 0,
            hasWriteToolCalls: session.toolCalls?.some(call => 
                call.tool === 'Write' || call.tool === 'MultiEdit') || false
        };
    }

    /**
     * Check for phantom work patterns
     */
    checkForPhantomWork(session) {
        const phantoms = [];
        let riskLevel = 'LOW';

        // Pattern 1: Claims without tool execution
        if ((session.claims?.length || 0) > 0 && (session.toolCalls?.length || 0) === 0) {
            phantoms.push({
                type: 'NO_TOOL_EXECUTION',
                description: 'Agent made claims but executed no tools',
                severity: 'CRITICAL',
                impact: 'Agent likely performed no actual work'
            });
            riskLevel = 'CRITICAL';
        }

        // Pattern 2: File creation claims without Write tools
        const fileCreationClaims = session.claims?.filter(claim =>
            claim.type?.toLowerCase().includes('file') ||
            claim.type?.toLowerCase().includes('create') ||
            claim.description?.toLowerCase().includes('creat') ||
            claim.expectedFiles?.length > 0
        ) || [];

        const writeToolCalls = session.toolCalls?.filter(call =>
            call.tool === 'Write' || call.tool === 'MultiEdit'
        ) || [];

        if (fileCreationClaims.length > 0 && writeToolCalls.length === 0) {
            phantoms.push({
                type: 'PHANTOM_FILE_CREATION',
                description: `Agent claimed to create ${fileCreationClaims.length} files but made no Write tool calls`,
                severity: 'CRITICAL',
                impact: 'Files claimed to be created likely do not exist'
            });
            riskLevel = 'CRITICAL';
        }

        // Pattern 3: High claim-to-execution ratio
        const claimToExecutionRatio = (session.claims?.length || 0) / Math.max(session.toolCalls?.length || 0, 1);
        if (claimToExecutionRatio > 3) {
            phantoms.push({
                type: 'HIGH_CLAIM_RATIO',
                description: `Agent made ${session.claims?.length || 0} claims with only ${session.toolCalls?.length || 0} tool calls`,
                severity: 'WARNING',
                impact: 'Agent may be over-claiming work relative to actual execution'
            });
            if (riskLevel === 'LOW') riskLevel = 'WARNING';
        }

        // Pattern 4: Successful tool calls vs failed verification
        const successfulToolCalls = session.toolCalls?.filter(call => call.success !== false) || [];
        const expectedFiles = session.claims?.flatMap(claim => claim.expectedFiles || []) || [];
        const existingFiles = expectedFiles.filter(file => fs.existsSync(path.join(this.projectRoot, file)));
        
        if (successfulToolCalls.length > 0 && expectedFiles.length > 0 && existingFiles.length / expectedFiles.length < 0.5) {
            phantoms.push({
                type: 'EXECUTION_RESULT_MISMATCH',
                description: `Tool calls succeeded but only ${existingFiles.length}/${expectedFiles.length} expected files exist`,
                severity: 'HIGH',
                impact: 'Tool calls may not have produced expected results'
            });
            if (riskLevel !== 'CRITICAL') riskLevel = 'HIGH';
        }

        return {
            detected: phantoms.length > 0,
            count: phantoms.length,
            riskLevel,
            phantoms
        };
    }

    /**
     * Verify claimed files actually exist
     */
    verifyFiles(session) {
        const expectedFiles = session.claims?.flatMap(claim => claim.expectedFiles || []) || [];
        const fileResults = [];
        
        for (const file of expectedFiles) {
            const fullPath = path.resolve(this.projectRoot, file);
            const exists = fs.existsSync(fullPath);
            let content = null;
            let size = 0;
            
            if (exists) {
                try {
                    const stats = fs.statSync(fullPath);
                    size = stats.size;
                    
                    // Read first 500 chars for verification
                    if (stats.size > 0 && stats.size < 50000) {
                        content = fs.readFileSync(fullPath, 'utf8').substring(0, 500);
                    }
                } catch (err) {
                    content = `Error reading file: ${err.message}`;
                }
            }
            
            fileResults.push({
                file,
                fullPath,
                exists,
                size,
                isEmpty: size === 0,
                content
            });
        }

        const existingFiles = fileResults.filter(f => f.exists);
        const missingFiles = fileResults.filter(f => !f.exists);
        const emptyFiles = fileResults.filter(f => f.exists && f.isEmpty);

        return {
            expectedCount: expectedFiles.length,
            existingCount: existingFiles.length,
            missingCount: missingFiles.length,
            emptyCount: emptyFiles.length,
            successRate: expectedFiles.length > 0 ? existingFiles.length / expectedFiles.length : 1,
            files: fileResults,
            missingFiles: missingFiles.map(f => f.file),
            emptyFiles: emptyFiles.map(f => f.file)
        };
    }

    /**
     * Run relevant tests to verify functionality
     */
    async runTests(session, options = {}) {
        if (options.skipTests) {
            return { skipped: true, reason: 'Tests skipped by user option' };
        }

        const results = {
            attempted: false,
            success: false,
            output: '',
            error: null,
            testFiles: []
        };

        try {
            // Detect test files related to agent work
            const testFiles = this.findRelevantTests(session);
            results.testFiles = testFiles;
            
            if (testFiles.length === 0) {
                return { 
                    skipped: true, 
                    reason: 'No relevant test files found',
                    testFiles: []
                };
            }

            results.attempted = true;

            // Try running Python tests first
            if (testFiles.some(f => f.endsWith('.py'))) {
                try {
                    const pythonOutput = execSync(
                        `PYTHONPATH=. python3 -m pytest ${testFiles.filter(f => f.endsWith('.py')).join(' ')} -v`,
                        { 
                            cwd: this.projectRoot,
                            encoding: 'utf8',
                            timeout: 30000
                        }
                    );
                    results.output += 'PYTHON TESTS:\n' + pythonOutput + '\n\n';
                    results.success = true;
                } catch (err) {
                    results.error = 'Python tests failed: ' + err.message;
                    results.output += 'PYTHON TESTS FAILED:\n' + err.stdout + '\n\n';
                }
            }

            // Try running JavaScript/TypeScript tests
            const jsTests = testFiles.filter(f => f.endsWith('.test.ts') || f.endsWith('.test.js'));
            if (jsTests.length > 0 && fs.existsSync(path.join(this.projectRoot, 'mini-claude-web', 'package.json'))) {
                try {
                    const jsOutput = execSync(
                        `cd mini-claude-web && npm test -- ${jsTests.map(f => f.replace('mini-claude-web/', '')).join(' ')}`,
                        { 
                            cwd: this.projectRoot,
                            encoding: 'utf8',
                            timeout: 30000
                        }
                    );
                    results.output += 'JAVASCRIPT TESTS:\n' + jsOutput + '\n\n';
                    results.success = true;
                } catch (err) {
                    if (!results.error) results.error = '';
                    results.error += 'JavaScript tests failed: ' + err.message;
                    results.output += 'JAVASCRIPT TESTS FAILED:\n' + err.stdout + '\n\n';
                }
            }

        } catch (err) {
            results.error = err.message;
            results.output = err.stdout || '';
        }

        return results;
    }

    /**
     * Verify build still works after agent changes
     */
    async verifyBuild(session, options = {}) {
        if (options.skipBuild) {
            return { skipped: true, reason: 'Build verification skipped by user option' };
        }

        const results = {
            attempted: false,
            success: false,
            output: '',
            error: null
        };

        try {
            // Try TypeScript build if available
            if (fs.existsSync(path.join(this.projectRoot, 'mini-claude-web', 'tsconfig.json'))) {
                results.attempted = true;
                
                const buildOutput = execSync(
                    'cd mini-claude-web && npm run type-check',
                    { 
                        cwd: this.projectRoot,
                        encoding: 'utf8',
                        timeout: 30000
                    }
                );
                
                results.output = buildOutput;
                results.success = true;
            }

            // Try Python syntax check for Python files
            const pythonFiles = this.findPythonFiles(session);
            if (pythonFiles.length > 0) {
                for (const file of pythonFiles.slice(0, 5)) { // Limit to 5 files
                    try {
                        execSync(`python3 -m py_compile ${file}`, {
                            cwd: this.projectRoot,
                            timeout: 10000
                        });
                    } catch (err) {
                        throw new Error(`Python syntax error in ${file}: ${err.message}`);
                    }
                }
                results.success = true;
                results.output += `Python syntax check passed for ${pythonFiles.length} files\n`;
            }

        } catch (err) {
            results.error = err.message;
            results.output += err.stdout || '';
        }

        return results;
    }

    /**
     * Verify git state consistency
     */
    verifyGitState(session) {
        try {
            const currentStatus = execSync('git status --porcelain', {
                cwd: this.projectRoot,
                encoding: 'utf8'
            });

            const untrackedFiles = currentStatus
                .split('\n')
                .filter(line => line.startsWith('??'))
                .map(line => line.substring(3));

            const modifiedFiles = currentStatus
                .split('\n')
                .filter(line => line.startsWith(' M') || line.startsWith('M '))
                .map(line => line.substring(3));

            return {
                hasUncommittedChanges: currentStatus.trim().length > 0,
                untrackedFiles,
                modifiedFiles,
                totalChanges: untrackedFiles.length + modifiedFiles.length,
                status: currentStatus
            };
        } catch (err) {
            return {
                error: err.message,
                hasUncommittedChanges: false
            };
        }
    }

    /**
     * Calculate overall verification score
     */
    calculateOverallScore(verification) {
        let score = 1.0;
        const issues = [];

        // Phantom work is critical
        if (verification.phantomWorkCheck.detected) {
            const criticalPhantoms = verification.phantomWorkCheck.phantoms.filter(p => p.severity === 'CRITICAL');
            if (criticalPhantoms.length > 0) {
                score = 0.0;
                issues.push('CRITICAL: Phantom work detected');
            } else {
                score -= 0.3;
                issues.push('WARNING: Suspicious work patterns detected');
            }
        }

        // File verification
        if (verification.fileVerification.missingCount > 0) {
            const missingRatio = verification.fileVerification.missingCount / verification.fileVerification.expectedCount;
            if (missingRatio > 0.5) {
                score -= 0.4;
                issues.push(`FAILED: ${verification.fileVerification.missingCount} claimed files missing`);
            } else {
                score -= 0.2;
                issues.push(`WARNING: ${verification.fileVerification.missingCount} claimed files missing`);
            }
        }

        // Test verification
        if (verification.testVerification.attempted && !verification.testVerification.success) {
            score -= 0.2;
            issues.push('WARNING: Tests failed after agent work');
        }

        // Build verification
        if (verification.buildVerification.attempted && !verification.buildVerification.success) {
            score -= 0.3;
            issues.push('FAILED: Build broken after agent work');
        }

        // Ensure score doesn't go negative
        score = Math.max(0, score);

        // Determine status
        let status;
        if (score >= 0.8) status = 'PASS';
        else if (score >= 0.6) status = 'WARNING';
        else status = 'FAIL';

        verification.overall = { score, status, issues };
    }

    /**
     * Generate detailed verification report
     */
    generateVerificationReport(verification, session) {
        const v = verification;
        const s = session;

        return `
🔍 AGENT WORK VERIFICATION REPORT
==================================

SESSION: ${s.sessionId}
AGENT: ${s.agentType}
TASK: ${s.taskDescription}
DURATION: ${v.sessionInfo.duration}

OVERALL STATUS: ${v.overall.status} (Score: ${(v.overall.score * 100).toFixed(1)}%)

${v.overall.issues.length > 0 ? 'ISSUES FOUND:\n' + v.overall.issues.map(i => `❌ ${i}`).join('\n') + '\n' : ''}

PHANTOM WORK ANALYSIS:
${v.phantomWorkCheck.detected ? '🚨 PHANTOM WORK DETECTED' : '✅ No phantom work detected'}
Risk Level: ${v.phantomWorkCheck.riskLevel}
${v.phantomWorkCheck.phantoms.map(p => `- ${p.type}: ${p.description} (${p.severity})`).join('\n')}

TOOL EXECUTION SUMMARY:
- Claims Made: ${v.sessionInfo.claimsCount}
- Tool Calls: ${v.sessionInfo.toolCallsCount}
- Write Tools Used: ${v.sessionInfo.hasWriteToolCalls ? 'YES' : 'NO'}

FILE VERIFICATION:
${v.fileVerification.expectedCount === 0 ? 'No files claimed' : `
- Expected Files: ${v.fileVerification.expectedCount}
- Existing Files: ${v.fileVerification.existingCount}
- Missing Files: ${v.fileVerification.missingCount}
- Empty Files: ${v.fileVerification.emptyCount}
- Success Rate: ${(v.fileVerification.successRate * 100).toFixed(1)}%`}

${v.fileVerification.missingFiles.length > 0 ? 
  'Missing Files:\n' + v.fileVerification.missingFiles.map(f => `  ❌ ${f}`).join('\n') + '\n' : ''}

TEST VERIFICATION:
${v.testVerification.skipped ? `⏭️  ${v.testVerification.reason}` :
  v.testVerification.attempted ? 
    (v.testVerification.success ? '✅ Tests passed' : '❌ Tests failed') :
    '⚠️  No tests attempted'}

BUILD VERIFICATION:
${v.buildVerification.skipped ? `⏭️  ${v.buildVerification.reason}` :
  v.buildVerification.attempted ? 
    (v.buildVerification.success ? '✅ Build successful' : '❌ Build failed') :
    '⚠️  No build attempted'}

GIT STATE:
${v.gitVerification.hasUncommittedChanges ? 
  `📝 Uncommitted changes: ${v.gitVerification.totalChanges} files` : 
  '✅ Working directory clean'}

RECOMMENDATIONS:
${this.generateRecommendations(verification).join('\n')}

DETAILED OUTPUT:
${v.testVerification.output ? '--- Test Output ---\n' + v.testVerification.output + '\n' : ''}
${v.buildVerification.output ? '--- Build Output ---\n' + v.buildVerification.output + '\n' : ''}
`;
    }

    /**
     * Generate actionable recommendations
     */
    generateRecommendations(verification) {
        const recommendations = [];

        if (verification.overall.status === 'FAIL') {
            recommendations.push('🚨 CRITICAL: This agent work FAILED verification - DO NOT COMMIT');
        }

        if (verification.phantomWorkCheck.detected) {
            recommendations.push('⚠️  This agent exhibited phantom work behavior - verify manually');
            recommendations.push('🔄 Consider re-running this agent with monitoring enabled');
        }

        if (verification.fileVerification.missingCount > 0) {
            recommendations.push('📁 Manually create missing files or re-run agent');
        }

        if (verification.testVerification.attempted && !verification.testVerification.success) {
            recommendations.push('🧪 Fix failing tests before proceeding');
        }

        if (verification.buildVerification.attempted && !verification.buildVerification.success) {
            recommendations.push('🔨 Fix build errors before committing');
        }

        if (verification.overall.status === 'PASS') {
            recommendations.push('✅ Agent work verified successfully - safe to commit');
        }

        return recommendations;
    }

    /**
     * Find test files relevant to agent work
     */
    findRelevantTests(session) {
        const testDirs = [
            'tests',
            'mini-claude-web/tests',
            'test',
            'spec'
        ];

        const testFiles = [];
        const agentType = session.agentType?.toLowerCase() || '';

        for (const dir of testDirs) {
            const fullDir = path.join(this.projectRoot, dir);
            if (fs.existsSync(fullDir)) {
                this.walkDirectory(fullDir, testFiles, (file) => {
                    // Include Python test files
                    if (file.endsWith('.test.py') || file.endsWith('_test.py')) return true;
                    
                    // Include JS/TS test files
                    if (file.endsWith('.test.js') || file.endsWith('.test.ts')) return true;
                    
                    // Include files matching agent type
                    if (agentType && file.toLowerCase().includes(agentType)) return true;
                    
                    return false;
                });
            }
        }

        return testFiles;
    }

    /**
     * Find Python files related to agent work
     */
    findPythonFiles(session) {
        const pythonFiles = [];
        const expectedFiles = session.claims?.flatMap(claim => claim.expectedFiles || []) || [];
        
        for (const file of expectedFiles) {
            if (file.endsWith('.py') && fs.existsSync(path.join(this.projectRoot, file))) {
                pythonFiles.push(file);
            }
        }

        return pythonFiles;
    }

    /**
     * Walk directory and collect files matching filter
     */
    walkDirectory(dir, files, filter) {
        try {
            const items = fs.readdirSync(dir);
            for (const item of items) {
                const fullPath = path.join(dir, item);
                const stats = fs.statSync(fullPath);
                
                if (stats.isDirectory()) {
                    this.walkDirectory(fullPath, files, filter);
                } else if (filter(fullPath)) {
                    files.push(fullPath);
                }
            }
        } catch (err) {
            // Skip directories we can't read
        }
    }

    /**
     * Calculate duration between timestamps
     */
    calculateDuration(start, end) {
        if (!start) return 'Unknown';
        
        const startTime = new Date(start);
        const endTime = end ? new Date(end) : new Date();
        const durationMs = endTime - startTime;
        
        const minutes = Math.floor(durationMs / 60000);
        const seconds = Math.floor((durationMs % 60000) / 1000);
        
        return `${minutes}m ${seconds}s`;
    }

    /**
     * Save verification results
     */
    saveVerificationResults(sessionId, verification) {
        const resultsFile = path.join(this.logsDir, `${sessionId}_verification.json`);
        fs.writeFileSync(resultsFile, JSON.stringify(verification, null, 2));
    }

    /**
     * List all available sessions
     */
    listSessions() {
        const sessions = AgentExecutionMonitor.getAllSessions(this.projectRoot);
        return sessions.map(session => ({
            sessionId: session.sessionId,
            agentType: session.agentType,
            startTime: session.startTime,
            status: session.verificationStatus || 'unverified',
            duration: this.calculateDuration(session.startTime, session.endTime)
        }));
    }
}

// CLI interface
if (require.main === module) {
    const verifier = new AgentVerifier();
    
    const command = process.argv[2];
    
    switch (command) {
        case 'verify':
            const sessionId = process.argv[3];
            if (!sessionId) {
                console.error('Usage: verify-agent.js verify <session-id> [--skip-tests] [--skip-build]');
                process.exit(1);
            }
            
            const options = {
                skipTests: process.argv.includes('--skip-tests'),
                skipBuild: process.argv.includes('--skip-build')
            };
            
            verifier.verifySession(sessionId, options)
                .then(result => {
                    console.log(result.report);
                    process.exit(result.status === 'FAIL' ? 1 : 0);
                })
                .catch(err => {
                    console.error('Verification failed:', err.message);
                    process.exit(1);
                });
            break;
            
        case 'list':
            const sessions = verifier.listSessions();
            console.log('Available sessions:');
            sessions.forEach(session => {
                console.log(`${session.sessionId}: ${session.agentType} - ${session.status} (${session.duration})`);
            });
            break;
            
        default:
            console.log('Agent Work Verification CLI');
            console.log('');
            console.log('Usage:');
            console.log('  verify-agent.js verify <session-id> [options]  - Verify agent work');
            console.log('  verify-agent.js list                           - List all sessions');
            console.log('');
            console.log('Options:');
            console.log('  --skip-tests    Skip test execution');
            console.log('  --skip-build    Skip build verification');
            console.log('');
            console.log('Examples:');
            console.log('  verify-agent.js verify vercel-deployment-specialist_1641234567890');
            console.log('  verify-agent.js verify my-session --skip-tests');
    }
}

module.exports = AgentVerifier;