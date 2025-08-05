#!/usr/bin/env node

/**
 * Pre-commit Hook for Agent Work Validation
 * 
 * Prevents commits of phantom work or unverified agent deliverables.
 * Automatically runs verification before allowing commits.
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');
const AgentExecutionMonitor = require('./execution-monitor');
const AgentVerifier = require('./verify-agent');

class PreCommitValidator {
    constructor() {
        this.projectRoot = this.findProjectRoot();
        this.guardrailsDir = path.join(this.projectRoot, '.claude', 'guardrails');
    }

    findProjectRoot() {
        let current = process.cwd();
        while (current !== '/' && !fs.existsSync(path.join(current, 'CLAUDE.md'))) {
            current = path.dirname(current);
        }
        return current;
    }

    /**
     * Main validation function called by git pre-commit hook
     */
    async validateCommit() {
        console.log('🔍 Running agent work validation before commit...');
        
        try {
            // Get staged files
            const stagedFiles = this.getStagedFiles();
            console.log(`📁 Found ${stagedFiles.length} staged files`);

            // Check for recent agent sessions
            const recentSessions = this.getRecentUnverifiedSessions();
            
            if (recentSessions.length === 0) {
                console.log('✅ No recent agent sessions found - commit allowed');
                return { allowed: true, reason: 'No agent work to verify' };
            }

            console.log(`🤖 Found ${recentSessions.length} recent agent sessions to verify`);

            // Verify each session
            const verificationResults = [];
            for (const session of recentSessions) {
                console.log(`\n🔍 Verifying session: ${session.sessionId}`);
                
                try {
                    const verifier = new AgentVerifier();
                    const result = await verifier.verifySession(session.sessionId, {
                        skipTests: false,
                        skipBuild: false
                    });
                    
                    verificationResults.push(result);
                    
                    if (result.status === 'FAIL') {
                        console.log(`❌ Session ${session.sessionId} FAILED verification`);
                    } else if (result.status === 'WARNING') {
                        console.log(`⚠️  Session ${session.sessionId} has warnings`);
                    } else {
                        console.log(`✅ Session ${session.sessionId} passed verification`);
                    }
                } catch (err) {
                    console.error(`💥 Failed to verify session ${session.sessionId}:`, err.message);
                    verificationResults.push({
                        sessionId: session.sessionId,
                        status: 'ERROR',
                        error: err.message
                    });
                }
            }

            // Analyze results
            const analysis = this.analyzeVerificationResults(verificationResults);
            
            // Generate report
            const report = this.generatePreCommitReport(analysis, stagedFiles);
            console.log('\n' + report);

            // Decide whether to allow commit
            return this.makeCommitDecision(analysis);

        } catch (err) {
            console.error('💥 Pre-commit validation failed:', err.message);
            return {
                allowed: false,
                reason: `Validation error: ${err.message}`,
                error: true
            };
        }
    }

    /**
     * Get files staged for commit
     */
    getStagedFiles() {
        try {
            const output = execSync('git diff --cached --name-only', {
                cwd: this.projectRoot,
                encoding: 'utf8'
            });
            
            return output.trim().split('\n').filter(file => file.length > 0);
        } catch (err) {
            return [];
        }
    }

    /**
     * Get recent agent sessions that haven't been verified yet
     */
    getRecentUnverifiedSessions() {
        const sessions = AgentExecutionMonitor.getAllSessions(this.projectRoot);
        const recentThreshold = Date.now() - (24 * 60 * 60 * 1000); // 24 hours ago
        
        return sessions.filter(session => {
            // Check if session is recent
            const sessionTime = new Date(session.endTime || session.startTime || 0).getTime();
            const isRecent = sessionTime > recentThreshold;
            
            // Check if session has ended (has endTime)
            const hasEnded = !!session.endTime;
            
            // Check if verification results exist in separate file
            const verificationFile = path.join(this.guardrailsDir, 'logs', `${session.sessionId}_verification.json`);
            const hasVerificationFile = fs.existsSync(verificationFile);
            
            // Include recent completed sessions that haven't been verified separately
            return isRecent && hasEnded && !hasVerificationFile;
        });
    }

    /**
     * Analyze verification results
     */
    analyzeVerificationResults(results) {
        const totalSessions = results.length;
        const failedSessions = results.filter(r => r.status === 'FAIL');
        const errorSessions = results.filter(r => r.status === 'ERROR');
        const warningSessions = results.filter(r => r.status === 'WARNING');
        const passedSessions = results.filter(r => r.status === 'PASS');
        
        // Check for phantom work
        const phantomWorkSessions = results.filter(r => 
            r.verification?.phantomWorkCheck?.detected
        );
        
        // Calculate scores
        const scores = results
            .filter(r => r.verification?.overall?.score !== undefined)
            .map(r => r.verification.overall.score);
        
        const averageScore = scores.length > 0 ? 
            scores.reduce((a, b) => a + b, 0) / scores.length : 0;

        return {
            totalSessions,
            failedSessions,
            errorSessions,
            warningSessions,
            passedSessions,
            phantomWorkSessions,
            averageScore,
            results
        };
    }

    /**
     * Generate pre-commit report
     */
    generatePreCommitReport(analysis, stagedFiles) {
        const { totalSessions, failedSessions, phantomWorkSessions, averageScore } = analysis;
        
        return `
🚦 PRE-COMMIT AGENT VALIDATION REPORT
=====================================

Staged Files: ${stagedFiles.length}
Agent Sessions Verified: ${totalSessions}

VERIFICATION SUMMARY:
✅ Passed: ${analysis.passedSessions.length}
⚠️  Warnings: ${analysis.warningSessions.length}
❌ Failed: ${failedSessions.length}
💥 Errors: ${analysis.errorSessions.length}

${phantomWorkSessions.length > 0 ? `🚨 PHANTOM WORK DETECTED: ${phantomWorkSessions.length} sessions` : ''}

Average Score: ${(averageScore * 100).toFixed(1)}%

${failedSessions.length > 0 ? `
FAILED SESSIONS:
${failedSessions.map(s => `❌ ${s.sessionId} - ${s.verification?.overall?.issues?.join(', ') || 'Unknown failure'}`).join('\n')}
` : ''}

${phantomWorkSessions.length > 0 ? `
PHANTOM WORK DETECTED:
${phantomWorkSessions.map(s => `🚨 ${s.sessionId} - Phantom work patterns found`).join('\n')}
` : ''}
`;
    }

    /**
     * Make commit decision based on analysis
     */
    makeCommitDecision(analysis) {
        const { failedSessions, phantomWorkSessions, averageScore } = analysis;
        
        // Critical failures block commit
        if (phantomWorkSessions.length > 0) {
            return {
                allowed: false,
                reason: `Phantom work detected in ${phantomWorkSessions.length} sessions - commit blocked`,
                critical: true
            };
        }

        if (failedSessions.length > 0) {
            return {
                allowed: false,
                reason: `${failedSessions.length} agent sessions failed verification - commit blocked`,
                critical: true
            };
        }

        // Low average score blocks commit
        if (averageScore < 0.6) {
            return {
                allowed: false,
                reason: `Low verification score (${(averageScore * 100).toFixed(1)}%) - commit blocked`,
                critical: false
            };
        }

        // Warnings allow commit but notify
        if (analysis.warningSessions.length > 0) {
            return {
                allowed: true,
                reason: `Commit allowed with ${analysis.warningSessions.length} warnings - monitor closely`,
                warnings: true
            };
        }

        // All good
        return {
            allowed: true,
            reason: 'All agent work verified successfully',
            clean: true
        };
    }

    /**
     * Install pre-commit hook
     */
    installHook() {
        const gitHooksDir = path.join(this.projectRoot, '.git', 'hooks');
        const preCommitHook = path.join(gitHooksDir, 'pre-commit');
        
        if (!fs.existsSync(gitHooksDir)) {
            console.error('Git hooks directory not found. Is this a git repository?');
            return false;
        }

        const hookContent = `#!/bin/sh
# Agent Work Validation Pre-commit Hook
# Generated by mini-claude guardrails system

node "${path.resolve(__filename)}" validate

exit $?
`;

        fs.writeFileSync(preCommitHook, hookContent);
        
        // Make executable on Unix systems
        if (process.platform !== 'win32') {
            try {
                execSync(`chmod +x "${preCommitHook}"`);
            } catch (err) {
                console.warn('Could not make pre-commit hook executable:', err.message);
            }
        }

        console.log('✅ Pre-commit hook installed successfully');
        console.log(`   Location: ${preCommitHook}`);
        return true;
    }

    /**
     * Uninstall pre-commit hook
     */
    uninstallHook() {
        const preCommitHook = path.join(this.projectRoot, '.git', 'hooks', 'pre-commit');
        
        if (fs.existsSync(preCommitHook)) {
            fs.unlinkSync(preCommitHook);
            console.log('✅ Pre-commit hook removed');
            return true;
        } else {
            console.log('⚠️  No pre-commit hook found');
            return false;
        }
    }
}

// CLI interface
if (require.main === module) {
    const validator = new PreCommitValidator();
    
    const command = process.argv[2];
    
    switch (command) {
        case 'validate':
            // Called by git pre-commit hook
            validator.validateCommit().then(result => {
                if (result.allowed) {
                    console.log('✅ Commit validation passed');
                    if (result.warnings) {
                        console.log('⚠️  Note: Commit has warnings - monitor deployment closely');
                    }
                    process.exit(0);
                } else {
                    console.log('\n❌ COMMIT BLOCKED');
                    console.log(`Reason: ${result.reason}`);
                    console.log('\nTo fix:');
                    if (result.critical) {
                        console.log('1. Review failed agent sessions');
                        console.log('2. Fix or re-run agents as needed');
                        console.log('3. Verify all agent work manually');
                    }
                    console.log('4. Run: npm run verify-agent <session-id>');
                    console.log('5. Try committing again');
                    process.exit(1);
                }
            }).catch(err => {
                console.error('💥 Validation error:', err.message);
                process.exit(1);
            });
            break;
            
        case 'install':
            if (validator.installHook()) {
                console.log('Pre-commit hook installed. Agent work will be validated before commits.');
            } else {
                process.exit(1);
            }
            break;
            
        case 'uninstall':
            validator.uninstallHook();
            break;
            
        case 'test':
            // Test the validation without actually committing
            console.log('🧪 Testing pre-commit validation...');
            validator.validateCommit().then(result => {
                console.log(`Result: ${result.allowed ? 'WOULD ALLOW' : 'WOULD BLOCK'}`);
                console.log(`Reason: ${result.reason}`);
            }).catch(err => {
                console.error('Test failed:', err.message);
            });
            break;
            
        default:
            console.log('Pre-commit Hook for Agent Validation');
            console.log('');
            console.log('Usage:');
            console.log('  pre-commit-hook.js install     - Install git pre-commit hook');
            console.log('  pre-commit-hook.js uninstall   - Remove git pre-commit hook');
            console.log('  pre-commit-hook.js test        - Test validation without committing');
            console.log('  pre-commit-hook.js validate    - Run validation (called by git)');
            console.log('');
            console.log('The pre-commit hook will automatically verify agent work before');
            console.log('allowing commits, preventing phantom work from entering the repo.');
    }
}

module.exports = PreCommitValidator;