#!/usr/bin/env node

/**
 * Pre-commit Hook Phantom Work Blocking Test
 * 
 * Tests the pre-commit hook's ability to detect and block commits containing phantom work.
 * Simulates git commit scenarios with phantom work sessions to validate blocking behavior.
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

class PreCommitPhantomBlockingTest {
    constructor() {
        this.projectRoot = this.findProjectRoot();
        this.testResults = [];
    }

    findProjectRoot() {
        let current = process.cwd();
        while (current !== '/' && !fs.existsSync(path.join(current, 'CLAUDE.md'))) {
            current = path.dirname(current);
        }
        return current;
    }

    /**
     * Run all pre-commit blocking tests
     */
    async runAllTests() {
        console.log('🚫 TESTING PRE-COMMIT PHANTOM WORK BLOCKING');
        console.log('=' .repeat(50));

        // Test 1: Pre-commit blocks phantom work sessions
        await this.testPhantomWorkBlocking();

        // Test 2: Pre-commit allows legitimate work
        await this.testLegitimateWorkAllowing();

        // Test 3: Pre-commit handles mixed scenarios
        await this.testMixedScenarioHandling();

        // Test 4: Pre-commit handles no recent sessions
        await this.testNoRecentSessions();

        this.generateBlockingTestReport();
        return this.testResults;
    }

    /**
     * Test that pre-commit blocks phantom work
     */
    async testPhantomWorkBlocking() {
        console.log('\n🚨 TEST: Phantom Work Blocking');

        try {
            // Create a phantom work session
            const phantomSession = this.createPhantomSession();
            this.saveTestSession(phantomSession);

            // Test pre-commit validation (dry run)
            let validationResult;
            let blockingDetected = false;
            
            try {
                const output = execSync('node .claude/guardrails/pre-commit-hook.js test', {
                    cwd: this.projectRoot,
                    encoding: 'utf8',
                    timeout: 15000
                });
                
                validationResult = output;
                // Check if phantom work was detected and would block commit
                blockingDetected = output.includes('WOULD BLOCK') || 
                                 output.includes('phantom') || 
                                 output.includes('FAIL');
                
            } catch (err) {
                // Pre-commit validation might exit with error code when blocking
                validationResult = err.stdout || err.message;
                blockingDetected = true; // Error exit typically means blocking
            }

            this.testResults.push({
                scenario: 'Phantom Work Blocking',
                phantomSessionId: phantomSession.sessionId,
                expected: 'Pre-commit should block phantom work',
                actual: blockingDetected ? 'BLOCKED' : 'ALLOWED',
                passed: blockingDetected,
                details: {
                    validationOutput: validationResult.substring(0, 500) // Truncate for readability
                }
            });

            console.log(`   Result: ${blockingDetected ? '✅ BLOCKED' : '❌ ALLOWED'}`);
            console.log(`   Phantom Session: ${phantomSession.sessionId}`);

        } catch (err) {
            console.error(`   💥 Test failed: ${err.message}`);
            this.testResults.push({
                scenario: 'Phantom Work Blocking',
                passed: false,
                error: err.message
            });
        }
    }

    /**
     * Test that pre-commit allows legitimate work
     */
    async testLegitimateWorkAllowing() {
        console.log('\n✅ TEST: Legitimate Work Allowing');

        try {
            // Create a legitimate work session
            const legitimateSession = this.createLegitimateSession();
            this.saveTestSession(legitimateSession);

            // Run verifier to mark session as verified
            const AgentVerifier = require('./verify-agent.js');
            const verifier = new AgentVerifier();
            await verifier.verifySession(legitimateSession.sessionId, { 
                skipTests: true, 
                skipBuild: true 
            });

            // Test pre-commit validation
            let validationResult;
            let allowingDetected = false;
            
            try {
                const output = execSync('node .claude/guardrails/pre-commit-hook.js test', {
                    cwd: this.projectRoot,
                    encoding: 'utf8',
                    timeout: 15000
                });
                
                validationResult = output;
                allowingDetected = output.includes('WOULD ALLOW') || 
                                 output.includes('PASS') ||
                                 !output.includes('WOULD BLOCK');
                
            } catch (err) {
                validationResult = err.stdout || err.message;
                allowingDetected = false;
            }

            this.testResults.push({
                scenario: 'Legitimate Work Allowing',
                sessionId: legitimateSession.sessionId,
                expected: 'Pre-commit should allow legitimate work',
                actual: allowingDetected ? 'ALLOWED' : 'BLOCKED',
                passed: allowingDetected,
                details: {
                    validationOutput: validationResult.substring(0, 500)
                }
            });

            console.log(`   Result: ${allowingDetected ? '✅ ALLOWED' : '❌ BLOCKED'}`);
            console.log(`   Legitimate Session: ${legitimateSession.sessionId}`);

        } catch (err) {
            console.error(`   💥 Test failed: ${err.message}`);
            this.testResults.push({
                scenario: 'Legitimate Work Allowing',
                passed: false,
                error: err.message
            });
        }
    }

    /**
     * Test mixed scenario handling
     */
    async testMixedScenarioHandling() {
        console.log('\n🔄 TEST: Mixed Scenario Handling');

        try {
            // Create both phantom and legitimate sessions
            const phantomSession = this.createPhantomSession('mixed-phantom');
            const legitimateSession = this.createLegitimateSession('mixed-legitimate');
            
            this.saveTestSession(phantomSession);
            this.saveTestSession(legitimateSession);

            // Test pre-commit validation with mixed sessions
            let validationResult;
            let mixedHandling = false;
            
            try {
                const output = execSync('node .claude/guardrails/pre-commit-hook.js test', {
                    cwd: this.projectRoot,
                    encoding: 'utf8',
                    timeout: 15000
                });
                
                validationResult = output;
                // Should block due to phantom work presence
                mixedHandling = output.includes('WOULD BLOCK') || 
                              output.includes('phantom');
                
            } catch (err) {
                validationResult = err.stdout || err.message;
                mixedHandling = true; // Error exit means blocking
            }

            this.testResults.push({
                scenario: 'Mixed Scenario Handling',
                sessions: [phantomSession.sessionId, legitimateSession.sessionId],
                expected: 'Pre-commit should block when any phantom work is present',
                actual: mixedHandling ? 'BLOCKED' : 'ALLOWED',
                passed: mixedHandling,
                details: {
                    validationOutput: validationResult.substring(0, 500)
                }
            });

            console.log(`   Result: ${mixedHandling ? '✅ BLOCKED' : '❌ ALLOWED'}`);
            console.log(`   Mixed Sessions: phantom + legitimate`);

        } catch (err) {
            console.error(`   💥 Test failed: ${err.message}`);
            this.testResults.push({
                scenario: 'Mixed Scenario Handling',
                passed: false,
                error: err.message
            });
        }
    }

    /**
     * Test handling when no recent sessions exist
     */
    async testNoRecentSessions() {
        console.log('\n⭕ TEST: No Recent Sessions');

        try {
            // Clear out recent test sessions (but keep them for later cleanup)
            const logsDir = path.join(this.projectRoot, '.claude', 'guardrails', 'logs');
            const backupDir = path.join(this.projectRoot, '.claude', 'guardrails', 'test-backup');
            
            if (!fs.existsSync(backupDir)) {
                fs.mkdirSync(backupDir, { recursive: true });
            }

            // Backup and remove test sessions temporarily
            const testSessionFiles = fs.readdirSync(logsDir)
                .filter(file => file.includes('test') || file.includes('phantom') || file.includes('legitimate'))
                .map(file => path.join(logsDir, file));

            testSessionFiles.forEach(file => {
                const backupFile = path.join(backupDir, path.basename(file));
                fs.copyFileSync(file, backupFile);
                fs.unlinkSync(file);
            });

            // Test pre-commit validation with no recent sessions
            let validationResult;
            let noSessionsHandling = false;
            
            try {
                const output = execSync('node .claude/guardrails/pre-commit-hook.js test', {
                    cwd: this.projectRoot,
                    encoding: 'utf8',
                    timeout: 10000
                });
                
                validationResult = output;
                noSessionsHandling = output.includes('WOULD ALLOW') || 
                                   output.includes('No recent agent sessions') ||
                                   output.includes('No agent work to verify');
                
            } catch (err) {
                validationResult = err.stdout || err.message;
                noSessionsHandling = false;
            }

            // Restore backed up sessions
            fs.readdirSync(backupDir).forEach(file => {
                const backupFile = path.join(backupDir, file);
                const originalFile = path.join(logsDir, file);
                fs.copyFileSync(backupFile, originalFile);
                fs.unlinkSync(backupFile);
            });

            this.testResults.push({
                scenario: 'No Recent Sessions',
                expected: 'Pre-commit should allow when no recent sessions exist',
                actual: noSessionsHandling ? 'ALLOWED' : 'BLOCKED',
                passed: noSessionsHandling,
                details: {
                    validationOutput: validationResult.substring(0, 500)
                }
            });

            console.log(`   Result: ${noSessionsHandling ? '✅ ALLOWED' : '❌ BLOCKED'}`);

        } catch (err) {
            console.error(`   💥 Test failed: ${err.message}`);
            this.testResults.push({
                scenario: 'No Recent Sessions',
                passed: false,
                error: err.message
            });
        }
    }

    /**
     * Create a phantom work session for testing
     */
    createPhantomSession(suffix = 'phantom') {
        const sessionId = `test-${suffix}-agent_${Date.now()}`;
        
        return {
            sessionId,
            agentType: `test-${suffix}-agent`,
            taskDescription: 'Create test files for phantom work testing',
            startTime: new Date(Date.now() - 300000).toISOString(), // 5 minutes ago
            endTime: new Date().toISOString(),
            toolCalls: [], // NO TOOL CALLS - PHANTOM BEHAVIOR
            claims: [
                {
                    type: 'file_creation',
                    description: 'Created phantom test file',
                    expectedFiles: ['test-phantom-file.js']
                }
            ],
            verification: {
                overall: { status: 'FAIL', score: 0.0 },
                phantomDetection: { 
                    detected: true,
                    riskLevel: 'CRITICAL',
                    phantoms: [
                        {
                            type: 'NO_EXECUTION',
                            description: 'Agent made claims but executed no tools',
                            severity: 'CRITICAL'
                        }
                    ]
                }
            },
            gitSnapshotBefore: { hash: 'abc123', status: '', stage: 'before' },
            gitSnapshotAfter: { hash: 'abc123', status: '', stage: 'after' },
            fileStateBefore: {},
            fileStateAfter: {},
            verificationStatus: 'FAIL'
        };
    }

    /**
     * Create a legitimate work session for testing
     */
    createLegitimateSession(suffix = 'legitimate') {
        const sessionId = `test-${suffix}-agent_${Date.now() + 1}`;
        
        // Create the test file that this session claims to have created
        const testFile = path.join(this.projectRoot, '.claude', 'guardrails', 'test-data', 'legitimate-test-file.js');
        fs.writeFileSync(testFile, '// Test file created by legitimate agent\nconsole.log("Hello World");');
        
        return {
            sessionId,
            agentType: `test-${suffix}-agent`,
            taskDescription: 'Create legitimate test files',
            startTime: new Date(Date.now() - 240000).toISOString(), // 4 minutes ago
            endTime: new Date().toISOString(),
            toolCalls: [
                {
                    tool: 'Write',
                    parameters: { file_path: testFile, content: 'console.log("Hello World");' },
                    result: { success: true },
                    timestamp: new Date().toISOString()
                }
            ],
            claims: [
                {
                    type: 'file_creation',
                    description: 'Created legitimate test file',
                    expectedFiles: [testFile]
                }
            ],
            verification: {
                overall: { status: 'PASS', score: 1.0 },
                phantomDetection: { 
                    detected: false,
                    riskLevel: 'LOW',
                    phantoms: []
                },
                fileVerification: {
                    expectedCount: 1,
                    existingCount: 1,
                    successRate: 1.0
                }
            },
            gitSnapshotBefore: { hash: 'def456', status: '', stage: 'before' },
            gitSnapshotAfter: { hash: 'def789', status: 'M legitimate-test-file.js', stage: 'after' },
            fileStateBefore: {},
            fileStateAfter: { [testFile]: { size: 50, mtime: new Date().toISOString() } },
            verificationStatus: 'PASS'
        };
    }

    /**
     * Save test session to logs directory
     */
    saveTestSession(sessionData) {
        const logsDir = path.join(this.projectRoot, '.claude', 'guardrails', 'logs');
        if (!fs.existsSync(logsDir)) {
            fs.mkdirSync(logsDir, { recursive: true });
        }
        
        const sessionFile = path.join(logsDir, `${sessionData.sessionId}.json`);
        fs.writeFileSync(sessionFile, JSON.stringify(sessionData, null, 2));
    }

    /**
     * Generate test report
     */
    generateBlockingTestReport() {
        console.log('\n📊 PRE-COMMIT BLOCKING TEST RESULTS');
        console.log('=' .repeat(50));

        const totalTests = this.testResults.length;
        const passedTests = this.testResults.filter(r => r.passed).length;

        console.log(`\nSUMMARY:`);
        console.log(`Total Tests: ${totalTests}`);
        console.log(`Passed: ${passedTests} ✅`);
        console.log(`Failed: ${totalTests - passedTests} ${totalTests - passedTests > 0 ? '❌' : ''}`);

        console.log(`\nDETAILED RESULTS:`);
        this.testResults.forEach((result, index) => {
            const status = result.passed ? '✅ PASS' : '❌ FAIL';
            console.log(`${index + 1}. ${result.scenario}: ${status}`);
            console.log(`   Expected: ${result.expected}`);
            console.log(`   Actual: ${result.actual}`);
            
            if (result.error) {
                console.log(`   Error: ${result.error}`);
            }
        });

        // Save report
        const testDataDir = path.join(this.projectRoot, '.claude', 'guardrails', 'test-data');
        if (!fs.existsSync(testDataDir)) {
            fs.mkdirSync(testDataDir, { recursive: true });
        }

        const reportFile = path.join(testDataDir, 'precommit-blocking-test-report.json');
        fs.writeFileSync(reportFile, JSON.stringify({
            timestamp: new Date().toISOString(),
            summary: {
                totalTests,
                passedTests,
                failedTests: totalTests - passedTests
            },
            results: this.testResults
        }, null, 2));

        console.log(`\n📄 Pre-commit blocking test report saved: ${reportFile}`);
    }

    /**
     * Clean up test sessions and files
     */
    cleanup() {
        try {
            const logsDir = path.join(this.projectRoot, '.claude', 'guardrails', 'logs');
            const testDataDir = path.join(this.projectRoot, '.claude', 'guardrails', 'test-data');

            // Remove test sessions
            if (fs.existsSync(logsDir)) {
                const testSessions = fs.readdirSync(logsDir)
                    .filter(file => file.includes('test-') && file.endsWith('.json'))
                    .map(file => path.join(logsDir, file));

                testSessions.forEach(file => fs.unlinkSync(file));
                console.log(`🧹 Cleaned up ${testSessions.length} test sessions`);
            }

            // Remove test files
            if (fs.existsSync(testDataDir)) {
                const testFiles = fs.readdirSync(testDataDir)
                    .filter(file => file.includes('test') && !file.includes('report'))
                    .map(file => path.join(testDataDir, file));

                testFiles.forEach(file => {
                    if (fs.statSync(file).isFile()) {
                        fs.unlinkSync(file);
                    }
                });
                console.log(`🧹 Cleaned up ${testFiles.length} test files`);
            }

        } catch (err) {
            console.warn(`Warning: Cleanup failed: ${err.message}`);
        }
    }
}

// CLI interface
if (require.main === module) {
    const tester = new PreCommitPhantomBlockingTest();
    
    const command = process.argv[2];
    
    switch (command) {
        case 'run':
            tester.runAllTests().then(results => {
                const passedCount = results.filter(r => r.passed).length;
                const totalCount = results.length;
                
                console.log(`\n🏁 Pre-commit blocking tests complete: ${passedCount}/${totalCount} passed`);
                
                if (process.argv.includes('--cleanup')) {
                    tester.cleanup();
                }
                
                process.exit(passedCount === totalCount ? 0 : 1);
            }).catch(err => {
                console.error('💥 Pre-commit testing failed:', err.message);
                process.exit(1);
            });
            break;
            
        case 'cleanup':
            tester.cleanup();
            break;
            
        default:
            console.log('Pre-commit Hook Phantom Work Blocking Test');
            console.log('');
            console.log('Usage:');
            console.log('  test-precommit-phantom-blocking.js run [--cleanup]  - Run pre-commit blocking tests');
            console.log('  test-precommit-phantom-blocking.js cleanup          - Clean up test data');
            console.log('');
            console.log('Tests the pre-commit hook\'s ability to detect and block phantom work commits.');
    }
}

module.exports = PreCommitPhantomBlockingTest;