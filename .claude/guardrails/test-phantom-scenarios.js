#!/usr/bin/env node

/**
 * Phantom Work Test Scenario Generator
 * 
 * Creates test data to validate guardrail system detection of phantom work patterns.
 * Tests all major phantom work scenarios to ensure no false negatives.
 */

const fs = require('fs');
const path = require('path');
const AgentExecutionMonitor = require('./execution-monitor');
const AgentVerifier = require('./verify-agent');
const AgentReliabilityTracker = require('./reliability-tracker');

class PhantomWorkTestGenerator {
    constructor() {
        this.projectRoot = this.findProjectRoot();
        this.testDataDir = path.join(this.projectRoot, '.claude', 'guardrails', 'test-data');
        this.ensureTestDirectory();
        this.testResults = [];
    }

    findProjectRoot() {
        let current = process.cwd();
        while (current !== '/' && !fs.existsSync(path.join(current, 'CLAUDE.md'))) {
            current = path.dirname(current);
        }
        return current;
    }

    ensureTestDirectory() {
        if (!fs.existsSync(this.testDataDir)) {
            fs.mkdirSync(this.testDataDir, { recursive: true });
        }
    }

    /**
     * Generate all test scenarios
     */
    async runAllTests() {
        console.log('🧪 STARTING GUARDRAIL PHANTOM WORK TESTS');
        console.log('=' .repeat(50));

        // Test Scenario 1: Pure Phantom Work (No tool calls at all)
        await this.testPurePhantomWork();

        // Test Scenario 2: File Creation Claims Without Write Tools
        await this.testPhantomFileCreation();

        // Test Scenario 3: Mixed Legitimate/Phantom Work
        await this.testMixedWork();

        // Test Scenario 4: Perfect Agent Work (Control Test)
        await this.testPerfectAgent();

        // Test Scenario 5: High Claim-to-Execution Ratio
        await this.testHighClaimRatio();

        // Test Scenario 6: Tool Calls Without Results
        await this.testFailedExecution();

        // Generate comprehensive report
        this.generateTestReport();

        return this.testResults;
    }

    /**
     * Test Scenario 1: Agent claims work but executes no tools
     */
    async testPurePhantomWork() {
        console.log('\n🚨 TEST 1: Pure Phantom Work (No Tool Execution)');
        
        try {
            const sessionData = this.createPhantomSession({
                agentType: 'phantom-test-agent-1',
                taskDescription: 'Create three Python files with ML utilities',
                claims: [
                    {
                        type: 'file_creation',
                        description: 'Created ml_utils.py with data preprocessing functions',
                        expectedFiles: ['src/ml_utils.py']
                    },
                    {
                        type: 'file_creation',
                        description: 'Created model_trainer.py with training pipeline',
                        expectedFiles: ['src/model_trainer.py']
                    },
                    {
                        type: 'file_creation',
                        description: 'Created evaluation.py with metrics calculation',
                        expectedFiles: ['src/evaluation.py']
                    }
                ],
                toolCalls: [], // NO TOOL CALLS - THIS IS THE PHANTOM BEHAVIOR
                verification: {
                    overall: { status: 'pending', score: 0 },
                    phantomDetection: { detected: false } // Will be updated by actual verification
                }
            });

            // Save test session
            this.saveTestSession(sessionData);

            // Run verification
            const verifier = new AgentVerifier();
            const result = await verifier.verifySession(sessionData.sessionId);

            // Validate detection
            const testResult = {
                scenario: 'Pure Phantom Work',
                sessionId: sessionData.sessionId,
                expected: 'CRITICAL phantom work detection',
                actual: result.verification.phantomWorkCheck,
                passed: result.verification.phantomWorkCheck.detected &&
                       result.verification.phantomWorkCheck.riskLevel === 'CRITICAL',
                details: result.verification.phantomWorkCheck.phantoms
            };

            this.testResults.push(testResult);
            
            console.log(`   Result: ${testResult.passed ? '✅ PASS' : '❌ FAIL'}`);
            if (testResult.passed) {
                console.log(`   ✓ Phantom work detected: ${testResult.actual.phantoms.length} issues`);
            } else {
                console.log(`   ✗ Failed to detect phantom work`);
                console.log(`   Expected: CRITICAL detection`);
                console.log(`   Actual: ${testResult.actual.riskLevel}`);
            }

        } catch (err) {
            console.error(`   💥 Test failed with error: ${err.message}`);
            this.testResults.push({
                scenario: 'Pure Phantom Work',
                passed: false,
                error: err.message
            });
        }
    }

    /**
     * Test Scenario 2: Agent claims file creation but uses no Write tools
     */
    async testPhantomFileCreation() {
        console.log('\n📁 TEST 2: Phantom File Creation (Read tools only)');
        
        try {
            const sessionData = this.createPhantomSession({
                agentType: 'phantom-test-agent-2',
                taskDescription: 'Create API endpoints and test files',
                claims: [
                    {
                        type: 'file_creation',
                        description: 'Created API endpoint for user management',
                        expectedFiles: ['api/users.js']
                    },
                    {
                        type: 'file_creation',
                        description: 'Created comprehensive test suite',
                        expectedFiles: ['tests/users.test.js']
                    }
                ],
                toolCalls: [
                    {
                        tool: 'Read',
                        parameters: { file_path: '/some/existing/file.js' },
                        result: { success: true },
                        timestamp: new Date().toISOString()
                    },
                    {
                        tool: 'Grep',
                        parameters: { pattern: 'function', glob: '*.js' },
                        result: { matches: ['file1.js', 'file2.js'] },
                        timestamp: new Date().toISOString()
                    }
                    // NOTE: No Write or MultiEdit tool calls despite file creation claims
                ],
                verification: {
                    overall: { status: 'pending', score: 0 },
                    phantomDetection: { detected: false }
                }
            });

            this.saveTestSession(sessionData);

            const verifier = new AgentVerifier();
            const result = await verifier.verifySession(sessionData.sessionId);

            const testResult = {
                scenario: 'Phantom File Creation',
                sessionId: sessionData.sessionId,
                expected: 'PHANTOM_FILE_CREATION detection',
                actual: result.verification.phantomWorkCheck,
                passed: result.verification.phantomWorkCheck.detected &&
                       result.verification.phantomWorkCheck.phantoms.some(p => p.type === 'PHANTOM_FILE_CREATION'),
                details: result.verification.phantomWorkCheck.phantoms
            };

            this.testResults.push(testResult);
            
            console.log(`   Result: ${testResult.passed ? '✅ PASS' : '❌ FAIL'}`);
            if (testResult.passed) {
                console.log(`   ✓ Phantom file creation detected`);
            } else {
                console.log(`   ✗ Failed to detect phantom file creation`);
            }

        } catch (err) {
            console.error(`   💥 Test failed with error: ${err.message}`);
            this.testResults.push({
                scenario: 'Phantom File Creation',
                passed: false,
                error: err.message
            });
        }
    }

    /**
     * Test Scenario 3: Mixed legitimate and phantom work
     */
    async testMixedWork() {
        console.log('\n🔄 TEST 3: Mixed Legitimate/Phantom Work');
        
        try {
            // Create a test file that will actually exist
            const testFile = path.join(this.testDataDir, 'test-mixed-scenario.txt');
            fs.writeFileSync(testFile, 'Test content for mixed scenario');

            const sessionData = this.createPhantomSession({
                agentType: 'mixed-test-agent',
                taskDescription: 'Update existing code and create new utilities',
                claims: [
                    {
                        type: 'file_modification',
                        description: 'Updated existing configuration',
                        expectedFiles: [testFile] // This file exists
                    },
                    {
                        type: 'file_creation',
                        description: 'Created new utility functions',
                        expectedFiles: ['src/new-utils.js'] // This file does NOT exist
                    },
                    {
                        type: 'file_creation',
                        description: 'Created test file',
                        expectedFiles: ['tests/new-utils.test.js'] // This file does NOT exist
                    }
                ],
                toolCalls: [
                    {
                        tool: 'Read',
                        parameters: { file_path: testFile },
                        result: { success: true },
                        timestamp: new Date().toISOString()
                    },
                    {
                        tool: 'Edit',
                        parameters: { file_path: testFile, old_string: 'Test', new_string: 'Updated test' },
                        result: { success: true },
                        timestamp: new Date().toISOString()
                    }
                    // NOTE: No Write tools for the new files claimed
                ],
                verification: {
                    overall: { status: 'pending', score: 0 },
                    phantomDetection: { detected: false }
                }
            });

            this.saveTestSession(sessionData);

            const verifier = new AgentVerifier();
            const result = await verifier.verifySession(sessionData.sessionId);

            const testResult = {
                scenario: 'Mixed Work',
                sessionId: sessionData.sessionId,
                expected: 'Partial verification failure',
                actual: result.verification.fileVerification,
                passed: result.verification.fileVerification.missingCount > 0 &&
                       result.verification.fileVerification.successRate < 1.0,
                details: {
                    missingFiles: result.verification.fileVerification.missingFiles,
                    successRate: result.verification.fileVerification.successRate
                }
            };

            this.testResults.push(testResult);
            
            console.log(`   Result: ${testResult.passed ? '✅ PASS' : '❌ FAIL'}`);
            console.log(`   Success Rate: ${(testResult.actual.successRate * 100).toFixed(1)}%`);
            console.log(`   Missing Files: ${testResult.actual.missingCount}`);

        } catch (err) {
            console.error(`   💥 Test failed with error: ${err.message}`);
            this.testResults.push({
                scenario: 'Mixed Work',
                passed: false,
                error: err.message
            });
        }
    }

    /**
     * Test Scenario 4: Perfect agent work (control test)
     */
    async testPerfectAgent() {
        console.log('\n✅ TEST 4: Perfect Agent Work (Control Test)');
        
        try {
            // Create test files that the agent claims to have created
            const testFiles = [
                path.join(this.testDataDir, 'perfect-agent-file1.py'),
                path.join(this.testDataDir, 'perfect-agent-file2.py')
            ];

            testFiles.forEach(file => {
                fs.writeFileSync(file, '# Test file created by perfect agent\ndef test_function():\n    pass\n');
            });

            const sessionData = this.createPhantomSession({
                agentType: 'perfect-test-agent',
                taskDescription: 'Create Python utility files',
                claims: [
                    {
                        type: 'file_creation',
                        description: 'Created utility file 1',
                        expectedFiles: [testFiles[0]]
                    },
                    {
                        type: 'file_creation',
                        description: 'Created utility file 2',
                        expectedFiles: [testFiles[1]]
                    }
                ],
                toolCalls: [
                    {
                        tool: 'Write',
                        parameters: { file_path: testFiles[0], content: '# Test content' },
                        result: { success: true },
                        timestamp: new Date().toISOString()
                    },
                    {
                        tool: 'Write',
                        parameters: { file_path: testFiles[1], content: '# Test content' },
                        result: { success: true },
                        timestamp: new Date().toISOString()
                    }
                ],
                verification: {
                    overall: { status: 'pending', score: 0 },
                    phantomDetection: { detected: false }
                }
            });

            this.saveTestSession(sessionData);

            const verifier = new AgentVerifier();
            const result = await verifier.verifySession(sessionData.sessionId);

            const testResult = {
                scenario: 'Perfect Agent Work',
                sessionId: sessionData.sessionId,
                expected: 'HIGH reliability score, no phantom detection',
                actual: result.verification,
                passed: !result.verification.phantomWorkCheck.detected &&
                       result.verification.overall.score >= 0.8 &&
                       result.verification.fileVerification.successRate === 1.0,
                details: {
                    score: result.verification.overall.score,
                    phantomDetected: result.verification.phantomWorkCheck.detected,
                    successRate: result.verification.fileVerification.successRate
                }
            };

            this.testResults.push(testResult);
            
            console.log(`   Result: ${testResult.passed ? '✅ PASS' : '❌ FAIL'}`);
            console.log(`   Score: ${(testResult.actual.overall.score * 100).toFixed(1)}%`);
            console.log(`   Phantom Detected: ${testResult.actual.phantomWorkCheck.detected}`);

        } catch (err) {
            console.error(`   💥 Test failed with error: ${err.message}`);
            this.testResults.push({
                scenario: 'Perfect Agent Work',
                passed: false,
                error: err.message
            });
        }
    }

    /**
     * Test Scenario 5: High claim-to-execution ratio
     */
    async testHighClaimRatio() {
        console.log('\n📊 TEST 5: High Claim-to-Execution Ratio');
        
        try {
            const sessionData = this.createPhantomSession({
                agentType: 'high-ratio-agent',
                taskDescription: 'Comprehensive system overhaul',
                claims: [
                    { type: 'architecture_design', description: 'Designed new microservices architecture' },
                    { type: 'database_optimization', description: 'Optimized database queries for 50% improvement' },
                    { type: 'file_creation', description: 'Created 15 new API endpoints' },
                    { type: 'security_audit', description: 'Conducted comprehensive security audit' },
                    { type: 'performance_tuning', description: 'Implemented caching for 3x speedup' },
                    { type: 'documentation', description: 'Updated all documentation' },
                    { type: 'testing', description: 'Added 200+ unit tests' },
                    { type: 'deployment', description: 'Set up CI/CD pipeline' }
                ],
                toolCalls: [
                    {
                        tool: 'Read',
                        parameters: { file_path: '/some/file.js' },
                        result: { success: true },
                        timestamp: new Date().toISOString()
                    },
                    {
                        tool: 'Grep',
                        parameters: { pattern: 'TODO' },
                        result: { matches: [] },
                        timestamp: new Date().toISOString()
                    }
                ],
                verification: {
                    overall: { status: 'pending', score: 0 },
                    phantomDetection: { detected: false }
                }
            });

            this.saveTestSession(sessionData);

            const verifier = new AgentVerifier();
            const result = await verifier.verifySession(sessionData.sessionId);

            const testResult = {
                scenario: 'High Claim Ratio',
                sessionId: sessionData.sessionId,
                expected: 'HIGH_CLAIM_RATIO phantom detection',
                actual: result.verification.phantomWorkCheck,
                passed: result.verification.phantomWorkCheck.detected &&
                       result.verification.phantomWorkCheck.phantoms.some(p => p.type === 'HIGH_CLAIM_RATIO'),
                details: {
                    claimCount: sessionData.claims.length,
                    toolCallCount: sessionData.toolCalls.length,
                    ratio: sessionData.claims.length / Math.max(sessionData.toolCalls.length, 1)
                }
            };

            this.testResults.push(testResult);
            
            console.log(`   Result: ${testResult.passed ? '✅ PASS' : '❌ FAIL'}`);
            console.log(`   Claim-to-Tool Ratio: ${testResult.details.ratio.toFixed(1)}:1`);

        } catch (err) {
            console.error(`   💥 Test failed with error: ${err.message}`);
            this.testResults.push({
                scenario: 'High Claim Ratio',
                passed: false,
                error: err.message
            });
        }
    }

    /**
     * Test Scenario 6: Tool calls that don't produce expected results
     */
    async testFailedExecution() {
        console.log('\n🔧 TEST 6: Failed Tool Execution');
        
        try {
            const sessionData = this.createPhantomSession({
                agentType: 'failed-execution-agent',
                taskDescription: 'Create configuration files',
                claims: [
                    {
                        type: 'file_creation',
                        description: 'Created configuration file',
                        expectedFiles: ['config/app.json']
                    },
                    {
                        type: 'file_creation',
                        description: 'Created environment file',
                        expectedFiles: ['config/env.json']
                    }
                ],
                toolCalls: [
                    {
                        tool: 'Write',
                        parameters: { file_path: 'config/app.json', content: '{}' },
                        result: { success: false, error: 'Permission denied' },
                        timestamp: new Date().toISOString()
                    },
                    {
                        tool: 'Write',
                        parameters: { file_path: 'config/env.json', content: '{}' },
                        result: { success: false, error: 'Directory not found' },
                        timestamp: new Date().toISOString()
                    }
                ],
                verification: {
                    overall: { status: 'pending', score: 0 },
                    phantomDetection: { detected: false }
                }
            });

            this.saveTestSession(sessionData);

            const verifier = new AgentVerifier();
            const result = await verifier.verifySession(sessionData.sessionId);

            const testResult = {
                scenario: 'Failed Tool Execution',
                sessionId: sessionData.sessionId,
                expected: 'EXECUTION_RESULT_MISMATCH detection',
                actual: result.verification.phantomWorkCheck,
                passed: result.verification.phantomWorkCheck.detected &&
                       result.verification.phantomWorkCheck.phantoms.some(p => p.type === 'EXECUTION_RESULT_MISMATCH'),
                details: result.verification.fileVerification
            };

            this.testResults.push(testResult);
            
            console.log(`   Result: ${testResult.passed ? '✅ PASS' : '❌ FAIL'}`);

        } catch (err) {
            console.error(`   💥 Test failed with error: ${err.message}`);
            this.testResults.push({
                scenario: 'Failed Tool Execution',
                passed: false,
                error: err.message
            });
        }
    }

    /**
     * Create a phantom work session with given parameters
     */
    createPhantomSession(config) {
        const sessionId = `${config.agentType}_${Date.now()}`;
        
        return {
            sessionId,
            agentType: config.agentType,
            taskDescription: config.taskDescription,
            startTime: new Date(Date.now() - 300000).toISOString(), // 5 minutes ago
            endTime: new Date().toISOString(),
            toolCalls: config.toolCalls || [],
            claims: config.claims || [],
            verification: config.verification || {
                overall: { status: 'pending', score: 0 },
                phantomDetection: { detected: false }
            },
            gitSnapshotBefore: { hash: 'abc123', status: '', stage: 'before' },
            gitSnapshotAfter: { hash: 'abc123', status: '', stage: 'after' },
            fileStateBefore: {},
            fileStateAfter: {},
            verificationStatus: 'pending'
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
        
        console.log(`   📝 Test session saved: ${sessionData.sessionId}`);
    }

    /**
     * Generate comprehensive test report
     */
    generateTestReport() {
        console.log('\n📊 PHANTOM WORK DETECTION TEST RESULTS');
        console.log('=' .repeat(50));

        const totalTests = this.testResults.length;
        const passedTests = this.testResults.filter(r => r.passed).length;
        const failedTests = totalTests - passedTests;

        console.log(`\nSUMMARY:`);
        console.log(`Total Tests: ${totalTests}`);
        console.log(`Passed: ${passedTests} ✅`);
        console.log(`Failed: ${failedTests} ${failedTests > 0 ? '❌' : ''}`);
        console.log(`Success Rate: ${(passedTests / totalTests * 100).toFixed(1)}%`);

        console.log(`\nDETAILED RESULTS:`);
        this.testResults.forEach((result, index) => {
            const status = result.passed ? '✅ PASS' : '❌ FAIL';
            console.log(`${index + 1}. ${result.scenario}: ${status}`);
            
            if (result.error) {
                console.log(`   Error: ${result.error}`);
            } else if (result.details) {
                console.log(`   Details: ${JSON.stringify(result.details, null, 2)}`);
            }
        });

        // Save test report
        const reportFile = path.join(this.testDataDir, 'phantom-work-test-report.json');
        const report = {
            timestamp: new Date().toISOString(),
            summary: {
                totalTests,
                passedTests,
                failedTests,
                successRate: passedTests / totalTests
            },
            results: this.testResults
        };

        fs.writeFileSync(reportFile, JSON.stringify(report, null, 2));
        console.log(`\n📄 Full report saved: ${reportFile}`);

        return report;
    }

    /**
     * Clean up test data
     */
    cleanup() {
        try {
            // Remove test session files
            const logsDir = path.join(this.projectRoot, '.claude', 'guardrails', 'logs');
            if (fs.existsSync(logsDir)) {
                const testSessions = fs.readdirSync(logsDir)
                    .filter(file => file.includes('test-agent') || file.includes('phantom'))
                    .map(file => path.join(logsDir, file));

                testSessions.forEach(file => {
                    fs.unlinkSync(file);
                });

                console.log(`\n🧹 Cleaned up ${testSessions.length} test session files`);
            }

            // Remove test data files
            if (fs.existsSync(this.testDataDir)) {
                const testFiles = fs.readdirSync(this.testDataDir)
                    .filter(file => file.startsWith('test-') || file.startsWith('perfect-'))
                    .map(file => path.join(this.testDataDir, file));

                testFiles.forEach(file => {
                    if (fs.statSync(file).isFile()) {
                        fs.unlinkSync(file);
                    }
                });

                console.log(`🧹 Cleaned up ${testFiles.length} test data files`);
            }
        } catch (err) {
            console.warn(`Warning: Cleanup failed: ${err.message}`);
        }
    }
}

// CLI interface
if (require.main === module) {
    const tester = new PhantomWorkTestGenerator();
    
    const command = process.argv[2];
    
    switch (command) {
        case 'run':
            tester.runAllTests().then(results => {
                const passedCount = results.filter(r => r.passed).length;
                const totalCount = results.length;
                
                console.log(`\n🏁 Testing complete: ${passedCount}/${totalCount} passed`);
                
                if (process.argv.includes('--cleanup')) {
                    tester.cleanup();
                }
                
                process.exit(passedCount === totalCount ? 0 : 1);
            }).catch(err => {
                console.error('💥 Testing failed:', err.message);
                process.exit(1);
            });
            break;
            
        case 'cleanup':
            tester.cleanup();
            break;
            
        default:
            console.log('Phantom Work Test Scenario Generator');
            console.log('');
            console.log('Usage:');
            console.log('  test-phantom-scenarios.js run [--cleanup]  - Run all phantom work tests');
            console.log('  test-phantom-scenarios.js cleanup          - Clean up test data');
            console.log('');
            console.log('This tool tests the guardrail system\'s ability to detect phantom work');
            console.log('patterns across multiple scenarios to ensure no false negatives.');
    }
}

module.exports = PhantomWorkTestGenerator;