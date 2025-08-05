#!/usr/bin/env node

/**
 * Individual Guardrail Component Testing
 * 
 * Tests each guardrail component separately to validate specific functionality.
 * Faster and more focused than the comprehensive phantom scenario tests.
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

class GuardrailComponentTester {
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
     * Run all component tests
     */
    async runAllTests() {
        console.log('🔧 TESTING INDIVIDUAL GUARDRAIL COMPONENTS');
        console.log('=' .repeat(50));

        // Test 1: Execution Monitor CLI
        await this.testExecutionMonitorCLI();

        // Test 2: Agent Verifier CLI
        await this.testAgentVerifierCLI();

        // Test 3: Pre-commit Hook CLI
        await this.testPreCommitHookCLI();

        // Test 4: Reliability Tracker CLI
        await this.testReliabilityTrackerCLI();

        // Test 5: Component Integration
        await this.testComponentIntegration();

        this.generateComponentTestReport();
        return this.testResults;
    }

    /**
     * Test Execution Monitor CLI functionality
     */
    async testExecutionMonitorCLI() {
        console.log('\n🔍 TEST: Execution Monitor CLI');

        try {
            // Test help command
            const helpOutput = execSync('node .claude/guardrails/execution-monitor.js', {
                cwd: this.projectRoot,
                encoding: 'utf8',
                timeout: 5000
            });

            const hasUsage = helpOutput.includes('Usage') && helpOutput.includes('start');
            
            // Test session list command
            const reportOutput = execSync('node .claude/guardrails/execution-monitor.js report', {
                cwd: this.projectRoot,
                encoding: 'utf8',
                timeout: 5000
            });

            const hasSessionCount = reportOutput.includes('Found') && reportOutput.includes('sessions');

            this.testResults.push({
                component: 'Execution Monitor CLI',
                tests: [
                    { name: 'Help Command', passed: hasUsage },
                    { name: 'Report Command', passed: hasSessionCount }
                ],
                overall: hasUsage && hasSessionCount
            });

            console.log(`   Help Command: ${hasUsage ? '✅' : '❌'}`);
            console.log(`   Report Command: ${hasSessionCount ? '✅' : '❌'}`);

        } catch (err) {
            console.log(`   ❌ CLI test failed: ${err.message}`);
            this.testResults.push({
                component: 'Execution Monitor CLI',
                tests: [],
                overall: false,
                error: err.message
            });
        }
    }

    /**
     * Test Agent Verifier CLI functionality
     */
    async testAgentVerifierCLI() {
        console.log('\n🔍 TEST: Agent Verifier CLI');

        try {
            // Test help command
            const helpOutput = execSync('node .claude/guardrails/verify-agent.js', {
                cwd: this.projectRoot,
                encoding: 'utf8',
                timeout: 5000
            });

            const hasUsage = helpOutput.includes('Usage') && helpOutput.includes('verify');

            // Test list command
            const listOutput = execSync('node .claude/guardrails/verify-agent.js list', {
                cwd: this.projectRoot,
                encoding: 'utf8',
                timeout: 5000
            });

            const hasList = listOutput.includes('Available sessions');

            this.testResults.push({
                component: 'Agent Verifier CLI',
                tests: [
                    { name: 'Help Command', passed: hasUsage },
                    { name: 'List Command', passed: hasList }
                ],
                overall: hasUsage && hasList
            });

            console.log(`   Help Command: ${hasUsage ? '✅' : '❌'}`);
            console.log(`   List Command: ${hasList ? '✅' : '❌'}`);

        } catch (err) {
            console.log(`   ❌ CLI test failed: ${err.message}`);
            this.testResults.push({
                component: 'Agent Verifier CLI',
                tests: [],
                overall: false,
                error: err.message
            });
        }
    }

    /**
     * Test Pre-commit Hook CLI functionality
     */
    async testPreCommitHookCLI() {
        console.log('\n🔍 TEST: Pre-commit Hook CLI');

        try {
            // Test help command
            const helpOutput = execSync('node .claude/guardrails/pre-commit-hook.js', {
                cwd: this.projectRoot,
                encoding: 'utf8',
                timeout: 5000
            });

            const hasUsage = helpOutput.includes('Usage') && helpOutput.includes('install');

            // Test validation command (dry run)
            const testOutput = execSync('node .claude/guardrails/pre-commit-hook.js test', {
                cwd: this.projectRoot,
                encoding: 'utf8',
                timeout: 10000
            });

            const hasTestOutput = testOutput.includes('Testing pre-commit validation') || 
                                testOutput.includes('Result:');

            this.testResults.push({
                component: 'Pre-commit Hook CLI',
                tests: [
                    { name: 'Help Command', passed: hasUsage },
                    { name: 'Test Command', passed: hasTestOutput }
                ],
                overall: hasUsage && hasTestOutput
            });

            console.log(`   Help Command: ${hasUsage ? '✅' : '❌'}`);
            console.log(`   Test Command: ${hasTestOutput ? '✅' : '❌'}`);

        } catch (err) {
            console.log(`   ❌ CLI test failed: ${err.message}`);
            this.testResults.push({
                component: 'Pre-commit Hook CLI',
                tests: [],
                overall: false,
                error: err.message
            });
        }
    }

    /**
     * Test Reliability Tracker CLI functionality
     */
    async testReliabilityTrackerCLI() {
        console.log('\n🔍 TEST: Reliability Tracker CLI');

        try {
            // Test help command
            const helpOutput = execSync('node .claude/guardrails/reliability-tracker.js', {
                cwd: this.projectRoot,
                encoding: 'utf8',
                timeout: 5000
            });

            const hasUsage = helpOutput.includes('Usage') && helpOutput.includes('report');

            // Test report command
            const reportOutput = execSync('node .claude/guardrails/reliability-tracker.js report', {
                cwd: this.projectRoot,
                encoding: 'utf8',
                timeout: 10000
            });

            const hasReport = reportOutput.includes('AGENT RELIABILITY REPORT');

            // Test sync command
            const syncOutput = execSync('node .claude/guardrails/reliability-tracker.js sync', {
                cwd: this.projectRoot,
                encoding: 'utf8',
                timeout: 10000
            });

            const hasSync = syncOutput.includes('Synced reliability data');

            this.testResults.push({
                component: 'Reliability Tracker CLI',
                tests: [
                    { name: 'Help Command', passed: hasUsage },
                    { name: 'Report Command', passed: hasReport },
                    { name: 'Sync Command', passed: hasSync }
                ],
                overall: hasUsage && hasReport && hasSync
            });

            console.log(`   Help Command: ${hasUsage ? '✅' : '❌'}`);
            console.log(`   Report Command: ${hasReport ? '✅' : '❌'}`);
            console.log(`   Sync Command: ${hasSync ? '✅' : '❌'}`);

        } catch (err) {
            console.log(`   ❌ CLI test failed: ${err.message}`);
            this.testResults.push({
                component: 'Reliability Tracker CLI',
                tests: [],
                overall: false,
                error: err.message
            });
        }
    }

    /**
     * Test component integration
     */
    async testComponentIntegration() {
        console.log('\n🔍 TEST: Component Integration');

        try {
            // Create a simple test session
            const AgentExecutionMonitor = require('./execution-monitor.js');
            const monitor = new AgentExecutionMonitor();

            // Start a test session
            const sessionId = monitor.startSession('integration-test-agent', 'Test integration');
            
            // Record some activity
            monitor.recordClaim('test_claim', 'Integration test claim');
            monitor.recordToolCall('Read', { file_path: 'test.txt' }, { success: true });

            // End session
            const result = await monitor.endSession();

            const integrationPassed = result && result.sessionId === sessionId;

            // Test if verifier can load the session
            const AgentVerifier = require('./verify-agent.js');
            const verifier = new AgentVerifier();
            
            const sessions = verifier.listSessions();
            const sessionFound = sessions.some(s => s.sessionId === sessionId);

            this.testResults.push({
                component: 'Component Integration',
                tests: [
                    { name: 'Monitor Session Creation', passed: integrationPassed },
                    { name: 'Verifier Session Discovery', passed: sessionFound }
                ],
                overall: integrationPassed && sessionFound
            });

            console.log(`   Monitor Session Creation: ${integrationPassed ? '✅' : '❌'}`);
            console.log(`   Verifier Session Discovery: ${sessionFound ? '✅' : '❌'}`);

        } catch (err) {
            console.log(`   ❌ Integration test failed: ${err.message}`);
            this.testResults.push({
                component: 'Component Integration',
                tests: [],
                overall: false,
                error: err.message
            });
        }
    }

    /**
     * Test specific phantom detection algorithms
     */
    async testPhantomDetectionAlgorithms() {
        console.log('\n🚨 TEST: Phantom Detection Algorithms');

        try {
            const AgentExecutionMonitor = require('./execution-monitor.js');
            const monitor = new AgentExecutionMonitor();

            // Test 1: No tool calls detection
            monitor.currentSession = {
                sessionId: 'test-phantom-1',
                claims: [{ type: 'file_creation', description: 'Created file' }],
                toolCalls: []
            };

            const phantomResult1 = monitor.detectPhantomWork();
            const noToolCallsDetected = phantomResult1.phantoms.some(p => p.type === 'NO_EXECUTION');

            // Test 2: File creation without Write tools
            monitor.currentSession = {
                sessionId: 'test-phantom-2',
                claims: [{ type: 'file_creation', description: 'Created file' }],
                toolCalls: [{ tool: 'Read', parameters: {}, result: { success: true } }]
            };

            const phantomResult2 = monitor.detectPhantomWork();
            const phantomFileDetected = phantomResult2.phantoms.some(p => p.type === 'PHANTOM_FILE_CREATION');

            this.testResults.push({
                component: 'Phantom Detection Algorithms',
                tests: [
                    { name: 'No Tool Calls Detection', passed: noToolCallsDetected },
                    { name: 'Phantom File Creation Detection', passed: phantomFileDetected }
                ],
                overall: noToolCallsDetected && phantomFileDetected
            });

            console.log(`   No Tool Calls Detection: ${noToolCallsDetected ? '✅' : '❌'}`);
            console.log(`   Phantom File Creation Detection: ${phantomFileDetected ? '✅' : '❌'}`);

        } catch (err) {
            console.log(`   ❌ Algorithm test failed: ${err.message}`);
            this.testResults.push({
                component: 'Phantom Detection Algorithms',
                tests: [],
                overall: false,
                error: err.message
            });
        }
    }

    /**
     * Generate component test report
     */
    generateComponentTestReport() {
        console.log('\n📊 GUARDRAIL COMPONENT TEST RESULTS');
        console.log('=' .repeat(50));

        const totalComponents = this.testResults.length;
        const passedComponents = this.testResults.filter(r => r.overall).length;

        console.log(`\nCOMPONENT SUMMARY:`);
        console.log(`Total Components: ${totalComponents}`);
        console.log(`Passed: ${passedComponents} ✅`);
        console.log(`Failed: ${totalComponents - passedComponents} ${totalComponents - passedComponents > 0 ? '❌' : ''}`);

        console.log(`\nDETAILED RESULTS:`);
        this.testResults.forEach(result => {
            const status = result.overall ? '✅ PASS' : '❌ FAIL';
            console.log(`\n${result.component}: ${status}`);
            
            if (result.tests && result.tests.length > 0) {
                result.tests.forEach(test => {
                    console.log(`  ${test.name}: ${test.passed ? '✅' : '❌'}`);
                });
            }
            
            if (result.error) {
                console.log(`  Error: ${result.error}`);
            }
        });

        // Generate CLI compatibility report
        console.log(`\n🖥️  CLI COMPATIBILITY:`);
        const cliComponents = this.testResults.filter(r => r.component.includes('CLI'));
        const workingCLIs = cliComponents.filter(r => r.overall).length;
        console.log(`Working CLIs: ${workingCLIs}/${cliComponents.length}`);

        // Save report
        const testDataDir = path.join(this.projectRoot, '.claude', 'guardrails', 'test-data');
        if (!fs.existsSync(testDataDir)) {
            fs.mkdirSync(testDataDir, { recursive: true });
        }

        const reportFile = path.join(testDataDir, 'component-test-report.json');
        fs.writeFileSync(reportFile, JSON.stringify({
            timestamp: new Date().toISOString(),
            summary: {
                totalComponents,
                passedComponents,
                failedComponents: totalComponents - passedComponents
            },
            results: this.testResults
        }, null, 2));

        console.log(`\n📄 Component test report saved: ${reportFile}`);
    }
}

// CLI interface
if (require.main === module) {
    const tester = new GuardrailComponentTester();
    
    const command = process.argv[2];
    
    switch (command) {
        case 'run':
            tester.runAllTests().then(results => {
                const passedCount = results.filter(r => r.overall).length;
                const totalCount = results.length;
                
                console.log(`\n🏁 Component testing complete: ${passedCount}/${totalCount} passed`);
                process.exit(passedCount === totalCount ? 0 : 1);
            }).catch(err => {
                console.error('💥 Component testing failed:', err.message);
                process.exit(1);
            });
            break;
            
        case 'algorithms':
            // Test just the phantom detection algorithms
            tester.testPhantomDetectionAlgorithms().then(() => {
                console.log('Algorithm testing complete');
            });
            break;
            
        default:
            console.log('Guardrail Component Testing Tool');
            console.log('');
            console.log('Usage:');
            console.log('  test-guardrail-components.js run        - Test all guardrail components');
            console.log('  test-guardrail-components.js algorithms - Test phantom detection algorithms');
            console.log('');
            console.log('Tests individual guardrail components for functionality and CLI compatibility.');
    }
}

module.exports = GuardrailComponentTester;