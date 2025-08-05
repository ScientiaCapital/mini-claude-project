#!/usr/bin/env node

/**
 * Comprehensive Guardrail System Test Runner
 * 
 * Executes all guardrail tests and generates a comprehensive validation report.
 * This is the main entry point for testing the entire phantom work detection system.
 */

const fs = require('fs');
const path = require('path');
const { execSync, spawn } = require('child_process');

class GuardrailTestRunner {
    constructor() {
        this.projectRoot = this.findProjectRoot();
        this.testDataDir = path.join(this.projectRoot, '.claude', 'guardrails', 'test-data');
        this.results = {
            overall: {
                totalTests: 0,
                passedTests: 0,
                failedTests: 0,
                startTime: new Date().toISOString(),
                endTime: null
            },
            components: {},
            phantomDetection: {},
            precommitBlocking: {},
            reliabilityScoring: {},
            integration: {},
            recommendations: []
        };
        this.ensureTestDirectory();
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
     * Run all guardrail tests
     */
    async runAllTests() {
        console.log('🔍 COMPREHENSIVE GUARDRAIL SYSTEM VALIDATION');
        console.log('=' .repeat(60));
        console.log(`Started: ${this.results.overall.startTime}`);
        console.log('');

        try {
            // Phase 1: Component Testing
            console.log('📋 PHASE 1: Component Functionality Testing');
            console.log('-' .repeat(40));
            await this.runComponentTests();

            // Phase 2: Phantom Detection Testing
            console.log('\n🚨 PHASE 2: Phantom Work Detection Testing');
            console.log('-' .repeat(40));
            await this.runPhantomDetectionTests();

            // Phase 3: Pre-commit Hook Testing
            console.log('\n🚫 PHASE 3: Pre-commit Hook Blocking Testing');
            console.log('-' .repeat(40));
            await this.runPrecommitTests();

            // Phase 4: Reliability Scoring Testing
            console.log('\n📊 PHASE 4: Reliability Tracker Scoring Testing');
            console.log('-' .repeat(40));
            await this.runReliabilityTests();

            // Phase 5: Integration Testing
            console.log('\n🔗 PHASE 5: System Integration Testing');
            console.log('-' .repeat(40));
            await this.runIntegrationTests();

            // Finalize results
            this.results.overall.endTime = new Date().toISOString();
            this.analyzeResults();
            this.generateComprehensiveReport();

            return this.results;

        } catch (err) {
            console.error(`💥 Test runner failed: ${err.message}`);
            this.results.overall.endTime = new Date().toISOString();
            this.results.error = err.message;
            throw err;
        }
    }

    /**
     * Run component functionality tests
     */
    async runComponentTests() {
        try {
            console.log('🔧 Testing individual guardrail components...');
            
            const componentResult = await this.runTestScript('test-guardrail-components.js', 'run');
            
            this.results.components = {
                passed: componentResult.exitCode === 0,
                details: this.loadTestReport('component-test-report.json'),
                timestamp: new Date().toISOString()
            };

            this.updateOverallStats(this.results.components);
            
            console.log(`   Result: ${this.results.components.passed ? '✅ PASS' : '❌ FAIL'}`);

        } catch (err) {
            console.error(`   💥 Component tests failed: ${err.message}`);
            this.results.components = { passed: false, error: err.message };
            this.updateOverallStats(this.results.components);
        }
    }

    /**
     * Run phantom work detection tests
     */
    async runPhantomDetectionTests() {
        try {
            console.log('🚨 Testing phantom work detection algorithms...');
            
            // Test phantom detection algorithms specifically
            const algorithmResult = await this.runTestScript('test-guardrail-components.js', 'algorithms');
            
            this.results.phantomDetection = {
                algorithmsPassed: algorithmResult.exitCode === 0,
                timestamp: new Date().toISOString()
            };

            this.updateOverallStats(this.results.phantomDetection);
            
            console.log(`   Algorithm Tests: ${this.results.phantomDetection.algorithmsPassed ? '✅ PASS' : '❌ FAIL'}`);

        } catch (err) {
            console.error(`   💥 Phantom detection tests failed: ${err.message}`);
            this.results.phantomDetection = { algorithmsPassed: false, error: err.message };
            this.updateOverallStats(this.results.phantomDetection);
        }
    }

    /**
     * Run pre-commit hook tests
     */
    async runPrecommitTests() {
        try {
            console.log('🚫 Testing pre-commit hook blocking behavior...');
            
            const precommitResult = await this.runTestScript('test-precommit-phantom-blocking.js', 'run');
            
            this.results.precommitBlocking = {
                passed: precommitResult.exitCode === 0,
                details: this.loadTestReport('precommit-blocking-test-report.json'),
                timestamp: new Date().toISOString()
            };

            this.updateOverallStats(this.results.precommitBlocking);
            
            console.log(`   Result: ${this.results.precommitBlocking.passed ? '✅ PASS' : '❌ FAIL'}`);
            
            // Note pre-commit issues for recommendations
            if (!this.results.precommitBlocking.passed) {
                this.results.recommendations.push({
                    type: 'CRITICAL',
                    component: 'Pre-commit Hook',
                    issue: 'Pre-commit hook not properly blocking phantom work',
                    action: 'Review pre-commit hook session verification logic'
                });
            }

        } catch (err) {
            console.error(`   💥 Pre-commit tests failed: ${err.message}`);
            this.results.precommitBlocking = { passed: false, error: err.message };
            this.updateOverallStats(this.results.precommitBlocking);
        }
    }

    /**
     * Run reliability tracker tests
     */
    async runReliabilityTests() {
        try {
            console.log('📊 Testing reliability tracker scoring...');
            
            const reliabilityResult = await this.runTestScript('test-reliability-scoring.js', 'run');
            
            this.results.reliabilityScoring = {
                passed: reliabilityResult.exitCode === 0,
                details: this.loadTestReport('reliability-scoring-test-report.json'),
                timestamp: new Date().toISOString()
            };

            this.updateOverallStats(this.results.reliabilityScoring);
            
            console.log(`   Result: ${this.results.reliabilityScoring.passed ? '✅ PASS' : '❌ FAIL'}`);

        } catch (err) {
            console.error(`   💥 Reliability tests failed: ${err.message}`);
            this.results.reliabilityScoring = { passed: false, error: err.message };
            this.updateOverallStats(this.results.reliabilityScoring);
        }
    }

    /**
     * Run integration tests
     */
    async runIntegrationTests() {
        try {
            console.log('🔗 Testing system integration...');
            
            // Test CLI tools integration
            const cliTests = await this.testCLIIntegration();
            
            // Test data flow between components
            const dataFlowTests = await this.testDataFlow();
            
            this.results.integration = {
                cliIntegration: cliTests,
                dataFlow: dataFlowTests,
                passed: cliTests.passed && dataFlowTests.passed,
                timestamp: new Date().toISOString()
            };

            this.updateOverallStats(this.results.integration);
            
            console.log(`   CLI Integration: ${cliTests.passed ? '✅ PASS' : '❌ FAIL'}`);
            console.log(`   Data Flow: ${dataFlowTests.passed ? '✅ PASS' : '❌ FAIL'}`);

        } catch (err) {
            console.error(`   💥 Integration tests failed: ${err.message}`);
            this.results.integration = { passed: false, error: err.message };
            this.updateOverallStats(this.results.integration);
        }
    }

    /**
     * Test CLI tool integration
     */
    async testCLIIntegration() {
        try {
            const cliTools = [
                'execution-monitor.js',
                'verify-agent.js', 
                'pre-commit-hook.js',
                'reliability-tracker.js'
            ];

            let workingCLIs = 0;
            const cliResults = [];

            for (const cli of cliTools) {
                try {
                    const output = execSync(`node .claude/guardrails/${cli}`, {
                        cwd: this.projectRoot,
                        encoding: 'utf8',
                        timeout: 5000
                    });
                    
                    const working = output.includes('Usage') || output.includes('help');
                    if (working) workingCLIs++;
                    
                    cliResults.push({ cli, working, output: output.substring(0, 100) });
                } catch (err) {
                    cliResults.push({ cli, working: false, error: err.message });
                }
            }

            return {
                passed: workingCLIs === cliTools.length,
                workingCLIs,
                totalCLIs: cliTools.length,
                details: cliResults
            };

        } catch (err) {
            return { passed: false, error: err.message };
        }
    }

    /**
     * Test data flow between components
     */
    async testDataFlow() {
        try {
            // Test: Monitor -> Verifier -> Reliability Tracker flow
            const AgentExecutionMonitor = require('./execution-monitor.js');
            const AgentVerifier = require('./verify-agent.js');
            const AgentReliabilityTracker = require('./reliability-tracker.js');

            const monitor = new AgentExecutionMonitor();
            const verifier = new AgentVerifier();
            const tracker = new AgentReliabilityTracker();

            // 1. Create session with monitor
            const sessionId = monitor.startSession('integration-test', 'Test data flow');
            monitor.recordClaim('test_work', 'Test integration claim');
            monitor.recordToolCall('Read', { file_path: 'test.txt' }, { success: true });
            const sessionResult = await monitor.endSession();

            // 2. Verify session with verifier
            const verificationResult = await verifier.verifySession(sessionId, { 
                skipTests: true, 
                skipBuild: true 
            });

            // 3. Update reliability tracker
            tracker.updateFromSession(sessionId);
            const reliability = tracker.getAgentReliability('integration-test');

            const dataFlowWorking = sessionResult && 
                                  verificationResult && 
                                  reliability.exists;

            return {
                passed: dataFlowWorking,
                sessionCreated: !!sessionResult,
                verificationCompleted: !!verificationResult,
                reliabilityUpdated: reliability.exists,
                sessionId
            };

        } catch (err) {
            return { passed: false, error: err.message };
        }
    }

    /**
     * Run a test script and return result
     */
    async runTestScript(scriptName, command) {
        return new Promise((resolve, reject) => {
            const scriptPath = path.join(this.projectRoot, '.claude', 'guardrails', scriptName);
            
            const child = spawn('node', [scriptPath, command], {
                cwd: this.projectRoot,
                stdio: ['pipe', 'pipe', 'pipe']
            });

            let stdout = '';
            let stderr = '';

            child.stdout.on('data', (data) => {
                stdout += data.toString();
            });

            child.stderr.on('data', (data) => {
                stderr += data.toString();
            });

            child.on('close', (exitCode) => {
                resolve({
                    exitCode,
                    stdout,
                    stderr,
                    success: exitCode === 0
                });
            });

            child.on('error', (err) => {
                reject(err);
            });

            // Timeout after 2 minutes
            setTimeout(() => {
                child.kill();
                reject(new Error(`Test script ${scriptName} timed out`));
            }, 120000);
        });
    }

    /**
     * Load test report if it exists
     */
    loadTestReport(reportFile) {
        try {
            const reportPath = path.join(this.testDataDir, reportFile);
            if (fs.existsSync(reportPath)) {
                return JSON.parse(fs.readFileSync(reportPath, 'utf8'));
            }
        } catch (err) {
            console.warn(`Could not load report ${reportFile}: ${err.message}`);
        }
        return null;
    }

    /**
     * Update overall statistics
     */
    updateOverallStats(componentResult) {
        this.results.overall.totalTests++;
        if (componentResult.passed) {
            this.results.overall.passedTests++;
        } else {
            this.results.overall.failedTests++;
        }
    }

    /**
     * Analyze all results and generate recommendations
     */
    analyzeResults() {
        const { overall } = this.results;
        overall.successRate = overall.totalTests > 0 ? overall.passedTests / overall.totalTests : 0;
        overall.duration = this.calculateDuration(overall.startTime, overall.endTime);

        // Generate recommendations based on results
        if (overall.successRate >= 0.9) {
            this.results.recommendations.push({
                type: 'SUCCESS',
                message: 'Guardrail system validation passed with excellent results',
                action: 'System is ready for production use'
            });
        } else if (overall.successRate >= 0.7) {
            this.results.recommendations.push({
                type: 'WARNING',
                message: 'Guardrail system validation passed with some issues',
                action: 'Review failed components before production deployment'
            });
        } else {
            this.results.recommendations.push({
                type: 'CRITICAL',
                message: 'Guardrail system validation failed',
                action: 'Do not deploy until all issues are resolved'
            });
        }

        // Component-specific recommendations
        if (!this.results.components.passed) {
            this.results.recommendations.push({
                type: 'HIGH',
                component: 'Core Components',
                issue: 'Basic component functionality failed',
                action: 'Review CLI interfaces and basic operations'
            });
        }

        if (!this.results.phantomDetection.algorithmsPassed) {
            this.results.recommendations.push({
                type: 'CRITICAL',
                component: 'Phantom Detection',
                issue: 'Phantom work detection algorithms failed',
                action: 'Fix phantom detection logic before deployment'
            });
        }

        if (!this.results.reliabilityScoring.passed) {
            this.results.recommendations.push({
                type: 'HIGH',
                component: 'Reliability Scoring',
                issue: 'Agent reliability scoring failed',
                action: 'Review scoring algorithms and deployment recommendations'
            });
        }

        if (!this.results.integration.passed) {
            this.results.recommendations.push({
                type: 'HIGH',
                component: 'System Integration',
                issue: 'Component integration failed',
                action: 'Review data flow between guardrail components'
            });
        }
    }

    /**
     * Generate comprehensive validation report
     */
    generateComprehensiveReport() {
        console.log('\n📊 COMPREHENSIVE GUARDRAIL VALIDATION REPORT');
        console.log('=' .repeat(60));

        const { overall } = this.results;
        
        console.log(`\nOVERALL RESULTS:`);
        console.log(`Duration: ${overall.duration}`);
        console.log(`Total Tests: ${overall.totalTests}`);
        console.log(`Passed: ${overall.passedTests} ✅`);
        console.log(`Failed: ${overall.failedTests} ${overall.failedTests > 0 ? '❌' : ''}`);
        console.log(`Success Rate: ${(overall.successRate * 100).toFixed(1)}%`);

        console.log(`\nCOMPONENT BREAKDOWN:`);
        console.log(`🔧 Core Components: ${this.results.components.passed ? '✅ PASS' : '❌ FAIL'}`);
        console.log(`🚨 Phantom Detection: ${this.results.phantomDetection.algorithmsPassed ? '✅ PASS' : '❌ FAIL'}`);
        console.log(`🚫 Pre-commit Blocking: ${this.results.precommitBlocking.passed ? '✅ PASS' : '❌ FAIL'}`);
        console.log(`📊 Reliability Scoring: ${this.results.reliabilityScoring.passed ? '✅ PASS' : '❌ FAIL'}`);
        console.log(`🔗 System Integration: ${this.results.integration.passed ? '✅ PASS' : '❌ FAIL'}`);

        if (this.results.integration.cliIntegration) {
            const cli = this.results.integration.cliIntegration;
            console.log(`\nCLI COMPATIBILITY:`);
            console.log(`Working CLIs: ${cli.workingCLIs}/${cli.totalCLIs}`);
        }

        console.log(`\nRECOMMENDATIONS:`);
        if (this.results.recommendations.length === 0) {
            console.log('✅ No issues found - system ready for deployment');
        } else {
            this.results.recommendations.forEach((rec, index) => {
                const icon = rec.type === 'CRITICAL' ? '🚨' : 
                           rec.type === 'HIGH' ? '⚠️' : 
                           rec.type === 'WARNING' ? '⚠️' : '✅';
                console.log(`${index + 1}. ${icon} ${rec.type}: ${rec.message || rec.issue}`);
                if (rec.action) {
                    console.log(`   Action: ${rec.action}`);
                }
                if (rec.component) {
                    console.log(`   Component: ${rec.component}`);
                }
            });
        }

        // Save comprehensive report
        const reportFile = path.join(this.testDataDir, 'comprehensive-validation-report.json');
        fs.writeFileSync(reportFile, JSON.stringify(this.results, null, 2));

        console.log(`\n📄 Full validation report saved: ${reportFile}`);

        // Generate summary for CI/CD
        const summaryFile = path.join(this.testDataDir, 'validation-summary.json');
        const summary = {
            timestamp: overall.endTime,
            success: overall.successRate >= 0.7,
            successRate: overall.successRate,
            criticalIssues: this.results.recommendations.filter(r => r.type === 'CRITICAL').length,
            readyForProduction: overall.successRate >= 0.9 && 
                               this.results.recommendations.filter(r => r.type === 'CRITICAL').length === 0
        };
        fs.writeFileSync(summaryFile, JSON.stringify(summary, null, 2));

        console.log(`\n🎯 VALIDATION SUMMARY:`);
        console.log(`Ready for Production: ${summary.readyForProduction ? 'YES ✅' : 'NO ❌'}`);
        console.log(`Critical Issues: ${summary.criticalIssues}`);
    }

    /**
     * Calculate duration between timestamps
     */
    calculateDuration(start, end) {
        if (!start || !end) return 'Unknown';
        
        const startTime = new Date(start);
        const endTime = new Date(end);
        const durationMs = endTime - startTime;
        
        const minutes = Math.floor(durationMs / 60000);
        const seconds = Math.floor((durationMs % 60000) / 1000);
        
        return `${minutes}m ${seconds}s`;
    }

    /**
     * Clean up all test data
     */
    async cleanup() {
        try {
            console.log('\n🧹 Cleaning up test data...');
            
            // Clean up each test suite
            const cleanupScripts = [
                'test-phantom-scenarios.js cleanup',
                'test-precommit-phantom-blocking.js cleanup',
                'test-reliability-scoring.js cleanup'
            ];

            for (const script of cleanupScripts) {
                try {
                    execSync(`node .claude/guardrails/${script}`, {
                        cwd: this.projectRoot,
                        timeout: 10000
                    });
                } catch (err) {
                    console.warn(`Cleanup warning: ${script} - ${err.message}`);
                }
            }

            console.log('✅ Test data cleanup completed');

        } catch (err) {
            console.warn(`Cleanup warning: ${err.message}`);
        }
    }
}

// CLI interface
if (require.main === module) {
    const runner = new GuardrailTestRunner();
    
    const command = process.argv[2];
    
    switch (command) {
        case 'run':
            runner.runAllTests().then(results => {
                const success = results.overall.successRate >= 0.7;
                
                console.log(`\n🏁 Guardrail validation complete: ${success ? 'SUCCESS' : 'FAILURE'}`);
                
                if (process.argv.includes('--cleanup')) {
                    runner.cleanup();
                }
                
                process.exit(success ? 0 : 1);
            }).catch(err => {
                console.error('💥 Guardrail validation failed:', err.message);
                process.exit(1);
            });
            break;
            
        case 'cleanup':
            runner.cleanup().then(() => {
                console.log('Cleanup completed');
            });
            break;
            
        default:
            console.log('Comprehensive Guardrail System Test Runner');
            console.log('');
            console.log('Usage:');
            console.log('  test-runner.js run [--cleanup]  - Run full guardrail validation');
            console.log('  test-runner.js cleanup          - Clean up all test data');
            console.log('');
            console.log('This tool validates the entire guardrail system for phantom work detection,');
            console.log('pre-commit blocking, reliability scoring, and component integration.');
            console.log('');
            console.log('Use this before deploying the guardrail system to production.');
    }
}

module.exports = GuardrailTestRunner;