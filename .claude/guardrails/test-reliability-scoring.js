#!/usr/bin/env node

/**
 * Reliability Tracker Scoring Test
 * 
 * Tests the reliability tracker's scoring accuracy and agent recommendations.
 * Validates that scoring algorithms correctly identify reliable vs unreliable agents.
 */

const fs = require('fs');
const path = require('path');

class ReliabilityTrackerScoringTest {
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
     * Run all reliability scoring tests
     */
    async runAllTests() {
        console.log('📊 TESTING RELIABILITY TRACKER SCORING');
        console.log('=' .repeat(50));

        // Test 1: Excellent Agent Scoring
        await this.testExcellentAgentScoring();

        // Test 2: Phantom Work Agent Scoring  
        await this.testPhantomWorkAgentScoring();

        // Test 3: Mixed Performance Agent Scoring
        await this.testMixedPerformanceScoring();

        // Test 4: Trend Analysis
        await this.testTrendAnalysis();

        // Test 5: Deployment Recommendations
        await this.testDeploymentRecommendations();

        this.generateScoringTestReport();
        return this.testResults;
    }

    /**
     * Test scoring for excellent agent performance
     */
    async testExcellentAgentScoring() {
        console.log('\n⭐ TEST: Excellent Agent Scoring');

        try {
            const AgentReliabilityTracker = require('./reliability-tracker.js');
            const tracker = new AgentReliabilityTracker();

            // Create excellent agent sessions
            const excellentSessions = [
                this.createTestSession('excellent-agent', 1.0, 'PASS', false),
                this.createTestSession('excellent-agent', 0.95, 'PASS', false),
                this.createTestSession('excellent-agent', 0.9, 'PASS', false),
                this.createTestSession('excellent-agent', 0.85, 'PASS', false)
            ];

            // Update tracker with sessions
            for (const session of excellentSessions) {
                this.saveTestSession(session);
                tracker.updateFromSession(session.sessionId);
            }

            // Get reliability metrics
            const reliability = tracker.getAgentReliability('excellent-agent');

            const expectedExcellent = reliability.metrics.reliabilityLevel === 'excellent' &&
                                    reliability.metrics.averageScore >= 0.8 &&
                                    reliability.metrics.successRate >= 0.8 &&
                                    reliability.recommendation.action === 'DEPLOY';

            this.testResults.push({
                scenario: 'Excellent Agent Scoring',
                agentType: 'excellent-agent',
                expected: 'excellent reliability, DEPLOY recommendation',
                actual: {
                    reliabilityLevel: reliability.metrics.reliabilityLevel,
                    averageScore: reliability.metrics.averageScore,
                    successRate: reliability.metrics.successRate,
                    recommendation: reliability.recommendation.action
                },
                passed: expectedExcellent,
                details: reliability.metrics
            });

            console.log(`   Result: ${expectedExcellent ? '✅ PASS' : '❌ FAIL'}`);
            console.log(`   Reliability Level: ${reliability.metrics.reliabilityLevel}`);
            console.log(`   Average Score: ${(reliability.metrics.averageScore * 100).toFixed(1)}%`);
            console.log(`   Recommendation: ${reliability.recommendation.action}`);

        } catch (err) {
            console.error(`   💥 Test failed: ${err.message}`);
            this.testResults.push({
                scenario: 'Excellent Agent Scoring',
                passed: false,
                error: err.message
            });
        }
    }

    /**
     * Test scoring for phantom work agent
     */
    async testPhantomWorkAgentScoring() {
        console.log('\n🚨 TEST: Phantom Work Agent Scoring');

        try {
            const AgentReliabilityTracker = require('./reliability-tracker.js');
            const tracker = new AgentReliabilityTracker();

            // Create phantom work agent sessions
            const phantomSessions = [
                this.createTestSession('phantom-agent', 0.0, 'FAIL', true),
                this.createTestSession('phantom-agent', 0.1, 'FAIL', true),
                this.createTestSession('phantom-agent', 0.0, 'FAIL', true),
                this.createTestSession('phantom-agent', 0.2, 'FAIL', true)
            ];

            // Update tracker with sessions
            for (const session of phantomSessions) {
                this.saveTestSession(session);
                tracker.updateFromSession(session.sessionId);
            }

            // Get reliability metrics
            const reliability = tracker.getAgentReliability('phantom-agent');

            const expectedPoor = (reliability.metrics.reliabilityLevel === 'poor' || 
                                reliability.metrics.reliabilityLevel === 'unreliable') &&
                               reliability.metrics.phantomRate > 0 &&
                               reliability.recommendation.action === 'AVOID';

            this.testResults.push({
                scenario: 'Phantom Work Agent Scoring',
                agentType: 'phantom-agent',
                expected: 'poor/unreliable reliability, AVOID recommendation',
                actual: {
                    reliabilityLevel: reliability.metrics.reliabilityLevel,
                    phantomRate: reliability.metrics.phantomRate,
                    averageScore: reliability.metrics.averageScore,
                    recommendation: reliability.recommendation.action
                },
                passed: expectedPoor,
                details: reliability.metrics
            });

            console.log(`   Result: ${expectedPoor ? '✅ PASS' : '❌ FAIL'}`);
            console.log(`   Reliability Level: ${reliability.metrics.reliabilityLevel}`);
            console.log(`   Phantom Rate: ${(reliability.metrics.phantomRate * 100).toFixed(1)}%`);
            console.log(`   Recommendation: ${reliability.recommendation.action}`);

        } catch (err) {
            console.error(`   💥 Test failed: ${err.message}`);
            this.testResults.push({
                scenario: 'Phantom Work Agent Scoring',
                passed: false,
                error: err.message
            });
        }
    }

    /**
     * Test scoring for mixed performance agent
     */
    async testMixedPerformanceScoring() {
        console.log('\n🔄 TEST: Mixed Performance Scoring');

        try {
            const AgentReliabilityTracker = require('./reliability-tracker.js');
            const tracker = new AgentReliabilityTracker();

            // Create mixed performance sessions
            const mixedSessions = [
                this.createTestSession('mixed-agent', 0.8, 'PASS', false),
                this.createTestSession('mixed-agent', 0.5, 'WARNING', false),
                this.createTestSession('mixed-agent', 0.7, 'PASS', false),
                this.createTestSession('mixed-agent', 0.4, 'FAIL', false),
                this.createTestSession('mixed-agent', 0.6, 'WARNING', false)
            ];

            // Update tracker with sessions
            for (const session of mixedSessions) {
                this.saveTestSession(session);
                tracker.updateFromSession(session.sessionId);
            }

            // Get reliability metrics
            const reliability = tracker.getAgentReliability('mixed-agent');

            const expectedFairGood = ['fair', 'good'].includes(reliability.metrics.reliabilityLevel) &&
                                   reliability.metrics.averageScore >= 0.4 &&
                                   reliability.metrics.averageScore <= 0.8 &&
                                   ['DEPLOY_WITH_MONITORING', 'DEPLOY_WITH_CAUTION'].includes(reliability.recommendation.action);

            this.testResults.push({
                scenario: 'Mixed Performance Scoring',
                agentType: 'mixed-agent',
                expected: 'fair/good reliability, cautious deployment',
                actual: {
                    reliabilityLevel: reliability.metrics.reliabilityLevel,
                    averageScore: reliability.metrics.averageScore,
                    successRate: reliability.metrics.successRate,
                    recommendation: reliability.recommendation.action
                },
                passed: expectedFairGood,
                details: reliability.metrics
            });

            console.log(`   Result: ${expectedFairGood ? '✅ PASS' : '❌ FAIL'}`);
            console.log(`   Reliability Level: ${reliability.metrics.reliabilityLevel}`);
            console.log(`   Average Score: ${(reliability.metrics.averageScore * 100).toFixed(1)}%`);
            console.log(`   Recommendation: ${reliability.recommendation.action}`);

        } catch (err) {
            console.error(`   💥 Test failed: ${err.message}`);
            this.testResults.push({
                scenario: 'Mixed Performance Scoring',
                passed: false,
                error: err.message
            });
        }
    }

    /**
     * Test trend analysis functionality
     */
    async testTrendAnalysis() {
        console.log('\n📈 TEST: Trend Analysis');

        try {
            const AgentReliabilityTracker = require('./reliability-tracker.js');
            const tracker = new AgentReliabilityTracker();

            // Create improving trend sessions (scores increase over time)
            const improvingSessions = [
                this.createTestSession('improving-agent', 0.3, 'FAIL', false, 1),
                this.createTestSession('improving-agent', 0.4, 'FAIL', false, 2),
                this.createTestSession('improving-agent', 0.6, 'WARNING', false, 3),
                this.createTestSession('improving-agent', 0.7, 'PASS', false, 4),
                this.createTestSession('improving-agent', 0.8, 'PASS', false, 5)
            ];

            // Create declining trend sessions (scores decrease over time)
            const decliningSessions = [
                this.createTestSession('declining-agent', 0.9, 'PASS', false, 1),
                this.createTestSession('declining-agent', 0.8, 'PASS', false, 2),
                this.createTestSession('declining-agent', 0.6, 'WARNING', false, 3),
                this.createTestSession('declining-agent', 0.4, 'FAIL', false, 4),
                this.createTestSession('declining-agent', 0.2, 'FAIL', false, 5)
            ];

            // Update tracker with sessions
            for (const session of [...improvingSessions, ...decliningSessions]) {
                this.saveTestSession(session);
                tracker.updateFromSession(session.sessionId);
            }

            // Test trend detection
            const improvingReliability = tracker.getAgentReliability('improving-agent');
            const decliningReliability = tracker.getAgentReliability('declining-agent');

            const trendDetected = improvingReliability.metrics.trendDirection === 'improving' &&
                                decliningReliability.metrics.trendDirection === 'declining';

            this.testResults.push({
                scenario: 'Trend Analysis',
                expected: 'Detect improving and declining trends',
                actual: {
                    improvingTrend: improvingReliability.metrics.trendDirection,
                    decliningTrend: decliningReliability.metrics.trendDirection
                },
                passed: trendDetected,
                details: {
                    improving: improvingReliability.metrics,
                    declining: decliningReliability.metrics
                }
            });

            console.log(`   Result: ${trendDetected ? '✅ PASS' : '❌ FAIL'}`);
            console.log(`   Improving Agent Trend: ${improvingReliability.metrics.trendDirection}`);
            console.log(`   Declining Agent Trend: ${decliningReliability.metrics.trendDirection}`);

        } catch (err) {
            console.error(`   💥 Test failed: ${err.message}`);
            this.testResults.push({
                scenario: 'Trend Analysis',
                passed: false,
                error: err.message
            });
        }
    }

    /**
     * Test deployment recommendation logic
     */
    async testDeploymentRecommendations() {
        console.log('\n🚀 TEST: Deployment Recommendations');

        try {
            const AgentReliabilityTracker = require('./reliability-tracker.js');
            const tracker = new AgentReliabilityTracker();

            // Test deployment recommendations for different reliability levels
            const testCases = [
                { agent: 'deploy-excellent', score: 0.9, status: 'PASS', phantom: false, expectedAction: 'DEPLOY' },
                { agent: 'deploy-good', score: 0.7, status: 'PASS', phantom: false, expectedAction: 'DEPLOY_WITH_MONITORING' },
                { agent: 'deploy-fair', score: 0.5, status: 'WARNING', phantom: false, expectedAction: 'DEPLOY_WITH_CAUTION' },
                { agent: 'deploy-poor', score: 0.3, status: 'FAIL', phantom: false, expectedAction: 'AVOID' },
                { agent: 'deploy-phantom', score: 0.8, status: 'FAIL', phantom: true, expectedAction: 'AVOID' }
            ];

            let correctRecommendations = 0;
            const results = [];

            for (const testCase of testCases) {
                // Create sessions for this test case
                const sessions = Array(3).fill().map((_, i) => 
                    this.createTestSession(testCase.agent, testCase.score, testCase.status, testCase.phantom, i + 1)
                );

                for (const session of sessions) {
                    this.saveTestSession(session);
                    tracker.updateFromSession(session.sessionId);
                }

                const shouldDeploy = tracker.shouldDeployAgent(testCase.agent);
                const actualAction = shouldDeploy.shouldDeploy ? 
                    (shouldDeploy.monitoring ? 'DEPLOY_WITH_MONITORING' : 'DEPLOY') : 'AVOID';

                const correct = testCase.expectedAction.includes('DEPLOY') === shouldDeploy.shouldDeploy;
                if (correct) correctRecommendations++;

                results.push({
                    agent: testCase.agent,
                    expected: testCase.expectedAction,
                    actual: actualAction,
                    correct
                });

                console.log(`   ${testCase.agent}: ${correct ? '✅' : '❌'} (${actualAction})`);
            }

            const recommendationsCorrect = correctRecommendations >= testCases.length * 0.8; // 80% accuracy

            this.testResults.push({
                scenario: 'Deployment Recommendations',
                expected: 'Correct deployment recommendations for different reliability levels',
                actual: `${correctRecommendations}/${testCases.length} correct`,
                passed: recommendationsCorrect,
                details: results
            });

            console.log(`   Result: ${recommendationsCorrect ? '✅ PASS' : '❌ FAIL'}`);
            console.log(`   Accuracy: ${correctRecommendations}/${testCases.length} (${(correctRecommendations/testCases.length*100).toFixed(1)}%)`);

        } catch (err) {
            console.error(`   💥 Test failed: ${err.message}`);
            this.testResults.push({
                scenario: 'Deployment Recommendations',
                passed: false,
                error: err.message
            });
        }
    }

    /**
     * Create a test session with specific characteristics
     */
    createTestSession(agentType, score, status, phantomDetected, sequenceNumber = 1) {
        const sessionId = `${agentType}_reliability_test_${Date.now()}_${sequenceNumber}`;
        const isSuccessful = status === 'PASS';
        
        return {
            sessionId,
            agentType,
            taskDescription: `Reliability test session ${sequenceNumber}`,
            startTime: new Date(Date.now() - (300000 * sequenceNumber)).toISOString(),
            endTime: new Date(Date.now() - (240000 * sequenceNumber)).toISOString(),
            toolCalls: phantomDetected ? [] : [
                {
                    tool: 'Write',
                    parameters: { file_path: 'test.txt', content: 'test' },
                    result: { success: true },
                    timestamp: new Date().toISOString()
                }
            ],
            claims: [
                {
                    type: 'test_work',
                    description: `Test work for ${agentType}`,
                    expectedFiles: phantomDetected ? ['nonexistent.txt'] : []
                }
            ],
            verification: {
                overall: { 
                    status, 
                    score,
                    issues: isSuccessful ? [] : ['Test failure']
                },
                phantomDetection: { 
                    detected: phantomDetected,
                    riskLevel: phantomDetected ? 'HIGH' : 'LOW',
                    phantoms: phantomDetected ? [
                        {
                            type: 'NO_EXECUTION',
                            description: 'Test phantom work',
                            severity: 'HIGH'
                        }
                    ] : []
                },
                toolCallVerification: {
                    score: phantomDetected ? 0 : 1,
                    hasWriteToolCalls: !phantomDetected
                },
                fileVerification: {
                    score: phantomDetected ? 0 : 1,
                    successRate: phantomDetected ? 0 : 1
                },
                claimVerification: {
                    score: phantomDetected ? 0 : 1
                }
            },
            verificationStatus: status
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
     * Generate reliability scoring test report
     */
    generateScoringTestReport() {
        console.log('\n📊 RELIABILITY SCORING TEST RESULTS');
        console.log('=' .repeat(50));

        const totalTests = this.testResults.length;
        const passedTests = this.testResults.filter(r => r.passed).length;

        console.log(`\nSUMMARY:`);
        console.log(`Total Tests: ${totalTests}`);
        console.log(`Passed: ${passedTests} ✅`);
        console.log(`Failed: ${totalTests - passedTests} ${totalTests - passedTests > 0 ? '❌' : ''}`);
        console.log(`Accuracy: ${(passedTests / totalTests * 100).toFixed(1)}%`);

        console.log(`\nDETAILED RESULTS:`);
        this.testResults.forEach((result, index) => {
            const status = result.passed ? '✅ PASS' : '❌ FAIL';
            console.log(`${index + 1}. ${result.scenario}: ${status}`);
            
            if (result.agentType) {
                console.log(`   Agent: ${result.agentType}`);
            }
            console.log(`   Expected: ${result.expected}`);
            console.log(`   Actual: ${typeof result.actual === 'string' ? result.actual : JSON.stringify(result.actual)}`);
            
            if (result.error) {
                console.log(`   Error: ${result.error}`);
            }
        });

        // Generate scoring accuracy analysis
        console.log(`\n🎯 SCORING ACCURACY ANALYSIS:`);
        const scoringTests = this.testResults.filter(r => r.scenario.includes('Scoring') || r.scenario.includes('Recommendations'));
        const scoringAccuracy = scoringTests.filter(r => r.passed).length / scoringTests.length;
        console.log(`Scoring Algorithm Accuracy: ${(scoringAccuracy * 100).toFixed(1)}%`);

        // Save report
        const testDataDir = path.join(this.projectRoot, '.claude', 'guardrails', 'test-data');
        if (!fs.existsSync(testDataDir)) {
            fs.mkdirSync(testDataDir, { recursive: true });
        }

        const reportFile = path.join(testDataDir, 'reliability-scoring-test-report.json');
        fs.writeFileSync(reportFile, JSON.stringify({
            timestamp: new Date().toISOString(),
            summary: {
                totalTests,
                passedTests,
                failedTests: totalTests - passedTests,
                accuracy: passedTests / totalTests,
                scoringAccuracy
            },
            results: this.testResults
        }, null, 2));

        console.log(`\n📄 Reliability scoring test report saved: ${reportFile}`);
    }

    /**
     * Clean up test sessions
     */
    cleanup() {
        try {
            const logsDir = path.join(this.projectRoot, '.claude', 'guardrails', 'logs');
            
            if (fs.existsSync(logsDir)) {
                const testSessions = fs.readdirSync(logsDir)
                    .filter(file => file.includes('reliability_test') && file.endsWith('.json'))
                    .map(file => path.join(logsDir, file));

                testSessions.forEach(file => fs.unlinkSync(file));
                console.log(`🧹 Cleaned up ${testSessions.length} reliability test sessions`);
            }

        } catch (err) {
            console.warn(`Warning: Cleanup failed: ${err.message}`);
        }
    }
}

// CLI interface
if (require.main === module) {
    const tester = new ReliabilityTrackerScoringTest();
    
    const command = process.argv[2];
    
    switch (command) {
        case 'run':
            tester.runAllTests().then(results => {
                const passedCount = results.filter(r => r.passed).length;
                const totalCount = results.length;
                
                console.log(`\n🏁 Reliability scoring tests complete: ${passedCount}/${totalCount} passed`);
                
                if (process.argv.includes('--cleanup')) {
                    tester.cleanup();
                }
                
                process.exit(passedCount === totalCount ? 0 : 1);
            }).catch(err => {
                console.error('💥 Reliability testing failed:', err.message);
                process.exit(1);
            });
            break;
            
        case 'cleanup':
            tester.cleanup();
            break;
            
        default:
            console.log('Reliability Tracker Scoring Test');
            console.log('');
            console.log('Usage:');
            console.log('  test-reliability-scoring.js run [--cleanup]  - Run reliability scoring tests');
            console.log('  test-reliability-scoring.js cleanup          - Clean up test data');
            console.log('');
            console.log('Tests the reliability tracker\'s scoring accuracy and deployment recommendations.');
    }
}

module.exports = ReliabilityTrackerScoringTest;