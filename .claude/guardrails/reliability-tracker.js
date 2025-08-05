#!/usr/bin/env node

/**
 * Agent Reliability Tracker
 * 
 * Tracks and scores agent reliability over time to identify patterns
 * and prevent repeated deployment of unreliable agents.
 */

const fs = require('fs');
const path = require('path');
const AgentExecutionMonitor = require('./execution-monitor');

class AgentReliabilityTracker {
    constructor() {
        this.projectRoot = this.findProjectRoot();
        this.reliabilityFile = path.join(this.projectRoot, '.claude', 'guardrails', 'reliability-data.json');
        this.data = this.loadReliabilityData();
    }

    findProjectRoot() {
        let current = process.cwd();
        while (current !== '/' && !fs.existsSync(path.join(current, 'CLAUDE.md'))) {
            current = path.dirname(current);
        }
        return current;
    }

    /**
     * Load existing reliability data
     */
    loadReliabilityData() {
        if (fs.existsSync(this.reliabilityFile)) {
            try {
                return JSON.parse(fs.readFileSync(this.reliabilityFile, 'utf8'));
            } catch (err) {
                console.warn('Failed to load reliability data, starting fresh');
            }
        }

        return {
            agents: {},
            summary: {
                totalSessions: 0,
                successfulSessions: 0,
                phantomWorkDetected: 0,
                lastUpdated: new Date().toISOString()
            }
        };
    }

    /**
     * Save reliability data
     */
    saveReliabilityData() {
        const dir = path.dirname(this.reliabilityFile);
        if (!fs.existsSync(dir)) {
            fs.mkdirSync(dir, { recursive: true });
        }
        
        this.data.summary.lastUpdated = new Date().toISOString();
        fs.writeFileSync(this.reliabilityFile, JSON.stringify(this.data, null, 2));
    }

    /**
     * Update reliability metrics from session data
     */
    updateFromSession(sessionId) {
        const session = AgentExecutionMonitor.loadSession(sessionId, this.projectRoot);
        if (!session || !session.verification) {
            console.warn(`Session ${sessionId} not found or not verified`);
            return;
        }

        const agentType = session.agentType;
        const verification = session.verification;

        // Initialize agent data if not exists
        if (!this.data.agents[agentType]) {
            this.data.agents[agentType] = {
                totalSessions: 0,
                successfulSessions: 0,
                phantomWorkCount: 0,
                averageScore: 0,
                lastScore: 0,
                reliabilityTrend: [],
                issues: [],
                firstSeen: new Date().toISOString(),
                lastSeen: new Date().toISOString()
            };
        }

        const agent = this.data.agents[agentType];
        
        // Update session counts
        agent.totalSessions++;
        this.data.summary.totalSessions++;

        // Update success metrics
        const isSuccessful = verification.overall?.status === 'PASS';
        if (isSuccessful) {
            agent.successfulSessions++;
            this.data.summary.successfulSessions++;
        }

        // Update phantom work detection
        if (verification.phantomDetection?.detected) {
            agent.phantomWorkCount++;
            this.data.summary.phantomWorkDetected++;
        }

        // Update scores
        const score = verification.overall?.score || 0;
        agent.lastScore = score;
        agent.averageScore = ((agent.averageScore * (agent.totalSessions - 1)) + score) / agent.totalSessions;

        // Update reliability trend (keep last 10 sessions)
        agent.reliabilityTrend.push({
            sessionId,
            score,
            status: verification.overall?.status,
            timestamp: session.endTime || new Date().toISOString(),
            phantomDetected: verification.phantomDetection?.detected || false
        });

        if (agent.reliabilityTrend.length > 10) {
            agent.reliabilityTrend = agent.reliabilityTrend.slice(-10);
        }

        // Track issues
        if (verification.overall?.issues?.length > 0) {
            for (const issue of verification.overall.issues) {
                const existingIssue = agent.issues.find(i => i.type === issue);
                if (existingIssue) {
                    existingIssue.count++;
                    existingIssue.lastSeen = new Date().toISOString();
                } else {
                    agent.issues.push({
                        type: issue,
                        count: 1,
                        firstSeen: new Date().toISOString(),
                        lastSeen: new Date().toISOString()
                    });
                }
            }
        }

        // Update timestamps
        agent.lastSeen = new Date().toISOString();

        // Save updated data
        this.saveReliabilityData();

        console.log(`Updated reliability data for ${agentType}: ${(score * 100).toFixed(1)}% score`);
    }

    /**
     * Get reliability metrics for a specific agent
     */
    getAgentReliability(agentType) {
        const agent = this.data.agents[agentType];
        if (!agent) {
            return {
                agentType,
                exists: false,
                message: 'No reliability data available'
            };
        }

        const successRate = agent.totalSessions > 0 ? agent.successfulSessions / agent.totalSessions : 0;
        const phantomRate = agent.totalSessions > 0 ? agent.phantomWorkCount / agent.totalSessions : 0;
        
        // Calculate trend direction
        const recentScores = agent.reliabilityTrend.slice(-5).map(t => t.score);
        const trendDirection = this.calculateTrend(recentScores);

        // Determine reliability level
        const reliabilityLevel = this.determineReliabilityLevel(agent.averageScore, successRate, phantomRate);

        return {
            agentType,
            exists: true,
            metrics: {
                totalSessions: agent.totalSessions,
                successRate: successRate,
                phantomRate: phantomRate,
                averageScore: agent.averageScore,
                lastScore: agent.lastScore,
                reliabilityLevel,
                trendDirection
            },
            issues: agent.issues,
            recentTrend: agent.reliabilityTrend.slice(-5),
            recommendation: this.getRecommendation(reliabilityLevel, phantomRate, trendDirection)
        };
    }

    /**
     * Calculate trend direction from scores
     */
    calculateTrend(scores) {
        if (scores.length < 2) return 'stable';
        
        const firstHalf = scores.slice(0, Math.floor(scores.length / 2));
        const secondHalf = scores.slice(Math.floor(scores.length / 2));
        
        const firstAvg = firstHalf.reduce((a, b) => a + b, 0) / firstHalf.length;
        const secondAvg = secondHalf.reduce((a, b) => a + b, 0) / secondHalf.length;
        
        const difference = secondAvg - firstAvg;
        
        if (difference > 0.1) return 'improving';
        if (difference < -0.1) return 'declining';
        return 'stable';
    }

    /**
     * Determine reliability level
     */
    determineReliabilityLevel(averageScore, successRate, phantomRate) {
        if (phantomRate > 0.2) return 'unreliable';
        if (averageScore >= 0.8 && successRate >= 0.8) return 'excellent';
        if (averageScore >= 0.6 && successRate >= 0.6) return 'good';
        if (averageScore >= 0.4 && successRate >= 0.4) return 'fair';
        return 'poor';
    }

    /**
     * Get deployment recommendation
     */
    getRecommendation(level, phantomRate, trend) {
        if (phantomRate > 0.3) {
            return {
                action: 'AVOID',
                reason: 'High phantom work rate detected',
                color: 'red'
            };
        }

        switch (level) {
            case 'excellent':
                return {
                    action: 'DEPLOY',
                    reason: 'Highly reliable agent',
                    color: 'green'
                };
            case 'good':
                if (trend === 'improving') {
                    return {
                        action: 'DEPLOY',
                        reason: 'Good reliability with improving trend',
                        color: 'green'
                    };
                } else {
                    return {
                        action: 'DEPLOY_WITH_MONITORING',
                        reason: 'Good reliability but monitor closely',
                        color: 'yellow'
                    };
                }
            case 'fair':
                return {
                    action: 'DEPLOY_WITH_CAUTION',
                    reason: 'Fair reliability - verify all deliverables',
                    color: 'orange'
                };
            case 'poor':
            case 'unreliable':
                return {
                    action: 'AVOID',
                    reason: 'Poor reliability - do not deploy',
                    color: 'red'
                };
            default:
                return {
                    action: 'UNKNOWN',
                    reason: 'Insufficient data',
                    color: 'gray'
                };
        }
    }

    /**
     * Generate comprehensive reliability report
     */
    generateReport() {
        const agents = Object.keys(this.data.agents).sort();
        
        let report = `
🔍 AGENT RELIABILITY REPORT
===========================

Generated: ${new Date().toISOString()}
Total Sessions: ${this.data.summary.totalSessions}
Overall Success Rate: ${this.data.summary.totalSessions > 0 ? 
    (this.data.summary.successfulSessions / this.data.summary.totalSessions * 100).toFixed(1) : 0}%
Phantom Work Incidents: ${this.data.summary.phantomWorkDetected}

AGENT RELIABILITY BREAKDOWN:
`;

        for (const agentType of agents) {
            const reliability = this.getAgentReliability(agentType);
            const metrics = reliability.metrics;
            const rec = reliability.recommendation;
            
            const statusIcon = rec.color === 'green' ? '✅' : 
                              rec.color === 'yellow' ? '⚠️' : 
                              rec.color === 'orange' ? '🟡' : '❌';

            report += `
${statusIcon} ${agentType.toUpperCase()}
   Sessions: ${metrics.totalSessions}
   Success Rate: ${(metrics.successRate * 100).toFixed(1)}%
   Average Score: ${(metrics.averageScore * 100).toFixed(1)}%
   Phantom Rate: ${(metrics.phantomRate * 100).toFixed(1)}%
   Reliability: ${metrics.reliabilityLevel.toUpperCase()}
   Trend: ${metrics.trendDirection.toUpperCase()}
   Recommendation: ${rec.action} - ${rec.reason}
`;

            if (reliability.issues.length > 0) {
                report += `   Common Issues:\n`;
                reliability.issues.slice(0, 3).forEach(issue => {
                    report += `     - ${issue.type} (${issue.count}x)\n`;
                });
            }
        }

        report += `
DEPLOYMENT RECOMMENDATIONS:
`;

        const excellent = agents.filter(a => this.getAgentReliability(a).metrics.reliabilityLevel === 'excellent');
        const good = agents.filter(a => this.getAgentReliability(a).metrics.reliabilityLevel === 'good');
        const problematic = agents.filter(a => ['poor', 'unreliable'].includes(this.getAgentReliability(a).metrics.reliabilityLevel));

        if (excellent.length > 0) {
            report += `✅ SAFE TO DEPLOY: ${excellent.join(', ')}\n`;
        }
        if (good.length > 0) {
            report += `⚠️  DEPLOY WITH MONITORING: ${good.join(', ')}\n`;
        }
        if (problematic.length > 0) {
            report += `❌ AVOID DEPLOYMENT: ${problematic.join(', ')}\n`;
        }

        report += `
SUMMARY:
- Most Reliable: ${this.getMostReliableAgent()}
- Least Reliable: ${this.getLeastReliableAgent()}
- Trending Up: ${this.getImprovingAgents().join(', ') || 'None'}
- Trending Down: ${this.getDecliningAgents().join(', ') || 'None'}
`;

        return report;
    }

    /**
     * Get most reliable agent
     */
    getMostReliableAgent() {
        const agents = Object.keys(this.data.agents);
        if (agents.length === 0) return 'None';

        return agents.reduce((best, current) => {
            const bestMetrics = this.getAgentReliability(best).metrics;
            const currentMetrics = this.getAgentReliability(current).metrics;
            
            if (currentMetrics.averageScore > bestMetrics.averageScore) {
                return current;
            }
            return best;
        });
    }

    /**
     * Get least reliable agent
     */
    getLeastReliableAgent() {
        const agents = Object.keys(this.data.agents);
        if (agents.length === 0) return 'None';

        return agents.reduce((worst, current) => {
            const worstMetrics = this.getAgentReliability(worst).metrics;
            const currentMetrics = this.getAgentReliability(current).metrics;
            
            if (currentMetrics.averageScore < worstMetrics.averageScore) {
                return current;
            }
            return worst;
        });
    }

    /**
     * Get improving agents
     */
    getImprovingAgents() {
        return Object.keys(this.data.agents).filter(agentType => {
            const reliability = this.getAgentReliability(agentType);
            return reliability.metrics.trendDirection === 'improving';
        });
    }

    /**
     * Get declining agents
     */
    getDecliningAgents() {
        return Object.keys(this.data.agents).filter(agentType => {
            const reliability = this.getAgentReliability(agentType);
            return reliability.metrics.trendDirection === 'declining';
        });
    }

    /**
     * Check if agent should be deployed
     */
    shouldDeployAgent(agentType) {
        const reliability = this.getAgentReliability(agentType);
        
        if (!reliability.exists) {
            return {
                shouldDeploy: true,
                reason: 'No prior data - proceed with monitoring',
                confidence: 'low'
            };
        }

        const rec = reliability.recommendation;
        
        return {
            shouldDeploy: rec.action === 'DEPLOY' || rec.action === 'DEPLOY_WITH_MONITORING',
            reason: rec.reason,
            confidence: reliability.metrics.totalSessions >= 3 ? 'high' : 'medium',
            monitoring: rec.action.includes('MONITORING') || rec.action.includes('CAUTION'),
            metrics: reliability.metrics
        };
    }

    /**
     * Sync with all available sessions
     */
    syncWithAllSessions() {
        const sessions = AgentExecutionMonitor.getAllSessions(this.projectRoot);
        let updated = 0;

        for (const session of sessions) {
            if (session.verification) {
                try {
                    this.updateFromSession(session.sessionId);
                    updated++;
                } catch (err) {
                    console.warn(`Failed to update from session ${session.sessionId}:`, err.message);
                }
            }
        }

        console.log(`Synced reliability data with ${updated} verified sessions`);
        return updated;
    }
}

// CLI interface
if (require.main === module) {
    const tracker = new AgentReliabilityTracker();
    
    const command = process.argv[2];
    
    switch (command) {
        case 'report':
            console.log(tracker.generateReport());
            break;
            
        case 'agent':
            const agentType = process.argv[3];
            if (!agentType) {
                console.error('Usage: reliability-tracker.js agent <agent-type>');
                process.exit(1);
            }
            
            const reliability = tracker.getAgentReliability(agentType);
            if (reliability.exists) {
                console.log(`\n🔍 ${agentType.toUpperCase()} RELIABILITY:`);
                console.log(`Sessions: ${reliability.metrics.totalSessions}`);
                console.log(`Success Rate: ${(reliability.metrics.successRate * 100).toFixed(1)}%`);
                console.log(`Average Score: ${(reliability.metrics.averageScore * 100).toFixed(1)}%`);
                console.log(`Reliability Level: ${reliability.metrics.reliabilityLevel.toUpperCase()}`);
                console.log(`Recommendation: ${reliability.recommendation.action} - ${reliability.recommendation.reason}`);
            } else {
                console.log(`No reliability data for ${agentType}`);
            }
            break;
            
        case 'should-deploy':
            const deployAgentType = process.argv[3];
            if (!deployAgentType) {
                console.error('Usage: reliability-tracker.js should-deploy <agent-type>');
                process.exit(1);
            }
            
            const deploymentAdvice = tracker.shouldDeployAgent(deployAgentType);
            console.log(`\n🚀 DEPLOYMENT ADVICE FOR ${deployAgentType}:`);
            console.log(`Should Deploy: ${deploymentAdvice.shouldDeploy ? 'YES' : 'NO'}`);
            console.log(`Reason: ${deploymentAdvice.reason}`);
            console.log(`Confidence: ${deploymentAdvice.confidence.toUpperCase()}`);
            if (deploymentAdvice.monitoring) {
                console.log(`⚠️  Requires close monitoring`);
            }
            process.exit(deploymentAdvice.shouldDeploy ? 0 : 1);
            break;
            
        case 'sync':
            tracker.syncWithAllSessions();
            break;
            
        case 'update':
            const sessionId = process.argv[3];
            if (!sessionId) {
                console.error('Usage: reliability-tracker.js update <session-id>');
                process.exit(1);
            }
            tracker.updateFromSession(sessionId);
            break;
            
        default:
            console.log('Agent Reliability Tracker');
            console.log('');
            console.log('Usage:');
            console.log('  reliability-tracker.js report                    - Generate full reliability report');
            console.log('  reliability-tracker.js agent <type>             - Get specific agent reliability');
            console.log('  reliability-tracker.js should-deploy <type>     - Check if agent should be deployed');
            console.log('  reliability-tracker.js sync                     - Sync with all verified sessions');
            console.log('  reliability-tracker.js update <session-id>      - Update from specific session');
    }
}

module.exports = AgentReliabilityTracker;