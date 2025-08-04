/**
 * Reliability Tracker API
 * Provides programmatic access to reliability tracking functions
 */

const AgentReliabilityTracker = require('./reliability-tracker');

class ReliabilityTrackerAPI {
  constructor() {
    this.tracker = new AgentReliabilityTracker();
  }

  /**
   * Record a successful agent operation
   */
  recordSuccess(agentType, operationType) {
    // Create a mock session for recording success
    const sessionData = {
      sessionId: `mock_${agentType}_${Date.now()}`,
      agentType: agentType,
      verification: {
        overall: {
          status: 'PASS',
          score: 1.0,
          issues: []
        },
        phantomDetection: {
          detected: false
        }
      },
      endTime: new Date().toISOString()
    };
    
    // Update reliability data
    this.updateFromMockSession(sessionData);
  }

  /**
   * Record a failed agent operation
   */
  recordFailure(agentType, failureType) {
    // Create a mock session for recording failure
    const sessionData = {
      sessionId: `mock_${agentType}_${Date.now()}`,
      agentType: agentType,
      verification: {
        overall: {
          status: 'FAIL',
          score: 0.3,
          issues: [failureType]
        },
        phantomDetection: {
          detected: failureType === 'phantom_work'
        }
      },
      endTime: new Date().toISOString()
    };
    
    // Update reliability data
    this.updateFromMockSession(sessionData);
  }

  /**
   * Get agent reliability score
   */
  getAgentScore(agentType) {
    const reliability = this.tracker.getAgentReliability(agentType);
    
    if (!reliability.exists) {
      return {
        reliability: 100, // Default to 100 for new agents
        sessions: 0,
        level: 'unknown'
      };
    }
    
    return {
      reliability: reliability.metrics.averageScore * 100,
      sessions: reliability.metrics.totalSessions,
      level: reliability.metrics.reliabilityLevel,
      successRate: reliability.metrics.successRate * 100,
      phantomRate: reliability.metrics.phantomRate * 100
    };
  }

  /**
   * Update from mock session data
   */
  updateFromMockSession(sessionData) {
    // Initialize agent data if needed
    if (!this.tracker.data.agents[sessionData.agentType]) {
      this.tracker.data.agents[sessionData.agentType] = {
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

    const agent = this.tracker.data.agents[sessionData.agentType];
    const verification = sessionData.verification;
    
    // Update metrics
    agent.totalSessions++;
    this.tracker.data.summary.totalSessions++;

    if (verification.overall.status === 'PASS') {
      agent.successfulSessions++;
      this.tracker.data.summary.successfulSessions++;
    }

    if (verification.phantomDetection?.detected) {
      agent.phantomWorkCount++;
      this.tracker.data.summary.phantomWorkDetected++;
    }

    // Update scores
    const score = verification.overall.score;
    agent.lastScore = score;
    agent.averageScore = ((agent.averageScore * (agent.totalSessions - 1)) + score) / agent.totalSessions;

    // Update trend
    agent.reliabilityTrend.push({
      sessionId: sessionData.sessionId,
      score,
      status: verification.overall.status,
      timestamp: sessionData.endTime,
      phantomDetected: verification.phantomDetection?.detected || false
    });

    if (agent.reliabilityTrend.length > 10) {
      agent.reliabilityTrend = agent.reliabilityTrend.slice(-10);
    }

    // Track issues
    if (verification.overall.issues?.length > 0) {
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

    agent.lastSeen = new Date().toISOString();
    
    // Save data
    this.tracker.saveReliabilityData();
  }

  /**
   * Generate reliability report
   */
  generateReport() {
    return this.tracker.generateReport();
  }

  /**
   * Check if agent should be deployed
   */
  shouldDeployAgent(agentType) {
    return this.tracker.shouldDeployAgent(agentType);
  }
}

// Export singleton instance
const api = new ReliabilityTrackerAPI();

module.exports = {
  recordSuccess: (agentType, operationType) => api.recordSuccess(agentType, operationType),
  recordFailure: (agentType, failureType) => api.recordFailure(agentType, failureType),
  getAgentScore: (agentType) => api.getAgentScore(agentType),
  generateReport: () => api.generateReport(),
  shouldDeployAgent: (agentType) => api.shouldDeployAgent(agentType),
  ReliabilityTrackerAPI
};