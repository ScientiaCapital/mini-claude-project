/**
 * Verify Agent API Module
 * Provides programmatic access to agent verification functions
 */

const AgentVerifier = require('./verify-agent');
const fs = require('fs');
const path = require('path');

class VerifyAgentAPI {
  constructor() {
    this.verifier = new AgentVerifier();
  }

  /**
   * Verify work based on claimed actions vs actual tool calls
   * @param {Object} workData - Object containing claimedActions and actualToolCalls
   * @returns {Object} Verification result
   */
  async verifyWork(workData) {
    const { claimedActions = [], actualToolCalls = [] } = workData;
    
    // Analyze phantom work patterns
    const phantomWork = [];
    const verifiedActions = [];
    
    // Check each claimed action
    for (const claim of claimedActions) {
      let isPhantom = false;
      
      // Check if claim involves file creation
      if (claim.toLowerCase().includes('create') || 
          claim.toLowerCase().includes('file') ||
          claim.toLowerCase().includes('wrote')) {
        
        // Look for corresponding Write tool call
        const hasWriteTool = actualToolCalls.some(call => 
          call.tool === 'Write' || call.tool === 'MultiEdit' || call.tool === 'Edit'
        );
        
        if (!hasWriteTool) {
          isPhantom = true;
          phantomWork.push({
            claimed: claim,
            reason: 'No Write tool call found for file creation claim',
            severity: 'CRITICAL'
          });
        }
      }
      
      // Check if claim involves reading/analyzing
      else if (claim.toLowerCase().includes('read') || 
               claim.toLowerCase().includes('analyz') ||
               claim.toLowerCase().includes('review')) {
        
        // Look for corresponding Read tool call
        const hasReadTool = actualToolCalls.some(call => 
          call.tool === 'Read' || call.tool === 'Grep' || call.tool === 'LS'
        );
        
        if (!hasReadTool) {
          isPhantom = true;
          phantomWork.push({
            claimed: claim,
            reason: 'No Read tool call found for analysis claim',
            severity: 'WARNING'
          });
        }
      }
      
      if (!isPhantom) {
        verifiedActions.push(claim);
      }
    }
    
    // Check for tool calls without claims (good practice)
    const unclaimedTools = actualToolCalls.filter(call => {
      const toolMentioned = claimedActions.some(claim => 
        claim.toLowerCase().includes(call.tool.toLowerCase()) ||
        (call.tool === 'Write' && claim.toLowerCase().includes('create')) ||
        (call.tool === 'Read' && claim.toLowerCase().includes('read'))
      );
      return !toolMentioned;
    });
    
    return {
      verified: phantomWork.length === 0,
      phantomWork,
      verifiedActions,
      unclaimedTools,
      score: claimedActions.length > 0 ? 
        (verifiedActions.length / claimedActions.length) * 100 : 100,
      summary: {
        totalClaims: claimedActions.length,
        verifiedClaims: verifiedActions.length,
        phantomClaims: phantomWork.length,
        toolCalls: actualToolCalls.length
      }
    };
  }

  /**
   * Verify session work
   */
  async verifySession(sessionId, options = {}) {
    return this.verifier.verifySession(sessionId, options);
  }

  /**
   * List all sessions
   */
  listSessions() {
    return this.verifier.listSessions();
  }
}

// Export singleton instance
const api = new VerifyAgentAPI();

module.exports = {
  verifyWork: (workData) => api.verifyWork(workData),
  verifySession: (sessionId, options) => api.verifySession(sessionId, options),
  listSessions: () => api.listSessions(),
  VerifyAgentAPI
};