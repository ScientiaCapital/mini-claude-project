#!/usr/bin/env node

/**
 * Checkpoint Manager for Safe Agent Rollbacks
 * 
 * Creates git snapshots before agent deployment and provides rollback capability
 * when agent work fails verification.
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

class CheckpointManager {
    constructor() {
        this.projectRoot = this.findProjectRoot();
        this.checkpointsFile = path.join(this.projectRoot, '.claude', 'guardrails', 'checkpoints.json');
        this.checkpoints = this.loadCheckpoints();
    }

    findProjectRoot() {
        let current = process.cwd();
        while (current !== '/' && !fs.existsSync(path.join(current, 'CLAUDE.md'))) {
            current = path.dirname(current);
        }
        return current;
    }

    /**
     * Load existing checkpoints
     */
    loadCheckpoints() {
        if (fs.existsSync(this.checkpointsFile)) {
            try {
                return JSON.parse(fs.readFileSync(this.checkpointsFile, 'utf8'));
            } catch (err) {
                console.warn('Failed to load checkpoints, starting fresh');
            }
        }

        return {
            checkpoints: [],
            lastCleanup: new Date().toISOString()
        };
    }

    /**
     * Save checkpoints data
     */
    saveCheckpoints() {
        const dir = path.dirname(this.checkpointsFile);
        if (!fs.existsSync(dir)) {
            fs.mkdirSync(dir, { recursive: true });
        }
        
        fs.writeFileSync(this.checkpointsFile, JSON.stringify(this.checkpoints, null, 2));
    }

    /**
     * Create a new checkpoint before agent deployment
     */
    createCheckpoint(agentType, description = '') {
        console.log(`📸 Creating checkpoint for ${agentType}...`);

        try {
            // Get current git state
            const gitHash = this.getCurrentGitHash();
            const gitStatus = this.getGitStatus();
            const hasUncommittedChanges = gitStatus.trim().length > 0;

            // Create stash if there are uncommitted changes
            let stashRef = null;
            if (hasUncommittedChanges) {
                console.log('💾 Stashing uncommitted changes...');
                stashRef = this.createGitStash(agentType);
            }

            // Create checkpoint record
            const checkpoint = {
                id: `${agentType}_${Date.now()}`,
                agentType,
                description,
                timestamp: new Date().toISOString(),
                gitHash,
                stashRef,
                hasUncommittedChanges,
                fileCount: this.getFileCount(),
                workingDirectory: this.captureWorkingDirectoryState()
            };

            // Save checkpoint
            this.checkpoints.checkpoints.push(checkpoint);
            this.saveCheckpoints();

            console.log(`✅ Checkpoint created: ${checkpoint.id}`);
            console.log(`   Git Hash: ${gitHash}`);
            console.log(`   Stashed: ${stashRef ? 'Yes' : 'No'}`);
            console.log(`   Files: ${checkpoint.fileCount}`);

            return checkpoint;

        } catch (err) {
            console.error(`❌ Failed to create checkpoint: ${err.message}`);
            throw err;
        }
    }

    /**
     * Rollback to a specific checkpoint
     */
    rollbackToCheckpoint(checkpointId) {
        console.log(`🔄 Rolling back to checkpoint: ${checkpointId}`);

        const checkpoint = this.checkpoints.checkpoints.find(cp => cp.id === checkpointId);
        if (!checkpoint) {
            throw new Error(`Checkpoint ${checkpointId} not found`);
        }

        try {
            // Warn about data loss
            console.log('⚠️  WARNING: This will discard all changes since the checkpoint');
            console.log(`   Checkpoint: ${checkpoint.timestamp}`);
            console.log(`   Agent: ${checkpoint.agentType}`);
            
            // Check if we need confirmation (in interactive mode)
            if (process.stdin.isTTY && !process.argv.includes('--force')) {
                const readline = require('readline');
                const rl = readline.createInterface({
                    input: process.stdin,
                    output: process.stdout
                });

                return new Promise((resolve, reject) => {
                    rl.question('Continue with rollback? (yes/no): ', (answer) => {
                        rl.close();
                        if (answer.toLowerCase() !== 'yes' && answer.toLowerCase() !== 'y') {
                            console.log('Rollback cancelled');
                            resolve(false);
                            return;
                        }
                        
                        this.performRollback(checkpoint)
                            .then(resolve)
                            .catch(reject);
                    });
                });
            } else {
                return this.performRollback(checkpoint);
            }

        } catch (err) {
            console.error(`❌ Rollback failed: ${err.message}`);
            throw err;
        }
    }

    /**
     * Perform the actual rollback
     */
    async performRollback(checkpoint) {
        console.log('🔄 Performing rollback...');

        try {
            // Reset to the git hash
            console.log(`📍 Resetting to git hash: ${checkpoint.gitHash}`);
            execSync(`git reset --hard ${checkpoint.gitHash}`, {
                cwd: this.projectRoot,
                stdio: 'pipe'
            });

            // Restore stash if it exists
            if (checkpoint.stashRef) {
                console.log(`💾 Restoring stash: ${checkpoint.stashRef}`);
                try {
                    execSync(`git stash apply ${checkpoint.stashRef}`, {
                        cwd: this.projectRoot,
                        stdio: 'pipe'
                    });
                } catch (err) {
                    console.warn(`⚠️  Could not apply stash: ${err.message}`);
                }
            }

            // Clean up untracked files
            console.log('🧹 Cleaning up untracked files...');
            execSync('git clean -fd', {
                cwd: this.projectRoot,
                stdio: 'pipe'
            });

            // Verify rollback
            const newHash = this.getCurrentGitHash();
            const success = newHash === checkpoint.gitHash;

            if (success) {
                console.log('✅ Rollback completed successfully');
                console.log(`   Current git hash: ${newHash}`);
                console.log(`   Files restored: ${this.getFileCount()}`);
                
                // Create rollback record
                this.recordRollback(checkpoint);
                
                return true;
            } else {
                console.error('❌ Rollback verification failed');
                console.error(`   Expected: ${checkpoint.gitHash}`);
                console.error(`   Actual: ${newHash}`);
                return false;
            }

        } catch (err) {
            console.error(`💥 Rollback operation failed: ${err.message}`);
            throw err;
        }
    }

    /**
     * Get current git hash
     */
    getCurrentGitHash() {
        try {
            return execSync('git rev-parse HEAD', {
                cwd: this.projectRoot,
                encoding: 'utf8'
            }).trim();
        } catch (err) {
            throw new Error(`Failed to get git hash: ${err.message}`);
        }
    }

    /**
     * Get git status
     */
    getGitStatus() {
        try {
            return execSync('git status --porcelain', {
                cwd: this.projectRoot,
                encoding: 'utf8'
            });
        } catch (err) {
            return '';
        }
    }

    /**
     * Create git stash
     */
    createGitStash(agentType) {
        try {
            const stashMessage = `Checkpoint stash for ${agentType} - ${new Date().toISOString()}`;
            execSync(`git stash push -m "${stashMessage}"`, {
                cwd: this.projectRoot,
                stdio: 'pipe'
            });

            // Get the stash reference
            const stashList = execSync('git stash list --oneline', {
                cwd: this.projectRoot,
                encoding: 'utf8'
            });

            const latestStash = stashList.split('\n')[0];
            const stashRef = latestStash ? latestStash.split(':')[0] : null;

            return stashRef;
        } catch (err) {
            console.warn(`Could not create stash: ${err.message}`);
            return null;
        }
    }

    /**
     * Get file count in project
     */
    getFileCount() {
        try {
            const output = execSync('find . -type f | wc -l', {
                cwd: this.projectRoot,
                encoding: 'utf8'
            });
            return parseInt(output.trim());
        } catch (err) {
            return 0;
        }
    }

    /**
     * Capture working directory state
     */
    captureWorkingDirectoryState() {
        try {
            // Get key directories and their file counts
            const keyDirs = ['src', 'tests', 'mini-claude-web/src', 'mini-claude-web/tests', '.claude'];
            const state = {};

            for (const dir of keyDirs) {
                const fullPath = path.join(this.projectRoot, dir);
                if (fs.existsSync(fullPath)) {
                    try {
                        const files = execSync(`find "${fullPath}" -type f | wc -l`, {
                            encoding: 'utf8'
                        });
                        state[dir] = parseInt(files.trim());
                    } catch (err) {
                        state[dir] = 0;
                    }
                }
            }

            return state;
        } catch (err) {
            return {};
        }
    }

    /**
     * Record rollback operation
     */
    recordRollback(checkpoint) {
        checkpoint.rolledBackAt = new Date().toISOString();
        checkpoint.rollbackSuccessful = true;
        this.saveCheckpoints();
    }

    /**
     * List all checkpoints
     */
    listCheckpoints() {
        return this.checkpoints.checkpoints.map(cp => ({
            id: cp.id,
            agentType: cp.agentType,
            description: cp.description,
            timestamp: cp.timestamp,
            gitHash: cp.gitHash.substring(0, 8),
            hasStash: !!cp.stashRef,
            rolledBack: !!cp.rolledBackAt,
            age: this.getAge(cp.timestamp)
        }));
    }

    /**
     * Get age of checkpoint
     */
    getAge(timestamp) {
        const now = new Date();
        const then = new Date(timestamp);
        const diffMs = now - then;
        
        const minutes = Math.floor(diffMs / 60000);
        const hours = Math.floor(minutes / 60);
        const days = Math.floor(hours / 24);
        
        if (days > 0) return `${days}d ago`;
        if (hours > 0) return `${hours}h ago`;
        return `${minutes}m ago`;
    }

    /**
     * Clean up old checkpoints
     */
    cleanupOldCheckpoints(maxAge = 7) {
        const cutoffDate = new Date();
        cutoffDate.setDate(cutoffDate.getDate() - maxAge);
        
        const initialCount = this.checkpoints.checkpoints.length;
        this.checkpoints.checkpoints = this.checkpoints.checkpoints.filter(cp => {
            const checkpointDate = new Date(cp.timestamp);
            return checkpointDate > cutoffDate;
        });
        
        const removedCount = initialCount - this.checkpoints.checkpoints.length;
        
        if (removedCount > 0) {
            this.checkpoints.lastCleanup = new Date().toISOString();
            this.saveCheckpoints();
            console.log(`🧹 Cleaned up ${removedCount} old checkpoints`);
        }
        
        return removedCount;
    }

    /**
     * Get the latest checkpoint for an agent
     */
    getLatestCheckpointForAgent(agentType) {
        const agentCheckpoints = this.checkpoints.checkpoints
            .filter(cp => cp.agentType === agentType)
            .sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
        
        return agentCheckpoints[0] || null;
    }

    /**
     * Get a specific checkpoint by session and ID
     */
    getCheckpoint(sessionId, checkpointId) {
        return this.checkpoints.checkpoints.find(cp => 
            cp.id === checkpointId || 
            (cp.sessionId === sessionId && cp.id === checkpointId)
        );
    }

    /**
     * List checkpoints for a specific session
     */
    listCheckpoints(sessionId) {
        return this.checkpoints.checkpoints.filter(cp => 
            cp.sessionId === sessionId || cp.agentType === sessionId
        );
    }

    /**
     * Rollback to latest checkpoint for specific agent
     */
    rollbackLatestForAgent(agentType) {
        const checkpoint = this.getLatestCheckpointForAgent(agentType);
        if (!checkpoint) {
            throw new Error(`No checkpoints found for agent: ${agentType}`);
        }
        
        console.log(`Rolling back to latest checkpoint for ${agentType}`);
        return this.rollbackToCheckpoint(checkpoint.id);
    }
}

// CLI interface
if (require.main === module) {
    const manager = new CheckpointManager();
    
    const command = process.argv[2];
    
    switch (command) {
        case 'create':
            const agentType = process.argv[3];
            const description = process.argv[4] || '';
            
            if (!agentType) {
                console.error('Usage: checkpoint-manager.js create <agent-type> [description]');
                process.exit(1);
            }
            
            try {
                const checkpoint = manager.createCheckpoint(agentType, description);
                console.log(`Checkpoint ID: ${checkpoint.id}`);
            } catch (err) {
                console.error('Failed to create checkpoint:', err.message);
                process.exit(1);
            }
            break;
            
        case 'rollback':
            const checkpointId = process.argv[3];
            
            if (!checkpointId) {
                console.error('Usage: checkpoint-manager.js rollback <checkpoint-id> [--force]');
                process.exit(1);
            }
            
            manager.rollbackToCheckpoint(checkpointId)
                .then(success => {
                    if (success) {
                        console.log('Rollback completed');
                        process.exit(0);
                    } else {
                        console.log('Rollback cancelled or failed');
                        process.exit(1);
                    }
                })
                .catch(err => {
                    console.error('Rollback failed:', err.message);
                    process.exit(1);
                });
            break;
            
        case 'rollback-agent':
            const targetAgent = process.argv[3];
            
            if (!targetAgent) {
                console.error('Usage: checkpoint-manager.js rollback-agent <agent-type> [--force]');
                process.exit(1);
            }
            
            manager.rollbackLatestForAgent(targetAgent)
                .then(success => {
                    process.exit(success ? 0 : 1);
                })
                .catch(err => {
                    console.error('Rollback failed:', err.message);
                    process.exit(1);
                });
            break;
            
        case 'list':
            const checkpoints = manager.listCheckpoints();
            if (checkpoints.length === 0) {
                console.log('No checkpoints found');
            } else {
                console.log('Available checkpoints:');
                console.log('');
                checkpoints.forEach(cp => {
                    const status = cp.rolledBack ? '🔄' : '📸';
                    const stash = cp.hasStash ? '💾' : '  ';
                    console.log(`${status} ${stash} ${cp.id}`);
                    console.log(`    Agent: ${cp.agentType}`);
                    console.log(`    Time: ${cp.timestamp} (${cp.age})`);
                    console.log(`    Git: ${cp.gitHash}`);
                    if (cp.description) {
                        console.log(`    Desc: ${cp.description}`);
                    }
                    console.log('');
                });
            }
            break;
            
        case 'cleanup':
            const maxAge = parseInt(process.argv[3]) || 7;
            const removed = manager.cleanupOldCheckpoints(maxAge);
            console.log(`Cleanup completed. Removed ${removed} checkpoints older than ${maxAge} days.`);
            break;
            
        default:
            console.log('Checkpoint Manager for Safe Agent Rollbacks');
            console.log('');
            console.log('Usage:');
            console.log('  checkpoint-manager.js create <agent-type> [description]  - Create checkpoint');
            console.log('  checkpoint-manager.js rollback <checkpoint-id>          - Rollback to checkpoint');
            console.log('  checkpoint-manager.js rollback-agent <agent-type>       - Rollback to latest checkpoint for agent');
            console.log('  checkpoint-manager.js list                              - List all checkpoints');
            console.log('  checkpoint-manager.js cleanup [days]                    - Remove old checkpoints (default: 7 days)');
            console.log('');
            console.log('Examples:');
            console.log('  checkpoint-manager.js create vercel-deployment-specialist "Before deployment work"');
            console.log('  checkpoint-manager.js rollback vercel-deployment-specialist_1641234567890');
            console.log('  checkpoint-manager.js rollback-agent vercel-deployment-specialist');
    }
}

module.exports = CheckpointManager;