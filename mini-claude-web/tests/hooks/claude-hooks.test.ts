/**
 * TDD Tests for Claude Code Hook System
 * RED: Write failing tests first
 * GREEN: Implement to pass tests
 * REFACTOR: Improve while keeping tests green
 */

import { exec } from 'child_process'
import { promisify } from 'util'
import fs from 'fs/promises'
import path from 'path'

const execAsync = promisify(exec)

describe('Claude Code Hook System', () => {
  const settingsPath = path.join(process.cwd(), '.claude', 'settings.json')
  const hooksDir = path.join(process.cwd(), '.claude', 'hooks')

  beforeEach(async () => {
    // Ensure clean state
    await fs.mkdir(path.dirname(settingsPath), { recursive: true })
  })

  describe('Settings Configuration', () => {
    test('should have valid Claude Code settings.json', async () => {
      const settings = await fs.readFile(settingsPath, 'utf-8')
      const config = JSON.parse(settings)
      
      expect(config).toHaveProperty('hooks')
      expect(config.hooks).toHaveProperty('PreToolUse')
      expect(config.hooks).toHaveProperty('PostToolUse')
    })

    test('should configure Task tool hooks for agents', async () => {
      const settings = await fs.readFile(settingsPath, 'utf-8')
      const config = JSON.parse(settings)
      
      const taskHooks = config.hooks.PreToolUse?.find((h: any) => h.matcher === 'Task')
      expect(taskHooks).toBeDefined()
      expect(taskHooks.hooks).toContainEqual({
        type: 'command',
        command: expect.stringContaining('pre-task-context.sh')
      })
    })
  })

  describe('Pre-Task Context Hook', () => {
    test('should detect agent type from Task parameters', async () => {
      const scriptPath = path.join(hooksDir, 'pre-task-context.sh')
      const scriptExists = await fs.access(scriptPath).then(() => true).catch(() => false)
      expect(scriptExists).toBe(true)

      // Test agent type detection
      const mockInput = JSON.stringify({
        tool: 'Task',
        params: { subagent_type: 'neon-database-architect' }
      })
      
      const { stdout } = await execAsync(
        `echo '${mockInput}' | ${scriptPath}`,
        { cwd: process.cwd() }
      )
      
      expect(stdout).toContain('Loading context for: neon-database-architect')
    })

    test('should load agent-specific context', async () => {
      const agentTypes = [
        'neon-database-architect',
        'vercel-deployment-specialist',
        'security-auditor-expert',
        'api-integration-specialist',
        'nextjs-performance-optimizer'
      ]

      for (const agentType of agentTypes) {
        const mockInput = JSON.stringify({
          tool: 'Task',
          params: { subagent_type: agentType }
        })
        
        const { stdout } = await execAsync(
          `echo '${mockInput}' | ${path.join(hooksDir, 'pre-task-context.sh')}`,
          { cwd: process.cwd() }
        )
        
        expect(stdout).toContain(`Loading context for: ${agentType}`)
      }
    })
  })

  describe('Post-Task Knowledge Update Hook', () => {
    test('should save agent-specific knowledge', async () => {
      const scriptPath = path.join(hooksDir, 'post-agent-update.sh')
      const scriptExists = await fs.access(scriptPath).then(() => true).catch(() => false)
      expect(scriptExists).toBe(true)

      const mockInput = JSON.stringify({
        tool: 'Task',
        params: { subagent_type: 'project-docs-curator' },
        output: 'Documentation updated successfully'
      })
      
      const { stdout } = await execAsync(
        `echo '${mockInput}' | ${scriptPath}`,
        { cwd: process.cwd() }
      )
      
      expect(stdout).toContain('Saving knowledge for: project-docs-curator')
    })

    test('should create agent-specific knowledge directories', async () => {
      const agentKnowledgeDir = path.join(
        process.cwd(),
        '.claude',
        'knowledge',
        'agents',
        'neon-database-architect'
      )
      
      const dirExists = await fs.access(agentKnowledgeDir).then(() => true).catch(() => false)
      expect(dirExists).toBe(true)
    })
  })

  describe('Agent-Specific Validation', () => {
    test('should validate deployment health for vercel-deployment-specialist', async () => {
      const scriptPath = path.join(hooksDir, 'validate-agent-work.sh')
      const mockInput = JSON.stringify({
        tool: 'Task',
        params: { subagent_type: 'vercel-deployment-specialist' },
        output: 'Deployment completed'
      })
      
      const { stdout } = await execAsync(
        `echo '${mockInput}' | ${scriptPath}`,
        { cwd: process.cwd() }
      )
      
      expect(stdout).toContain('Validating deployment health')
    })

    test('should validate database connections for neon-database-architect', async () => {
      const mockInput = JSON.stringify({
        tool: 'Task',
        params: { subagent_type: 'neon-database-architect' },
        output: 'Schema updated'
      })
      
      const { stdout } = await execAsync(
        `echo '${mockInput}' | ${path.join(hooksDir, 'validate-agent-work.sh')}`,
        { cwd: process.cwd() }
      )
      
      expect(stdout).toContain('Validating database connections')
    })
  })
})