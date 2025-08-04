/**
 * TDD Environment Setup Tests
 * These tests ensure all required environment variables are properly configured
 * Following Red -> Green -> Refactor cycle
 */
import { describe, test, expect } from '@jest/globals'

describe('Environment Configuration', () => {
  test('required environment variables are defined', () => {
    const requiredEnvVars = [
      'NEON_DATABASE_URL',
      'ANTHROPIC_API_KEY',
      'NEXTAUTH_SECRET',
    ]
    
    requiredEnvVars.forEach(envVar => {
      expect(process.env[envVar]).toBeDefined()
      expect(process.env[envVar]).not.toBe('')
    })
  })

  test('database URL format is valid', () => {
    const dbUrl = process.env.NEON_DATABASE_URL
    expect(dbUrl).toMatch(/^postgresql:\/\//)
    
    // Should contain required components
    expect(dbUrl).toMatch(/postgresql:\/\/[^:]+:[^@]+@[^:]+:[0-9]+\/[^?]+/)
  })

  test('API keys have correct format', () => {
    const anthropicKey = process.env.ANTHROPIC_API_KEY
    expect(anthropicKey).toBeDefined()
    expect(anthropicKey!).toMatch(/^[a-zA-Z0-9-_]+$/)
    expect(anthropicKey!.length).toBeGreaterThan(10)
  })

  test('NextAuth secret is sufficiently complex', () => {
    const secret = process.env.NEXTAUTH_SECRET
    expect(secret).toBeDefined()
    expect(secret!.length).toBeGreaterThanOrEqual(32)
  })

  test('optional environment variables have defaults', () => {
    // Test that optional vars have sensible defaults
    const nodeEnv = process.env.NODE_ENV || 'development'
    expect(['development', 'test', 'production']).toContain(nodeEnv)
    
    const port = process.env.PORT || '3000'
    expect(parseInt(port)).toBeGreaterThan(0)
    expect(parseInt(port)).toBeLessThan(65536)
  })
})

describe('Development Environment', () => {
  test('should load .env.local in development', () => {
    if (process.env.NODE_ENV === 'development') {
      // In development, we should prefer .env.local values
      expect(process.env.ANTHROPIC_API_KEY).not.toBe('test-key-anthropic')
    }
  })

  test('should use test values in test environment', () => {
    if (process.env.NODE_ENV === 'test') {
      expect(process.env.ANTHROPIC_API_KEY).toBe('test-key-anthropic')
      expect(process.env.NEON_DATABASE_URL).toContain('localhost')
    }
  })
})

describe('Configuration Validation', () => {
  test('should validate database connection string components', () => {
    const dbUrl = process.env.NEON_DATABASE_URL
    expect(dbUrl).toBeDefined()
    const url = new URL(dbUrl!)
    
    expect(url.protocol).toBe('postgresql:')
    expect(url.hostname).toBeDefined()
    expect(url.port).toBeDefined()
    expect(url.pathname).toMatch(/^\/\w+$/) // Database name
    expect(url.username).toBeDefined()
    expect(url.password).toBeDefined()
  })

  test('should validate API key formats match expected patterns', () => {
    const keys = {
      ANTHROPIC_API_KEY: /^[a-zA-Z0-9-_]+$/,
    }
    
    Object.entries(keys).forEach((entry) => {
      const [envVar, pattern] = entry
      const value = process.env[envVar]
      expect(value).toBeDefined()
      expect(value!).toMatch(pattern)
    })
  })
})