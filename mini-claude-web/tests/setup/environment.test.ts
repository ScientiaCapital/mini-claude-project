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
      'GOOGLE_API_KEY',
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
    
    // NEON uses pooler endpoints with format: postgresql://user:pass@endpoint/db?params
    // Handle both standard port format and NEON pooler format (no explicit port)
    expect(dbUrl).toMatch(/postgresql:\/\/[^:]+:[^@]+@[^?/]+(\/[^?]+)?(\?.*)?$/)
  })

  test('API keys have correct format', () => {
    const googleKey = process.env.GOOGLE_API_KEY
    expect(googleKey).toBeDefined()
    // Google API keys can contain alphanumeric, hyphens, and underscores
    expect(googleKey!).toMatch(/^[a-zA-Z0-9-_]+$/)
    expect(googleKey!.length).toBeGreaterThan(10)
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
      expect(process.env.GOOGLE_API_KEY).not.toBe('test-key-google-gemini')
    } else {
      // Test passes in non-development environments
      expect(true).toBe(true)
    }
  })

  test('should use test values in test environment', () => {
    if (process.env.NODE_ENV === 'test') {
      // Match the actual test values set in jest.setup.js
      expect(process.env.GOOGLE_API_KEY).toBe('test-key-google-gemini')
      expect(process.env.ELEVENLABS_API_KEY).toBe('test-key-elevenlabs')
      // In test environment, we use the real NEON database for TDD integration tests
      expect(process.env.NEON_DATABASE_URL).toContain('neon.tech')
    } else {
      // Test passes in non-test environments
      expect(true).toBe(true)
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
    expect(url.hostname.length).toBeGreaterThan(0)
    // NEON pooler endpoints may not have explicit port, so check if port exists or is empty
    if (url.port) {
      expect(parseInt(url.port)).toBeGreaterThan(0)
    }
    expect(url.pathname).toMatch(/^\/\w+$/) // Database name
    expect(url.username).toBeDefined()
    expect(url.username.length).toBeGreaterThan(0)
    expect(url.password).toBeDefined()
    expect(url.password.length).toBeGreaterThan(0)
  })

  test('should validate API key formats match expected patterns', () => {
    const keys = {
      GOOGLE_API_KEY: /^[a-zA-Z0-9-_]+$/,
      ELEVENLABS_API_KEY: /^[a-zA-Z0-9-_]+$/,
    }
    
    Object.entries(keys).forEach((entry) => {
      const [envVar, pattern] = entry
      const value = process.env[envVar]
      expect(value).toBeDefined()
      expect(value!).toMatch(pattern)
      expect(value!.length).toBeGreaterThan(5)
    })
  })

  test('should validate NEON database URL query parameters', () => {
    const dbUrl = process.env.NEON_DATABASE_URL
    expect(dbUrl).toBeDefined()
    
    // NEON URLs should contain SSL mode or other query parameters
    if (dbUrl!.includes('?')) {
      const url = new URL(dbUrl!)
      expect(url.searchParams).toBeDefined()
      // Common NEON parameters: sslmode, channel_binding
      const hasValidParams = url.searchParams.has('sslmode') || 
                           url.searchParams.has('channel_binding') ||
                           url.searchParams.size === 0
      expect(hasValidParams).toBe(true)
    } else {
      // URL without query params is also valid
      expect(true).toBe(true)
    }
  })
})