// Only import jest-dom for DOM tests
if (typeof document !== 'undefined') {
  require('@testing-library/jest-dom')
}

const dotenv = require('dotenv')

// Load environment variables - .env.test first, then .env.local can override
dotenv.config({ path: '.env.test' })
dotenv.config({ path: '.env.local' })

// Polyfill for Next.js Request/Response in tests
const { TextEncoder, TextDecoder } = require('util')

global.TextEncoder = TextEncoder
global.TextDecoder = TextDecoder

// Polyfill fetch for Node environment (needed for Neon serverless driver)
if (typeof globalThis.fetch === 'undefined') {
  const fetch = require('node-fetch')
  globalThis.fetch = fetch
  globalThis.Headers = fetch.Headers
  globalThis.Request = fetch.Request
  globalThis.Response = fetch.Response
}

// Mock NextResponse.json for test environment
const mockNextResponse = {
  json: (data, init) => new Response(JSON.stringify(data), {
    ...init,
    headers: {
      'content-type': 'application/json',
      ...init?.headers,
    },
  }),
}

global.NextResponse = mockNextResponse

// Override environment for testing (keep test keys for safety)
// Use real NEON_DATABASE_URL from .env.local for database tests
process.env.GOOGLE_API_KEY = process.env.GOOGLE_API_KEY || 'test-key-google-gemini'
process.env.ELEVENLABS_API_KEY = process.env.ELEVENLABS_API_KEY || 'test-key-elevenlabs'
process.env.NEXTAUTH_SECRET = process.env.NEXTAUTH_SECRET || 'test-secret-minimum-32-characters-long'
// NEXTAUTH_URL is handled by Vercel automatically