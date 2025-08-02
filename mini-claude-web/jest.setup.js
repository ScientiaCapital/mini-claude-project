import '@testing-library/jest-dom'
import dotenv from 'dotenv'

// Load environment variables from .env.local for testing
dotenv.config({ path: '.env.local' })

// Polyfill for Next.js Request/Response in tests
import { TextEncoder, TextDecoder } from 'util'

global.TextEncoder = TextEncoder
global.TextDecoder = TextDecoder

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

// Polyfill fetch API for tests
import 'whatwg-fetch'

// Override environment for testing (keep test keys for safety)
process.env.GOOGLE_API_KEY = 'test-key-google-gemini'
process.env.ELEVENLABS_API_KEY = 'test-key-elevenlabs'
process.env.NEXTAUTH_SECRET = 'test-secret-minimum-32-characters-long'
process.env.NEXTAUTH_URL = 'http://localhost:3000'

// Mock external API calls in tests (but use real database)
global.fetch = jest.fn()

// Mock Google Generative AI SDK for testing (prevent real API calls)
jest.mock('@google/generative-ai', () => ({
  GoogleGenerativeAI: jest.fn(() => ({
    getGenerativeModel: jest.fn(() => ({
      generateContent: jest.fn().mockResolvedValue({
        response: {
          text: jest.fn().mockReturnValue('Test response from Gemini')
        }
      }),
    })),
  })),
}))