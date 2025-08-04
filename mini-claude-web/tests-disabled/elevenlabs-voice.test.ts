/**
 * TDD Tests for ElevenLabs Voice Synthesis Integration
 * Following strict RED -> GREEN -> REFACTOR cycle
 * Tests must be written before implementation
 */
import { describe, test, expect, beforeEach, jest } from '@jest/globals'
import { NextRequest } from 'next/server'

// Mock ElevenLabs SDK
jest.mock('@elevenlabs/elevenlabs-js', () => ({
  ElevenLabs: jest.fn(),
  play: jest.fn()
}))

// Mock Next.js server components
jest.mock('next/server', () => ({
  NextRequest: class NextRequest extends Request {
    constructor(input, init) {
      super(input, init)
    }
  },
  NextResponse: {
    json: (data, init) => new Response(JSON.stringify(data), {
      status: init?.status || 200,
      headers: {
        'content-type': 'application/json',
        ...init?.headers,
      },
    }),
  },
}))

describe('ElevenLabs Voice Synthesis Integration', () => {
  beforeEach(() => {
    jest.clearAllMocks()
    // Set up environment variable
    process.env.ELEVENLABS_API_KEY = 'test-api-key'
  })

  test('should synthesize voice when voice_enabled is true', async () => {
    const { POST } = await import('@/app/api/chat/route')
    
    // Mock ElevenLabs SDK response
    const mockAudioStream = Buffer.from('mock-audio-data')
    const mockTextToSpeech = jest.fn().mockResolvedValue(mockAudioStream)
    
    jest.doMock('@elevenlabs/elevenlabs-js', () => ({
      ElevenLabs: jest.fn(() => ({
        textToSpeech: {
          convert: mockTextToSpeech
        }
      }))
    }))

    // Mock Google Gemini response
    const mockGeminiResponse = {
      response: {
        text: jest.fn().mockReturnValue('This is a test response for voice synthesis.')
      }
    }

    jest.doMock('@google/generative-ai', () => ({
      GoogleGenerativeAI: jest.fn(() => ({
        getGenerativeModel: jest.fn(() => ({
          generateContent: jest.fn().mockResolvedValue(mockGeminiResponse)
        }))
      }))
    }))

    // Mock database
    const mockSql = jest.fn()
      .mockResolvedValueOnce([{ id: '123e4567-e89b-12d3-a456-426614174000' }]) // conversation
      .mockResolvedValueOnce([]) // history
      .mockResolvedValueOnce([{ id: 'msg_user' }]) // user message
      .mockResolvedValueOnce([{ id: 'msg_assistant' }]) // assistant message
      .mockResolvedValueOnce([]) // update timestamp

    jest.doMock('@/lib/database', () => ({
      getNeonConnection: jest.fn().mockResolvedValue(mockSql)
    }))

    const request = new Request('http://localhost:3000/api/chat', {
      method: 'POST',
      body: JSON.stringify({ 
        message: 'Hello, please speak this response',
        conversation_id: '123e4567-e89b-12d3-a456-426614174000',
        voice_enabled: true,
        voice_id: 'rachel' // ElevenLabs voice ID
      }),
      headers: { 'content-type': 'application/json' },
    }) as NextRequest

    const response = await POST(request)
    expect(response.status).toBe(200)
    
    const data = await response.json()
    expect(data.reply).toBe('This is a test response for voice synthesis.')
    expect(data.audio_url).toBeDefined()
    expect(data.audio_url).toMatch(/^https:\/\//)
    
    // Verify ElevenLabs was called with correct parameters
    expect(mockTextToSpeech).toHaveBeenCalledWith({
      text: 'This is a test response for voice synthesis.',
      voice_id: 'rachel',
      model_id: 'eleven_multilingual_v2'
    })
  })

  test('should use default voice when voice_id not provided', async () => {
    const { POST } = await import('@/app/api/chat/route')
    
    const mockTextToSpeech = jest.fn().mockResolvedValue(Buffer.from('mock-audio'))
    
    jest.doMock('@elevenlabs/elevenlabs-js', () => ({
      ElevenLabs: jest.fn(() => ({
        textToSpeech: {
          convert: mockTextToSpeech
        }
      }))
    }))

    // Mock other dependencies
    jest.doMock('@google/generative-ai', () => ({
      GoogleGenerativeAI: jest.fn(() => ({
        getGenerativeModel: jest.fn(() => ({
          generateContent: jest.fn().mockResolvedValue({
            response: { text: jest.fn().mockReturnValue('Default voice test') }
          })
        }))
      }))
    }))

    const mockSql = jest.fn()
      .mockResolvedValueOnce([{ id: '123e4567-e89b-12d3-a456-426614174000' }])
      .mockResolvedValueOnce([])
      .mockResolvedValueOnce([{ id: 'msg_user' }])
      .mockResolvedValueOnce([{ id: 'msg_assistant' }])
      .mockResolvedValueOnce([])

    jest.doMock('@/lib/database', () => ({
      getNeonConnection: jest.fn().mockResolvedValue(mockSql)
    }))

    const request = new Request('http://localhost:3000/api/chat', {
      method: 'POST',
      body: JSON.stringify({ 
        message: 'Test default voice',
        conversation_id: '123e4567-e89b-12d3-a456-426614174000',
        voice_enabled: true
        // No voice_id provided
      }),
      headers: { 'content-type': 'application/json' },
    }) as NextRequest

    const response = await POST(request)
    expect(response.status).toBe(200)
    
    const data = await response.json()
    expect(data.audio_url).toBeDefined()
    
    // Should use default voice ID
    expect(mockTextToSpeech).toHaveBeenCalledWith({
      text: 'Default voice test',
      voice_id: 'rachel', // Default voice
      model_id: 'eleven_multilingual_v2'
    })
  })

  test('should handle ElevenLabs API errors gracefully', async () => {
    const { POST } = await import('@/app/api/chat/route')
    
    // Mock ElevenLabs error
    const mockTextToSpeech = jest.fn().mockRejectedValue(new Error('ElevenLabs API error: Rate limit exceeded'))
    
    jest.doMock('@elevenlabs/elevenlabs-js', () => ({
      ElevenLabs: jest.fn(() => ({
        textToSpeech: {
          convert: mockTextToSpeech
        }
      }))
    }))

    // Mock successful Gemini response
    jest.doMock('@google/generative-ai', () => ({
      GoogleGenerativeAI: jest.fn(() => ({
        getGenerativeModel: jest.fn(() => ({
          generateContent: jest.fn().mockResolvedValue({
            response: { text: jest.fn().mockReturnValue('Response that fails voice synthesis') }
          })
        }))
      }))
    }))

    const mockSql = jest.fn()
      .mockResolvedValueOnce([{ id: '123e4567-e89b-12d3-a456-426614174000' }])
      .mockResolvedValueOnce([])
      .mockResolvedValueOnce([{ id: 'msg_user' }])
      .mockResolvedValueOnce([{ id: 'msg_assistant' }])
      .mockResolvedValueOnce([])

    jest.doMock('@/lib/database', () => ({
      getNeonConnection: jest.fn().mockResolvedValue(mockSql)
    }))

    const request = new Request('http://localhost:3000/api/chat', {
      method: 'POST',
      body: JSON.stringify({ 
        message: 'Test voice error handling',
        conversation_id: '123e4567-e89b-12d3-a456-426614174000',
        voice_enabled: true,
        voice_id: 'rachel'
      }),
      headers: { 'content-type': 'application/json' },
    }) as NextRequest

    const response = await POST(request)
    expect(response.status).toBe(200) // Should still return 200
    
    const data = await response.json()
    expect(data.reply).toBe('Response that fails voice synthesis')
    expect(data.audio_url).toBeUndefined() // No audio URL due to error
    // Should not throw error, just omit audio_url
  })

  test('should not synthesize voice when voice_enabled is false', async () => {
    const { POST } = await import('@/app/api/chat/route')
    
    const mockTextToSpeech = jest.fn()
    
    jest.doMock('@elevenlabs/elevenlabs-js', () => ({
      ElevenLabs: jest.fn(() => ({
        textToSpeech: {
          convert: mockTextToSpeech
        }
      }))
    }))

    // Mock other dependencies
    jest.doMock('@google/generative-ai', () => ({
      GoogleGenerativeAI: jest.fn(() => ({
        getGenerativeModel: jest.fn(() => ({
          generateContent: jest.fn().mockResolvedValue({
            response: { text: jest.fn().mockReturnValue('No voice needed') }
          })
        }))
      }))
    }))

    const mockSql = jest.fn()
      .mockResolvedValueOnce([{ id: '123e4567-e89b-12d3-a456-426614174000' }])
      .mockResolvedValueOnce([])
      .mockResolvedValueOnce([{ id: 'msg_user' }])
      .mockResolvedValueOnce([{ id: 'msg_assistant' }])
      .mockResolvedValueOnce([])

    jest.doMock('@/lib/database', () => ({
      getNeonConnection: jest.fn().mockResolvedValue(mockSql)
    }))

    const request = new Request('http://localhost:3000/api/chat', {
      method: 'POST',
      body: JSON.stringify({ 
        message: 'Test without voice',
        conversation_id: '123e4567-e89b-12d3-a456-426614174000',
        voice_enabled: false
      }),
      headers: { 'content-type': 'application/json' },
    }) as NextRequest

    const response = await POST(request)
    expect(response.status).toBe(200)
    
    const data = await response.json()
    expect(data.reply).toBe('No voice needed')
    expect(data.audio_url).toBeUndefined()
    
    // ElevenLabs should not be called
    expect(mockTextToSpeech).not.toHaveBeenCalled()
  })

  test('should handle missing ELEVENLABS_API_KEY gracefully', async () => {
    // Remove API key
    delete process.env.ELEVENLABS_API_KEY
    
    const { POST } = await import('@/app/api/chat/route')
    
    // Mock other dependencies
    jest.doMock('@google/generative-ai', () => ({
      GoogleGenerativeAI: jest.fn(() => ({
        getGenerativeModel: jest.fn(() => ({
          generateContent: jest.fn().mockResolvedValue({
            response: { text: jest.fn().mockReturnValue('Response without voice due to missing API key') }
          })
        }))
      }))
    }))

    const mockSql = jest.fn()
      .mockResolvedValueOnce([{ id: '123e4567-e89b-12d3-a456-426614174000' }])
      .mockResolvedValueOnce([])
      .mockResolvedValueOnce([{ id: 'msg_user' }])
      .mockResolvedValueOnce([{ id: 'msg_assistant' }])
      .mockResolvedValueOnce([])

    jest.doMock('@/lib/database', () => ({
      getNeonConnection: jest.fn().mockResolvedValue(mockSql)
    }))

    const request = new Request('http://localhost:3000/api/chat', {
      method: 'POST',
      body: JSON.stringify({ 
        message: 'Test missing API key',
        conversation_id: '123e4567-e89b-12d3-a456-426614174000',
        voice_enabled: true,
        voice_id: 'rachel'
      }),
      headers: { 'content-type': 'application/json' },
    }) as NextRequest

    const response = await POST(request)
    expect(response.status).toBe(200)
    
    const data = await response.json()
    expect(data.reply).toBe('Response without voice due to missing API key')
    expect(data.audio_url).toBeUndefined() // No audio due to missing API key
  })

  test('should handle long text by chunking for voice synthesis', async () => {
    const { POST } = await import('@/app/api/chat/route')
    
    // Create a long response text
    const longText = 'This is a very long response. '.repeat(100) // ~3000 characters
    
    const mockTextToSpeech = jest.fn().mockResolvedValue(Buffer.from('mock-audio-long'))
    
    jest.doMock('@elevenlabs/elevenlabs-js', () => ({
      ElevenLabs: jest.fn(() => ({
        textToSpeech: {
          convert: mockTextToSpeech
        }
      }))
    }))

    jest.doMock('@google/generative-ai', () => ({
      GoogleGenerativeAI: jest.fn(() => ({
        getGenerativeModel: jest.fn(() => ({
          generateContent: jest.fn().mockResolvedValue({
            response: { text: jest.fn().mockReturnValue(longText) }
          })
        }))
      }))
    }))

    const mockSql = jest.fn()
      .mockResolvedValueOnce([{ id: '123e4567-e89b-12d3-a456-426614174000' }])
      .mockResolvedValueOnce([])
      .mockResolvedValueOnce([{ id: 'msg_user' }])
      .mockResolvedValueOnce([{ id: 'msg_assistant' }])
      .mockResolvedValueOnce([])

    jest.doMock('@/lib/database', () => ({
      getNeonConnection: jest.fn().mockResolvedValue(mockSql)
    }))

    const request = new Request('http://localhost:3000/api/chat', {
      method: 'POST',
      body: JSON.stringify({ 
        message: 'Generate a long response',
        conversation_id: '123e4567-e89b-12d3-a456-426614174000',
        voice_enabled: true,
        voice_id: 'rachel'
      }),
      headers: { 'content-type': 'application/json' },
    }) as NextRequest

    const response = await POST(request)
    expect(response.status).toBe(200)
    
    const data = await response.json()
    expect(data.reply).toBe(longText)
    expect(data.audio_url).toBeDefined()
    
    // Should handle long text appropriately
    expect(mockTextToSpeech).toHaveBeenCalled()
  })
})