/**
 * TDD API Route Tests for Chat Endpoint with Google Gemini
 * These tests define the behavior of /api/chat using Google Gemini
 * Following Red -> Green -> Refactor cycle
 */
import { describe, test, expect, beforeEach, jest } from '@jest/globals'
import { NextRequest } from 'next/server'
import { GoogleGenerativeAI } from '@google/generative-ai'
import type { MockSqlFunction } from '../types/mocks'

// Mock Next.js server components for testing
jest.mock('next/server', () => ({
  NextRequest: class NextRequest extends Request {
    constructor(input: any, init?: any) {
      super(input, init)
    }
  },
  NextResponse: {
    json: (data: any, init?: any) => new Response(JSON.stringify(data), {
      status: init?.status || 200,
      headers: {
        'content-type': 'application/json',
        ...init?.headers,
      },
    }),
  },
}))

describe('POST /api/chat', () => {
  beforeEach(() => {
    jest.clearAllMocks()
  })

  test('should return 400 for missing message', async () => {
    const { POST } = await import('@/app/api/chat/route')
    
    const request = new Request('http://localhost:3000/api/chat', {
      method: 'POST',
      body: JSON.stringify({}),
      headers: { 'content-type': 'application/json' },
    }) as NextRequest

    const response = await POST(request)
    expect(response.status).toBe(400)
    
    const data = await response.json()
    expect(data.error).toBe('Message is required')
  })

  test('should return 400 for empty message', async () => {
    const { POST } = await import('@/app/api/chat/route')
    
    const request = new Request('http://localhost:3000/api/chat', {
      method: 'POST',
      body: JSON.stringify({ message: '' }),
      headers: { 'content-type': 'application/json' },
    }) as NextRequest

    const response = await POST(request)
    expect(response.status).toBe(400)
    
    const data = await response.json()
    expect(data.error).toBe('Message cannot be empty')
  })

  test('should call Google Gemini API with user message', async () => {
    const { POST } = await import('@/app/api/chat/route')
    
    // Mock Google Gemini response
    const mockGeminiResponse = {
      response: {
        text: jest.fn().mockReturnValue('Hello! How can I help you today?')
      }
    }

    // Mock the Google Gemini SDK
    const mockGenerateContent = jest.fn().mockResolvedValue(mockGeminiResponse)
    const mockGetGenerativeModel = jest.fn().mockReturnValue({
      generateContent: mockGenerateContent
    })
    
    jest.doMock('@google/generative-ai', () => ({
      GoogleGenerativeAI: jest.fn(() => ({
        getGenerativeModel: mockGetGenerativeModel
      }))
    }))

    // Mock database to avoid real DB calls
    const mockSql = jest.fn() as MockSqlFunction
    mockSql
      .mockResolvedValueOnce([{ id: '123e4567-e89b-12d3-a456-426614174000' }]) // Get conversation history
      .mockResolvedValueOnce([{ id: 'msg_user' }]) // Save user message
      .mockResolvedValueOnce([{ id: 'msg_assistant' }]) // Save assistant message
      .mockResolvedValueOnce([]) // Update conversation timestamp

    jest.doMock('@/lib/database', () => ({
      getNeonConnection: jest.fn().mockResolvedValue(mockSql)
    }))

    const request = new Request('http://localhost:3000/api/chat', {
      method: 'POST',
      body: JSON.stringify({ 
        message: 'Hello AI',
        conversation_id: '123e4567-e89b-12d3-a456-426614174000'
      }),
      headers: { 'content-type': 'application/json' },
    }) as NextRequest

    const response = await POST(request)
    expect(response.status).toBe(200)
    
    const data = await response.json()
    expect(data.reply).toBe('Hello! How can I help you today?')
    expect(data.message_id).toBeDefined()
    expect(data.conversation_id).toBe('123e4567-e89b-12d3-a456-426614174000')
    
    // Verify Gemini was called with correct prompt
    expect(mockGenerateContent).toHaveBeenCalledWith('Hello AI')
    expect(mockGetGenerativeModel).toHaveBeenCalledWith({ model: 'gemini-1.5-flash' })
  })

  test('should save message and response to database', async () => {
    const { POST } = await import('@/app/api/chat/route')
    
    // Mock database SQL template literal function
    const mockSql = jest.fn() as MockSqlFunction
    mockSql
      .mockResolvedValueOnce([]) // Get conversation history (empty)
      .mockResolvedValueOnce([{ id: 'msg_user' }]) // Save user message
      .mockResolvedValueOnce([{ id: 'msg_assistant' }]) // Save assistant message
      .mockResolvedValueOnce([]) // Update conversation timestamp

    jest.doMock('@/lib/database', () => ({
      getNeonConnection: jest.fn().mockResolvedValue(mockSql)
    }))

    // Mock Google Gemini
    const mockGeminiResponse = {
      response: { text: jest.fn().mockReturnValue('2+2 equals 4') }
    }
    jest.doMock('@google/generative-ai', () => ({
      GoogleGenerativeAI: jest.fn(() => ({
        getGenerativeModel: jest.fn(() => ({
          generateContent: jest.fn().mockResolvedValue(mockGeminiResponse)
        }))
      }))
    }))

    const request = new Request('http://localhost:3000/api/chat', {
      method: 'POST',
      body: JSON.stringify({ 
        message: 'What is 2+2?',
        conversation_id: '123e4567-e89b-12d3-a456-426614174000'
      }),
      headers: { 'content-type': 'application/json' },
    }) as NextRequest

    const response = await POST(request)
    expect(response.status).toBe(200)
    
    const data = await response.json()
    expect(data.reply).toBe('2+2 equals 4')
    expect(data.message_id).toBeDefined()
    
    // Verify database calls were made (using Neon SQL template literals)
    expect(mockSql).toHaveBeenCalledTimes(4) // history, user msg, assistant msg, update timestamp
  })

  test('should handle conversation history context', async () => {
    const { POST } = await import('@/app/api/chat/route')
    
    // Mock conversation history
    const mockHistory = [
      { role: 'user', content: 'Hello' },
      { role: 'assistant', content: 'Hi there!' },
      { role: 'user', content: 'What is your name?' }
    ]

    const mockSql = jest.fn() as MockSqlFunction
    mockSql
      .mockResolvedValueOnce(mockHistory) // Get conversation history
      .mockResolvedValueOnce([{ id: 'msg_user' }]) // Save user message
      .mockResolvedValueOnce([{ id: 'msg_assistant' }]) // Save assistant message
      .mockResolvedValueOnce([]) // Update conversation timestamp

    jest.doMock('@/lib/database', () => ({
      getNeonConnection: jest.fn().mockResolvedValue(mockSql)
    }))

    // Mock Google Gemini
    const mockGenerateContent = jest.fn().mockResolvedValue({
      response: { text: jest.fn().mockReturnValue('Yes, we discussed greetings and names.') }
    })

    jest.doMock('@google/generative-ai', () => ({
      GoogleGenerativeAI: jest.fn(() => ({
        getGenerativeModel: jest.fn(() => ({
          generateContent: mockGenerateContent
        }))
      }))
    }))

    const request = new Request('http://localhost:3000/api/chat', {
      method: 'POST',
      body: JSON.stringify({ 
        message: 'Can you remember what we talked about?',
        conversation_id: '123e4567-e89b-12d3-a456-426614174000'
      }),
      headers: { 'content-type': 'application/json' },
    }) as NextRequest

    const response = await POST(request)
    expect(response.status).toBe(200)
    
    // Verify context was included in Gemini call
    const expectedPrompt = `Previous conversation:
User: Hello
Assistant: Hi there!
User: What is your name?

Current message: Can you remember what we talked about?`
    
    expect(mockGenerateContent).toHaveBeenCalledWith(expectedPrompt)
  })

  test('should handle Google Gemini API errors gracefully', async () => {
    const { POST } = await import('@/app/api/chat/route')
    
    // Mock database
    const mockSql = jest.fn() as MockSqlFunction
    mockSql.mockResolvedValueOnce([]) // Get conversation history (empty)
    
    jest.doMock('@/lib/database', () => ({
      getNeonConnection: jest.fn().mockResolvedValue(mockSql)
    }))
    
    // Mock Gemini error
    const mockGenerateContent = jest.fn().mockRejectedValue(new Error('API quota exceeded'))
    jest.doMock('@google/generative-ai', () => ({
      GoogleGenerativeAI: jest.fn(() => ({
        getGenerativeModel: jest.fn(() => ({
          generateContent: mockGenerateContent
        }))
      }))
    }))

    const request = new Request('http://localhost:3000/api/chat', {
      method: 'POST',
      body: JSON.stringify({ 
        message: 'Hello',
        conversation_id: '123e4567-e89b-12d3-a456-426614174000'
      }),
      headers: { 'content-type': 'application/json' },
    }) as NextRequest

    const response = await POST(request)
    expect(response.status).toBe(500)
    
    const data = await response.json()
    expect(data.error).toBe('Failed to generate response')
    expect(data.details).toBe('API quota exceeded')
  })

  test('should validate message length limits', async () => {
    const { POST } = await import('@/app/api/chat/route')
    
    const longMessage = 'a'.repeat(5001) // Exceed 5000 character limit
    
    const request = new Request('http://localhost:3000/api/chat', {
      method: 'POST',
      body: JSON.stringify({ 
        message: longMessage,
        conversation_id: '123e4567-e89b-12d3-a456-426614174000'
      }),
      headers: { 'content-type': 'application/json' },
    }) as NextRequest

    const response = await POST(request)
    expect(response.status).toBe(400)
    
    const data = await response.json()
    expect(data.error).toBe('Message too long (max 5000 characters)')
  })

  test('should create new conversation if none provided', async () => {
    const { POST } = await import('@/app/api/chat/route')
    
    const mockSql = jest.fn() as MockSqlFunction
    mockSql
      .mockResolvedValueOnce([{ id: '456e7890-e89b-12d3-a456-426614174001' }]) // Create new conversation
      .mockResolvedValueOnce([]) // Get conversation history (empty for new conversation)
      .mockResolvedValueOnce([{ id: 'msg_user' }]) // Save user message
      .mockResolvedValueOnce([{ id: 'msg_assistant' }]) // Save assistant message
      .mockResolvedValueOnce([]) // Update conversation timestamp

    jest.doMock('@/lib/database', () => ({
      getNeonConnection: jest.fn().mockResolvedValue(mockSql)
    }))

    // Mock Google Gemini
    jest.doMock('@google/generative-ai', () => ({
      GoogleGenerativeAI: jest.fn(() => ({
        getGenerativeModel: jest.fn(() => ({
          generateContent: jest.fn().mockResolvedValue({
            response: { text: jest.fn().mockReturnValue('Welcome! How can I assist you?') }
          })
        }))
      }))
    }))

    const request = new Request('http://localhost:3000/api/chat', {
      method: 'POST',
      body: JSON.stringify({ 
        message: 'Start new conversation'
      }),
      headers: { 'content-type': 'application/json' },
    }) as NextRequest

    const response = await POST(request)
    expect(response.status).toBe(200)
    
    const data = await response.json()
    expect(data.conversation_id).toBe('456e7890-e89b-12d3-a456-426614174001')
    expect(data.reply).toBe('Welcome! How can I assist you?')
    
    // Verify database calls were made for new conversation
    expect(mockSql).toHaveBeenCalledTimes(5) // create conv, get history, save user msg, save assistant msg, update timestamp
  })
})