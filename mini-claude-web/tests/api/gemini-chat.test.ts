/**
 * TDD Tests for Google Gemini API Integration
 * RED Phase: Write failing tests that define expected behavior
 * These tests will fail until we implement Google Gemini integration
 */
import { describe, test, expect, beforeEach, jest } from '@jest/globals'
import { NextRequest } from 'next/server'
import type { MockSqlFunction } from '../types/mocks'

describe('POST /api/chat with Google Gemini', () => {
  beforeEach(() => {
    jest.clearAllMocks()
    // Clear any existing environment variables
    delete process.env.ANTHROPIC_API_KEY
    process.env.GOOGLE_API_KEY = 'test_google_api_key'
  })

  test('should use Google Gemini API instead of Anthropic', async () => {
    // Mock Google Generative AI response
    const mockGeminiResponse = {
      response: {
        text: () => 'Hello from Gemini! How can I assist you today?'
      }
    }

    // Mock the Google Generative AI SDK
    const mockGenerateContent = jest.fn().mockResolvedValue(mockGeminiResponse)
    const mockGetGenerativeModel = jest.fn().mockReturnValue({
      generateContent: mockGenerateContent
    })

    jest.doMock('@google/generative-ai', () => ({
      GoogleGenerativeAI: jest.fn(() => ({
        getGenerativeModel: mockGetGenerativeModel
      }))
    }))

    // Mock database operations
    const mockSql = jest.fn() as MockSqlFunction
    mockSql
      .mockResolvedValueOnce([{ id: 'conv_123' }]) // Create conversation
      .mockResolvedValueOnce([]) // Get history (empty)
      .mockResolvedValueOnce([{ id: 'msg_user_123' }]) // Save user message
      .mockResolvedValueOnce([{ id: 'msg_assistant_123' }]) // Save assistant message
      .mockResolvedValueOnce([]) // Update conversation

    jest.doMock('@/lib/database', () => ({
      getNeonConnection: jest.fn().mockResolvedValue(mockSql)
    }))

    const { POST } = await import('@/app/api/chat/route')
    
    const request = new NextRequest('http://localhost:3000/api/chat', {
      method: 'POST',
      body: JSON.stringify({ 
        message: 'Hello Gemini',
        conversation_id: 'conv_123'
      }),
      headers: { 'content-type': 'application/json' },
    })

    const response = await POST(request)
    expect(response.status).toBe(200)
    
    const data = await response.json()
    expect(data.reply).toBe('Hello from Gemini! How can I assist you today?')
    expect(data.message_id).toBe('msg_assistant_123')
    expect(data.conversation_id).toBe('conv_123')
    
    // Verify Google Gemini API was called correctly
    expect(mockGetGenerativeModel).toHaveBeenCalledWith({ model: 'gemini-1.5-flash' })
    expect(mockGenerateContent).toHaveBeenCalledWith('Hello Gemini')
  })

  test('should handle conversation history with Gemini format', async () => {
    const mockGeminiResponse = {
      response: {
        text: () => 'I remember our conversation about greetings!'
      }
    }

    const mockGenerateContent = jest.fn().mockResolvedValue(mockGeminiResponse)
    const mockGetGenerativeModel = jest.fn().mockReturnValue({
      generateContent: mockGenerateContent
    })

    jest.doMock('@google/generative-ai', () => ({
      GoogleGenerativeAI: jest.fn(() => ({
        getGenerativeModel: mockGetGenerativeModel
      }))
    }))

    // Mock conversation history
    const mockHistory = [
      { role: 'user', content: 'Hello' },
      { role: 'assistant', content: 'Hi there!' }
    ]

    const mockSql = jest.fn() as MockSqlFunction
    mockSql
      .mockResolvedValueOnce(mockHistory) // Get conversation history
      .mockResolvedValueOnce([{ id: 'msg_user_123' }]) // Save user message
      .mockResolvedValueOnce([{ id: 'msg_assistant_123' }]) // Save assistant message
      .mockResolvedValueOnce([]) // Update conversation

    jest.doMock('@/lib/database', () => ({
      getNeonConnection: jest.fn().mockResolvedValue(mockSql)
    }))

    const { POST } = await import('@/app/api/chat/route')
    
    const request = new NextRequest('http://localhost:3000/api/chat', {
      method: 'POST',
      body: JSON.stringify({ 
        message: 'Do you remember what we talked about?',
        conversation_id: 'conv_123'
      }),
      headers: { 'content-type': 'application/json' },
    })

    const response = await POST(request)
    expect(response.status).toBe(200)
    
    // Verify Gemini was called with proper context format
    const expectedContext = "Previous conversation:\nUser: Hello\nAssistant: Hi there!\n\nCurrent message: Do you remember what we talked about?"
    expect(mockGenerateContent).toHaveBeenCalledWith(expectedContext)
  })

  test('should handle Google Gemini API errors gracefully', async () => {
    const mockError = new Error('Gemini API quota exceeded')
    const mockGenerateContent = jest.fn().mockRejectedValue(mockError)
    const mockGetGenerativeModel = jest.fn().mockReturnValue({
      generateContent: mockGenerateContent
    })

    jest.doMock('@google/generative-ai', () => ({
      GoogleGenerativeAI: jest.fn(() => ({
        getGenerativeModel: mockGetGenerativeModel
      }))
    }))

    const mockSql = jest.fn() as MockSqlFunction
    mockSql.mockResolvedValueOnce([]) // Get history (empty)

    jest.doMock('@/lib/database', () => ({
      getNeonConnection: jest.fn().mockResolvedValue(mockSql)
    }))

    const { POST } = await import('@/app/api/chat/route')
    
    const request = new NextRequest('http://localhost:3000/api/chat', {
      method: 'POST',
      body: JSON.stringify({ 
        message: 'Hello',
        conversation_id: 'conv_123'
      }),
      headers: { 'content-type': 'application/json' },
    })

    const response = await POST(request)
    expect(response.status).toBe(500)
    
    const data = await response.json()
    expect(data.error).toBe('Failed to generate response')
    expect(data.details).toBe('Gemini API quota exceeded')
  })

  test('should use GOOGLE_API_KEY environment variable', async () => {
    const mockGoogleGenerativeAI = jest.fn()
    const mockGetGenerativeModel = jest.fn().mockReturnValue({
      generateContent: jest.fn().mockResolvedValue({
        response: { text: () => 'Test response' }
      })
    })
    
    mockGoogleGenerativeAI.mockReturnValue({
      getGenerativeModel: mockGetGenerativeModel
    })

    jest.doMock('@google/generative-ai', () => ({
      GoogleGenerativeAI: mockGoogleGenerativeAI
    }))

    const mockSql = jest.fn() as MockSqlFunction
    mockSql.mockResolvedValueOnce([]) // Get history
      .mockResolvedValueOnce([{ id: 'msg_user' }]) // Save user message
      .mockResolvedValueOnce([{ id: 'msg_assistant' }]) // Save assistant message
      .mockResolvedValueOnce([]) // Update conversation

    jest.doMock('@/lib/database', () => ({
      getNeonConnection: jest.fn().mockResolvedValue(mockSql)
    }))

    const { POST } = await import('@/app/api/chat/route')
    
    const request = new NextRequest('http://localhost:3000/api/chat', {
      method: 'POST',
      body: JSON.stringify({ 
        message: 'Test message',
        conversation_id: 'conv_123'
      }),
      headers: { 'content-type': 'application/json' },
    })

    await POST(request)
    
    // Verify Google AI was initialized with the correct API key
    expect(mockGoogleGenerativeAI).toHaveBeenCalledWith('test_google_api_key')
  })

  test('should use gemini-1.5-flash model by default', async () => {
    const mockGetGenerativeModel = jest.fn().mockReturnValue({
      generateContent: jest.fn().mockResolvedValue({
        response: { text: () => 'Model test response' }
      })
    })

    jest.doMock('@google/generative-ai', () => ({
      GoogleGenerativeAI: jest.fn(() => ({
        getGenerativeModel: mockGetGenerativeModel
      }))
    }))

    const mockSql = jest.fn() as MockSqlFunction
    mockSql.mockResolvedValueOnce([]) // Get history
      .mockResolvedValueOnce([{ id: 'msg_user' }]) // Save user message
      .mockResolvedValueOnce([{ id: 'msg_assistant' }]) // Save assistant message
      .mockResolvedValueOnce([]) // Update conversation

    jest.doMock('@/lib/database', () => ({
      getNeonConnection: jest.fn().mockResolvedValue(mockSql)
    }))

    const { POST } = await import('@/app/api/chat/route')
    
    const request = new NextRequest('http://localhost:3000/api/chat', {
      method: 'POST',
      body: JSON.stringify({ 
        message: 'What model are you?',
        conversation_id: 'conv_123'
      }),
      headers: { 'content-type': 'application/json' },
    })

    await POST(request)
    
    // Verify the correct model was requested
    expect(mockGetGenerativeModel).toHaveBeenCalledWith({ model: 'gemini-1.5-flash' })
  })

  test('should maintain same API response format as before', async () => {
    const mockGeminiResponse = {
      response: {
        text: () => 'Consistent API response format test'
      }
    }

    const mockGenerateContent = jest.fn().mockResolvedValue(mockGeminiResponse)
    const mockGetGenerativeModel = jest.fn().mockReturnValue({
      generateContent: mockGenerateContent
    })

    jest.doMock('@google/generative-ai', () => ({
      GoogleGenerativeAI: jest.fn(() => ({
        getGenerativeModel: mockGetGenerativeModel
      }))
    }))

    const mockSql = jest.fn() as MockSqlFunction
    mockSql.mockResolvedValueOnce([]) // Get history
      .mockResolvedValueOnce([{ id: 'msg_user_456' }]) // Save user message
      .mockResolvedValueOnce([{ id: 'msg_assistant_789' }]) // Save assistant message
      .mockResolvedValueOnce([]) // Update conversation

    jest.doMock('@/lib/database', () => ({
      getNeonConnection: jest.fn().mockResolvedValue(mockSql)
    }))

    const { POST } = await import('@/app/api/chat/route')
    
    const request = new NextRequest('http://localhost:3000/api/chat', {
      method: 'POST',
      body: JSON.stringify({ 
        message: 'API format test',
        conversation_id: 'conv_456'
      }),
      headers: { 'content-type': 'application/json' },
    })

    const response = await POST(request)
    expect(response.status).toBe(200)
    
    const data = await response.json()
    
    // Verify the response maintains the exact same structure
    expect(data).toHaveProperty('reply')
    expect(data).toHaveProperty('message_id')
    expect(data).toHaveProperty('conversation_id')
    expect(data.reply).toBe('Consistent API response format test')
    expect(data.message_id).toBe('msg_assistant_789')
    expect(data.conversation_id).toBe('conv_456')
  })
})