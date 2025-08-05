/**
 * Test Helper for Mock Setup
 * Provides utilities for setting up mocks correctly in tests
 */

import { jest } from '@jest/globals'

export function setupGoogleAIMock(responseText: string = 'Test response from Gemini') {
  const mockGenerateContent = jest.fn().mockResolvedValue({
    response: {
      text: jest.fn().mockReturnValue(responseText)
    }
  })

  const mockGetGenerativeModel = jest.fn().mockReturnValue({
    generateContent: mockGenerateContent
  })

  const mockGoogleGenerativeAI = jest.fn().mockImplementation(() => ({
    getGenerativeModel: mockGetGenerativeModel
  }))

  // Reset the module and set up fresh mock
  jest.resetModules()
  jest.doMock('@google/generative-ai', () => ({
    GoogleGenerativeAI: mockGoogleGenerativeAI
  }))

  return {
    mockGoogleGenerativeAI,
    mockGetGenerativeModel,
    mockGenerateContent
  }
}

export function setupDatabaseMock(mockSqlResponses: any[]) {
  const mockSql = jest.fn()
  
  // Set up sequential responses
  mockSqlResponses.forEach(response => {
    mockSql.mockResolvedValueOnce(response)
  })

  jest.doMock('@/lib/database', () => ({
    getNeonConnection: jest.fn().mockResolvedValue(mockSql)
  }))

  return mockSql
}

export function setupVoiceSynthesisMock(shouldSucceed: boolean = true) {
  const mockSynthesizeVoice = jest.fn().mockResolvedValue(
    shouldSucceed 
      ? { success: true, audioBuffer: Buffer.from('mock-audio') }
      : { success: false, error: 'Voice synthesis failed' }
  )

  const mockCreateAudioUrl = jest.fn().mockResolvedValue('https://storage.example.com/audio/test.mp3')

  jest.doMock('@/lib/voice-synthesis', () => ({
    synthesizeVoice: mockSynthesizeVoice,
    createAudioUrl: mockCreateAudioUrl,
    isVoiceEnabled: jest.fn().mockReturnValue(shouldSucceed)
  }))

  return { mockSynthesizeVoice, mockCreateAudioUrl }
}