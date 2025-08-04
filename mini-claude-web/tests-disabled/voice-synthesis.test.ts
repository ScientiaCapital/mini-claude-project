/**
 * TDD Tests for Voice Synthesis Utility Functions
 * Tests for the voice synthesis module that handles ElevenLabs integration
 */
import { describe, test, expect, beforeEach, jest } from '@jest/globals'
import type { MockSqlFunction } from '../types/mocks'

// Mock ElevenLabs SDK
jest.mock('@elevenlabs/elevenlabs-js', () => ({
  ElevenLabs: jest.fn()
}))

describe('Voice Synthesis Utility', () => {
  beforeEach(() => {
    jest.clearAllMocks()
    process.env.ELEVENLABS_API_KEY = 'test-api-key'
  })

  test('synthesizeVoice should convert text to audio successfully', async () => {
    // This will be imported from the implementation file
    const { synthesizeVoice } = await import('@/lib/voice-synthesis')
    
    const mockAudioBuffer = Buffer.from('mock-audio-data')
    const mockConvert = jest.fn().mockResolvedValue(mockAudioBuffer) as jest.Mock
    
    jest.doMock('@elevenlabs/elevenlabs-js', () => ({
      ElevenLabs: jest.fn(() => ({
        textToSpeech: {
          convert: mockConvert
        }
      }))
    }))

    const result = await synthesizeVoice({
      text: 'Hello, this is a test',
      voiceId: 'rachel'
    })

    expect(result.success).toBe(true)
    expect(result.audioBuffer).toBe(mockAudioBuffer)
    expect(result.error).toBeUndefined()
    
    expect(mockConvert).toHaveBeenCalledWith({
      text: 'Hello, this is a test',
      voice_id: 'rachel',
      model_id: 'eleven_multilingual_v2'
    })
  })

  test('synthesizeVoice should use default voice when not specified', async () => {
    const { synthesizeVoice } = await import('@/lib/voice-synthesis')
    
    const mockConvert = jest.fn().mockResolvedValue(Buffer.from('audio')) as jest.Mock
    
    jest.doMock('@elevenlabs/elevenlabs-js', () => ({
      ElevenLabs: jest.fn(() => ({
        textToSpeech: {
          convert: mockConvert
        }
      }))
    }))

    await synthesizeVoice({
      text: 'Test default voice'
    })

    expect(mockConvert).toHaveBeenCalledWith({
      text: 'Test default voice',
      voice_id: 'rachel', // Default voice
      model_id: 'eleven_multilingual_v2'
    })
  })

  test('synthesizeVoice should handle API errors gracefully', async () => {
    const { synthesizeVoice } = await import('@/lib/voice-synthesis')
    
    const mockConvert = jest.fn().mockRejectedValue(new Error('API rate limit exceeded')) as jest.Mock
    
    jest.doMock('@elevenlabs/elevenlabs-js', () => ({
      ElevenLabs: jest.fn(() => ({
        textToSpeech: {
          convert: mockConvert
        }
      }))
    }))

    const result = await synthesizeVoice({
      text: 'This will fail',
      voiceId: 'rachel'
    })

    expect(result.success).toBe(false)
    expect(result.audioBuffer).toBeUndefined()
    expect(result.error).toBe('Voice synthesis failed: API rate limit exceeded')
  })

  test('synthesizeVoice should return error when API key is missing', async () => {
    delete process.env.ELEVENLABS_API_KEY
    
    const { synthesizeVoice } = await import('@/lib/voice-synthesis')
    
    const result = await synthesizeVoice({
      text: 'No API key',
      voiceId: 'rachel'
    })

    expect(result.success).toBe(false)
    expect(result.error).toBe('ElevenLabs API key not configured')
  })

  test('synthesizeVoice should validate input text', async () => {
    const { synthesizeVoice } = await import('@/lib/voice-synthesis')
    
    // Test empty text
    const result1 = await synthesizeVoice({
      text: '',
      voiceId: 'rachel'
    })

    expect(result1.success).toBe(false)
    expect(result1.error).toBe('Text cannot be empty')

    // Test text that's too long (over 5000 chars)
    const longText = 'a'.repeat(5001)
    const result2 = await synthesizeVoice({
      text: longText,
      voiceId: 'rachel'
    })

    expect(result2.success).toBe(false)
    expect(result2.error).toBe('Text too long for voice synthesis (max 5000 characters)')
  })

  test('createAudioUrl should generate valid URL from audio buffer', async () => {
    const { createAudioUrl } = await import('@/lib/voice-synthesis')
    
    const audioBuffer = Buffer.from('test-audio-data')
    const url = await createAudioUrl(audioBuffer)

    expect(url).toMatch(/^https:\/\//)
    expect(url).toContain('audio')
  })

  test('createAudioUrl should handle storage errors', async () => {
    const { createAudioUrl } = await import('@/lib/voice-synthesis')
    
    // Mock storage failure
    jest.doMock('@/lib/audio-storage', () => ({
      uploadAudio: jest.fn().mockRejectedValue(new Error('Storage error')) as jest.Mock
    }))

    const audioBuffer = Buffer.from('test-audio-data')
    
    await expect(createAudioUrl(audioBuffer)).rejects.toThrow('Failed to upload audio: Storage error')
  })

  test('isVoiceEnabled should check environment and parameters correctly', async () => {
    const { isVoiceEnabled } = await import('@/lib/voice-synthesis')
    
    // With API key and voice enabled
    process.env.ELEVENLABS_API_KEY = 'test-key'
    expect(isVoiceEnabled(true)).toBe(true)
    expect(isVoiceEnabled(false)).toBe(false)
    
    // Without API key
    delete process.env.ELEVENLABS_API_KEY
    expect(isVoiceEnabled(true)).toBe(false)
    expect(isVoiceEnabled(false)).toBe(false)
  })
})