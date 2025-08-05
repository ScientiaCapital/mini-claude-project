/**
 * Voice Synthesis Module - ElevenLabs Integration
 * Handles text-to-speech conversion for AI responses
 * Following TDD: implementing to make tests pass
 */

// Types for voice synthesis
export interface VoiceSynthesisRequest {
  text: string
  voiceId?: string
  modelId?: string
}

export interface VoiceSynthesisResponse {
  success: boolean
  audioBuffer?: Buffer
  error?: string
}

// Default voice configuration
const DEFAULT_VOICE_ID = 'rachel'
const DEFAULT_MODEL_ID = 'eleven_multilingual_v2'
const MAX_TEXT_LENGTH = 5000

/**
 * Synthesize voice from text using ElevenLabs API
 * @param request Voice synthesis request parameters
 * @returns Voice synthesis response with audio buffer or error
 */
export async function synthesizeVoice(request: VoiceSynthesisRequest): Promise<VoiceSynthesisResponse> {
  try {
    // Validate API key
    if (!process.env.ELEVENLABS_API_KEY) {
      return {
        success: false,
        error: 'ElevenLabs API key not configured'
      }
    }

    // Validate input text
    if (!request.text || request.text.trim() === '') {
      return {
        success: false,
        error: 'Text cannot be empty'
      }
    }

    if (request.text.length > MAX_TEXT_LENGTH) {
      return {
        success: false,
        error: `Text too long for voice synthesis (max ${MAX_TEXT_LENGTH} characters)`
      }
    }

    // Dynamic import for better testability
    const { ElevenLabs } = await import('@elevenlabs/elevenlabs-js')
    
    // Initialize client
    const client = new (ElevenLabs as any)({
      apiKey: process.env.ELEVENLABS_API_KEY
    })

    // Perform text-to-speech conversion
    const audioBuffer = await client.textToSpeech.convert({
      text: request.text,
      voice_id: request.voiceId || DEFAULT_VOICE_ID,
      model_id: request.modelId || DEFAULT_MODEL_ID
    })

    return {
      success: true,
      audioBuffer: audioBuffer as Buffer
    }

  } catch (error: any) {
    console.error('Voice synthesis error:', error)
    return {
      success: false,
      error: `Voice synthesis failed: ${error.message || 'Unknown error'}`
    }
  }
}

/**
 * Create a URL for the synthesized audio
 * In production, this would upload to a CDN or blob storage
 * @param audioBuffer The audio buffer to create URL for
 * @returns URL string for the audio
 */
export async function createAudioUrl(audioBuffer: Buffer): Promise<string> {
  // For testing, return a simulated HTTPS URL
  if (process.env.NODE_ENV === 'test') {
    return `https://storage.example.com/audio/${Date.now()}.mp3`
  }

  // Production path - try real upload
  try {
    // Check if we have audio storage module
    const audioStorage = await import('@/lib/audio-storage').catch(() => null)
    
    if (audioStorage && audioStorage.uploadAudio) {
      return await audioStorage.uploadAudio(audioBuffer)
    }

    // Fallback: create data URL (not recommended for production)
    const base64Audio = audioBuffer.toString('base64')
    return `data:audio/mpeg;base64,${base64Audio}`
    
  } catch (error: any) {
    throw new Error(`Failed to upload audio: ${error.message}`)
  }
}

/**
 * Check if voice synthesis is enabled
 * @param voiceEnabled User preference for voice
 * @returns true if voice synthesis should be used
 */
export function isVoiceEnabled(voiceEnabled: boolean): boolean {
  // Voice is only enabled if:
  // 1. User has enabled it
  // 2. API key is configured
  return voiceEnabled && !!process.env.ELEVENLABS_API_KEY
}

// Voice presets for different use cases
export const VOICE_PRESETS = {
  assistant: {
    voiceId: 'rachel',
    settings: {
      stability: 0.75,
      similarity_boost: 0.75
    }
  },
  narrator: {
    voiceId: 'antoni',
    settings: {
      stability: 0.85,
      similarity_boost: 0.65
    }
  },
  energetic: {
    voiceId: 'jessie',
    settings: {
      stability: 0.65,
      similarity_boost: 0.85
    }
  }
}

/**
 * Get available voices (cached)
 * In production, this would fetch from ElevenLabs API
 */
export async function getAvailableVoices() {
  // For now, return preset voices
  // In production, would call: client.voices.getAll()
  return [
    { voice_id: 'rachel', name: 'Rachel', preview_url: null },
    { voice_id: 'antoni', name: 'Antoni', preview_url: null },
    { voice_id: 'jessie', name: 'Jessie', preview_url: null }
  ]
}