/**
 * Voice Synthesis Utility Functions
 * Handles ElevenLabs integration for text-to-speech conversion
 */

interface VoiceSynthesisRequest {
  text: string
  voiceId?: string
}

interface VoiceSynthesisResponse {
  success: boolean
  audioBuffer?: Buffer
  error?: string
}

/**
 * Convert text to speech using ElevenLabs API
 */
export async function synthesizeVoice({
  text,
  voiceId = 'rachel'
}: VoiceSynthesisRequest): Promise<VoiceSynthesisResponse> {
  try {
    // Check for API key
    if (!process.env.ELEVENLABS_API_KEY) {
      return {
        success: false,
        error: 'ElevenLabs API key not configured'
      }
    }

    // Validate input text
    if (!text || text.trim().length === 0) {
      return {
        success: false,
        error: 'Text cannot be empty'
      }
    }

    if (text.length > 5000) {
      return {
        success: false,
        error: 'Text too long for voice synthesis (max 5000 characters)'
      }
    }

    // Dynamic import to avoid issues during testing
    const ElevenLabsModule = await import('@elevenlabs/elevenlabs-js')
    
    // Type assertion for the constructor
    const ElevenLabsClass = ElevenLabsModule.ElevenLabs as any
    const client = new ElevenLabsClass({
      apiKey: process.env.ELEVENLABS_API_KEY
    })

    const audioBuffer = await client.textToSpeech.convert({
      text,
      voice_id: voiceId,
      model_id: 'eleven_multilingual_v2'
    })

    return {
      success: true,
      audioBuffer: audioBuffer as Buffer
    }
  } catch (error) {
    return {
      success: false,
      error: `Voice synthesis failed: ${error instanceof Error ? error.message : 'Unknown error'}`
    }
  }
}

/**
 * Create a URL for audio buffer (placeholder for storage implementation)
 */
export async function createAudioUrl(audioBuffer: Buffer): Promise<string> {
  try {
    // This would integrate with Vercel Blob or similar storage
    // For now, return a placeholder URL
    const { uploadAudio } = await import('@/lib/audio-storage')
    return await uploadAudio(audioBuffer)
  } catch (error) {
    throw new Error(`Failed to upload audio: ${error instanceof Error ? error.message : 'Unknown error'}`)
  }
}

/**
 * Check if voice synthesis is enabled
 */
export function isVoiceEnabled(voiceEnabledParam: boolean): boolean {
  return !!process.env.ELEVENLABS_API_KEY && voiceEnabledParam
}