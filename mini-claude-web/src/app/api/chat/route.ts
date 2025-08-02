/**
 * Chat API Route - /api/chat
 * Implements the chat endpoint with Google Gemini integration
 * Following TDD: implement the minimum to make tests pass
 */
import { NextRequest, NextResponse } from 'next/server'
import { GoogleGenerativeAI } from '@google/generative-ai'
import { getNeonConnection } from '@/lib/database'
import { z } from 'zod'

// Initialize Google Gemini client
const genAI = new GoogleGenerativeAI(process.env.GOOGLE_API_KEY!)
const model = genAI.getGenerativeModel({ model: 'gemini-1.5-flash' })

// Request validation schema
const chatRequestSchema = z.object({
  message: z.string().min(1, 'Message cannot be empty').max(5000, 'Message too long (max 5000 characters)'),
  conversation_id: z.string().uuid().optional(),
  voice_enabled: z.boolean().optional().default(false),
  voice_id: z.string().optional(),
})

// Response types
interface ChatResponse {
  reply: string
  message_id: string
  conversation_id: string
  audio_url?: string // URL to generated voice audio (future ElevenLabs integration)
}

interface ErrorResponse {
  error: string
  details?: string
}

export async function POST(request: NextRequest): Promise<NextResponse<ChatResponse | ErrorResponse>> {
  try {
    // Parse and validate request body
    const body = await request.json()
    
    // Validate required fields
    if (!body.message) {
      return NextResponse.json(
        { error: 'Message is required' },
        { status: 400 }
      )
    }

    const validatedData = chatRequestSchema.parse(body)
    const { message, conversation_id, voice_enabled, voice_id } = validatedData

    // Get database connection
    const sql = await getNeonConnection()

    // Get or create conversation
    let conversationId = conversation_id
    if (!conversationId) {
      // Create new conversation
      const newConversation = await sql`
        INSERT INTO conversations (title, created_at) 
        VALUES (${'New Conversation'}, NOW()) 
        RETURNING id
      `
      conversationId = newConversation[0].id
    }

    // Get conversation history for context
    const history = await sql`
      SELECT role, content 
      FROM messages 
      WHERE conversation_id = ${conversationId}
      ORDER BY created_at ASC
      LIMIT 20
    `

    // Build message context for Google Gemini
    let prompt = message
    if (history.length > 0) {
      const contextMessages = history.map((msg: any) => 
        `${msg.role === 'user' ? 'User' : 'Assistant'}: ${msg.content}`
      ).join('\n')
      prompt = `Previous conversation:\n${contextMessages}\n\nCurrent message: ${message}`
    }

    // Call Google Gemini API
    let geminiResponse
    try {
      geminiResponse = await model.generateContent(prompt)
    } catch (geminiError: any) {
      console.error('Gemini API error:', geminiError)
      return NextResponse.json(
        { 
          error: 'Failed to generate response',
          details: geminiError.message 
        },
        { status: 500 }
      )
    }

    const reply = geminiResponse.response.text()

    // Save user message to database
    const userMessage = await sql`
      INSERT INTO messages (conversation_id, role, content) 
      VALUES (${conversationId}, ${'user'}, ${message}) 
      RETURNING id
    `

    // Save assistant response to database
    const assistantMessage = await sql`
      INSERT INTO messages (conversation_id, role, content) 
      VALUES (${conversationId}, ${'assistant'}, ${reply}) 
      RETURNING id
    `

    // Update conversation timestamp
    await sql`
      UPDATE conversations 
      SET updated_at = NOW() 
      WHERE id = ${conversationId}
    `

    // Prepare response object
    const response: ChatResponse = {
      reply,
      message_id: assistantMessage[0].id,
      conversation_id: conversationId!
    }

    // TODO: ElevenLabs voice synthesis integration
    // This will be implemented in the next phase
    if (voice_enabled) {
      // Future implementation will:
      // 1. Call ElevenLabs API with the reply text
      // 2. Upload generated audio to storage (e.g., Vercel Blob)
      // 3. Add audio_url to response
      console.log('Voice synthesis requested but not yet implemented', { voice_id })
    }

    return NextResponse.json(response)

  } catch (error: any) {
    console.error('Chat API error:', error)
    
    // Handle validation errors
    if (error.name === 'ZodError') {
      const firstError = error.errors[0]
      return NextResponse.json(
        { error: firstError.message },
        { status: 400 }
      )
    }

    // Handle database errors
    if (error.message.includes('database')) {
      return NextResponse.json(
        { 
          error: 'Database error',
          details: error.message 
        },
        { status: 500 }
      )
    }

    // Generic error response
    return NextResponse.json(
      { 
        error: 'Internal server error',
        details: process.env.NODE_ENV === 'development' ? error.message : undefined
      },
      { status: 500 }
    )
  }
}

// Handle CORS preflight requests
export async function OPTIONS(request: NextRequest) {
  return new NextResponse(null, {
    status: 200,
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, Authorization',
    },
  })
}