/**
 * Health Check API Route - /api/health
 * Simple endpoint to verify the application is running
 * Used by deployment tests and monitoring
 */
import { NextRequest, NextResponse } from 'next/server'
import { testConnection } from '@/lib/database'

export async function GET(request: NextRequest) {
  try {
    // Test database connection
    const dbConnected = await testConnection()
    
    // Check environment variables
    const requiredEnvVars = [
      'NEON_DATABASE_URL',
      'GOOGLE_API_KEY',
      'NEXTAUTH_SECRET'
    ]
    
    const missingEnvVars = requiredEnvVars.filter(
      envVar => !process.env[envVar]
    )

    const status = {
      status: 'healthy',
      timestamp: new Date().toISOString(),
      database: dbConnected ? 'connected' : 'disconnected',
      environment: process.env.NODE_ENV || 'unknown',
      version: process.env.npm_package_version || '1.0.0',
      checks: {
        database: dbConnected,
        environment_variables: missingEnvVars.length === 0,
        api_keys: {
          google: !!process.env.GOOGLE_API_KEY,
          elevenlabs: !!process.env.ELEVENLABS_API_KEY,
        }
      }
    }

    // Return 503 if critical checks fail
    if (!dbConnected || missingEnvVars.length > 0) {
      return NextResponse.json(
        { ...status, status: 'unhealthy', missing_env_vars: missingEnvVars },
        { status: 503 }
      )
    }

    return NextResponse.json(status, { status: 200 })
  } catch (error: any) {
    return NextResponse.json(
      {
        status: 'error',
        timestamp: new Date().toISOString(),
        error: error.message,
        database: 'error',
        environment: process.env.NODE_ENV || 'unknown'
      },
      { status: 500 }
    )
  }
}