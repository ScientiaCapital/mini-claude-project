/**
 * Database connection and utilities for Neon PostgreSQL
 * This file implements the functions tested in tests/database/connection.test.ts
 * Following TDD: implement the minimum to make tests pass
 */
import { neon } from '@neondatabase/serverless'

// Types for our database schema
export interface DatabaseSchema {
  exists: boolean
  columns: string[]
  columnTypes: Record<string, string>
  foreignKeys: string[]
  constraints: Record<string, any>
}

// Cache for database connection
let cachedConnection: any = null

/**
 * Get Neon database connection
 * @param customUrl Optional custom database URL (for testing)
 * @returns Neon query function
 */
export async function getNeonConnection(customUrl?: string) {
  // Use cached connection if available
  if (cachedConnection && !customUrl) {
    return cachedConnection
  }

  const databaseUrl = customUrl || process.env.NEON_DATABASE_URL

  if (!databaseUrl) {
    throw new Error('NEON_DATABASE_URL is not defined')
  }

  // Validate URL format
  try {
    new URL(databaseUrl)
  } catch (error) {
    throw new Error('Invalid database URL')
  }

  if (!databaseUrl.startsWith('postgresql://')) {
    throw new Error('Invalid database URL')
  }

  try {
    const sql = neon(databaseUrl)
    
    // Cache the connection
    if (!customUrl) {
      cachedConnection = sql
    }
    
    return sql
  } catch (error: any) {
    console.error('Database connection creation failed:', error)
    throw new Error(`Failed to connect to database: ${error.message}`)
  }
}

/**
 * Test database connection with a simple query
 * @returns true if connection is successful
 */
export async function testConnection(): Promise<boolean> {
  try {
    const sql = await getNeonConnection()
    const result = await sql`SELECT 1 as test`
    return true
  } catch (error) {
    console.error('Database connection test failed:', error)
    return false
  }
}

/**
 * Get table schema information
 * @param tableName Name of the table to inspect
 * @returns Schema information for the table
 */
export async function getTableSchema(tableName: string): Promise<DatabaseSchema> {
  const sql = await getNeonConnection()
  
  try {
    // Check if table exists
    const tableExists = await sql`
      SELECT EXISTS (
        SELECT FROM information_schema.tables 
        WHERE table_name = ${tableName}
      ) as exists
    `
    
    if (!tableExists[0]?.exists) {
      return {
        exists: false,
        columns: [],
        columnTypes: {},
        foreignKeys: [],
        constraints: {}
      }
    }

    // Get column information
    const columns = await sql`
      SELECT column_name, data_type, is_nullable, column_default
      FROM information_schema.columns
      WHERE table_name = ${tableName}
      ORDER BY ordinal_position
    `
    
    // Get foreign key constraints
    const foreignKeys = await sql`
      SELECT 
        kcu.column_name,
        ccu.table_name AS foreign_table_name,
        ccu.column_name AS foreign_column_name
      FROM information_schema.table_constraints AS tc
      INNER JOIN information_schema.key_column_usage AS kcu
        ON tc.constraint_name = kcu.constraint_name
      INNER JOIN information_schema.constraint_column_usage AS ccu
        ON ccu.constraint_name = tc.constraint_name
      WHERE tc.constraint_type = 'FOREIGN KEY'
        AND tc.table_name = ${tableName}
    `

    // Get unique constraints
    const uniqueConstraints = await sql`
      SELECT column_name
      FROM information_schema.table_constraints tc
      INNER JOIN information_schema.key_column_usage kcu
        ON tc.constraint_name = kcu.constraint_name
      WHERE tc.constraint_type = 'UNIQUE'
        AND tc.table_name = ${tableName}
    `

    // Build schema object
    const schema: DatabaseSchema = {
      exists: true,
      columns: columns.map((col: any) => col.column_name),
      columnTypes: Object.fromEntries(
        columns.map((col: any) => [col.column_name, col.data_type])
      ),
      foreignKeys: foreignKeys.map((fk: any) => fk.column_name),
      constraints: {
        ...Object.fromEntries(
          foreignKeys.map((fk: any) => [
            fk.column_name,
            { references: `${fk.foreign_table_name}.${fk.foreign_column_name}` }
          ])
        ),
        ...Object.fromEntries(
          uniqueConstraints.map((uc: any) => [uc.column_name, { unique: true }])
        )
      }
    }

    return schema
  } catch (error) {
    console.error(`Failed to get schema for table ${tableName}:`, error)
    throw error
  }
}

/**
 * Initialize database schema (create tables if they don't exist)
 * This will be called during setup to ensure our schema exists
 */
export async function initializeSchema() {
  const sql = await getNeonConnection()

  try {
    // Create users table
    await sql`
      CREATE TABLE IF NOT EXISTS users (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        email VARCHAR(255) UNIQUE NOT NULL,
        name VARCHAR(255),
        created_at TIMESTAMPTZ DEFAULT NOW(),
        updated_at TIMESTAMPTZ DEFAULT NOW()
      )
    `

    // Create conversations table
    await sql`
      CREATE TABLE IF NOT EXISTS conversations (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        user_id UUID REFERENCES users(id),
        title TEXT DEFAULT 'New Conversation',
        created_at TIMESTAMPTZ DEFAULT NOW(),
        updated_at TIMESTAMPTZ DEFAULT NOW()
      )
    `

    // Create messages table
    await sql`
      CREATE TABLE IF NOT EXISTS messages (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        conversation_id UUID REFERENCES conversations(id) ON DELETE CASCADE,
        role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
        content TEXT NOT NULL,
        created_at TIMESTAMPTZ DEFAULT NOW()
      )
    `

    // Create indexes for better performance
    await sql`CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id)`
    await sql`CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at)`
    await sql`CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON conversations(user_id)`

    console.log('Database schema initialized successfully')
    return true
  } catch (error) {
    console.error('Failed to initialize database schema:', error)
    throw error
  }
}