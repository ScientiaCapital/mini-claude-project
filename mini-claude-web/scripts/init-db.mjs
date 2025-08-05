#!/usr/bin/env node

import dotenv from 'dotenv';
import { neon } from '@neondatabase/serverless';

// Load environment variables
dotenv.config({ path: '.env.local' });

const sql = neon(process.env.NEON_DATABASE_URL);

async function initializeSchema() {
  try {
    console.log('Initializing database schema...');
    
    // Create users table
    await sql`
      CREATE TABLE IF NOT EXISTS users (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        email VARCHAR(255) UNIQUE NOT NULL,
        name VARCHAR(255),
        created_at TIMESTAMPTZ DEFAULT NOW(),
        updated_at TIMESTAMPTZ DEFAULT NOW()
      )
    `;
    console.log('✓ Created users table');

    // Create conversations table
    await sql`
      CREATE TABLE IF NOT EXISTS conversations (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        user_id UUID REFERENCES users(id),
        title TEXT DEFAULT 'New Conversation',
        created_at TIMESTAMPTZ DEFAULT NOW(),
        updated_at TIMESTAMPTZ DEFAULT NOW()
      )
    `;
    console.log('✓ Created conversations table');

    // Create messages table
    await sql`
      CREATE TABLE IF NOT EXISTS messages (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        conversation_id UUID REFERENCES conversations(id) ON DELETE CASCADE,
        role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
        content TEXT NOT NULL,
        created_at TIMESTAMPTZ DEFAULT NOW()
      )
    `;
    console.log('✓ Created messages table');

    // Create indexes for better performance
    await sql`CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id)`;
    await sql`CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at)`;
    await sql`CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON conversations(user_id)`;
    console.log('✓ Created indexes');

    console.log('\nDatabase schema initialized successfully!');
  } catch (error) {
    console.error('Failed to initialize schema:', error);
    process.exit(1);
  }
}

initializeSchema();