#!/usr/bin/env node

// Initialize database schema
require('dotenv').config({ path: '.env.local' });

async function init() {
  const database = await import('../src/lib/database.ts');
  
  try {
    console.log('Initializing database schema...');
    await database.initializeSchema();
    console.log('Database schema initialized successfully!');
  } catch (error) {
    console.error('Failed to initialize schema:', error);
    process.exit(1);
  }
}

init();