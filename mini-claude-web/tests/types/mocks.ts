/**
 * Mock types for test files
 */
import { jest } from '@jest/globals'

// Database mock types  
export type MockSqlFunction = jest.MockedFunction<any>

// API response mock types
export interface MockGeminiResponse {
  response: {
    text: jest.MockedFunction<() => string>
  }
}

// Generic mock types
export type MockFunction<T = any> = jest.MockedFunction<(...args: any[]) => T>
export type MockAsyncFunction<T = any> = jest.MockedFunction<(...args: any[]) => Promise<T>>