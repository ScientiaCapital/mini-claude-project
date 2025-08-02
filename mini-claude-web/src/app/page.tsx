'use client'

import { useState, useEffect } from 'react'

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
}

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([])
  const [inputMessage, setInputMessage] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [conversationId, setConversationId] = useState<string | null>(null)

  const sendMessage = async () => {
    if (!inputMessage.trim() || isLoading) return

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: inputMessage,
      timestamp: new Date()
    }

    setMessages(prev => [...prev, userMessage])
    setInputMessage('')
    setIsLoading(true)

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: inputMessage,
          conversation_id: conversationId,
        }),
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()

      // Set conversation ID if this is the first message
      if (!conversationId) {
        setConversationId(data.conversation_id)
      }

      const assistantMessage: Message = {
        id: data.message_id,
        role: 'assistant',
        content: data.reply,
        timestamp: new Date()
      }

      setMessages(prev => [...prev, assistantMessage])
    } catch (error) {
      console.error('Failed to send message:', error)
      
      const errorMessage: Message = {
        id: Date.now().toString(),
        role: 'assistant',
        content: 'Sorry, I encountered an error. Please try again.',
        timestamp: new Date()
      }

      setMessages(prev => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  return (
    <div className="flex flex-col h-[calc(100vh-200px)]">
      {/* Chat Messages */}
      <div className="flex-1 overflow-y-auto mb-4 space-y-4">
        {messages.length === 0 ? (
          <div className="text-center text-gray-500 py-12">
            <p className="text-lg mb-2">👋 Welcome to Mini-Claude!</p>
            <p>Start a conversation by typing a message below.</p>
            <div className="mt-4 text-sm">
              <p>Built with:</p>
              <div className="flex justify-center space-x-4 mt-2">
                <span className="bg-blue-100 text-blue-800 px-2 py-1 rounded">Next.js</span>
                <span className="bg-green-100 text-green-800 px-2 py-1 rounded">Anthropic Claude</span>
                <span className="bg-purple-100 text-purple-800 px-2 py-1 rounded">Neon DB</span>
                <span className="bg-red-100 text-red-800 px-2 py-1 rounded">TDD</span>
              </div>
            </div>
          </div>
        ) : (
          messages.map((message) => (
            <div
              key={message.id}
              className={`chat-message ${message.role}`}
            >
              <div className="text-sm opacity-75 mb-1">
                {message.role === 'user' ? 'You' : 'Mini-Claude'}
              </div>
              <div className="whitespace-pre-wrap">{message.content}</div>
              <div className="text-xs opacity-50 mt-2">
                {message.timestamp.toLocaleTimeString()}
              </div>
            </div>
          ))
        )}
        
        {isLoading && (
          <div className="chat-message assistant">
            <div className="text-sm opacity-75 mb-1">Mini-Claude</div>
            <div className="flex items-center space-x-1">
              <span>Thinking</span>
              <div className="loading-animation"></div>
              <div className="loading-animation"></div>
              <div className="loading-animation"></div>
            </div>
          </div>
        )}
      </div>

      {/* Input Area */}
      <div className="border-t pt-4">
        <div className="flex space-x-4">
          <textarea
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Type your message... (Press Enter to send, Shift+Enter for new line)"
            className="flex-1 p-3 border border-gray-300 rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            rows={3}
            disabled={isLoading}
          />
          <button
            onClick={sendMessage}
            disabled={!inputMessage.trim() || isLoading}
            className="px-6 py-3 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
          >
            {isLoading ? (
              <div className="flex items-center space-x-2">
                <div className="loading-animation bg-white"></div>
                <span>Sending</span>
              </div>
            ) : (
              'Send'
            )}
          </button>
        </div>
        
        <div className="mt-2 text-xs text-gray-500">
          <p>
            💡 This is an educational AI assistant. Messages are saved to demonstrate database integration.
          </p>
        </div>
      </div>
    </div>
  )
}