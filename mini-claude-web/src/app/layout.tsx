import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'Mini-Claude AI Assistant',
  description: 'Educational AI chatbot built with Next.js, Anthropic Claude, and Neon database',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <div className="min-h-screen bg-gray-50">
          <header className="bg-white shadow-sm border-b">
            <div className="max-w-4xl mx-auto px-4 py-4">
              <h1 className="text-2xl font-bold text-gray-900">
                🤖 Mini-Claude AI Assistant
              </h1>
              <p className="text-sm text-gray-600 mt-1">
                Educational AI chatbot - Built with TDD principles
              </p>
            </div>
          </header>
          <main className="max-w-4xl mx-auto px-4 py-6">
            {children}
          </main>
          <footer className="mt-auto py-6 text-center text-sm text-gray-500">
            <p>Built following Test-Driven Development | Powered by Anthropic Claude & Neon</p>
          </footer>
        </div>
      </body>
    </html>
  )
}