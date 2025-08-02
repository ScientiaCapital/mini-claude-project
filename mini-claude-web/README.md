# Mini-Claude Web Interface

A Next.js-based web interface for the Mini-Claude AI assistant, built following **Test-Driven Development** principles.

## 🏗️ Architecture

- **Frontend**: Next.js 14 with TypeScript and Tailwind CSS
- **Backend**: Next.js API routes
- **Database**: Neon PostgreSQL (serverless)
- **AI**: Anthropic Claude API
- **Deployment**: Vercel
- **Testing**: Jest with React Testing Library

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   npm install
   ```

2. **Set up environment variables:**
   ```bash
   cp .env.example .env.local
   # Edit .env.local with your actual API keys
   ```

3. **Run tests (TDD approach):**
   ```bash
   npm test
   ```

4. **Run development server:**
   ```bash
   npm run dev
   ```

5. **Open [http://localhost:3000](http://localhost:3000)**

## 🧪 Test-Driven Development

This project follows strict TDD principles:

```bash
# Run tests in watch mode
npm test

# Run tests with coverage
npm run test:coverage

# Run tests in CI mode
npm run test:ci

# Type checking
npm run type-check
```

### Test Structure

```
tests/
├── api/               # API route tests
├── components/        # React component tests
├── database/         # Database connection tests
├── setup/            # Environment and configuration tests
└── e2e/              # End-to-end tests
```

## 📊 Database Schema

The app uses a simple PostgreSQL schema:

```sql
-- Users table
CREATE TABLE users (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  email VARCHAR(255) UNIQUE NOT NULL,
  name VARCHAR(255),
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Conversations table
CREATE TABLE conversations (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES users(id),
  title TEXT DEFAULT 'New Conversation',
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Messages table
CREATE TABLE messages (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  conversation_id UUID REFERENCES conversations(id),
  role TEXT CHECK (role IN ('user', 'assistant', 'system')),
  content TEXT NOT NULL,
  created_at TIMESTAMPTZ DEFAULT NOW()
);
```

## 🔧 API Endpoints

- `POST /api/chat` - Send message to AI assistant
- `GET /api/health` - Health check for monitoring

## 🌍 Environment Variables

```env
# Required
NEON_DATABASE_URL=postgresql://...
ANTHROPIC_API_KEY=sk-ant-...
NEXTAUTH_SECRET=your-secret-here

# Optional
ELEVENLABS_API_KEY=...
GOOGLE_API_KEY=...
```

## 📦 Deployment

### Vercel (Recommended)

1. **Connect to Vercel:**
   ```bash
   vercel link
   ```

2. **Set environment variables:**
   ```bash
   vercel env add NEON_DATABASE_URL
   vercel env add ANTHROPIC_API_KEY
   vercel env add NEXTAUTH_SECRET
   ```

3. **Deploy:**
   ```bash
   vercel --prod
   ```

### Manual Deployment

```bash
npm run build
npm start
```

## 🔍 Development Workflow

1. **Write tests first** (Red phase)
2. **Implement minimal code** to pass tests (Green phase)
3. **Refactor** while keeping tests green (Refactor phase)

Example workflow:
```bash
# 1. Write a test
echo 'test("should do something", () => { ... })' >> tests/feature.test.ts

# 2. Run test (should fail)
npm test tests/feature.test.ts

# 3. Implement code to make test pass
# Edit src/...

# 4. Run test again (should pass)
npm test tests/feature.test.ts

# 5. Refactor if needed
npm test  # All tests should still pass
```

## 🐛 Troubleshooting

### Common Issues

**Database connection fails:**
```bash
# Check environment variables
npm run type-check
# Verify Neon database URL format
```

**Tests failing:**
```bash
# Clear cache and reinstall
rm -rf node_modules .next
npm install
npm test
```

**Build errors:**
```bash
# Check TypeScript errors
npm run type-check
# Check for missing dependencies
npm install
```

## 📈 Performance

- **Response Time**: < 200ms for API routes
- **First Contentful Paint**: < 1.5s
- **Time to Interactive**: < 2.5s
- **Lighthouse Score**: > 90

## 🧮 Test Coverage

Maintain > 80% test coverage:

```bash
npm run test:coverage
```

Coverage reports are generated in `coverage/` directory.

## 🤖 AI Integration

The app integrates with Anthropic's Claude API:

- **Model**: claude-3-sonnet-20240229
- **Max Tokens**: 1000
- **Context**: Conversation history (last 20 messages)
- **Error Handling**: Graceful fallbacks and user feedback

## 🔐 Security

- Environment variables for API keys
- CORS configuration for API routes
- Input validation with Zod
- Database query parameterization
- Error handling without information leakage

## 📚 Learning Resources

This project demonstrates:
- Test-Driven Development (TDD)
- Next.js 14 App Router
- TypeScript best practices
- Database integration
- API design
- Modern React patterns
- Deployment strategies

## 🤝 Contributing

1. Follow TDD principles
2. Write tests before implementation
3. Maintain > 80% test coverage
4. Use TypeScript strictly
5. Follow existing code patterns

## 📄 License

MIT License - Educational use encouraged!