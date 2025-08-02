# 🚀 Deployment Guide: Mini-Claude TDD Infrastructure

## Pre-Deployment Checklist (TDD Approach)

Before deploying, ensure all tests pass:

```bash
cd mini-claude-web

# 1. Install dependencies
npm install

# 2. Run all tests (should pass)
npm run test:ci

# 3. Type check
npm run type-check

# 4. Build check
npm run build
```

## 🗄️ Database Setup (Neon)

### 1. Create Neon Project

```bash
# Using Neon CLI (if available)
neon create-project mini-claude-db

# Or manually at: https://console.neon.tech/
```

### 2. Get Connection String

From Neon Console:
- Go to your project dashboard
- Copy the connection string (looks like: `postgresql://user:pass@host/db?sslmode=require`)

### 3. Initialize Database Schema

```bash
# Set environment variable temporarily
export NEON_DATABASE_URL="your_connection_string_here"

# Run schema initialization
node -e "
const { initializeSchema } = require('./dist/lib/database.js');
initializeSchema().then(() => console.log('✅ Schema initialized'));
"
```

## 🌐 Vercel Deployment

### 1. Install Vercel CLI

```bash
npm install -g vercel
```

### 2. Link Project

```bash
cd mini-claude-web
vercel link
# Follow prompts to create/link project
```

### 3. Set Environment Variables

```bash
# Required variables
vercel env add NEON_DATABASE_URL
# Paste your Neon connection string

vercel env add ANTHROPIC_API_KEY  
# Paste your Anthropic API key (get from: https://console.anthropic.com/)

vercel env add NEXTAUTH_SECRET
# Generate: openssl rand -base64 32
```

### 4. Deploy to Preview

```bash
# Test deployment
vercel

# This creates a preview URL for testing
```

### 5. Test Preview Deployment

```bash
# Test health endpoint
curl https://your-preview-url.vercel.app/api/health

# Expected response:
{
  "status": "healthy",
  "database": "connected",
  "environment": "production"
}
```

### 6. Deploy to Production

```bash
# Deploy to production domain
vercel --prod
```

## 🧪 Post-Deployment Testing

### Automated Tests

Create a post-deployment test script:

```bash
# tests/deployment/verify-deployment.js
const fetch = require('node-fetch');

async function testDeployment(baseUrl) {
  console.log(`🧪 Testing deployment at: ${baseUrl}`);
  
  // Test health endpoint
  const healthResponse = await fetch(`${baseUrl}/api/health`);
  const health = await healthResponse.json();
  
  if (health.status !== 'healthy') {
    throw new Error(`Health check failed: ${JSON.stringify(health)}`);
  }
  console.log('✅ Health check passed');
  
  // Test chat endpoint
  const chatResponse = await fetch(`${baseUrl}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message: 'Hello, deployment test!' })
  });
  
  if (!chatResponse.ok) {
    throw new Error(`Chat API failed: ${chatResponse.status}`);
  }
  
  const chatData = await chatResponse.json();
  if (!chatData.reply) {
    throw new Error('Chat API returned invalid response');
  }
  console.log('✅ Chat API test passed');
  
  console.log('🎉 All deployment tests passed!');
}

// Run tests
const deploymentUrl = process.argv[2] || 'http://localhost:3000';
testDeployment(deploymentUrl).catch(console.error);
```

### Manual Testing Checklist

After deployment, verify:

- [ ] **Homepage loads**: Visit your Vercel URL
- [ ] **Health check**: `GET /api/health` returns 200
- [ ] **Chat functionality**: Send a message through the UI
- [ ] **Database persistence**: Check that messages are saved
- [ ] **Error handling**: Test with invalid inputs
- [ ] **Mobile responsiveness**: Test on mobile devices

## 🔧 CI/CD Integration (GitHub Actions)

Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy Mini-Claude

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '18'
          cache: 'npm'
          cache-dependency-path: mini-claude-web/package-lock.json
      
      - name: Install dependencies
        run: |
          cd mini-claude-web
          npm ci
      
      - name: Run tests
        run: |
          cd mini-claude-web
          npm run test:ci
        env:
          NEON_DATABASE_URL: ${{ secrets.NEON_DATABASE_URL }}
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
          NEXTAUTH_SECRET: ${{ secrets.NEXTAUTH_SECRET }}
      
      - name: Type check
        run: |
          cd mini-claude-web
          npm run type-check
      
      - name: Build
        run: |
          cd mini-claude-web
          npm run build

  deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
      
      - name: Deploy to Vercel
        uses: amondnet/vercel-action@v25
        with:
          vercel-token: ${{ secrets.VERCEL_TOKEN }}
          vercel-org-id: ${{ secrets.VERCEL_ORG_ID }}
          vercel-project-id: ${{ secrets.VERCEL_PROJECT_ID }}
          working-directory: mini-claude-web
```

## 🐛 Troubleshooting

### Common Issues

**"Database connection failed"**
```bash
# Check connection string format
echo $NEON_DATABASE_URL
# Should start with postgresql://

# Test connection
node -e "
const { testConnection } = require('./dist/lib/database.js');
testConnection().then(result => console.log('Connection:', result));
"
```

**"Anthropic API error"**
```bash
# Verify API key
curl https://api.anthropic.com/v1/messages \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "content-type: application/json" \
  -d '{"model":"claude-3-sonnet-20240229","max_tokens":10,"messages":[{"role":"user","content":"test"}]}'
```

**"Build failed on Vercel"**
- Check Node.js version compatibility
- Verify all dependencies are in package.json
- Check TypeScript errors: `npm run type-check`

### Performance Monitoring

Add Vercel Analytics:

```typescript
// src/app/layout.tsx
import { Analytics } from '@vercel/analytics/react';

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body>
        {children}
        <Analytics />
      </body>
    </html>
  )
}
```

## 🎯 Success Criteria

Your deployment is successful when:

1. **All tests pass** in CI/CD pipeline
2. **Health endpoint** returns healthy status
3. **Chat functionality** works end-to-end
4. **Database** saves and retrieves messages
5. **Performance** meets targets (< 2s response time)

## 🔄 Rollback Strategy

If deployment fails:

```bash
# Rollback to previous deployment
vercel rollback

# Or redeploy specific commit
vercel --prod --meta gitCommitSha=abc123
```

## 📊 Monitoring

Set up monitoring for:
- API response times
- Database connection health
- Error rates
- User activity

Use Vercel's built-in monitoring or integrate with:
- Sentry for error tracking
- DataDog for performance monitoring
- Uptime Robot for availability checks

## 🎉 You're Done!

Your Mini-Claude AI assistant is now deployed with:
- ✅ **TDD-validated** code quality
- ✅ **Production-ready** infrastructure
- ✅ **Scalable** Neon database
- ✅ **Fast** Vercel hosting
- ✅ **Reliable** CI/CD pipeline

Visit your deployment URL and start chatting with your AI assistant!