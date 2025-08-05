#!/bin/bash

echo "🔧 Fixing All Failing Tests..."
echo ""

# Count total tests before
BEFORE_COUNT=$(npm run test:ci -- --listTests 2>/dev/null | grep -c "test.ts$")
echo "📊 Total test files: $BEFORE_COUNT"
echo ""

# Run tests and check status
echo "🧪 Running initial test suite..."
npm run test:ci 2>&1 | tail -5
echo ""

# Check for tests-disabled directory
if [ -d "tests-disabled" ]; then
  echo "❌ ERROR: tests-disabled directory still exists!"
  echo "This is a violation of test integrity!"
  exit 1
fi

# Count actual test assertions
TOTAL_TESTS=$(npm run test:ci 2>&1 | grep "Tests:" | tail -1 | awk '{print $2}')
FAILED_TESTS=$(npm run test:ci 2>&1 | grep "Tests:" | tail -1 | awk '{print $4}')
PASSED_TESTS=$(npm run test:ci 2>&1 | grep "Tests:" | tail -1 | awk '{print $6}')

echo "📈 Test Status:"
echo "   Total: $TOTAL_TESTS"
echo "   Passed: $PASSED_TESTS" 
echo "   Failed: $FAILED_TESTS"
echo ""

# Ensure we have at least 60 tests
if [ "$TOTAL_TESTS" -lt 60 ]; then
  echo "❌ ERROR: Only $TOTAL_TESTS tests found, expected at least 60!"
  echo "Some tests may be disabled or missing!"
  exit 1
fi

echo "✅ Test count validation passed: $TOTAL_TESTS tests found"
echo ""

# Create test validation script
cat > scripts/validate-tests.js << 'EOF'
#!/usr/bin/env node

const MIN_TESTS = 60;
const { execSync } = require('child_process');

try {
  const output = execSync('npm run test:ci 2>&1', { encoding: 'utf8' });
  const match = output.match(/Tests:\s+(\d+)\s+failed,\s+(\d+)\s+passed,\s+(\d+)\s+total/);
  
  if (match) {
    const totalTests = parseInt(match[3]);
    
    if (totalTests < MIN_TESTS) {
      console.error(`❌ Test count validation failed: ${totalTests} tests found, minimum ${MIN_TESTS} required`);
      process.exit(1);
    }
    
    console.log(`✅ Test count validation passed: ${totalTests} tests found`);
    process.exit(0);
  }
} catch (error) {
  console.error('❌ Failed to validate test count');
  process.exit(1);
}
EOF

chmod +x scripts/validate-tests.js

echo "📝 Test validation script created"
echo ""
echo "🎯 Next: Fix individual test failures..."