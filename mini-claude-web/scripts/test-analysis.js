#!/usr/bin/env node

/**
 * Test Analysis Script
 * Analyzes failing tests to understand what needs to be fixed
 */

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

console.log('🔍 Analyzing Test Failures...\n');

// Run tests and capture output
try {
  const output = execSync('npm run test:ci -- --json --outputFile=test-results.json', {
    encoding: 'utf8',
    stdio: 'pipe'
  });
} catch (error) {
  // Tests will fail, but we still get the JSON output
}

// Read and analyze results
const results = JSON.parse(fs.readFileSync('test-results.json', 'utf8'));

console.log('📊 Test Summary:');
console.log(`Total Test Suites: ${results.numTotalTestSuites}`);
console.log(`Failed Test Suites: ${results.numFailedTestSuites}`);
console.log(`Total Tests: ${results.numTotalTests}`);
console.log(`Failed Tests: ${results.numFailedTests}`);
console.log(`Passed Tests: ${results.numPassedTests}`);
console.log('\n');

// Group failures by test file
const failuresByFile = {};

results.testResults.forEach(testFile => {
  if (testFile.numFailingTests > 0) {
    const fileName = path.basename(testFile.name);
    failuresByFile[fileName] = {
      failures: testFile.numFailingTests,
      total: testFile.assertionResults.length,
      failedTests: testFile.assertionResults
        .filter(test => test.status === 'failed')
        .map(test => ({
          title: test.title,
          error: test.failureMessages[0]?.split('\n')[0] || 'Unknown error'
        }))
    };
  }
});

console.log('❌ Failing Test Files:\n');

Object.entries(failuresByFile).forEach(([file, data]) => {
  console.log(`📄 ${file} (${data.failures}/${data.total} failing)`);
  data.failedTests.forEach(test => {
    console.log(`  ❌ ${test.title}`);
    console.log(`     ${test.error}\n`);
  });
  console.log('');
});

// Clean up
fs.unlinkSync('test-results.json');

console.log('\n🎯 Next Steps:');
console.log('1. Fix mock implementations to match actual API behavior');
console.log('2. Update error messages to match implementation');
console.log('3. Ensure all database mocks return expected data');
console.log('4. Verify environment variables are properly set in tests');