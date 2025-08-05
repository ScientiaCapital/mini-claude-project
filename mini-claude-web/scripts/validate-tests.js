#!/usr/bin/env node

const MIN_TESTS = 60;
const { execSync } = require('child_process');

try {
  const output = execSync('npm run test:ci 2>&1', { encoding: 'utf8' });
  const match = output.match(/Tests:\s+(\d+)\s+failed,\s+(\d+)\s+passed,\s+(\d+)\s+total/);
  
  if (match) {
    const failedTests = parseInt(match[1]);
    const passedTests = parseInt(match[2]);
    const totalTests = parseInt(match[3]);
    
    console.log(`📊 Test Summary:`);
    console.log(`   Total: ${totalTests}`);
    console.log(`   Passed: ${passedTests}`);
    console.log(`   Failed: ${failedTests}`);
    console.log(``);
    
    if (totalTests < MIN_TESTS) {
      console.error(`❌ Test count validation failed: ${totalTests} tests found, minimum ${MIN_TESTS} required`);
      console.error(`\n⚠️  This indicates tests may have been disabled or removed!`);
      process.exit(1);
    }
    
    console.log(`✅ Test count validation passed: ${totalTests} tests found (minimum: ${MIN_TESTS})`);
    
    if (failedTests > 0) {
      console.log(`\n⚠️  Warning: ${failedTests} tests are still failing!`);
    }
    
    process.exit(0);
  } else {
    console.error('❌ Could not parse test results');
    process.exit(1);
  }
} catch (error) {
  console.error('❌ Failed to run tests:', error.message);
  process.exit(1);
}