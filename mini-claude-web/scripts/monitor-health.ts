#!/usr/bin/env node
/**
 * Health Monitoring Script
 * Can be run locally or in CI/CD to check system health
 */
import fetch from 'node-fetch'

const BASE_URL = process.env.MONITORING_URL || 'https://mini-claude-web-scientia-capital.vercel.app'

interface HealthCheck {
  endpoint: string
  expectedStatus: number
  maxResponseTime: number
}

const healthChecks: HealthCheck[] = [
  {
    endpoint: '/api/health',
    expectedStatus: 200,
    maxResponseTime: 1000
  },
  {
    endpoint: '/api/monitoring/metrics',
    expectedStatus: 200,
    maxResponseTime: 2000
  },
  {
    endpoint: '/api/monitoring/dashboard',
    expectedStatus: 200,
    maxResponseTime: 2000
  }
]

async function checkEndpoint(check: HealthCheck) {
  const startTime = Date.now()
  
  try {
    const response = await fetch(`${BASE_URL}${check.endpoint}`, {
      timeout: 10000
    })
    
    const responseTime = Date.now() - startTime
    const data = await response.json()
    
    const result = {
      endpoint: check.endpoint,
      status: response.status,
      responseTime,
      success: response.status === check.expectedStatus && responseTime <= check.maxResponseTime,
      data
    }
    
    if (!result.success) {
      console.error(`❌ ${check.endpoint} - Status: ${response.status}, Time: ${responseTime}ms`)
      if (response.status !== check.expectedStatus) {
        console.error(`   Expected status ${check.expectedStatus}, got ${response.status}`)
      }
      if (responseTime > check.maxResponseTime) {
        console.error(`   Response time ${responseTime}ms exceeds threshold ${check.maxResponseTime}ms`)
      }
    } else {
      console.log(`✅ ${check.endpoint} - Status: ${response.status}, Time: ${responseTime}ms`)
    }
    
    return result
  } catch (error) {
    console.error(`❌ ${check.endpoint} - Error: ${error}`)
    return {
      endpoint: check.endpoint,
      status: 0,
      responseTime: Date.now() - startTime,
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error'
    }
  }
}

async function analyzeHealth(healthData: any) {
  console.log('\n📊 System Health Analysis:')
  
  if (healthData.status === 'healthy') {
    console.log('✅ Overall Status: HEALTHY')
  } else if (healthData.status === 'degraded') {
    console.log('⚠️  Overall Status: DEGRADED')
  } else {
    console.log('❌ Overall Status: UNHEALTHY')
  }
  
  // Check services
  console.log('\n🔧 Service Status:')
  Object.entries(healthData.services || {}).forEach(([service, status]: [string, any]) => {
    const icon = status.status === 'healthy' ? '✅' : status.status === 'degraded' ? '⚠️' : '❌'
    console.log(`${icon} ${service}: ${status.status} (${status.responseTime}ms)`)
  })
  
  // Check metrics
  if (healthData.metrics) {
    console.log('\n📈 Performance Metrics:')
    console.log(`API Success Rate: ${healthData.metrics.api?.successRate?.toFixed(2) || 'N/A'}%`)
    console.log(`Avg Response Time: ${healthData.metrics.api?.averageResponseTime?.toFixed(0) || 'N/A'}ms`)
    console.log(`DB Avg Query Time: ${healthData.metrics.database?.averageQueryTime?.toFixed(2) || 'N/A'}ms`)
    console.log(`Slow Queries: ${healthData.metrics.database?.slowQueries || 0}`)
  }
  
  // Check for issues
  if (healthData.issues && healthData.issues.length > 0) {
    console.log('\n⚠️  Active Issues:')
    healthData.issues.forEach((issue: string) => {
      console.log(`- ${issue}`)
    })
  }
}

async function checkAlerts() {
  try {
    const response = await fetch(`${BASE_URL}/api/monitoring/alerts?hours=1`)
    const data = await response.json()
    
    if (data.alerts && data.alerts.length > 0) {
      console.log('\n🚨 Recent Alerts:')
      data.alerts.slice(0, 5).forEach((alert: any) => {
        const icon = alert.severity === 'critical' ? '🔴' : '🟡'
        console.log(`${icon} ${alert.metric}: ${alert.value} - ${alert.message || 'No message'}`)
      })
    }
  } catch (error) {
    console.error('Failed to fetch alerts:', error)
  }
}

async function main() {
  console.log(`🔍 Monitoring Health Check - ${BASE_URL}`)
  console.log('=' . repeat(50))
  
  // Run all health checks
  const results = await Promise.all(healthChecks.map(checkEndpoint))
  
  // Check if all passed
  const allPassed = results.every(r => r.success)
  
  // Get detailed health data
  const healthResult = results.find(r => r.endpoint === '/api/health')
  if (healthResult?.data) {
    await analyzeHealth(healthResult.data)
  }
  
  // Check for alerts
  await checkAlerts()
  
  console.log('\n' + '=' . repeat(50))
  if (allPassed) {
    console.log('✅ All health checks passed!')
    process.exit(0)
  } else {
    console.log('❌ Some health checks failed!')
    process.exit(1)
  }
}

// Run the monitoring script
main().catch(error => {
  console.error('Monitoring script error:', error)
  process.exit(1)
})