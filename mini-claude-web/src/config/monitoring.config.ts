/**
 * Monitoring Configuration
 * Central configuration for all monitoring thresholds and settings
 */

export const monitoringConfig = {
  // Alert thresholds
  alerts: {
    errorRate: {
      warning: 5,    // 5% error rate triggers warning
      critical: 10   // 10% error rate triggers critical alert
    },
    responseTime: {
      warning: 2000,  // 2 seconds
      critical: 3000  // 3 seconds
    },
    availability: {
      warning: 99.5,  // Below 99.5% triggers warning
      critical: 95    // Below 95% triggers critical
    },
    databaseQueryTime: {
      warning: 50,    // 50ms
      critical: 150   // 150ms
    },
    memoryUsage: {
      warning: 80,    // 80% memory usage
      critical: 90    // 90% memory usage
    }
  },
  
  // Circuit breaker settings
  circuitBreakers: {
    gemini: {
      failureThreshold: 3,
      resetTimeout: 30000,     // 30 seconds
      monitoringPeriod: 300000 // 5 minutes
    },
    elevenlabs: {
      failureThreshold: 5,
      resetTimeout: 60000,     // 1 minute
      monitoringPeriod: 300000 // 5 minutes
    },
    database: {
      failureThreshold: 10,
      resetTimeout: 10000,     // 10 seconds
      monitoringPeriod: 300000 // 5 minutes
    }
  },
  
  // Performance targets
  performanceTargets: {
    api: {
      p50: 200,      // 50th percentile target: 200ms
      p95: 2000,     // 95th percentile target: 2s
      p99: 3000      // 99th percentile target: 3s
    },
    database: {
      averageQuery: 50,    // Average query time target
      slowQueryThreshold: 100 // Queries slower than this are logged
    },
    pageLoad: {
      fcp: 1500,     // First Contentful Paint
      lcp: 2500,     // Largest Contentful Paint
      tti: 3500      // Time to Interactive
    }
  },
  
  // Data retention settings
  dataRetention: {
    realTimeMetrics: 1000,     // Keep last 1000 data points
    aggregatedMetrics: 30,     // Keep 30 days of aggregated data
    alertHistory: 90,          // Keep 90 days of alert history
    performanceTrends: 7       // Keep 7 days of performance trends
  },
  
  // Health check intervals
  healthCheckIntervals: {
    internal: 60000,      // Internal health checks every minute
    external: 900000,     // External monitoring every 15 minutes
    dashboard: 5000       // Dashboard updates every 5 seconds
  },
  
  // Cost monitoring
  costLimits: {
    gemini: {
      dailyTokenLimit: 1000000,
      monthlyBudget: 100  // $100/month
    },
    elevenlabs: {
      dailyCharacterLimit: 100000,
      monthlyBudget: 50   // $50/month
    },
    vercel: {
      monthlyBudget: 20   // $20/month
    }
  },
  
  // Notification settings
  notifications: {
    channels: ['console', 'webhook'], // Available: console, email, webhook, slack
    webhookUrl: process.env.MONITORING_WEBHOOK_URL,
    cooldownPeriod: 300000, // 5 minutes between duplicate alerts
    escalationThreshold: 3  // Escalate after 3 occurrences
  }
}