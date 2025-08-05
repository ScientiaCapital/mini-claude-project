# Production Health Monitoring Operations Guide

## Overview

The mini-claude production health monitoring system provides comprehensive observability, real-time metrics collection, and intelligent alerting for maintaining 99.9% uptime.

## System Architecture

### Core Components

1. **MetricsCollector**: Tracks API performance, database queries, and AI service metrics
2. **HealthMonitor**: Performs comprehensive health checks on all services
3. **AlertManager**: Manages threshold-based alerting with cooldown periods
4. **PerformanceTracker**: Monitors operation timing and identifies bottlenecks
5. **SystemMetrics**: Aggregates all metrics for dashboard visualization

### Monitored Services

- **API Endpoints**: Response times, success rates, error tracking
- **Neon PostgreSQL**: Query performance, connection pool, slow queries
- **Google Gemini API**: Latency, token usage, rate limiting
- **ElevenLabs API**: Voice synthesis performance (when implemented)
- **Vercel Functions**: Memory usage, execution time, cold starts

## API Endpoints

### Health Check Endpoint
```
GET /api/health
```
Returns comprehensive system health including:
- Service status (healthy/degraded/unhealthy)
- Performance metrics summary
- Resource utilization
- Active issues list

### Metrics Endpoint
```
GET /api/monitoring/metrics
GET /api/monitoring/metrics?service=database
GET /api/monitoring/metrics?timeRange=24h
```
Provides detailed metrics for specific services or time ranges.

### Dashboard Endpoint
```
GET /api/monitoring/dashboard
Accept: text/event-stream  # For real-time updates
```
Returns dashboard data with optional Server-Sent Events for real-time monitoring.

### Alerts Endpoint
```
GET /api/monitoring/alerts
GET /api/monitoring/alerts?severity=critical&hours=24
POST /api/monitoring/alerts
```
Manages system alerts and allows manual alert creation.

## Monitoring Dashboard

Access the web-based dashboard at `/monitoring` for real-time visualization of:
- System uptime and request rates
- Service health status
- Performance metrics
- Recent alerts

## Alert Thresholds

| Metric | Warning | Critical |
|--------|---------|----------|
| Error Rate | > 5% | > 10% |
| Response Time | > 2000ms | > 3000ms |
| Database Query Time | > 50ms | > 150ms |
| Availability | < 99.5% | < 95% |
| Memory Usage | > 80% | > 90% |

## Circuit Breaker Configuration

Automatic failure recovery for external services:

- **Gemini API**: Opens after 3 failures, resets after 30s
- **ElevenLabs API**: Opens after 5 failures, resets after 60s
- **Database**: Opens after 10 failures, resets after 10s

## Monitoring Scripts

### Health Check Script
```bash
cd mini-claude-web
npm run monitor:health

# Or directly:
ts-node scripts/monitor-health.ts

# For production:
MONITORING_URL=https://mini-claude-web-scientia-capital.vercel.app ts-node scripts/monitor-health.ts
```

## Performance Optimization

### Identifying Bottlenecks
```typescript
const bottlenecks = performanceTracker.identifyBottlenecks()
// Returns operations taking > 1000ms on average
```

### Database Query Monitoring
- Queries > 50ms are flagged as slow
- Connection pool monitoring prevents exhaustion
- Automatic alerting on query performance degradation

### AI Service Optimization
- Token usage tracking for cost optimization
- Latency monitoring for user experience
- Rate limit detection and alerting

## Troubleshooting

### Common Issues

1. **High Error Rate**
   - Check `/api/monitoring/alerts` for specific errors
   - Review circuit breaker states
   - Verify external service availability

2. **Slow Response Times**
   - Check `/api/monitoring/metrics?service=api` for bottlenecks
   - Review database slow query log
   - Monitor AI service latencies

3. **Database Connection Issues**
   - Verify connection pool status in health endpoint
   - Check for connection leaks
   - Review database circuit breaker state

### Debug Mode
Enable detailed logging:
```typescript
// In development
process.env.NODE_ENV = 'development'
```

## Incident Response

### Severity Levels
- **Warning**: Performance degradation, non-critical issues
- **Critical**: Service failures, data loss risk, > 10% error rate

### Response Procedures

1. **Alert Received**
   - Check dashboard for overall system status
   - Identify affected services
   - Review recent changes

2. **Diagnosis**
   - Use `/api/monitoring/metrics` for detailed analysis
   - Check circuit breaker states
   - Review error logs

3. **Mitigation**
   - Circuit breakers provide automatic recovery
   - Manual intervention for persistent issues
   - Scale resources if needed

4. **Post-Incident**
   - Document root cause
   - Update monitoring thresholds
   - Implement preventive measures

## Best Practices

1. **Regular Health Checks**
   - Run automated health checks every 15 minutes
   - Monitor trends, not just current state
   - Set up alerts for degradation patterns

2. **Performance Budgets**
   - API response time: < 200ms (p50), < 2s (p95)
   - Database queries: < 50ms average
   - Page load time: < 1.5s

3. **Capacity Planning**
   - Monitor resource utilization trends
   - Plan for 2x peak traffic
   - Regular load testing

4. **Security Monitoring**
   - Track failed authentication attempts
   - Monitor for unusual API usage patterns
   - Regular security audits

## Integration with CI/CD

### GitHub Actions Integration
```yaml
- name: Health Check
  run: |
    npm run monitor:health
  env:
    MONITORING_URL: ${{ secrets.PRODUCTION_URL }}
```

### Deployment Verification
Always run health checks after deployment:
```bash
npm run deploy:verify
```

## Metrics Retention

- Real-time metrics: Last 1000 data points
- Aggregated metrics: 30 days
- Alerts history: 90 days
- Performance trends: 7 days

## Cost Optimization

1. **Monitor AI Token Usage**
   - Track via `/api/monitoring/metrics?service=ai`
   - Set budget alerts
   - Optimize prompt engineering

2. **Database Efficiency**
   - Monitor slow queries
   - Optimize indexes based on query patterns
   - Use connection pooling effectively

3. **Vercel Function Optimization**
   - Monitor execution time
   - Reduce cold starts
   - Optimize bundle size

## Future Enhancements

1. **Predictive Analytics**
   - ML-based anomaly detection
   - Predictive failure alerts
   - Capacity forecasting

2. **Advanced Visualizations**
   - Grafana integration
   - Custom dashboards
   - Mobile monitoring app

3. **Automated Remediation**
   - Self-healing capabilities
   - Automated scaling
   - Intelligent load balancing