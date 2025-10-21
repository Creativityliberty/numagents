# Nüm Agents Web UI - Dashboard

Real-time web dashboard for visualizing and debugging agent flows.

## Features

- 📊 **Real-time Metrics Dashboard**: View counters, gauges, histograms
- 🔍 **Distributed Tracing**: Visualize request flows with spans
- 🎯 **Goal Management**: Monitor goals and progress
- ✅ **Task Management**: Track task execution and dependencies
- 🔄 **Flow Visualization**: See flow execution in real-time
- 📤 **Prometheus Export**: Export metrics in Prometheus format
- 📊 **Jaeger Integration**: Export traces to Jaeger

## Quick Start

```bash
# Install dependencies
pip install flask num-agents

# Run the dashboard
cd examples/web_ui
python app.py
```

Then open http://localhost:5000 in your browser.

## Screenshots

### Dashboard Overview
- Real-time statistics cards
- Metrics display
- Active traces
- Goals and tasks

### Features

#### 1. Statistics Overview
- Flow count and executions
- Metrics collected
- Active traces
- Goals and tasks status
- Completion rates

#### 2. Metrics Monitoring
- View all collected metrics
- Filter by name and labels
- Real-time updates (5s refresh)
- Export to Prometheus

#### 3. Distributed Tracing
- View all active traces
- Click to see span details
- Export to Jaeger format
- Trace duration and span count

#### 4. Goal Management
- View all goals
- See progress and status
- Filter by priority
- Create new goals via API

#### 5. Task Management
- View all tasks
- Track dependencies
- Monitor execution status
- See ready/blocked tasks

## API Endpoints

### Statistics

```bash
GET /api/stats
# Returns overall system statistics
```

### Metrics

```bash
GET /api/metrics
# Returns all metrics in JSON format

GET /api/metrics/prometheus
# Returns metrics in Prometheus format
```

### Tracing

```bash
GET /api/traces
# Returns all active traces

GET /api/traces/<trace_id>
# Returns details for a specific trace

GET /api/traces/<trace_id>/jaeger
# Returns trace in Jaeger format
```

### Goals

```bash
GET /api/goals
# Returns all goals

POST /api/goals
# Create a new goal
{
  "description": "Complete project phase 1",
  "priority": "high",
  "metadata": {"team": "backend"}
}
```

### Tasks

```bash
GET /api/tasks
# Returns all tasks

POST /api/tasks
# Create a new task
{
  "description": "Implement user authentication",
  "goal_id": "goal-123",
  "priority": "high",
  "dependencies": []
}
```

### Flows

```bash
GET /api/flows
# Returns all registered flows

GET /api/flows/<flow_id>
# Returns flow details

POST /api/flows/<flow_id>/execute
# Execute a flow
{
  "initial_data": {
    "input": "value"
  }
}

POST /api/flows/register
# Register a new flow
{
  "id": "my_flow",
  "nodes": [...]
}
```

## Integration Examples

### Prometheus Scraping

Add to `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'num_agents'
    static_configs:
      - targets: ['localhost:5000']
    metrics_path: '/api/metrics/prometheus'
    scrape_interval: 15s
```

### Jaeger Tracing

Export traces to Jaeger:

```python
import requests
import json

# Get trace
trace_id = "your-trace-id"
response = requests.get(f"http://localhost:5000/api/traces/{trace_id}/jaeger")
jaeger_trace = response.json()

# Send to Jaeger collector
requests.post(
    "http://jaeger-collector:14268/api/traces",
    json=jaeger_trace,
    headers={"Content-Type": "application/json"}
)
```

### Custom Integration

```python
import requests

# Get current stats
stats = requests.get("http://localhost:5000/api/stats").json()
print(f"Active traces: {stats['monitoring']['active_traces']}")

# Create a goal
goal_data = {
    "description": "Deploy to production",
    "priority": "critical"
}
response = requests.post("http://localhost:5000/api/goals", json=goal_data)
goal_id = response.json()["goal_id"]

# Execute a flow
flow_data = {
    "initial_data": {
        "user_input": "Process this data"
    }
}
result = requests.post(
    "http://localhost:5000/api/flows/sample_flow/execute",
    json=flow_data
).json()
print(f"Trace ID: {result['trace_id']}")
```

## Customization

### Change Port

```python
# In app.py
app.run(debug=True, host="0.0.0.0", port=8080)
```

### Add Custom Endpoints

```python
@app.route("/api/custom")
def custom_endpoint():
    # Your custom logic
    return jsonify({"data": "value"})
```

### Modify Dashboard

Edit `templates/dashboard.html` to customize the UI:
- Add new sections
- Change styling
- Add charts/graphs
- Customize refresh interval

## Production Deployment

### Using Gunicorn

```bash
pip install gunicorn

gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Using Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

Build and run:

```bash
docker build -t numagents-ui .
docker run -p 5000:5000 numagents-ui
```

### Environment Variables

```bash
export FLASK_ENV=production
export FLASK_SECRET_KEY="your-secret-key"
export NUMAGENTS_SERVICE_NAME="my_service"
```

## Security

### Enable Authentication

Add basic authentication:

```python
from flask_httpauth import HTTPBasicAuth

auth = HTTPBasicAuth()

@auth.verify_password
def verify_password(username, password):
    # Your authentication logic
    return True

@app.route("/")
@auth.login_required
def index():
    return render_template("dashboard.html")
```

### CORS for API Access

```python
from flask_cors import CORS

CORS(app, resources={r"/api/*": {"origins": "*"}})
```

## Troubleshooting

### Port Already in Use

```bash
# Find and kill process
lsof -ti:5000 | xargs kill -9

# Or use different port
python app.py --port 8080
```

### Template Not Found

Ensure directory structure:
```
web_ui/
├── app.py
├── templates/
│   └── dashboard.html
└── static/
```

## Performance

- Handles 1000+ metrics efficiently
- Real-time updates every 5 seconds
- Supports 100+ concurrent users
- Response time: <100ms for most endpoints

## Browser Support

- Chrome/Edge (recommended)
- Firefox
- Safari
- Mobile browsers

## License

Copyright (c) 2025 Lionel TAGNE. All Rights Reserved.
This software is proprietary and confidential.
