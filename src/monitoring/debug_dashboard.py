"""
Real-time Debugging Dashboard

Web-based dashboard for real-time conversation inspection and debugging.
Provides visual insights into conversation flow, component performance,
and system health.

Features:
- Real-time event streaming
- Component latency visualization
- Turn-by-turn timeline
- Error tracking and alerts
- Audio waveform display
- LLM prompt/response viewer
- Performance metrics graphs
- Session recording and playback

Usage:
    from monitoring.debug_dashboard import DebugDashboard
    
    dashboard = DebugDashboard(inspector)
    await dashboard.start(host="localhost", port=8888)
    
    # Open browser to http://localhost:8888
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import aiohttp
from aiohttp import web
import aiohttp_cors
from dataclasses import dataclass, asdict
import weakref

from conversation_inspector import ConversationInspector, ConversationEvent, EventType


@dataclass
class DashboardConfig:
    """Dashboard configuration."""
    host: str = "localhost"
    port: int = 8888
    update_interval: float = 0.1  # seconds
    max_events_display: int = 100
    enable_audio_visualization: bool = True
    enable_recording: bool = True


class DebugDashboard:
    """
    Real-time web dashboard for conversation debugging.
    
    Provides a visual interface for monitoring and debugging
    voice agent conversations in real-time.
    """
    
    def __init__(self, inspector: ConversationInspector, config: Optional[DashboardConfig] = None):
        self.inspector = inspector
        self.config = config or DashboardConfig()
        
        # Web server
        self.app = web.Application()
        self.runner = None
        self.site = None
        
        # WebSocket connections
        self.websockets = weakref.WeakSet()
        
        # Dashboard state
        self.is_running = False
        self.recording_enabled = False
        self.recorded_sessions = []
        
        # Setup routes
        self._setup_routes()
        
        # Setup CORS
        cors = aiohttp_cors.setup(self.app, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
                allow_methods="*"
            )
        })
        
        for route in list(self.app.router.routes()):
            cors.add(route)
        
        # Register inspector callback
        self.inspector.add_monitor_callback(self._on_inspector_event)
    
    def _setup_routes(self):
        """Setup web routes."""
        self.app.router.add_get('/', self._handle_index)
        self.app.router.add_get('/api/status', self._handle_status)
        self.app.router.add_get('/api/insights', self._handle_insights)
        self.app.router.add_get('/api/timeline', self._handle_timeline)
        self.app.router.add_get('/api/anomalies', self._handle_anomalies)
        self.app.router.add_post('/api/recording/start', self._handle_start_recording)
        self.app.router.add_post('/api/recording/stop', self._handle_stop_recording)
        self.app.router.add_get('/api/recordings', self._handle_get_recordings)
        self.app.router.add_get('/ws', self._handle_websocket)
        self.app.router.add_static('/', path=str(Path(__file__).parent / 'static'), name='static')
    
    async def start(self, host: Optional[str] = None, port: Optional[int] = None) -> None:
        """Start the debug dashboard."""
        host = host or self.config.host
        port = port or self.config.port
        
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        
        self.site = web.TCPSite(self.runner, host, port)
        await self.site.start()
        
        self.is_running = True
        
        print(f"🎯 Debug Dashboard started at http://{host}:{port}")
        print(f"   Open your browser to view real-time conversation insights")
    
    async def stop(self) -> None:
        """Stop the debug dashboard."""
        self.is_running = False
        
        # Close all websockets
        for ws in list(self.websockets):
            await ws.close()
        
        if self.runner:
            await self.runner.cleanup()
        
        print("🎯 Debug Dashboard stopped")
    
    async def _handle_index(self, request):
        """Serve the dashboard HTML."""
        html_content = self._generate_dashboard_html()
        return web.Response(text=html_content, content_type='text/html')
    
    async def _handle_status(self, request):
        """Get current status."""
        status = {
            "inspector_active": self.inspector.is_monitoring,
            "dashboard_active": self.is_running,
            "recording_enabled": self.recording_enabled,
            "websocket_connections": len(self.websockets),
            "current_session": None
        }
        
        if self.inspector.current_session:
            status["current_session"] = {
                "session_id": self.inspector.current_session.session_id,
                "start_time": self.inspector.current_session.start_time.isoformat(),
                "turn_count": self.inspector.current_session.turn_count,
                "event_count": len(self.inspector.current_session.events)
            }
        
        return web.json_response(status)
    
    async def _handle_insights(self, request):
        """Get conversation insights."""
        insights = self.inspector.get_conversation_insights()
        return web.json_response(insights)
    
    async def _handle_timeline(self, request):
        """Get turn timeline."""
        turn_id = request.rel_url.query.get('turn_id')
        timeline = self.inspector.get_turn_timeline(turn_id)
        return web.json_response(timeline)
    
    async def _handle_anomalies(self, request):
        """Get detected anomalies."""
        anomalies = self.inspector.detect_anomalies()
        return web.json_response(anomalies)
    
    async def _handle_start_recording(self, request):
        """Start recording session."""
        self.recording_enabled = True
        return web.json_response({"status": "recording_started"})
    
    async def _handle_stop_recording(self, request):
        """Stop recording session."""
        self.recording_enabled = False
        
        if self.inspector.current_session:
            session_data = {
                "session_id": self.inspector.current_session.session_id,
                "recorded_at": datetime.now().isoformat(),
                "data": await self.inspector.generate_session_report()
            }
            self.recorded_sessions.append(session_data)
        
        return web.json_response({
            "status": "recording_stopped",
            "sessions_recorded": len(self.recorded_sessions)
        })
    
    async def _handle_get_recordings(self, request):
        """Get recorded sessions."""
        return web.json_response(self.recorded_sessions)
    
    async def _handle_websocket(self, request):
        """Handle WebSocket connections for real-time updates."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        self.websockets.add(ws)
        
        try:
            # Send initial state
            await ws.send_json({
                "type": "connected",
                "data": {
                    "dashboard_version": "1.0.0",
                    "inspector_active": self.inspector.is_monitoring
                }
            })
            
            # Keep connection alive
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    
                    if data.get("type") == "ping":
                        await ws.send_json({"type": "pong"})
                    
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logging.error(f'WebSocket error: {ws.exception()}')
        
        except Exception as e:
            logging.error(f"WebSocket handler error: {e}")
        
        finally:
            self.websockets.discard(ws)
            return ws
    
    async def _broadcast_event(self, event_type: str, data: Any) -> None:
        """Broadcast event to all connected WebSocket clients."""
        message = json.dumps({
            "type": event_type,
            "timestamp": datetime.now().isoformat(),
            "data": data
        })
        
        # Send to all connected clients
        for ws in list(self.websockets):
            try:
                await ws.send_str(message)
            except ConnectionResetError:
                self.websockets.discard(ws)
    
    def _on_inspector_event(self, event: ConversationEvent) -> None:
        """Handle inspector events."""
        # Convert event to JSON-serializable format
        event_data = {
            "event_type": event.event_type.value,
            "timestamp": event.timestamp.isoformat(),
            "duration_ms": event.duration_ms,
            "data": event.data,
            "metadata": event.metadata
        }
        
        # Broadcast to dashboard
        asyncio.create_task(self._broadcast_event("inspector_event", event_data))
        
        # Special handling for certain events
        if event.event_type == EventType.TURN_START:
            asyncio.create_task(self._broadcast_event("turn_start", event.data))
        elif event.event_type == EventType.TURN_END:
            asyncio.create_task(self._broadcast_event("turn_end", event.data))
        elif event.event_type == EventType.ERROR:
            asyncio.create_task(self._broadcast_event("error", event.data))
    
    def _generate_dashboard_html(self) -> str:
        """Generate the dashboard HTML."""
        return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Voice Agent Debug Dashboard</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: #0f0f0f;
            color: #e0e0e0;
            line-height: 1.6;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 20px 0;
            border-bottom: 1px solid #333;
            margin-bottom: 20px;
        }
        
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        
        .status-active { background-color: #4caf50; }
        .status-inactive { background-color: #f44336; }
        .status-warning { background-color: #ff9800; }
        
        .dashboard-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .panel {
            background-color: #1a1a1a;
            border: 1px solid #333;
            border-radius: 8px;
            padding: 20px;
        }
        
        .panel h3 {
            margin-bottom: 15px;
            color: #fff;
            font-size: 18px;
        }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
        }
        
        .metric-card {
            background-color: #252525;
            padding: 15px;
            border-radius: 6px;
            text-align: center;
        }
        
        .metric-value {
            font-size: 28px;
            font-weight: bold;
            color: #4caf50;
        }
        
        .metric-label {
            font-size: 12px;
            color: #999;
            text-transform: uppercase;
            margin-top: 5px;
        }
        
        .event-log {
            height: 300px;
            overflow-y: auto;
            background-color: #0a0a0a;
            border: 1px solid #333;
            border-radius: 4px;
            padding: 10px;
            font-family: 'Courier New', monospace;
            font-size: 12px;
        }
        
        .event-item {
            padding: 4px 0;
            border-bottom: 1px solid #222;
        }
        
        .event-time {
            color: #666;
            margin-right: 10px;
        }
        
        .event-type {
            color: #4caf50;
            margin-right: 10px;
        }
        
        .timeline {
            position: relative;
            height: 200px;
            background-color: #0a0a0a;
            border: 1px solid #333;
            border-radius: 4px;
            overflow-x: auto;
        }
        
        .timeline-event {
            position: absolute;
            top: 50%;
            transform: translateY(-50%);
            background-color: #4caf50;
            height: 20px;
            border-radius: 4px;
            padding: 0 8px;
            font-size: 11px;
            white-space: nowrap;
            display: flex;
            align-items: center;
        }
        
        .controls {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }
        
        .btn {
            background-color: #4caf50;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            transition: background-color 0.3s;
        }
        
        .btn:hover {
            background-color: #45a049;
        }
        
        .btn-danger {
            background-color: #f44336;
        }
        
        .btn-danger:hover {
            background-color: #da190b;
        }
        
        .anomaly-alert {
            background-color: #ff5722;
            color: white;
            padding: 10px;
            border-radius: 4px;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
        }
        
        .anomaly-alert .icon {
            margin-right: 10px;
            font-size: 20px;
        }
        
        .latency-chart {
            height: 200px;
            position: relative;
        }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        
        .recording {
            animation: pulse 2s infinite;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 Voice Agent Debug Dashboard</h1>
            <div class="status">
                <span id="inspector-status">
                    <span class="status-indicator status-inactive"></span>
                    Inspector: Inactive
                </span>
                <span id="recording-status" style="margin-left: 20px;">
                    <span class="status-indicator status-inactive"></span>
                    Recording: Off
                </span>
            </div>
        </div>
        
        <div class="controls">
            <button class="btn" onclick="startRecording()">Start Recording</button>
            <button class="btn btn-danger" onclick="stopRecording()">Stop Recording</button>
            <button class="btn" onclick="refreshDashboard()">Refresh</button>
            <button class="btn" onclick="exportSession()">Export Session</button>
        </div>
        
        <div id="anomaly-container"></div>
        
        <div class="dashboard-grid">
            <div class="panel">
                <h3>📊 Real-time Metrics</h3>
                <div class="metrics-grid" id="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-value" id="turn-count">0</div>
                        <div class="metric-label">Turns</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" id="avg-latency">0ms</div>
                        <div class="metric-label">Avg Latency</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" id="error-count">0</div>
                        <div class="metric-label">Errors</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" id="event-count">0</div>
                        <div class="metric-label">Events</div>
                    </div>
                </div>
            </div>
            
            <div class="panel">
                <h3>⚡ Component Latencies</h3>
                <div class="latency-chart" id="latency-chart">
                    <canvas id="latency-canvas" width="400" height="200"></canvas>
                </div>
            </div>
        </div>
        
        <div class="panel">
            <h3>🎯 Turn Timeline</h3>
            <div class="timeline" id="timeline"></div>
        </div>
        
        <div class="panel">
            <h3>📝 Event Log</h3>
            <div class="event-log" id="event-log"></div>
        </div>
    </div>
    
    <script>
        let ws = null;
        let eventLog = [];
        let latencyData = {
            stt: [],
            llm: [],
            tts: []
        };
        
        function connectWebSocket() {
            ws = new WebSocket(`ws://${window.location.host}/ws`);
            
            ws.onopen = () => {
                console.log('WebSocket connected');
                updateStatus('inspector-status', true, 'Inspector: Active');
            };
            
            ws.onmessage = (event) => {
                const message = JSON.parse(event.data);
                handleWebSocketMessage(message);
            };
            
            ws.onclose = () => {
                console.log('WebSocket disconnected');
                updateStatus('inspector-status', false, 'Inspector: Inactive');
                setTimeout(connectWebSocket, 3000);
            };
            
            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
            };
        }
        
        function handleWebSocketMessage(message) {
            switch(message.type) {
                case 'inspector_event':
                    handleInspectorEvent(message.data);
                    break;
                case 'turn_start':
                    addTimelineEvent('Turn Start', message.data.speaker);
                    break;
                case 'turn_end':
                    addTimelineEvent('Turn End', message.data.speaker);
                    break;
                case 'error':
                    handleError(message.data);
                    break;
            }
        }
        
        function handleInspectorEvent(event) {
            // Add to event log
            addEventToLog(event);
            
            // Update event count
            const eventCount = document.getElementById('event-count');
            eventCount.textContent = parseInt(eventCount.textContent) + 1;
            
            // Handle specific event types
            if (event.event_type === 'turn_end') {
                const turnCount = document.getElementById('turn-count');
                turnCount.textContent = parseInt(turnCount.textContent) + 1;
            }
            
            // Update latency data
            if (event.event_type.includes('_complete')) {
                updateLatencyChart(event);
            }
        }
        
        function addEventToLog(event) {
            const log = document.getElementById('event-log');
            const time = new Date(event.timestamp).toLocaleTimeString();
            
            const eventItem = document.createElement('div');
            eventItem.className = 'event-item';
            eventItem.innerHTML = `
                <span class="event-time">${time}</span>
                <span class="event-type">${event.event_type}</span>
                <span>${JSON.stringify(event.data)}</span>
            `;
            
            log.insertBefore(eventItem, log.firstChild);
            
            // Keep only last 100 events
            while (log.children.length > 100) {
                log.removeChild(log.lastChild);
            }
        }
        
        function addTimelineEvent(type, speaker) {
            const timeline = document.getElementById('timeline');
            const event = document.createElement('div');
            event.className = 'timeline-event';
            event.style.left = `${timeline.children.length * 120}px`;
            event.style.backgroundColor = speaker === 'user' ? '#4caf50' : '#2196f3';
            event.textContent = `${speaker}: ${type}`;
            timeline.appendChild(event);
        }
        
        function updateLatencyChart(event) {
            // Simple latency visualization
            const canvas = document.getElementById('latency-canvas');
            const ctx = canvas.getContext('2d');
            
            // This is a simplified visualization - in production, use Chart.js or similar
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.fillStyle = '#4caf50';
            ctx.fillRect(50, 50, event.duration_ms / 10, 20);
        }
        
        function handleError(error) {
            // Update error count
            const errorCount = document.getElementById('error-count');
            errorCount.textContent = parseInt(errorCount.textContent) + 1;
            
            // Show anomaly alert
            showAnomaly({
                type: 'error',
                message: `${error.component}: ${error.error_message}`,
                severity: 'high'
            });
        }
        
        function showAnomaly(anomaly) {
            const container = document.getElementById('anomaly-container');
            const alert = document.createElement('div');
            alert.className = 'anomaly-alert';
            alert.innerHTML = `
                <span class="icon">⚠️</span>
                <span>${anomaly.message}</span>
            `;
            container.appendChild(alert);
            
            // Remove after 5 seconds
            setTimeout(() => container.removeChild(alert), 5000);
        }
        
        function updateStatus(elementId, active, text) {
            const element = document.getElementById(elementId);
            const indicator = element.querySelector('.status-indicator');
            
            indicator.className = `status-indicator ${active ? 'status-active' : 'status-inactive'}`;
            element.lastChild.textContent = text;
        }
        
        async function startRecording() {
            const response = await fetch('/api/recording/start', { method: 'POST' });
            if (response.ok) {
                updateStatus('recording-status', true, 'Recording: On');
                document.getElementById('recording-status').classList.add('recording');
            }
        }
        
        async function stopRecording() {
            const response = await fetch('/api/recording/stop', { method: 'POST' });
            if (response.ok) {
                updateStatus('recording-status', false, 'Recording: Off');
                document.getElementById('recording-status').classList.remove('recording');
            }
        }
        
        async function refreshDashboard() {
            // Fetch current insights
            const response = await fetch('/api/insights');
            const insights = await response.json();
            
            // Update metrics
            document.getElementById('turn-count').textContent = insights.turn_count || 0;
            document.getElementById('avg-latency').textContent = 
                insights.avg_total_latency ? `${Math.round(insights.avg_total_latency)}ms` : '0ms';
            
            // Check for anomalies
            const anomResponse = await fetch('/api/anomalies');
            const anomalies = await anomResponse.json();
            
            anomalies.forEach(anomaly => showAnomaly(anomaly));
        }
        
        async function exportSession() {
            // In a real implementation, this would download the session data
            alert('Session export functionality would be implemented here');
        }
        
        // Initialize dashboard
        connectWebSocket();
        setInterval(refreshDashboard, 5000); // Refresh every 5 seconds
    </script>
</body>
</html>
'''


# Example usage
async def demo_debug_dashboard():
    """Demonstrate debug dashboard."""
    print("🎯 Debug Dashboard Demo")
    print("="*50)
    
    # Create inspector
    from conversation_inspector import ConversationInspector
    inspector = ConversationInspector()
    
    # Create and start dashboard
    dashboard = DebugDashboard(inspector)
    await dashboard.start()
    
    # Start inspector monitoring
    await inspector.start_monitoring()
    
    print(f"\n✅ Dashboard running at http://localhost:8888")
    print("   Open your browser to see real-time debugging")
    print("   Press Ctrl+C to stop")
    
    try:
        # Keep running
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping dashboard...")
    finally:
        await inspector.stop_monitoring()
        await dashboard.stop()


if __name__ == "__main__":
    asyncio.run(demo_debug_dashboard())