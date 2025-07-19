'use client';

import { useState, useEffect } from 'react';
import { BarChart3, Clock, Cpu, Mic, Volume2, Zap } from 'lucide-react';
import { cn, formatDuration } from '@/lib/utils';

interface MetricsData {
  llm: {
    total_requests: number;
    avg_tokens_per_second: number;
    avg_time_to_first_token: number;
    total_tokens: number;
  };
  stt: {
    total_requests: number;
    avg_duration: number;
    streaming_percentage: number;
  };
  tts: {
    total_requests: number;
    avg_time_to_first_byte: number;
    avg_duration: number;
  };
}

interface MetricsDashboardProps {
  className?: string;
  isVisible: boolean;
  onClose: () => void;
}

export default function MetricsDashboard({ className, isVisible, onClose }: MetricsDashboardProps) {
  const [metrics, setMetrics] = useState<MetricsData | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  // Mock metrics data for demonstration
  const fetchMetrics = async () => {
    setIsLoading(true);
    
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 500));
    
    // Mock data
    const mockMetrics: MetricsData = {
      llm: {
        total_requests: 15,
        avg_tokens_per_second: 25.3,
        avg_time_to_first_token: 0.42,
        total_tokens: 1250
      },
      stt: {
        total_requests: 18,
        avg_duration: 1.8,
        streaming_percentage: 94.4
      },
      tts: {
        total_requests: 15,
        avg_time_to_first_byte: 0.23,
        avg_duration: 2.1
      }
    };
    
    setMetrics(mockMetrics);
    setIsLoading(false);
  };

  useEffect(() => {
    if (isVisible) {
      fetchMetrics();
    }
  }, [isVisible]);

  if (!isVisible) return null;

  const MetricCard = ({ 
    title, 
    value, 
    unit, 
    icon: Icon, 
    description,
    status = 'good' 
  }: {
    title: string;
    value: string | number;
    unit?: string;
    icon: React.ComponentType<{ className?: string }>;
    description: string;
    status?: 'good' | 'warning' | 'error';
  }) => (
    <div className="bg-white rounded-lg p-4 border border-gray-200 shadow-sm">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center space-x-2">
          <Icon className={cn(
            'w-4 h-4',
            {
              'text-green-600': status === 'good',
              'text-yellow-600': status === 'warning',
              'text-red-600': status === 'error'
            }
          )} />
          <span className="text-sm font-medium text-gray-700">{title}</span>
        </div>
        <div className={cn(
          'w-2 h-2 rounded-full',
          {
            'bg-green-500': status === 'good',
            'bg-yellow-500': status === 'warning',
            'bg-red-500': status === 'error'
          }
        )} />
      </div>
      
      <div className="flex items-baseline space-x-1 mb-1">
        <span className="text-2xl font-semibold text-gray-900">
          {typeof value === 'number' ? value.toFixed(2) : value}
        </span>
        {unit && <span className="text-sm text-gray-500">{unit}</span>}
      </div>
      
      <p className="text-xs text-gray-500">{description}</p>
    </div>
  );

  return (
    <div className={cn(
      'fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50',
      className
    )}>
      <div className="bg-white rounded-xl shadow-2xl max-w-4xl w-full max-h-[90vh] overflow-hidden m-4">
        {/* Header */}
        <div className="bg-gray-50 px-6 py-4 border-b border-gray-200">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <BarChart3 className="w-6 h-6 text-primary-600" />
              <h2 className="text-xl font-semibold text-gray-900">Performance Metrics</h2>
            </div>
            <button
              onClick={onClose}
              className="text-gray-400 hover:text-gray-600 transition-colors"
            >
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="p-6 overflow-y-auto max-h-[calc(90vh-80px)]">
          {isLoading ? (
            <div className="flex items-center justify-center py-12">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600"></div>
              <span className="ml-2 text-gray-600">Loading metrics...</span>
            </div>
          ) : metrics ? (
            <div className="space-y-6">
              {/* LLM Metrics */}
              <div>
                <h3 className="text-lg font-medium text-gray-900 mb-4 flex items-center">
                  <Cpu className="w-5 h-5 mr-2 text-primary-600" />
                  Language Model Performance
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                  <MetricCard
                    title="Total Requests"
                    value={metrics.llm.total_requests}
                    icon={Zap}
                    description="Number of LLM requests"
                    status="good"
                  />
                  <MetricCard
                    title="Tokens per Second"
                    value={metrics.llm.avg_tokens_per_second}
                    unit="tokens/s"
                    icon={Clock}
                    description="Average generation speed"
                    status={metrics.llm.avg_tokens_per_second > 20 ? 'good' : 'warning'}
                  />
                  <MetricCard
                    title="Time to First Token"
                    value={metrics.llm.avg_time_to_first_token}
                    unit="seconds"
                    icon={Clock}
                    description="Average response latency"
                    status={metrics.llm.avg_time_to_first_token < 0.5 ? 'good' : 'warning'}
                  />
                  <MetricCard
                    title="Total Tokens"
                    value={metrics.llm.total_tokens}
                    icon={BarChart3}
                    description="Total tokens generated"
                    status="good"
                  />
                </div>
              </div>

              {/* STT Metrics */}
              <div>
                <h3 className="text-lg font-medium text-gray-900 mb-4 flex items-center">
                  <Mic className="w-5 h-5 mr-2 text-blue-600" />
                  Speech-to-Text Performance
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <MetricCard
                    title="Total Requests"
                    value={metrics.stt.total_requests}
                    icon={Zap}
                    description="Number of STT requests"
                    status="good"
                  />
                  <MetricCard
                    title="Average Duration"
                    value={metrics.stt.avg_duration}
                    unit="seconds"
                    icon={Clock}
                    description="Processing time per request"
                    status={metrics.stt.avg_duration < 2 ? 'good' : 'warning'}
                  />
                  <MetricCard
                    title="Streaming Rate"
                    value={metrics.stt.streaming_percentage}
                    unit="%"
                    icon={BarChart3}
                    description="Percentage using streaming"
                    status={metrics.stt.streaming_percentage > 90 ? 'good' : 'warning'}
                  />
                </div>
              </div>

              {/* TTS Metrics */}
              <div>
                <h3 className="text-lg font-medium text-gray-900 mb-4 flex items-center">
                  <Volume2 className="w-5 h-5 mr-2 text-green-600" />
                  Text-to-Speech Performance
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <MetricCard
                    title="Total Requests"
                    value={metrics.tts.total_requests}
                    icon={Zap}
                    description="Number of TTS requests"
                    status="good"
                  />
                  <MetricCard
                    title="Time to First Byte"
                    value={metrics.tts.avg_time_to_first_byte}
                    unit="seconds"
                    icon={Clock}
                    description="Audio generation latency"
                    status={metrics.tts.avg_time_to_first_byte < 0.3 ? 'good' : 'warning'}
                  />
                  <MetricCard
                    title="Average Duration"
                    value={metrics.tts.avg_duration}
                    unit="seconds"
                    icon={Clock}
                    description="Total generation time"
                    status={metrics.tts.avg_duration < 3 ? 'good' : 'warning'}
                  />
                </div>
              </div>

              {/* Performance Summary */}
              <div className="bg-gradient-to-r from-primary-50 to-blue-50 rounded-lg p-4 border border-primary-200">
                <h3 className="text-lg font-medium text-gray-900 mb-2">Performance Summary</h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                  <div>
                    <span className="font-medium text-gray-700">Overall Latency:</span>
                    <span className="ml-2 text-primary-600">
                      {(metrics.llm.avg_time_to_first_token + metrics.tts.avg_time_to_first_byte).toFixed(2)}s
                    </span>
                  </div>
                  <div>
                    <span className="font-medium text-gray-700">Processing Efficiency:</span>
                    <span className="ml-2 text-green-600">
                      {metrics.stt.streaming_percentage > 90 ? 'Excellent' : 'Good'}
                    </span>
                  </div>
                  <div>
                    <span className="font-medium text-gray-700">Response Quality:</span>
                    <span className="ml-2 text-green-600">
                      {metrics.llm.avg_tokens_per_second > 20 ? 'High' : 'Moderate'}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <div className="text-center py-12">
              <BarChart3 className="w-12 h-12 text-gray-300 mx-auto mb-4" />
              <h3 className="text-lg font-medium text-gray-900 mb-2">No Metrics Available</h3>
              <p className="text-gray-500">Start a conversation to see performance data.</p>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="bg-gray-50 px-6 py-3 border-t border-gray-200">
          <div className="flex items-center justify-between text-sm text-gray-600">
            <span>Real-time performance monitoring</span>
            <button
              onClick={fetchMetrics}
              disabled={isLoading}
              className="text-primary-600 hover:text-primary-800 font-medium"
            >
              {isLoading ? 'Refreshing...' : 'Refresh'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}