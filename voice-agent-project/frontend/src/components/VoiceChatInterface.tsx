'use client';

import { useState } from 'react';
import { Mic, MicOff, Phone, PhoneOff, Settings, Trash2, Volume2, BarChart3 } from 'lucide-react';
import { cn, formatTime } from '@/lib/utils';
import { useVoiceAgent } from '@/hooks/useVoiceAgent';
import Avatar from './Avatar';
import MetricsDashboard from './MetricsDashboard';
import type { ConversationMessage } from '@/types/voice-agent';

interface VoiceChatInterfaceProps {
  className?: string;
}

export default function VoiceChatInterface({ className }: VoiceChatInterfaceProps) {
  const {
    connectionState,
    voiceActivity,
    conversation,
    isConnecting,
    error,
    connect,
    disconnect,
    toggleMicrophone,
    clearConversation,
    isConnected,
    isMicrophoneEnabled
  } = useVoiceAgent();

  const [showSettings, setShowSettings] = useState(false);
  const [showMetrics, setShowMetrics] = useState(false);

  const handleConnect = async () => {
    await connect({
      participant_name: 'User',
      voice_instructions: 'Be helpful and conversational. Keep responses concise for voice chat.'
    });
  };

  const MessageBubble = ({ message }: { message: ConversationMessage }) => (
    <div className={cn(
      'flex mb-4',
      {
        'justify-end': message.type === 'user',
        'justify-start': message.type === 'assistant',
        'justify-center': message.type === 'system'
      }
    )}>
      <div className={cn(
        'max-w-xs lg:max-w-md px-4 py-2 rounded-lg text-sm',
        {
          'bg-primary-600 text-white': message.type === 'user',
          'bg-white text-gray-800 border border-gray-200': message.type === 'assistant',
          'bg-gray-100 text-gray-600 text-xs': message.type === 'system'
        }
      )}>
        <p>{message.content}</p>
        <p className={cn(
          'text-xs mt-1 opacity-70',
          {
            'text-primary-200': message.type === 'user',
            'text-gray-500': message.type !== 'user'
          }
        )}>
          {formatTime(message.timestamp)}
        </p>
      </div>
    </div>
  );

  return (
    <div className={cn('flex flex-col h-full max-w-4xl mx-auto', className)}>
      {/* Header */}
      <div className="bg-white shadow-sm border-b border-gray-200 px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 bg-primary-600 rounded-full flex items-center justify-center">
              <Volume2 className="w-4 h-4 text-white" />
            </div>
            <div>
              <h1 className="text-lg font-semibold text-gray-900">Voice Agent</h1>
              <p className="text-sm text-gray-500">
                {isConnected ? 'Connected' : 'Disconnected'}
                {connectionState.session && ` • Room: ${connectionState.session.room_name}`}
              </p>
            </div>
          </div>
          
          <div className="flex items-center space-x-2">
            <button
              onClick={() => setShowMetrics(true)}
              className="p-2 text-gray-400 hover:text-gray-600 transition-colors"
              title="Performance metrics"
            >
              <BarChart3 className="w-5 h-5" />
            </button>
            
            <button
              onClick={() => setShowSettings(!showSettings)}
              className="p-2 text-gray-400 hover:text-gray-600 transition-colors"
            >
              <Settings className="w-5 h-5" />
            </button>
            
            {conversation.length > 0 && (
              <button
                onClick={clearConversation}
                className="p-2 text-gray-400 hover:text-red-600 transition-colors"
                title="Clear conversation"
              >
                <Trash2 className="w-5 h-5" />
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex">
        {/* Avatar Section */}
        <div className="w-80 bg-gradient-to-br from-primary-50 to-primary-100 flex flex-col items-center justify-center p-8 border-r border-gray-200">
          <Avatar 
            voiceActivity={voiceActivity}
            size="xl"
            className="mb-6"
          />
          
          <div className="text-center space-y-2">
            <h2 className="text-xl font-semibold text-gray-800">AI Assistant</h2>
            <p className="text-sm text-gray-600">
              {voiceActivity.isSpeaking ? 'Speaking...' :
               voiceActivity.isListening ? 'Listening...' :
               voiceActivity.isProcessing ? 'Thinking...' :
               isConnected ? 'Ready to chat' : 'Not connected'}
            </p>
            
            {/* Voice Activity Indicator */}
            {voiceActivity.volume > 0 && (
              <div className="flex items-center justify-center space-x-1">
                {[...Array(5)].map((_, i) => (
                  <div
                    key={i}
                    className={cn(
                      'w-1 bg-primary-500 rounded-full transition-all duration-150',
                      {
                        'h-2': voiceActivity.volume * 5 > i,
                        'h-1 opacity-30': voiceActivity.volume * 5 <= i
                      }
                    )}
                  />
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Chat Section */}
        <div className="flex-1 flex flex-col">
          {/* Conversation */}
          <div className="flex-1 overflow-y-auto p-6 bg-gray-50">
            {conversation.length === 0 ? (
              <div className="flex items-center justify-center h-full text-center">
                <div>
                  <div className="w-16 h-16 bg-gray-200 rounded-full flex items-center justify-center mx-auto mb-4">
                    <Mic className="w-8 h-8 text-gray-400" />
                  </div>
                  <h3 className="text-lg font-medium text-gray-900 mb-2">
                    Ready to start chatting?
                  </h3>
                  <p className="text-gray-500 max-w-md">
                    Connect to the voice agent and start speaking. The AI will listen and respond naturally.
                  </p>
                </div>
              </div>
            ) : (
              <div className="space-y-4">
                {conversation.map((message) => (
                  <MessageBubble key={message.id} message={message} />
                ))}
              </div>
            )}
          </div>

          {/* Controls */}
          <div className="bg-white border-t border-gray-200 p-6">
            {error && (
              <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg">
                <p className="text-sm text-red-600">{error}</p>
              </div>
            )}

            <div className="flex items-center justify-center space-x-4">
              {!isConnected ? (
                <button
                  onClick={handleConnect}
                  disabled={isConnecting}
                  className={cn(
                    'flex items-center space-x-2 px-6 py-3 rounded-full font-medium transition-all',
                    isConnecting
                      ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                      : 'bg-primary-600 text-white hover:bg-primary-700 shadow-lg hover:shadow-xl'
                  )}
                >
                  <Phone className="w-5 h-5" />
                  <span>{isConnecting ? 'Connecting...' : 'Connect'}</span>
                </button>
              ) : (
                <>
                  <button
                    onClick={toggleMicrophone}
                    className={cn(
                      'p-4 rounded-full transition-all shadow-lg',
                      isMicrophoneEnabled
                        ? 'bg-primary-600 text-white hover:bg-primary-700'
                        : 'bg-red-600 text-white hover:bg-red-700'
                    )}
                    title={isMicrophoneEnabled ? 'Mute microphone' : 'Unmute microphone'}
                  >
                    {isMicrophoneEnabled ? (
                      <Mic className="w-6 h-6" />
                    ) : (
                      <MicOff className="w-6 h-6" />
                    )}
                  </button>

                  <button
                    onClick={disconnect}
                    className="p-4 bg-red-600 text-white rounded-full hover:bg-red-700 transition-all shadow-lg"
                    title="Disconnect"
                  >
                    <PhoneOff className="w-6 h-6" />
                  </button>
                </>
              )}
            </div>

            {isConnected && (
              <div className="mt-4 text-center">
                <p className="text-sm text-gray-500">
                  {isMicrophoneEnabled 
                    ? 'Microphone is on • Start speaking'
                    : 'Microphone is muted'
                  }
                </p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Metrics Dashboard */}
      <MetricsDashboard
        isVisible={showMetrics}
        onClose={() => setShowMetrics(false)}
      />
    </div>
  );
}