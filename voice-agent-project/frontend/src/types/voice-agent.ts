export interface SessionRequest {
  participant_name?: string;
  room_name?: string;
  voice_instructions?: string;
}

export interface SessionResponse {
  session_id: string;
  room_name: string;
  token: string;
  livekit_url: string;
  participant_identity: string;
}

export interface SessionStatus {
  session_id: string;
  room_name: string;
  participant_identity: string;
  status: string;
  created_at: string;
  is_active: boolean;
}

export interface VoiceConfig {
  livekit_url: string;
  available_voices: Record<string, string>;
  available_models: string[];
}

export interface ConnectionState {
  status: 'disconnected' | 'connecting' | 'connected' | 'error';
  room?: any;
  session?: SessionResponse;
  error?: string;
}

export interface ConversationMessage {
  id: string;
  type: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
}

export interface VoiceActivityState {
  isSpeaking: boolean;
  isListening: boolean;
  isProcessing: boolean;
  volume: number;
}

export interface AvatarState {
  isAnimating: boolean;
  expression: 'neutral' | 'speaking' | 'listening' | 'thinking';
  mood: 'happy' | 'neutral' | 'focused';
}