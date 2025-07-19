import { useState, useEffect, useCallback, useRef } from 'react';
import { Room, RoomEvent, AudioTrack, Participant } from 'livekit-client';
import { VoiceAgentAPI } from '@/lib/api';
import type {
  ConnectionState,
  SessionRequest,
  VoiceActivityState,
  ConversationMessage
} from '@/types/voice-agent';
import { generateId } from '@/lib/utils';

export function useVoiceAgent() {
  const [connectionState, setConnectionState] = useState<ConnectionState>({
    status: 'disconnected'
  });
  
  const [voiceActivity, setVoiceActivity] = useState<VoiceActivityState>({
    isSpeaking: false,
    isListening: false,
    isProcessing: false,
    volume: 0
  });
  
  const [conversation, setConversation] = useState<ConversationMessage[]>([]);
  const [isConnecting, setIsConnecting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  const roomRef = useRef<Room | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const animationFrameRef = useRef<number | null>(null);

  // Add message to conversation
  const addMessage = useCallback((type: 'user' | 'assistant' | 'system', content: string) => {
    const message: ConversationMessage = {
      id: generateId(),
      type,
      content,
      timestamp: new Date()
    };
    
    setConversation(prev => [...prev, message]);
  }, []);

  // Monitor audio levels for visual feedback
  const monitorAudioLevels = useCallback(() => {
    if (!analyserRef.current) return;
    
    const analyser = analyserRef.current;
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    
    const updateLevels = () => {
      analyser.getByteFrequencyData(dataArray);
      
      // Calculate average volume
      const average = dataArray.reduce((sum, value) => sum + value, 0) / bufferLength;
      const volume = average / 255;
      
      setVoiceActivity(prev => ({
        ...prev,
        volume
      }));
      
      animationFrameRef.current = requestAnimationFrame(updateLevels);
    };
    
    updateLevels();
  }, []);

  // Set up audio monitoring
  const setupAudioMonitoring = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioContextRef.current = new AudioContext();
      analyserRef.current = audioContextRef.current.createAnalyser();
      
      const source = audioContextRef.current.createMediaStreamSource(stream);
      source.connect(analyserRef.current);
      
      analyserRef.current.fftSize = 256;
      monitorAudioLevels();
    } catch (error) {
      console.error('Failed to set up audio monitoring:', error);
    }
  }, [monitorAudioLevels]);

  // Connect to voice agent
  const connect = useCallback(async (sessionRequest?: SessionRequest) => {
    if (isConnecting || connectionState.status === 'connected') {
      return;
    }

    setIsConnecting(true);
    setError(null);
    setConnectionState(prev => ({ ...prev, status: 'connecting' }));

    try {
      // Create session
      const session = await VoiceAgentAPI.createSession(
        sessionRequest || { participant_name: 'User' }
      );

      // Create LiveKit room
      const room = new Room();
      roomRef.current = room;

      // Set up room event handlers
      room.on(RoomEvent.Connected, () => {
        console.log('Connected to room');
        setConnectionState({
          status: 'connected',
          room,
          session
        });
        addMessage('system', 'Connected to voice agent');
      });

      room.on(RoomEvent.Disconnected, () => {
        console.log('Disconnected from room');
        setConnectionState(prev => ({ ...prev, status: 'disconnected' }));
        addMessage('system', 'Disconnected from voice agent');
      });

      room.on(RoomEvent.TrackSubscribed, (track, publication, participant) => {
        if (track.kind === 'audio' && participant.isAgent) {
          console.log('Agent audio track subscribed');
          setVoiceActivity(prev => ({ ...prev, isSpeaking: true }));
        }
      });

      room.on(RoomEvent.TrackUnsubscribed, (track, publication, participant) => {
        if (track.kind === 'audio' && participant.isAgent) {
          console.log('Agent audio track unsubscribed');
          setVoiceActivity(prev => ({ ...prev, isSpeaking: false }));
        }
      });

      room.on(RoomEvent.AudioPlaybackStatusChanged, () => {
        const isPlaying = room.canPlaybackAudio;
        setVoiceActivity(prev => ({ ...prev, isSpeaking: isPlaying }));
      });

      // Connect to room
      await room.connect(session.livekit_url, session.token);
      
      // Enable microphone
      await room.localParticipant.enableCameraAndMicrophone(false, true);
      
      // Set up audio monitoring
      await setupAudioMonitoring();

    } catch (error) {
      console.error('Failed to connect:', error);
      const errorMessage = error instanceof Error ? error.message : 'Failed to connect to voice agent';
      setError(errorMessage);
      setConnectionState(prev => ({ 
        ...prev, 
        status: 'error',
        error: errorMessage 
      }));
    } finally {
      setIsConnecting(false);
    }
  }, [isConnecting, connectionState.status, addMessage, setupAudioMonitoring]);

  // Disconnect from voice agent
  const disconnect = useCallback(async () => {
    try {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      
      if (audioContextRef.current) {
        await audioContextRef.current.close();
        audioContextRef.current = null;
      }
      
      if (roomRef.current) {
        await roomRef.current.disconnect();
        roomRef.current = null;
      }
      
      if (connectionState.session) {
        await VoiceAgentAPI.endSession(connectionState.session.session_id);
      }

      setConnectionState({ status: 'disconnected' });
      setVoiceActivity({
        isSpeaking: false,
        isListening: false,
        isProcessing: false,
        volume: 0
      });
      
      addMessage('system', 'Disconnected from voice agent');
    } catch (error) {
      console.error('Error during disconnect:', error);
    }
  }, [connectionState.session, addMessage]);

  // Toggle microphone
  const toggleMicrophone = useCallback(async () => {
    if (!roomRef.current) return;

    const isEnabled = roomRef.current.localParticipant.isMicrophoneEnabled;
    await roomRef.current.localParticipant.setMicrophoneEnabled(!isEnabled);
    
    setVoiceActivity(prev => ({ ...prev, isListening: !isEnabled }));
  }, []);

  // Clear conversation
  const clearConversation = useCallback(() => {
    setConversation([]);
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      
      if (audioContextRef.current) {
        audioContextRef.current.close();
      }
      
      if (roomRef.current) {
        roomRef.current.disconnect();
      }
    };
  }, []);

  return {
    connectionState,
    voiceActivity,
    conversation,
    isConnecting,
    error,
    connect,
    disconnect,
    toggleMicrophone,
    clearConversation,
    isConnected: connectionState.status === 'connected',
    isMicrophoneEnabled: roomRef.current?.localParticipant.isMicrophoneEnabled ?? false
  };
}