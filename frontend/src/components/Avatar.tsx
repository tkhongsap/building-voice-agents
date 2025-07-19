'use client';

import { useEffect, useRef, useState } from 'react';
import { cn } from '@/lib/utils';
import type { AvatarState, VoiceActivityState } from '@/types/voice-agent';

interface AvatarProps {
  voiceActivity: VoiceActivityState;
  className?: string;
  size?: 'sm' | 'md' | 'lg' | 'xl';
}

export default function Avatar({ voiceActivity, className, size = 'lg' }: AvatarProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number | null>(null);
  const [avatarState, setAvatarState] = useState<AvatarState>({
    isAnimating: false,
    expression: 'neutral',
    mood: 'neutral'
  });

  const sizeClasses = {
    sm: 'w-16 h-16',
    md: 'w-24 h-24',
    lg: 'w-32 h-32',
    xl: 'w-48 h-48'
  };

  // Update avatar state based on voice activity
  useEffect(() => {
    if (voiceActivity.isSpeaking) {
      setAvatarState(prev => ({
        ...prev,
        expression: 'speaking',
        isAnimating: true,
        mood: 'happy'
      }));
    } else if (voiceActivity.isListening) {
      setAvatarState(prev => ({
        ...prev,
        expression: 'listening',
        isAnimating: true,
        mood: 'focused'
      }));
    } else if (voiceActivity.isProcessing) {
      setAvatarState(prev => ({
        ...prev,
        expression: 'thinking',
        isAnimating: true,
        mood: 'focused'
      }));
    } else {
      setAvatarState(prev => ({
        ...prev,
        expression: 'neutral',
        isAnimating: false,
        mood: 'neutral'
      }));
    }
  }, [voiceActivity]);

  // Draw avatar on canvas
  const drawAvatar = (ctx: CanvasRenderingContext2D, canvas: HTMLCanvasElement, timestamp: number) => {
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const baseRadius = Math.min(canvas.width, canvas.height) / 3;
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Background circle with gradient
    const gradient = ctx.createRadialGradient(centerX, centerY, 0, centerX, centerY, baseRadius * 1.5);
    
    if (avatarState.mood === 'happy') {
      gradient.addColorStop(0, 'rgba(34, 197, 94, 0.2)');
      gradient.addColorStop(1, 'rgba(34, 197, 94, 0.05)');
    } else if (avatarState.mood === 'focused') {
      gradient.addColorStop(0, 'rgba(59, 130, 246, 0.2)');
      gradient.addColorStop(1, 'rgba(59, 130, 246, 0.05)');
    } else {
      gradient.addColorStop(0, 'rgba(107, 114, 128, 0.2)');
      gradient.addColorStop(1, 'rgba(107, 114, 128, 0.05)');
    }
    
    ctx.fillStyle = gradient;
    ctx.beginPath();
    ctx.arc(centerX, centerY, baseRadius * 1.2, 0, Math.PI * 2);
    ctx.fill();
    
    // Main avatar circle
    const pulseEffect = avatarState.isAnimating ? Math.sin(timestamp * 0.005) * 0.1 + 1 : 1;
    const currentRadius = baseRadius * pulseEffect;
    
    const mainGradient = ctx.createRadialGradient(centerX, centerY - 10, 0, centerX, centerY, currentRadius);
    
    if (avatarState.mood === 'happy') {
      mainGradient.addColorStop(0, '#10b981');
      mainGradient.addColorStop(1, '#059669');
    } else if (avatarState.mood === 'focused') {
      mainGradient.addColorStop(0, '#3b82f6');
      mainGradient.addColorStop(1, '#2563eb');
    } else {
      mainGradient.addColorStop(0, '#6b7280');
      mainGradient.addColorStop(1, '#4b5563');
    }
    
    ctx.fillStyle = mainGradient;
    ctx.beginPath();
    ctx.arc(centerX, centerY, currentRadius, 0, Math.PI * 2);
    ctx.fill();
    
    // Eyes
    const eyeY = centerY - baseRadius * 0.2;
    const eyeRadius = baseRadius * 0.08;
    
    // Blinking animation
    const blinkPhase = Math.sin(timestamp * 0.003) > 0.98 ? 0.2 : 1;
    
    ctx.fillStyle = 'white';
    // Left eye
    ctx.beginPath();
    ctx.ellipse(centerX - baseRadius * 0.25, eyeY, eyeRadius, eyeRadius * blinkPhase, 0, 0, Math.PI * 2);
    ctx.fill();
    // Right eye
    ctx.beginPath();
    ctx.ellipse(centerX + baseRadius * 0.25, eyeY, eyeRadius, eyeRadius * blinkPhase, 0, 0, Math.PI * 2);
    ctx.fill();
    
    // Pupils
    if (blinkPhase > 0.5) {
      ctx.fillStyle = '#1f2937';
      const pupilRadius = eyeRadius * 0.6;
      
      // Left pupil
      ctx.beginPath();
      ctx.arc(centerX - baseRadius * 0.25, eyeY, pupilRadius, 0, Math.PI * 2);
      ctx.fill();
      // Right pupil
      ctx.beginPath();
      ctx.arc(centerX + baseRadius * 0.25, eyeY, pupilRadius, 0, Math.PI * 2);
      ctx.fill();
    }
    
    // Mouth based on expression
    const mouthY = centerY + baseRadius * 0.3;
    ctx.strokeStyle = 'white';
    ctx.lineWidth = baseRadius * 0.05;
    ctx.lineCap = 'round';
    
    ctx.beginPath();
    
    if (avatarState.expression === 'speaking') {
      // Animated speaking mouth
      const speakingAnimation = Math.sin(timestamp * 0.02) * 0.3 + 0.7;
      const mouthWidth = baseRadius * 0.3 * speakingAnimation;
      const mouthHeight = baseRadius * 0.15 * speakingAnimation;
      
      ctx.ellipse(centerX, mouthY, mouthWidth, mouthHeight, 0, 0, Math.PI * 2);
    } else if (avatarState.expression === 'listening') {
      // Small open circle
      ctx.arc(centerX, mouthY, baseRadius * 0.08, 0, Math.PI * 2);
    } else if (avatarState.expression === 'thinking') {
      // Slight curve
      ctx.arc(centerX, mouthY + baseRadius * 0.1, baseRadius * 0.2, 0.2 * Math.PI, 0.8 * Math.PI);
    } else {
      // Neutral smile
      ctx.arc(centerX, mouthY, baseRadius * 0.25, 0.2 * Math.PI, 0.8 * Math.PI);
    }
    
    ctx.stroke();
    
    // Voice activity visualization
    if (voiceActivity.isSpeaking && voiceActivity.volume > 0.01) {
      const waveCount = 3;
      const baseWaveRadius = currentRadius + 20;
      
      for (let i = 0; i < waveCount; i++) {
        const waveRadius = baseWaveRadius + (i * 15);
        const alpha = (1 - i / waveCount) * voiceActivity.volume * 0.5;
        
        ctx.strokeStyle = `rgba(34, 197, 94, ${alpha})`;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(centerX, centerY, waveRadius, 0, Math.PI * 2);
        ctx.stroke();
      }
    }
  };

  // Animation loop
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas size
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    canvas.style.width = rect.width + 'px';
    canvas.style.height = rect.height + 'px';
    
    ctx.scale(dpr, dpr);

    const animate = (timestamp: number) => {
      drawAvatar(ctx, canvas, timestamp);
      animationRef.current = requestAnimationFrame(animate);
    };

    animationRef.current = requestAnimationFrame(animate);

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [avatarState, voiceActivity]);

  return (
    <div className={cn(
      'relative flex items-center justify-center',
      sizeClasses[size],
      className
    )}>
      <canvas
        ref={canvasRef}
        className={cn(
          'rounded-full',
          sizeClasses[size]
        )}
      />
      
      {/* Status indicator */}
      <div className={cn(
        'absolute bottom-1 right-1 w-3 h-3 rounded-full border-2 border-white',
        {
          'bg-green-500': voiceActivity.isSpeaking,
          'bg-blue-500': voiceActivity.isListening,
          'bg-yellow-500 animate-pulse': voiceActivity.isProcessing,
          'bg-gray-400': !voiceActivity.isSpeaking && !voiceActivity.isListening && !voiceActivity.isProcessing
        }
      )} />
    </div>
  );
}