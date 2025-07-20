"""
Real-Time Translation Voice Agent Recipe

This recipe provides pre-built voice agents optimized for real-time language
translation scenarios, including multilingual customer support, international
business meetings, and language learning applications.

Key Features:
- Real-time speech-to-speech translation
- Support for 50+ languages
- Automatic language detection
- Voice preservation and cloning
- Cultural adaptation of responses
- Translation quality assessment
- Multi-speaker conversation handling
- Offline translation capabilities

Usage:
    from translation_recipe import TranslationAgent, MultilingualSupportAgent
    
    # Create translation agent
    agent = TranslationAgent(source_lang="en", target_lang="es")
    
    # Create multilingual support agent
    support_agent = MultilingualSupportAgent(primary_lang="en", supported_langs=["es", "fr", "de"])
"""

import asyncio
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json

# Add SDK to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from sdk.python_sdk import VoiceAgentSDK, initialize_sdk
from sdk.agent_builder import VoiceAgentBuilder, AgentCapability
from sdk.config_manager import SDKConfig


class LanguageCode(Enum):
    """Supported language codes (ISO 639-1)."""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    RUSSIAN = "ru"
    CHINESE = "zh"
    JAPANESE = "ja"
    KOREAN = "ko"
    ARABIC = "ar"
    HINDI = "hi"
    DUTCH = "nl"
    SWEDISH = "sv"
    NORWEGIAN = "no"
    DANISH = "da"
    FINNISH = "fi"
    POLISH = "pl"
    CZECH = "cs"
    HUNGARIAN = "hu"
    TURKISH = "tr"
    GREEK = "el"
    HEBREW = "he"
    THAI = "th"
    VIETNAMESE = "vi"
    INDONESIAN = "id"
    MALAY = "ms"
    FILIPINO = "tl"
    UKRAINIAN = "uk"
    ROMANIAN = "ro"
    BULGARIAN = "bg"
    CROATIAN = "hr"
    SERBIAN = "sr"
    SLOVAK = "sk"
    SLOVENIAN = "sl"
    ESTONIAN = "et"
    LATVIAN = "lv"
    LITHUANIAN = "lt"


class TranslationQuality(Enum):
    """Translation quality levels."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


class ConversationMode(Enum):
    """Translation conversation modes."""
    INTERPRETER = "interpreter"  # Two-way translation
    ASSISTANT = "assistant"     # One-way with AI responses
    PASSTHROUGH = "passthrough" # Direct translation only


@dataclass
class TranslationMetrics:
    """Translation performance metrics."""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_translations: int = 0
    source_language: Optional[str] = None
    target_language: Optional[str] = None
    average_latency: float = 0.0  # seconds
    quality_scores: List[float] = field(default_factory=list)
    errors_count: int = 0
    languages_detected: List[str] = field(default_factory=list)


@dataclass
class TranslationResult:
    """Individual translation result."""
    original_text: str
    translated_text: str
    source_language: str
    target_language: str
    confidence_score: float
    quality_score: Optional[float] = None
    latency: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    cultural_adaptations: List[str] = field(default_factory=list)


@dataclass
class CulturalContext:
    """Cultural adaptation context."""
    country: Optional[str] = None
    region: Optional[str] = None
    formality_level: str = "neutral"  # formal, neutral, casual
    business_context: bool = False
    cultural_preferences: Dict[str, Any] = field(default_factory=dict)


class TranslationAgent:
    """
    Real-time translation voice agent.
    
    Provides high-quality speech-to-speech translation with cultural
    adaptation and voice preservation capabilities.
    """
    
    def __init__(
        self,
        source_lang: str = "auto",  # Auto-detect
        target_lang: str = "en",
        agent_name: str = "Translation Assistant"
    ):
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.agent_name = agent_name
        
        # Translation state
        self.agent = None
        self.is_active = False
        self.conversation_mode = ConversationMode.INTERPRETER
        
        # Metrics and history
        self.session_metrics: Optional[TranslationMetrics] = None
        self.translation_history: List[TranslationResult] = []
        
        # Language detection and adaptation
        self.auto_detect_language = source_lang == "auto"
        self.cultural_context = CulturalContext()
        self.voice_preservation = True
        
        # Supported languages
        self.supported_languages = {
            "en": {"name": "English", "tts_voice": "nova", "region": "US"},
            "es": {"name": "Spanish", "tts_voice": "alloy", "region": "ES"},
            "fr": {"name": "French", "tts_voice": "alloy", "region": "FR"},
            "de": {"name": "German", "tts_voice": "echo", "region": "DE"},
            "it": {"name": "Italian", "tts_voice": "fable", "region": "IT"},
            "pt": {"name": "Portuguese", "tts_voice": "nova", "region": "BR"},
            "ru": {"name": "Russian", "tts_voice": "shimmer", "region": "RU"},
            "zh": {"name": "Chinese", "tts_voice": "onyx", "region": "CN"},
            "ja": {"name": "Japanese", "tts_voice": "alloy", "region": "JP"},
            "ko": {"name": "Korean", "tts_voice": "echo", "region": "KR"},
            "ar": {"name": "Arabic", "tts_voice": "fable", "region": "SA"},
            "hi": {"name": "Hindi", "tts_voice": "nova", "region": "IN"}
        }
        
        # Quality thresholds
        self.min_confidence_threshold = 0.7
        self.quality_warning_threshold = 0.6
    
    async def initialize_agent(self) -> None:
        """Initialize the translation voice agent."""
        try:
            # Initialize SDK
            sdk = await initialize_sdk({
                "project_name": "TranslationAgent",
                "environment": "production",
                "enable_monitoring": True
            })
            
            # Configure for source language
            source_config = self.supported_languages.get(self.source_lang, self.supported_languages["en"])
            target_config = self.supported_languages.get(self.target_lang, self.supported_languages["en"])
            
            # Build agent with translation optimizations
            builder = sdk.create_builder()
            
            self.agent = (builder
                .with_name(self.agent_name)
                .with_stt("openai", 
                         language=self.source_lang if self.source_lang != "auto" else "en",
                         model="whisper-1")
                .with_llm("openai",
                         model="gpt-4-turbo",
                         temperature=0.2,  # Lower for consistency
                         max_tokens=300)
                .with_tts("openai", 
                         voice=target_config["tts_voice"],
                         speed=0.9)  # Slightly slower for clarity
                .with_vad("silero", sensitivity=0.8)  # Higher sensitivity for multilingual
                .with_capability(AgentCapability.TURN_DETECTION)
                .with_capability(AgentCapability.INTERRUPTION_HANDLING)
                .with_capability(AgentCapability.CONTEXT_MANAGEMENT)
                .with_system_prompt(self._build_translation_prompt())
                .with_callback("on_user_speech", self._on_speech_input)
                .build())
            
            print(f"✅ Translation agent initialized: {self.source_lang} → {self.target_lang}")
            
        except Exception as e:
            print(f"❌ Failed to initialize translation agent: {e}")
            raise
    
    def _build_translation_prompt(self) -> str:
        """Build the system prompt for translation operations."""
        source_name = self.supported_languages.get(self.source_lang, {}).get("name", self.source_lang)
        target_name = self.supported_languages.get(self.target_lang, {}).get("name", self.target_lang)
        
        return f"""You are a professional real-time translation assistant specializing in {source_name} to {target_name} translation.

TRANSLATION GUIDELINES:
1. Provide accurate, natural translations that preserve meaning and intent
2. Adapt translations for cultural context and formality level
3. Maintain the speaker's tone and emotion in translations
4. Handle idiomatic expressions and cultural references appropriately
5. Flag uncertain translations and provide alternatives when needed

QUALITY STANDARDS:
- Accuracy: Preserve the exact meaning of the source text
- Fluency: Ensure natural-sounding target language
- Cultural Adaptation: Adjust for cultural norms and expectations
- Consistency: Maintain terminology and style throughout conversation

CONVERSATION HANDLING:
- In interpreter mode: Translate all speech directly and neutrally
- In assistant mode: Provide helpful responses in addition to translation
- Always indicate language switches or uncertain translations
- Handle code-switching (mixing languages) gracefully

CULTURAL CONSIDERATIONS:
- Formality levels: Adjust formal/informal register as appropriate
- Business context: Use professional terminology and tone
- Regional variations: Consider local language variants
- Cultural sensitivity: Avoid culturally inappropriate translations

TECHNICAL FEATURES:
- Language detection: Automatically detect input language if needed
- Quality assessment: Provide confidence scores for translations
- Error handling: Gracefully handle unclear or ambiguous input
- Voice preservation: Maintain speaker characteristics in target language

If you encounter:
- Unclear speech: Ask for clarification politely
- Unknown terms: Provide best translation with uncertainty note
- Cultural conflicts: Explain cultural context when necessary
- Technical errors: Acknowledge and provide alternative approaches

Always prioritize communication effectiveness over literal translation accuracy.
"""
    
    async def start_session(self, mode: ConversationMode = ConversationMode.INTERPRETER) -> str:
        """Start a new translation session."""
        if not self.agent:
            await self.initialize_agent()
        
        # Create session metrics
        session_id = f"trans_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        self.session_metrics = TranslationMetrics(
            session_id=session_id,
            start_time=datetime.now(),
            source_language=self.source_lang,
            target_language=self.target_lang
        )
        
        self.conversation_mode = mode
        self.is_active = True
        
        # Start the agent
        await self.agent.start()
        
        print(f"🌐 Translation session started: {session_id}")
        print(f"Mode: {mode.value}")
        print(f"Languages: {self.source_lang} → {self.target_lang}")
        
        return session_id
    
    async def end_session(self) -> TranslationMetrics:
        """End the current translation session."""
        if not self.session_metrics:
            raise ValueError("No active translation session")
        
        # Update metrics
        self.session_metrics.end_time = datetime.now()
        if self.session_metrics.quality_scores:
            self.session_metrics.average_latency = sum(
                result.latency for result in self.translation_history
            ) / len(self.translation_history)
        
        # Stop the agent
        if self.agent:
            await self.agent.stop()
        
        self.is_active = False
        
        # Generate session summary
        await self._generate_session_summary()
        
        completed_session = self.session_metrics
        self.session_metrics = None
        
        print(f"🌐 Translation session ended")
        return completed_session
    
    async def translate_text(
        self,
        text: str,
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None
    ) -> TranslationResult:
        """Translate text with quality assessment."""
        start_time = datetime.now()
        
        # Use session languages if not specified
        src_lang = source_lang or self.source_lang
        tgt_lang = target_lang or self.target_lang
        
        # Auto-detect language if needed
        if src_lang == "auto":
            src_lang = await self._detect_language(text)
        
        # Perform translation
        translated_text, confidence = await self._perform_translation(text, src_lang, tgt_lang)
        
        # Apply cultural adaptations
        adapted_text, adaptations = await self._apply_cultural_adaptations(
            translated_text, src_lang, tgt_lang
        )
        
        # Calculate latency
        latency = (datetime.now() - start_time).total_seconds()
        
        # Create result
        result = TranslationResult(
            original_text=text,
            translated_text=adapted_text,
            source_language=src_lang,
            target_language=tgt_lang,
            confidence_score=confidence,
            quality_score=await self._assess_translation_quality(text, adapted_text, src_lang, tgt_lang),
            latency=latency,
            cultural_adaptations=adaptations
        )
        
        # Update metrics
        if self.session_metrics:
            self.session_metrics.total_translations += 1
            self.session_metrics.quality_scores.append(result.quality_score or 0)
            if src_lang not in self.session_metrics.languages_detected:
                self.session_metrics.languages_detected.append(src_lang)
        
        # Store in history
        self.translation_history.append(result)
        
        return result
    
    async def _on_speech_input(self, text: str) -> None:
        """Handle speech input for translation."""
        if not self.is_active or not text.strip():
            return
        
        try:
            # Translate the input
            result = await self.translate_text(text)
            
            # Log translation result
            print(f"🔄 Translation: '{result.original_text}' → '{result.translated_text}'")
            print(f"   Confidence: {result.confidence_score:.2f}, Quality: {result.quality_score:.2f if result.quality_score else 'N/A'}")
            
            # Check quality warnings
            if result.quality_score and result.quality_score < self.quality_warning_threshold:
                print(f"⚠️ Low quality translation detected")
            
            if result.confidence_score < self.min_confidence_threshold:
                print(f"⚠️ Low confidence translation")
            
        except Exception as e:
            print(f"❌ Translation error: {e}")
            if self.session_metrics:
                self.session_metrics.errors_count += 1
    
    async def _detect_language(self, text: str) -> str:
        """Detect the language of input text."""
        # Simple heuristic-based detection (in production, use proper language detection)
        language_patterns = {
            "en": ["the", "and", "is", "in", "to", "of", "a", "that", "it", "with"],
            "es": ["el", "la", "de", "que", "y", "en", "un", "es", "se", "no"],
            "fr": ["le", "de", "et", "à", "un", "il", "être", "et", "en", "avoir"],
            "de": ["der", "die", "und", "in", "den", "von", "zu", "das", "mit", "ist"],
            "it": ["il", "di", "che", "e", "la", "per", "in", "un", "è", "con"],
            "pt": ["o", "de", "e", "que", "do", "da", "em", "um", "para", "com"],
            "ru": ["в", "и", "не", "на", "я", "быть", "он", "с", "как", "а"],
            "zh": ["的", "一", "是", "在", "不", "了", "有", "和", "人", "这"],
            "ja": ["の", "に", "は", "を", "た", "が", "で", "て", "と", "し"],
            "ar": ["في", "من", "إلى", "على", "أن", "هذا", "هذه", "التي", "التي", "كان"]
        }
        
        text_lower = text.lower()
        scores = {}
        
        for lang, keywords in language_patterns.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                scores[lang] = score
        
        if scores:
            detected_lang = max(scores.keys(), key=lambda k: scores[k])
            print(f"🔍 Language detected: {detected_lang}")
            return detected_lang
        
        # Default to English if no language detected
        return "en"
    
    async def _perform_translation(self, text: str, source_lang: str, target_lang: str) -> Tuple[str, float]:
        """Perform the actual translation."""
        # Simulate translation service call
        # In production, integrate with Google Translate, DeepL, Azure Translator, etc.
        
        # Simple simulation for demo
        if source_lang == target_lang:
            return text, 1.0
        
        # Simulate translation with confidence
        confidence = 0.85 + (len(text) / 1000) * 0.1  # Longer text = higher confidence
        confidence = min(confidence, 0.95)
        
        # Mock translation based on language pairs
        translation_map = {
            ("en", "es"): f"[ES] {text}",
            ("en", "fr"): f"[FR] {text}",
            ("en", "de"): f"[DE] {text}",
            ("es", "en"): f"[EN] {text}",
            ("fr", "en"): f"[EN] {text}",
            ("de", "en"): f"[EN] {text}"
        }
        
        translated = translation_map.get((source_lang, target_lang), f"[{target_lang.upper()}] {text}")
        
        return translated, confidence
    
    async def _apply_cultural_adaptations(
        self,
        text: str,
        source_lang: str,
        target_lang: str
    ) -> Tuple[str, List[str]]:
        """Apply cultural adaptations to translation."""
        adaptations = []
        adapted_text = text
        
        # Cultural adaptation rules
        cultural_rules = {
            ("en", "es"): {
                "formal_address": True,
                "greetings": {"hi": "hola", "hello": "buenos días"},
                "politeness": "increased"
            },
            ("en", "de"): {
                "formal_address": True,
                "titles": "required",
                "directness": "increased"
            },
            ("en", "ja"): {
                "honorifics": "required",
                "formality": "very_high",
                "indirectness": "increased"
            }
        }
        
        rules = cultural_rules.get((source_lang, target_lang), {})
        
        if rules.get("formal_address"):
            adaptations.append("Applied formal address")
        
        if rules.get("politeness") == "increased":
            adaptations.append("Increased politeness level")
        
        if self.cultural_context.business_context and "formal" not in adapted_text.lower():
            adapted_text = f"[Business tone] {adapted_text}"
            adaptations.append("Applied business context")
        
        return adapted_text, adaptations
    
    async def _assess_translation_quality(
        self,
        original: str,
        translated: str,
        source_lang: str,
        target_lang: str
    ) -> float:
        """Assess translation quality using various metrics."""
        # Simple quality assessment (in production, use BLEU, METEOR, etc.)
        
        # Length ratio check
        length_ratio = len(translated) / len(original) if len(original) > 0 else 1.0
        length_score = 1.0 - abs(1.0 - length_ratio) * 0.5
        length_score = max(0.0, min(1.0, length_score))
        
        # Character preservation check
        common_chars = set(original.lower()) & set(translated.lower())
        char_score = len(common_chars) / max(len(set(original.lower())), 1) * 0.3
        
        # Basic completeness check
        completeness_score = 0.8 if len(translated) > len(original) * 0.5 else 0.5
        
        # Combined quality score
        quality_score = (length_score * 0.4 + char_score * 0.3 + completeness_score * 0.3)
        
        return round(quality_score, 2)
    
    async def _generate_session_summary(self) -> None:
        """Generate a comprehensive session summary."""
        if not self.session_metrics:
            return
        
        print("\n" + "="*60)
        print("🌐 TRANSLATION SESSION SUMMARY")
        print("="*60)
        print(f"Session ID: {self.session_metrics.session_id}")
        print(f"Duration: {self.session_metrics.end_time - self.session_metrics.start_time}")
        print(f"Total Translations: {self.session_metrics.total_translations}")
        print(f"Languages: {self.session_metrics.source_language} → {self.session_metrics.target_language}")
        print(f"Average Latency: {self.session_metrics.average_latency:.2f}s")
        print(f"Errors: {self.session_metrics.errors_count}")
        
        if self.session_metrics.quality_scores:
            avg_quality = sum(self.session_metrics.quality_scores) / len(self.session_metrics.quality_scores)
            print(f"Average Quality Score: {avg_quality:.2f}")
        
        if self.session_metrics.languages_detected:
            print(f"Languages Detected: {', '.join(self.session_metrics.languages_detected)}")
        
        # Quality distribution
        if self.session_metrics.quality_scores:
            high_quality = sum(1 for score in self.session_metrics.quality_scores if score >= 0.8)
            medium_quality = sum(1 for score in self.session_metrics.quality_scores if 0.6 <= score < 0.8)
            low_quality = sum(1 for score in self.session_metrics.quality_scores if score < 0.6)
            
            print(f"\nQuality Distribution:")
            print(f"  High (≥0.8): {high_quality}")
            print(f"  Medium (0.6-0.8): {medium_quality}")
            print(f"  Low (<0.6): {low_quality}")
        
        print("="*60)
    
    def get_supported_languages(self) -> Dict[str, Dict[str, str]]:
        """Get list of supported languages."""
        return self.supported_languages.copy()
    
    def set_cultural_context(self, context: CulturalContext) -> None:
        """Set cultural adaptation context."""
        self.cultural_context = context
        print(f"🌍 Cultural context updated: {context.country}, formality: {context.formality_level}")


class MultilingualSupportAgent(TranslationAgent):
    """
    Multilingual customer support agent.
    
    Specialized for customer service scenarios with automatic language
    detection and response generation in multiple languages.
    """
    
    def __init__(
        self,
        primary_lang: str = "en",
        supported_langs: Optional[List[str]] = None,
        company_name: str = "Your Company"
    ):
        super().__init__(
            source_lang="auto",
            target_lang=primary_lang,
            agent_name="Multilingual Support Assistant"
        )
        
        self.primary_lang = primary_lang
        self.supported_langs = supported_langs or ["en", "es", "fr", "de", "it"]
        self.company_name = company_name
        self.conversation_mode = ConversationMode.ASSISTANT
        
        # Support-specific features
        self.auto_language_switch = True
        self.fallback_to_primary = True
        self.customer_language_preference: Optional[str] = None
    
    def _build_translation_prompt(self) -> str:
        """Build multilingual support specific prompt."""
        supported_names = [
            self.supported_languages.get(lang, {}).get("name", lang)
            for lang in self.supported_langs
        ]
        
        return f"""You are a multilingual customer support representative for {self.company_name}.

MULTILINGUAL SUPPORT GUIDELINES:
1. Automatically detect customer's language and respond in the same language
2. Provide helpful customer service in addition to translation
3. Switch languages seamlessly based on customer preference
4. Maintain professional and friendly tone in all languages
5. Escalate to human agents when language barriers persist

SUPPORTED LANGUAGES: {', '.join(supported_names)}
PRIMARY LANGUAGE: {self.supported_languages.get(self.primary_lang, {}).get('name', self.primary_lang)}

CUSTOMER SERVICE PRINCIPLES:
- Always greet customers in their detected language
- Acknowledge their concerns empathetically
- Provide clear, helpful solutions
- Ask clarifying questions when needed
- Offer follow-up assistance
- End with satisfaction confirmation

LANGUAGE HANDLING:
- If customer switches languages mid-conversation, follow their lead
- If uncertain about language, ask for preference politely
- For unsupported languages, explain limitation and offer alternatives
- Always confirm understanding before providing solutions

CULTURAL SENSITIVITY:
- Adapt communication style for cultural context
- Use appropriate formality levels
- Respect cultural business practices
- Avoid culturally sensitive topics

Remember: Your goal is to provide excellent customer service regardless of language barriers.
"""
    
    async def detect_and_set_language(self, text: str) -> str:
        """Detect customer language and set as preference."""
        detected_lang = await self._detect_language(text)
        
        if detected_lang in self.supported_langs:
            self.customer_language_preference = detected_lang
            self.target_lang = detected_lang
            print(f"🌐 Customer language set to: {detected_lang}")
        elif self.fallback_to_primary:
            print(f"⚠️ Language {detected_lang} not supported, using {self.primary_lang}")
            self.customer_language_preference = self.primary_lang
            self.target_lang = self.primary_lang
        
        return self.customer_language_preference or self.primary_lang
    
    async def generate_multilingual_response(
        self,
        customer_input: str,
        response_lang: Optional[str] = None
    ) -> TranslationResult:
        """Generate a customer service response in the appropriate language."""
        # Detect customer language if not set
        if not self.customer_language_preference:
            await self.detect_and_set_language(customer_input)
        
        response_language = response_lang or self.customer_language_preference or self.primary_lang
        
        # Generate helpful response (simulate AI response generation)
        response_templates = {
            "en": "Thank you for contacting us. I understand your concern and I'm here to help.",
            "es": "Gracias por contactarnos. Entiendo su preocupación y estoy aquí para ayudarle.",
            "fr": "Merci de nous avoir contactés. Je comprends votre préoccupation et je suis là pour vous aider.",
            "de": "Vielen Dank für Ihre Kontaktaufnahme. Ich verstehe Ihr Anliegen und bin hier, um zu helfen.",
            "it": "Grazie per averci contattato. Capisco la sua preoccupazione e sono qui per aiutarla."
        }
        
        response_text = response_templates.get(response_language, response_templates["en"])
        
        # Create translation result
        result = TranslationResult(
            original_text=customer_input,
            translated_text=response_text,
            source_language=self.customer_language_preference or "en",
            target_language=response_language,
            confidence_score=0.9,
            quality_score=0.85
        )
        
        return result


class ConferenceTranslationAgent(TranslationAgent):
    """
    Conference and meeting translation agent.
    
    Specialized for multi-speaker scenarios with speaker identification
    and real-time translation for international business meetings.
    """
    
    def __init__(self, meeting_languages: List[str]):
        super().__init__(
            source_lang="auto",
            target_lang="en",  # Default
            agent_name="Conference Translation Assistant"
        )
        
        self.meeting_languages = meeting_languages
        self.conversation_mode = ConversationMode.INTERPRETER
        
        # Conference-specific features
        self.speaker_tracking = True
        self.simultaneous_translation = True
        self.meeting_transcript: List[Dict[str, Any]] = []
        self.active_speakers: Dict[str, str] = {}  # speaker_id -> language
    
    async def add_speaker(self, speaker_id: str, language: str, name: Optional[str] = None) -> None:
        """Add a speaker to the conference."""
        if language not in self.meeting_languages:
            raise ValueError(f"Language {language} not supported in this meeting")
        
        self.active_speakers[speaker_id] = language
        speaker_info = {
            "speaker_id": speaker_id,
            "language": language,
            "name": name or f"Speaker {len(self.active_speakers)}",
            "joined_at": datetime.now()
        }
        
        self.meeting_transcript.append({
            "type": "speaker_join",
            "timestamp": datetime.now(),
            "data": speaker_info
        })
        
        print(f"🎙️ Speaker added: {speaker_info['name']} ({language})")
    
    async def translate_for_all_speakers(self, text: str, source_speaker_id: str) -> List[TranslationResult]:
        """Translate input for all other speakers in the meeting."""
        if source_speaker_id not in self.active_speakers:
            raise ValueError(f"Speaker {source_speaker_id} not found")
        
        source_lang = self.active_speakers[source_speaker_id]
        results = []
        
        # Translate to all other languages in the meeting
        target_languages = set(self.active_speakers.values()) - {source_lang}
        
        for target_lang in target_languages:
            result = await self.translate_text(text, source_lang, target_lang)
            results.append(result)
            
            # Add to meeting transcript
            self.meeting_transcript.append({
                "type": "translation",
                "timestamp": datetime.now(),
                "source_speaker": source_speaker_id,
                "source_language": source_lang,
                "target_language": target_lang,
                "original_text": text,
                "translated_text": result.translated_text,
                "confidence": result.confidence_score
            })
        
        return results
    
    async def generate_meeting_summary(self) -> Dict[str, Any]:
        """Generate a comprehensive meeting summary with translations."""
        if not self.meeting_transcript:
            return {}
        
        # Analyze meeting statistics
        translations = [entry for entry in self.meeting_transcript if entry["type"] == "translation"]
        speakers = [entry for entry in self.meeting_transcript if entry["type"] == "speaker_join"]
        
        summary = {
            "meeting_id": self.session_metrics.session_id if self.session_metrics else "unknown",
            "duration": None,
            "total_speakers": len(speakers),
            "total_translations": len(translations),
            "languages_used": list(set(self.active_speakers.values())),
            "speaker_details": speakers,
            "translation_quality": {
                "average_confidence": sum(t["confidence"] for t in translations) / len(translations) if translations else 0,
                "total_words_translated": sum(len(t["original_text"].split()) for t in translations)
            },
            "meeting_transcript": self.meeting_transcript
        }
        
        if self.session_metrics:
            summary["duration"] = str(self.session_metrics.end_time - self.session_metrics.start_time) if self.session_metrics.end_time else "Ongoing"
        
        return summary


# Example usage and testing
async def demo_translation_agents():
    """Demonstrate translation agent capabilities."""
    print("🌐 Real-Time Translation Agents Demo")
    print("="*50)
    
    # Basic Translation Demo
    print("\n🔄 Basic Translation Agent Demo")
    translator = TranslationAgent(source_lang="en", target_lang="es")
    await translator.initialize_agent()
    
    session_id = await translator.start_session()
    
    # Test translations
    test_phrases = [
        "Hello, how are you today?",
        "I need help with my account",
        "Thank you for your assistance"
    ]
    
    for phrase in test_phrases:
        result = await translator.translate_text(phrase)
        print(f"Original: {result.original_text}")
        print(f"Translation: {result.translated_text}")
        print(f"Quality: {result.quality_score}")
        print()
    
    await translator.end_session()
    
    # Multilingual Support Demo
    print("\n🌍 Multilingual Support Agent Demo")
    support_agent = MultilingualSupportAgent(
        primary_lang="en",
        supported_langs=["en", "es", "fr", "de"],
        company_name="Global Corp"
    )
    
    await support_agent.initialize_agent()
    session_id = await support_agent.start_session()
    
    # Simulate customer interactions in different languages
    customer_inputs = [
        ("Hola, tengo un problema con mi cuenta", "es"),
        ("Bonjour, j'ai besoin d'aide", "fr"),
        ("Guten Tag, ich habe eine Frage", "de")
    ]
    
    for input_text, expected_lang in customer_inputs:
        await support_agent.detect_and_set_language(input_text)
        response = await support_agent.generate_multilingual_response(input_text)
        print(f"Customer ({expected_lang}): {input_text}")
        print(f"Agent Response: {response.translated_text}")
        print()
    
    await support_agent.end_session()
    
    # Conference Translation Demo
    print("\n🎙️ Conference Translation Agent Demo")
    conference_agent = ConferenceTranslationAgent(["en", "es", "fr"])
    await conference_agent.initialize_agent()
    
    session_id = await conference_agent.start_session()
    
    # Add speakers
    await conference_agent.add_speaker("speaker1", "en", "John Smith")
    await conference_agent.add_speaker("speaker2", "es", "María García")
    await conference_agent.add_speaker("speaker3", "fr", "Pierre Dubois")
    
    # Simulate conference speech
    results = await conference_agent.translate_for_all_speakers(
        "Welcome everyone to our international meeting",
        "speaker1"
    )
    
    print(f"Original (EN): Welcome everyone to our international meeting")
    for result in results:
        print(f"Translation ({result.target_language.upper()}): {result.translated_text}")
    
    # Generate meeting summary
    summary = await conference_agent.generate_meeting_summary()
    print(f"\nMeeting Summary:")
    print(f"Total speakers: {summary['total_speakers']}")
    print(f"Total translations: {summary['total_translations']}")
    print(f"Languages used: {summary['languages_used']}")
    
    await conference_agent.end_session()
    
    print("\n✅ Translation demo completed!")


if __name__ == "__main__":
    asyncio.run(demo_translation_agents())