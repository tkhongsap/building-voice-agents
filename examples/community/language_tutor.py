"""
Language Tutor Example

An interactive voice-based language learning assistant that provides
personalized lessons, pronunciation feedback, and conversation practice.

Features:
- Multi-language support (Spanish, French, German, Italian, Chinese)
- Pronunciation analysis and feedback
- Adaptive difficulty levels
- Interactive conversation practice
- Grammar exercises
- Progress tracking
- Cultural context integration

Usage:
    python language_tutor.py --language spanish --level beginner
"""

import asyncio
import json
import os
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path

from src.sdk.python_sdk import VoiceAgentSDK
from src.components.stt.google_stt import GoogleSTT
from src.components.llm.openai_llm import OpenAILLM
from src.components.tts.elevenlabs_tts import ElevenLabsTTS
from src.components.vad.silero_vad import SileroVAD


class ProficiencyLevel(Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    NATIVE = "native"


class LessonType(Enum):
    VOCABULARY = "vocabulary"
    GRAMMAR = "grammar"
    PRONUNCIATION = "pronunciation"
    CONVERSATION = "conversation"
    LISTENING = "listening"
    CULTURAL = "cultural"


@dataclass
class LanguageConfig:
    """Configuration for a supported language."""
    language_code: str
    display_name: str
    tts_voice_id: str
    stt_language_code: str
    native_greeting: str
    cultural_context: str


@dataclass
class LearningProgress:
    """Student's learning progress."""
    student_id: str
    language: str
    level: ProficiencyLevel
    total_lessons: int = 0
    total_study_time: float = 0.0  # hours
    vocabulary_known: int = 0
    pronunciation_score: float = 0.0
    grammar_score: float = 0.0
    conversation_score: float = 0.0
    streak_days: int = 0
    last_session: Optional[datetime] = None
    strengths: List[str] = None
    areas_for_improvement: List[str] = None
    
    def __post_init__(self):
        if self.strengths is None:
            self.strengths = []
        if self.areas_for_improvement is None:
            self.areas_for_improvement = []


@dataclass
class LessonSession:
    """Individual lesson session."""
    session_id: str
    student_id: str
    language: str
    lesson_type: LessonType
    difficulty_level: ProficiencyLevel
    start_time: datetime
    end_time: Optional[datetime] = None
    exercises_completed: int = 0
    accuracy_score: float = 0.0
    pronunciation_feedback: List[str] = None
    vocabulary_learned: List[str] = None
    notes: Optional[str] = None
    
    def __post_init__(self):
        if self.pronunciation_feedback is None:
            self.pronunciation_feedback = []
        if self.vocabulary_learned is None:
            self.vocabulary_learned = []


class LanguageTutor:
    """
    AI-powered language tutor with voice interaction.
    
    Provides personalized language learning with:
    - Adaptive curriculum based on progress
    - Real-time pronunciation feedback
    - Interactive conversation practice
    - Cultural context integration
    - Progress tracking and analytics
    """
    
    def __init__(self, target_language: str = "spanish", level: str = "beginner"):
        self.sdk = VoiceAgentSDK()
        self.agent = None
        
        # Language configuration
        self.supported_languages = {
            "spanish": LanguageConfig(
                language_code="es",
                display_name="Spanish",
                tts_voice_id="spanish_male",
                stt_language_code="es-ES",
                native_greeting="¡Hola! Soy tu tutor de español.",
                cultural_context="Spanish and Latin American cultures emphasize family, respect, and warm social connections."
            ),
            "french": LanguageConfig(
                language_code="fr",
                display_name="French",
                tts_voice_id="french_female",
                stt_language_code="fr-FR", 
                native_greeting="Bonjour! Je suis votre professeur de français.",
                cultural_context="French culture values sophistication, culinary arts, and intellectual discourse."
            ),
            "german": LanguageConfig(
                language_code="de",
                display_name="German",
                tts_voice_id="german_male",
                stt_language_code="de-DE",
                native_greeting="Guten Tag! Ich bin Ihr Deutschlehrer.",
                cultural_context="German culture emphasizes precision, efficiency, and direct communication."
            ),
            "italian": LanguageConfig(
                language_code="it",
                display_name="Italian",
                tts_voice_id="italian_female",
                stt_language_code="it-IT",
                native_greeting="Ciao! Sono il tuo insegnante di italiano.",
                cultural_context="Italian culture celebrates art, food, family, and expressive communication."
            ),
            "chinese": LanguageConfig(
                language_code="zh",
                display_name="Mandarin Chinese",
                tts_voice_id="chinese_female",
                stt_language_code="zh-CN",
                native_greeting="你好！我是您的中文老师。",
                cultural_context="Chinese culture values harmony, respect for elders, and collective well-being."
            )
        }
        
        self.target_language = target_language
        self.language_config = self.supported_languages.get(target_language)
        if not self.language_config:
            raise ValueError(f"Unsupported language: {target_language}")
        
        self.current_level = ProficiencyLevel(level)
        self.current_session: Optional[LessonSession] = None
        self.student_progress: Optional[LearningProgress] = None
        
        # Data storage
        self.data_dir = Path("language_learning_data")
        self.data_dir.mkdir(exist_ok=True)
        
        # Learning content
        self.vocabulary_database = self._load_vocabulary()
        self.grammar_rules = self._load_grammar_rules()
        self.conversation_scenarios = self._load_conversation_scenarios()
        
        # Load or create student progress
        self._load_student_progress()
    
    def _load_vocabulary(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load vocabulary database for the target language."""
        # This would normally load from a comprehensive database
        vocabulary_db = {
            "spanish": {
                "beginner": [
                    {"word": "hola", "translation": "hello", "pronunciation": "OH-lah", "example": "Hola, ¿cómo estás?"},
                    {"word": "gracias", "translation": "thank you", "pronunciation": "GRAH-see-ahs", "example": "Gracias por tu ayuda."},
                    {"word": "agua", "translation": "water", "pronunciation": "AH-gwah", "example": "Necesito agua, por favor."},
                    {"word": "casa", "translation": "house", "pronunciation": "KAH-sah", "example": "Mi casa es pequeña."},
                    {"word": "comer", "translation": "to eat", "pronunciation": "koh-MEHR", "example": "Me gusta comer tacos."}
                ],
                "intermediate": [
                    {"word": "desarrollar", "translation": "to develop", "pronunciation": "deh-sah-roh-YAHR", "example": "Quiero desarrollar mis habilidades."},
                    {"word": "ambiente", "translation": "environment", "pronunciation": "ahm-bee-EHN-teh", "example": "Cuidemos el ambiente."},
                    {"word": "experiencia", "translation": "experience", "pronunciation": "eks-peh-ree-EHN-see-ah", "example": "Fue una experiencia increíble."}
                ]
            },
            "french": {
                "beginner": [
                    {"word": "bonjour", "translation": "hello", "pronunciation": "bone-ZHOOR", "example": "Bonjour, comment allez-vous?"},
                    {"word": "merci", "translation": "thank you", "pronunciation": "mer-SEE", "example": "Merci beaucoup!"},
                    {"word": "eau", "translation": "water", "pronunciation": "oh", "example": "Je voudrais de l'eau."}
                ]
            }
        }
        
        return vocabulary_db.get(self.target_language, {})
    
    def _load_grammar_rules(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load grammar rules for the target language."""
        grammar_db = {
            "spanish": {
                "beginner": [
                    {
                        "rule": "Gender of Nouns",
                        "explanation": "Spanish nouns are either masculine (el) or feminine (la)",
                        "examples": ["el libro (the book - masculine)", "la mesa (the table - feminine)"],
                        "exercise": "Choose the correct article: ___ casa (la/el)"
                    },
                    {
                        "rule": "Present Tense -AR Verbs",
                        "explanation": "Regular -ar verbs follow a pattern: yo hablo, tú hablas, él/ella habla",
                        "examples": ["Yo hablo español", "Tú hablas inglés", "Ella habla francés"],
                        "exercise": "Conjugate 'caminar' for 'yo': yo ___"
                    }
                ]
            }
        }
        
        return grammar_db.get(self.target_language, {})
    
    def _load_conversation_scenarios(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load conversation scenarios for practice."""
        scenarios_db = {
            "spanish": {
                "beginner": [
                    {
                        "title": "Café Order",
                        "context": "You're at a café in Madrid and want to order coffee and pastry",
                        "starter": "Buenos días, ¿qué le gustaría tomar?",
                        "vocabulary": ["café", "leche", "azúcar", "croissant", "cuenta"],
                        "expected_responses": ["Un café con leche, por favor", "¿Cuánto cuesta?"]
                    },
                    {
                        "title": "Hotel Check-in",
                        "context": "You're checking into a hotel in Barcelona",
                        "starter": "¡Bienvenido! ¿Tiene una reservación?",
                        "vocabulary": ["reservación", "habitación", "llaves", "equipaje"],
                        "expected_responses": ["Sí, tengo una reservación", "¿Dónde está mi habitación?"]
                    }
                ]
            }
        }
        
        return scenarios_db.get(self.target_language, {})
    
    def _load_student_progress(self):
        """Load or create student progress."""
        progress_file = self.data_dir / f"progress_{self.target_language}.json"
        
        if progress_file.exists():
            with open(progress_file, 'r') as f:
                progress_data = json.load(f)
                if progress_data.get('last_session'):
                    progress_data['last_session'] = datetime.fromisoformat(progress_data['last_session'])
                self.student_progress = LearningProgress(**progress_data)
        else:
            self.student_progress = LearningProgress(
                student_id="student_001",
                language=self.target_language,
                level=self.current_level
            )
            self._save_student_progress()
    
    def _save_student_progress(self):
        """Save student progress to file."""
        progress_file = self.data_dir / f"progress_{self.target_language}.json"
        progress_data = asdict(self.student_progress)
        
        with open(progress_file, 'w') as f:
            json.dump(progress_data, f, indent=2, default=str)
    
    async def setup(self):
        """Setup the language tutor."""
        print(f"🔧 Setting up {self.language_config.display_name} Language Tutor...")
        
        # Configure components for bilingual operation
        stt_config = {
            "api_key": os.getenv("GOOGLE_CLOUD_KEY"),
            "language": self.language_config.stt_language_code,
            "enable_automatic_punctuation": True,
            "alternative_language_codes": ["en-US"]  # Allow English for mixed conversation
        }
        
        llm_config = {
            "model": "gpt-4o",
            "api_key": os.getenv("OPENAI_API_KEY"),
            "temperature": 0.3,
            "system_prompt": f"""You are an expert {self.language_config.display_name} language tutor. Your role:

1. Teach {self.language_config.display_name} to English speakers
2. Provide clear explanations in English when needed
3. Give pronunciation feedback and corrections
4. Adapt to the student's level: {self.current_level.value}
5. Be encouraging and patient
6. Incorporate cultural context: {self.language_config.cultural_context}

Guidelines:
- Speak primarily in {self.language_config.display_name} for practice
- Switch to English for explanations and corrections
- Provide pronunciation tips using phonetic notation
- Give positive reinforcement for attempts
- Correct mistakes gently with explanations
- Ask follow-up questions to encourage practice

Current student level: {self.current_level.value}
Student progress: {self.student_progress.total_lessons} lessons completed"""
        }
        
        tts_config = {
            "api_key": os.getenv("ELEVENLABS_API_KEY"),
            "voice_id": self.language_config.tts_voice_id,
            "stability": 0.7,
            "clarity": 0.8,
            "multilingual": True  # Enable multilingual synthesis
        }
        
        # Create agent
        self.agent = await self.sdk.create_agent(
            stt_provider="google",
            stt_config=stt_config,
            llm_provider="openai",
            llm_config=llm_config,
            tts_provider="elevenlabs",
            tts_config=tts_config,
            vad_provider="silero",
            vad_config={"sensitivity": 0.6}
        )
        
        # Register language learning functions
        self._register_tutor_functions()
        
        print(f"✅ {self.language_config.display_name} Language Tutor ready!")
        print(f"   Student Level: {self.current_level.value.title()}")
        print(f"   Lessons Completed: {self.student_progress.total_lessons}")
    
    def _register_tutor_functions(self):
        """Register language tutoring functions."""
        
        @self.agent.function(
            name="start_vocabulary_lesson",
            description="Start a vocabulary lesson",
            parameters={
                "topic": {"type": "string", "description": "Vocabulary topic (e.g., 'food', 'family', 'travel')"},
                "word_count": {"type": "integer", "minimum": 3, "maximum": 10, "description": "Number of words to learn"}
            }
        )
        async def start_vocabulary_lesson(topic: str = "general", word_count: int = 5) -> str:
            """Start a vocabulary lesson."""
            level_key = self.current_level.value
            available_words = self.vocabulary_database.get(level_key, [])
            
            if not available_words:
                return f"I don't have vocabulary for {level_key} level yet. Let's practice conversation instead!"
            
            # Select random words
            selected_words = random.sample(available_words, min(word_count, len(available_words)))
            
            # Start lesson session
            self.current_session = LessonSession(
                session_id=f"vocab_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                student_id=self.student_progress.student_id,
                language=self.target_language,
                lesson_type=LessonType.VOCABULARY,
                difficulty_level=self.current_level,
                start_time=datetime.now()
            )
            
            lesson_text = f"¡Excelente! Let's learn {word_count} new {self.language_config.display_name} words about {topic}.\n\n"
            
            for i, word_data in enumerate(selected_words, 1):
                lesson_text += f"{i}. **{word_data['word']}** [{word_data['pronunciation']}] = {word_data['translation']}\n"
                lesson_text += f"   Example: {word_data['example']}\n\n"
                self.current_session.vocabulary_learned.append(word_data['word'])
            
            lesson_text += f"Now let's practice! Try using one of these words in a sentence in {self.language_config.display_name}."
            
            return lesson_text
        
        @self.agent.function(
            name="practice_pronunciation",
            description="Practice pronunciation of a word or phrase",
            parameters={
                "text": {"type": "string", "description": "Word or phrase to practice"},
                "user_pronunciation": {"type": "string", "description": "How the user pronounced it (phonetically)"}
            }
        )
        async def practice_pronunciation(text: str, user_pronunciation: str = None) -> str:
            """Practice pronunciation with feedback."""
            # This is a simplified pronunciation feedback system
            # In a real implementation, this would use speech analysis
            
            # Look up correct pronunciation
            correct_pronunciation = None
            for level_words in self.vocabulary_database.values():
                for word_data in level_words:
                    if word_data['word'].lower() == text.lower():
                        correct_pronunciation = word_data['pronunciation']
                        break
            
            if not correct_pronunciation:
                return f"Let me help you with '{text}'. Try breaking it down syllable by syllable. Can you repeat it slowly?"
            
            feedback = f"Great attempt at '{text}'!\n"
            feedback += f"Correct pronunciation: [{correct_pronunciation}]\n"
            feedback += f"Tip: Focus on the stressed syllables (shown in CAPS).\n"
            feedback += f"Try saying it again, and I'll listen!"
            
            if self.current_session:
                self.current_session.pronunciation_feedback.append(f"Practiced: {text}")
            
            return feedback
        
        @self.agent.function(
            name="start_conversation_practice",
            description="Start a conversation practice scenario",
            parameters={
                "scenario": {"type": "string", "description": "Conversation scenario (e.g., 'restaurant', 'hotel', 'shopping')"}
            }
        )
        async def start_conversation_practice(scenario: str = "general") -> str:
            """Start conversation practice."""
            level_key = self.current_level.value
            scenarios = self.conversation_scenarios.get(level_key, [])
            
            if not scenarios:
                return f"Let's have a general conversation in {self.language_config.display_name}! ¿Cómo estás hoy?"
            
            # Find matching scenario or use random one
            selected_scenario = None
            for scene in scenarios:
                if scenario.lower() in scene['title'].lower():
                    selected_scenario = scene
                    break
            
            if not selected_scenario:
                selected_scenario = random.choice(scenarios)
            
            # Start conversation session
            self.current_session = LessonSession(
                session_id=f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                student_id=self.student_progress.student_id,
                language=self.target_language,
                lesson_type=LessonType.CONVERSATION,
                difficulty_level=self.current_level,
                start_time=datetime.now()
            )
            
            context_text = f"🎭 **Conversation Practice: {selected_scenario['title']}**\n\n"
            context_text += f"**Context:** {selected_scenario['context']}\n\n"
            context_text += f"**Useful vocabulary:** {', '.join(selected_scenario['vocabulary'])}\n\n"
            context_text += f"I'll start the conversation:\n\n"
            context_text += f"*{selected_scenario['starter']}*\n\n"
            context_text += f"Your turn! Respond in {self.language_config.display_name}."
            
            return context_text
        
        @self.agent.function(
            name="grammar_explanation",
            description="Explain a grammar rule",
            parameters={
                "topic": {"type": "string", "description": "Grammar topic to explain"}
            }
        )
        async def grammar_explanation(topic: str) -> str:
            """Provide grammar explanation."""
            level_key = self.current_level.value
            grammar_rules = self.grammar_rules.get(level_key, [])
            
            # Find matching rule
            matching_rule = None
            for rule in grammar_rules:
                if topic.lower() in rule['rule'].lower():
                    matching_rule = rule
                    break
            
            if not matching_rule and grammar_rules:
                matching_rule = random.choice(grammar_rules)
            
            if not matching_rule:
                return f"Let me explain that grammar concept. What specific aspect of {self.language_config.display_name} grammar would you like to practice?"
            
            explanation = f"📚 **Grammar: {matching_rule['rule']}**\n\n"
            explanation += f"**Rule:** {matching_rule['explanation']}\n\n"
            explanation += f"**Examples:**\n"
            for example in matching_rule['examples']:
                explanation += f"• {example}\n"
            explanation += f"\n**Practice:** {matching_rule['exercise']}\n\n"
            explanation += f"Try answering the practice question!"
            
            return explanation
        
        @self.agent.function(
            name="end_lesson",
            description="End the current lesson and save progress"
        )
        async def end_lesson() -> str:
            """End the current lesson and update progress."""
            if not self.current_session:
                return "No active lesson to end."
            
            # End session
            self.current_session.end_time = datetime.now()
            duration = (self.current_session.end_time - self.current_session.start_time).total_seconds() / 3600
            
            # Update student progress
            self.student_progress.total_lessons += 1
            self.student_progress.total_study_time += duration
            self.student_progress.last_session = datetime.now()
            
            # Calculate streak
            if self.student_progress.last_session:
                days_since_last = (datetime.now() - self.student_progress.last_session).days
                if days_since_last <= 1:
                    self.student_progress.streak_days += 1
                else:
                    self.student_progress.streak_days = 1
            
            # Update vocabulary count
            self.student_progress.vocabulary_known += len(self.current_session.vocabulary_learned)
            
            # Save progress
            self._save_student_progress()
            
            # Generate summary
            summary = f"🎉 Lesson completed!\n\n"
            summary += f"**Session Summary:**\n"
            summary += f"• Duration: {duration:.1f} hours\n"
            summary += f"• Type: {self.current_session.lesson_type.value.title()}\n"
            summary += f"• New words learned: {len(self.current_session.vocabulary_learned)}\n\n"
            summary += f"**Overall Progress:**\n"
            summary += f"• Total lessons: {self.student_progress.total_lessons}\n"
            summary += f"• Study time: {self.student_progress.total_study_time:.1f} hours\n"
            summary += f"• Vocabulary: {self.student_progress.vocabulary_known} words\n"
            summary += f"• Streak: {self.student_progress.streak_days} days\n\n"
            summary += f"¡Excelente trabajo! Keep practicing every day!"
            
            self.current_session = None
            return summary
        
        @self.agent.function(
            name="get_progress_report",
            description="Get detailed progress report"
        )
        async def get_progress_report() -> str:
            """Generate detailed progress report."""
            progress = self.student_progress
            
            report = f"📊 **{self.language_config.display_name} Learning Progress Report**\n\n"
            report += f"**Current Level:** {progress.level.value.title()}\n"
            report += f"**Total Lessons:** {progress.total_lessons}\n"
            report += f"**Study Time:** {progress.total_study_time:.1f} hours\n"
            report += f"**Vocabulary Known:** {progress.vocabulary_known} words\n"
            report += f"**Current Streak:** {progress.streak_days} days\n"
            
            if progress.last_session:
                report += f"**Last Session:** {progress.last_session.strftime('%Y-%m-%d')}\n"
            
            # Recommendations
            report += f"\n**Recommendations:**\n"
            if progress.total_lessons < 5:
                report += f"• Focus on basic vocabulary and pronunciation\n"
            elif progress.total_lessons < 20:
                report += f"• Practice conversation scenarios\n"
                report += f"• Review grammar fundamentals\n"
            else:
                report += f"• Consider advancing to {ProficiencyLevel.INTERMEDIATE.value} level\n"
                report += f"• Practice with native speakers\n"
            
            if progress.streak_days == 0:
                report += f"• Try to study a little bit every day\n"
            elif progress.streak_days >= 7:
                report += f"• Great consistency! Keep up the daily practice\n"
            
            return report
    
    async def start_learning_session(self):
        """Start an interactive learning session."""
        if not self.agent:
            await self.setup()
        
        print(f"\n🎓 {self.language_config.display_name} Language Tutor")
        print(f"   {self.language_config.native_greeting}")
        print(f"   Level: {self.current_level.value.title()}")
        print(f"   Lessons completed: {self.student_progress.total_lessons}")
        print(f"   \nWhat would you like to practice today?")
        print(f"   • Vocabulary: 'Start vocabulary lesson about food'")
        print(f"   • Conversation: 'Practice restaurant conversation'")
        print(f"   • Pronunciation: 'Help me pronounce hola'")
        print(f"   • Grammar: 'Explain verb conjugation'")
        print(f"   • Progress: 'Show my progress report'")
        print(f"   \nPress Ctrl+C to end session\n")
        
        try:
            await self.agent.start_conversation()
        except KeyboardInterrupt:
            print(f"\n¡Hasta la vista! Thank you for learning {self.language_config.display_name}!")
        finally:
            if self.current_session:
                await self.agent.process_text("End lesson")
    
    async def demo_mode(self):
        """Run language tutor demonstration."""
        if not self.agent:
            await self.setup()
        
        print(f"\n🎭 {self.language_config.display_name} Language Tutor Demo")
        print("=" * 60)
        
        # Demo interactions
        demo_interactions = [
            ("Vocabulary Lesson", "Start vocabulary lesson about food with 3 words"),
            ("Pronunciation Practice", "Help me pronounce gracias"),
            ("Grammar Question", "Explain gender of nouns"),
            ("Conversation Practice", "Practice café conversation"),
            ("Progress Check", "Show my progress report"),
            ("End Lesson", "End lesson")
        ]
        
        for step_name, user_input in demo_interactions:
            print(f"\n📝 {step_name}")
            print(f"Student: {user_input}")
            
            response = await self.agent.process_text(user_input)
            print(f"Tutor: {response}")
            
            await asyncio.sleep(1)
        
        print(f"\n✅ {self.language_config.display_name} tutoring demo completed!")


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Language Tutor Example")
    parser.add_argument("--language", choices=["spanish", "french", "german", "italian", "chinese"], 
                       default="spanish", help="Target language to learn")
    parser.add_argument("--level", choices=["beginner", "intermediate", "advanced"], 
                       default="beginner", help="Current proficiency level")
    parser.add_argument("--mode", choices=["live", "demo"], default="demo",
                       help="Run mode: live tutoring or demo")
    
    args = parser.parse_args()
    
    print("🌍 AI Language Tutor Example")
    print("=" * 50)
    
    try:
        tutor = LanguageTutor(args.language, args.level)
        
        if args.mode == "live":
            await tutor.start_learning_session()
        else:  # demo mode
            await tutor.demo_mode()
            
    except ValueError as e:
        print(f"❌ Error: {e}")
        print("Available languages: spanish, french, german, italian, chinese")


if __name__ == "__main__":
    # Check for required environment variables
    required_vars = {
        "OPENAI_API_KEY": "OpenAI API key for LLM",
        "GOOGLE_CLOUD_KEY": "Google Cloud API key for Speech-to-Text",
        "ELEVENLABS_API_KEY": "ElevenLabs API key for multilingual TTS"
    }
    
    missing_vars = []
    for var, description in required_vars.items():
        if not os.getenv(var):
            missing_vars.append(f"  {var}: {description}")
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(var)
        print("\nSet them with: export VARIABLE_NAME='your-key-here'")
        exit(1)
    
    asyncio.run(main())