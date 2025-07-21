"""
Gaming NPC Example

An interactive voice-powered non-player character (NPC) for games.
Demonstrates dynamic character personalities, quest management, 
and immersive voice interaction for gaming applications.

Features:
- Dynamic character personalities and backgrounds
- Quest system with objectives and rewards
- Inventory and trading mechanics
- Contextual dialogue based on game state
- Emotion and mood simulation
- World knowledge and lore integration
- Character relationship tracking

Usage:
    python gaming_npc.py --character merchant --location tavern
"""

import asyncio
import json
import os
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path

from src.sdk.python_sdk import VoiceAgentSDK
from src.components.stt.openai_stt import OpenAISTT
from src.components.llm.openai_llm import OpenAILLM
from src.components.tts.elevenlabs_tts import ElevenLabsTTS
from src.components.vad.silero_vad import SileroVAD


class CharacterType(Enum):
    MERCHANT = "merchant"
    GUARD = "guard"
    INNKEEPER = "innkeeper"
    WIZARD = "wizard"
    BLACKSMITH = "blacksmith"
    QUEST_GIVER = "quest_giver"
    COMPANION = "companion"


class QuestStatus(Enum):
    AVAILABLE = "available"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"


class Mood(Enum):
    HAPPY = "happy"
    NEUTRAL = "neutral"
    ANGRY = "angry"
    SAD = "sad"
    EXCITED = "excited"
    FEARFUL = "fearful"
    SUSPICIOUS = "suspicious"


@dataclass
class Item:
    """Game item."""
    item_id: str
    name: str
    description: str
    value: int
    item_type: str  # weapon, armor, consumable, misc
    rarity: str = "common"  # common, uncommon, rare, epic, legendary
    quantity: int = 1


@dataclass
class Quest:
    """Game quest."""
    quest_id: str
    title: str
    description: str
    objectives: List[str]
    rewards: List[Item]
    prerequisites: List[str] = None
    status: QuestStatus = QuestStatus.AVAILABLE
    progress: Dict[str, int] = None
    
    def __post_init__(self):
        if self.prerequisites is None:
            self.prerequisites = []
        if self.progress is None:
            self.progress = {}


@dataclass
class CharacterPersonality:
    """Character personality traits."""
    name: str
    race: str
    profession: str
    background: str
    personality_traits: List[str]
    speech_patterns: List[str]
    likes: List[str]
    dislikes: List[str]
    voice_description: str
    default_mood: Mood = Mood.NEUTRAL


@dataclass
class PlayerRelationship:
    """Relationship with the player."""
    player_id: str
    relationship_level: int = 0  # -100 to 100
    trust_level: int = 0  # 0 to 100
    fear_level: int = 0  # 0 to 100
    interactions_count: int = 0
    last_interaction: Optional[datetime] = None
    reputation_tags: List[str] = None
    
    def __post_init__(self):
        if self.reputation_tags is None:
            self.reputation_tags = []


@dataclass
class GameState:
    """Current game state."""
    location: str
    time_of_day: str
    weather: str
    player_level: int
    player_class: str
    world_events: List[str] = None
    
    def __post_init__(self):
        if self.world_events is None:
            self.world_events = []


class GamingNPC:
    """
    Interactive voice-powered NPC for games.
    
    Provides dynamic character interaction with:
    - Contextual dialogue based on game state
    - Quest management and progression
    - Trading and inventory systems
    - Relationship tracking with players
    - Adaptive personality and mood
    """
    
    def __init__(self, character_type: str = "merchant", location: str = "tavern"):
        self.sdk = VoiceAgentSDK()
        self.agent = None
        
        # Character configuration
        self.character_type = CharacterType(character_type)
        self.location = location
        self.current_mood = Mood.NEUTRAL
        
        # Game state
        self.game_state = GameState(
            location=location,
            time_of_day="afternoon",
            weather="clear",
            player_level=5,
            player_class="warrior"
        )
        
        # Character data
        self.personality = self._create_character_personality()
        self.inventory: List[Item] = []
        self.available_quests: List[Quest] = []
        self.player_relationships: Dict[str, PlayerRelationship] = {}
        
        # Data storage
        self.data_dir = Path("gaming_npc_data")
        self.data_dir.mkdir(exist_ok=True)
        
        # Initialize character data
        self._initialize_character_data()
        self._load_world_lore()
    
    def _create_character_personality(self) -> CharacterPersonality:
        """Create character personality based on type."""
        personalities = {
            CharacterType.MERCHANT: CharacterPersonality(
                name="Gareth the Trader",
                race="Human",
                profession="Merchant",
                background="A seasoned trader who has traveled across many lands, collecting rare goods and stories.",
                personality_traits=["greedy", "charismatic", "knowledgeable", "opportunistic"],
                speech_patterns=["uses trade terminology", "mentions prices frequently", "name-drops exotic locations"],
                likes=["gold", "rare items", "good deals", "interesting stories"],
                dislikes=["thieves", "haggling", "competition", "broken promises"],
                voice_description="smooth and persuasive with a slight accent"
            ),
            
            CharacterType.GUARD: CharacterPersonality(
                name="Captain Marcus",
                race="Human",
                profession="City Guard",
                background="A veteran guard who has protected the city for over two decades.",
                personality_traits=["dutiful", "stern", "protective", "honorable"],
                speech_patterns=["military terminology", "direct commands", "formal address"],
                likes=["order", "justice", "loyalty", "peace"],
                dislikes=["criminals", "chaos", "corruption", "disrespect"],
                voice_description="authoritative and commanding"
            ),
            
            CharacterType.INNKEEPER: CharacterPersonality(
                name="Martha Hearthstone",
                race="Halfling",
                profession="Innkeeper",
                background="A warm-hearted innkeeper who knows everyone's business and serves the best ale in town.",
                personality_traits=["hospitable", "gossipy", "caring", "perceptive"],
                speech_patterns=["uses endearing terms", "shares local gossip", "offers food and drink"],
                likes=["good company", "stories", "cooking", "helping travelers"],
                dislikes=["troublemakers", "poor manners", "waste", "sadness"],
                voice_description="warm and motherly with a regional accent"
            ),
            
            CharacterType.WIZARD: CharacterPersonality(
                name="Archmage Zephyr",
                race="Elf",
                profession="Wizard",
                background="An ancient elf who has studied magic for centuries and guards dangerous knowledge.",
                personality_traits=["intellectual", "mysterious", "patient", "condescending"],
                speech_patterns=["archaic language", "magical terminology", "philosophical observations"],
                likes=["knowledge", "magic", "books", "solitude"],
                dislikes=["ignorance", "rash decisions", "interruptions", "dark magic"],
                voice_description="ethereal and ancient with measured speech"
            ),
            
            CharacterType.BLACKSMITH: CharacterPersonality(
                name="Thorin Ironforge",
                race="Dwarf",
                profession="Blacksmith",
                background="A master craftsman whose family has forged weapons for kings and heroes for generations.",
                personality_traits=["proud", "skilled", "stubborn", "hardworking"],
                speech_patterns=["technical smithing terms", "gruff expressions", "quality emphasis"],
                likes=["fine craftsmanship", "good metal", "hard work", "respect"],
                dislikes=["shoddy work", "impatience", "cheap materials", "disrespect for craft"],
                voice_description="gruff and booming with a heavy accent"
            )
        }
        
        return personalities.get(self.character_type, personalities[CharacterType.MERCHANT])
    
    def _initialize_character_data(self):
        """Initialize character inventory and quests."""
        # Initialize inventory based on character type
        if self.character_type == CharacterType.MERCHANT:
            self.inventory = [
                Item("healing_potion", "Healing Potion", "Restores 50 HP", 25, "consumable"),
                Item("iron_sword", "Iron Sword", "A well-crafted iron blade", 150, "weapon", "common"),
                Item("leather_armor", "Leather Armor", "Basic protection for adventurers", 75, "armor"),
                Item("magic_scroll", "Scroll of Fireball", "Single-use spell scroll", 200, "consumable", "uncommon"),
                Item("travel_rations", "Travel Rations", "Food for long journeys", 5, "consumable", quantity=10)
            ]
        
        elif self.character_type == CharacterType.BLACKSMITH:
            self.inventory = [
                Item("steel_sword", "Steel Sword", "Superior steel craftsmanship", 300, "weapon", "uncommon"),
                Item("chain_mail", "Chain Mail", "Flexible metal protection", 200, "armor"),
                Item("iron_ingot", "Iron Ingot", "Raw smithing material", 20, "misc", quantity=5),
                Item("smithing_hammer", "Master's Hammer", "Tool of the trade", 500, "misc", "rare")
            ]
        
        # Initialize quests
        if self.character_type == CharacterType.QUEST_GIVER:
            self.available_quests = [
                Quest(
                    quest_id="goblin_problem",
                    title="The Goblin Problem",
                    description="Goblins have been raiding merchant caravans. Clear out their camp.",
                    objectives=["Defeat 10 goblins", "Find the stolen goods", "Return to quest giver"],
                    rewards=[
                        Item("gold_coins", "Gold Coins", "Currency", 1, "misc", quantity=100),
                        Item("goblin_slayer", "Goblin Slayer Sword", "Effective against goblins", 250, "weapon", "uncommon")
                    ]
                )
            ]
    
    def _load_world_lore(self):
        """Load world lore and background information."""
        self.world_lore = {
            "locations": {
                "tavern": "The Prancing Pony - A popular gathering place for adventurers and locals",
                "market": "Central Market - The heart of trade in the city",
                "castle": "Ironhold Castle - Seat of power ruled by King Aldric",
                "forest": "Darkwood Forest - Ancient woods filled with magical creatures",
                "dungeon": "The Crypts of Shadows - Dangerous underground ruins"
            },
            "factions": {
                "merchants_guild": "Powerful organization controlling trade",
                "city_guard": "Protectors of law and order",
                "adventurers_guild": "Registry for monster hunters and treasure seekers",
                "mages_circle": "Ancient order of spellcasters"
            },
            "current_events": [
                "Dragon sightings reported in the northern mountains",
                "Merchant caravans going missing on the trade roads",
                "Strange magical anomalies appearing near the old ruins",
                "The harvest festival is approaching next month"
            ]
        }
    
    async def setup(self):
        """Setup the gaming NPC."""
        print(f"🔧 Setting up {self.personality.name} ({self.character_type.value})...")
        
        # Configure components
        stt_config = {
            "model": "whisper-1",
            "language": "en",
            "api_key": os.getenv("OPENAI_API_KEY")
        }
        
        # Create character-specific system prompt
        system_prompt = self._create_character_prompt()
        
        llm_config = {
            "model": "gpt-4o",
            "api_key": os.getenv("OPENAI_API_KEY"),
            "temperature": 0.7,  # Higher for more creative roleplay
            "system_prompt": system_prompt
        }
        
        tts_config = {
            "api_key": os.getenv("ELEVENLABS_API_KEY"),
            "voice_id": self._get_voice_id(),
            "stability": 0.6,
            "clarity": 0.8
        }
        
        # Create agent
        self.agent = await self.sdk.create_agent(
            stt_provider="openai",
            stt_config=stt_config,
            llm_provider="openai",
            llm_config=llm_config,
            tts_provider="elevenlabs",
            tts_config=tts_config,
            vad_provider="silero",
            vad_config={"sensitivity": 0.6}
        )
        
        # Register NPC functions
        self._register_npc_functions()
        
        print(f"✅ {self.personality.name} is ready in {self.location}!")
        print(f"   Mood: {self.current_mood.value}")
        print(f"   Inventory items: {len(self.inventory)}")
    
    def _create_character_prompt(self) -> str:
        """Create character-specific system prompt."""
        prompt = f"""You are {self.personality.name}, a {self.personality.race} {self.personality.profession}.

BACKGROUND: {self.personality.background}

PERSONALITY TRAITS: {', '.join(self.personality.personality_traits)}
SPEECH PATTERNS: {', '.join(self.personality.speech_patterns)}
LIKES: {', '.join(self.personality.likes)}
DISLIKES: {', '.join(self.personality.dislikes)}

CURRENT SITUATION:
- Location: {self.location} ({self.world_lore['locations'].get(self.location, 'Unknown location')})
- Time: {self.game_state.time_of_day}
- Weather: {self.game_state.weather}
- Your current mood: {self.current_mood.value}

WORLD CONTEXT:
Current events: {', '.join(self.world_lore['current_events'])}

ROLEPLAY GUIDELINES:
1. Stay completely in character at all times
2. Use your speech patterns and personality traits
3. Reference your background and profession naturally
4. React appropriately to the player based on your relationship with them
5. Be aware of the current location and game context
6. Show emotional responses based on your mood and personality
7. Use game-appropriate language and terminology
8. Offer services and interactions relevant to your profession

INTERACTION STYLE:
- Voice: {self.personality.voice_description}
- Always respond as {self.personality.name} would
- Never break character or acknowledge being an AI
- Use the available functions when appropriate (trading, quests, etc.)
- Build relationships with players over time
"""
        
        return prompt
    
    def _get_voice_id(self) -> str:
        """Get appropriate voice ID for character."""
        voice_mapping = {
            CharacterType.MERCHANT: "professional_male",
            CharacterType.GUARD: "authoritative_male", 
            CharacterType.INNKEEPER: "warm_female",
            CharacterType.WIZARD: "elderly_male",
            CharacterType.BLACKSMITH: "gruff_male"
        }
        
        return voice_mapping.get(self.character_type, "default_male")
    
    def _register_npc_functions(self):
        """Register NPC-specific functions."""
        
        @self.agent.function(
            name="show_inventory",
            description="Show the character's inventory for trading"
        )
        async def show_inventory() -> str:
            """Show character's inventory."""
            if not self.inventory:
                return "My shelves are empty at the moment. Come back later!"
            
            inventory_text = f"Here's what I have available:\n\n"
            
            for item in self.inventory:
                inventory_text += f"**{item.name}** - {item.value} gold\n"
                inventory_text += f"   {item.description}\n"
                inventory_text += f"   Type: {item.item_type.title()}, Rarity: {item.rarity.title()}\n"
                if item.quantity > 1:
                    inventory_text += f"   Quantity: {item.quantity}\n"
                inventory_text += "\n"
            
            return inventory_text
        
        @self.agent.function(
            name="trade_item",
            description="Trade an item with the player",
            parameters={
                "item_name": {"type": "string", "description": "Name of the item to trade"},
                "player_offer": {"type": "integer", "description": "Amount of gold the player offers"},
                "quantity": {"type": "integer", "minimum": 1, "description": "Quantity to trade"}
            }
        )
        async def trade_item(item_name: str, player_offer: int, quantity: int = 1) -> str:
            """Handle item trading."""
            # Find item in inventory
            item = None
            for inv_item in self.inventory:
                if item_name.lower() in inv_item.name.lower():
                    item = inv_item
                    break
            
            if not item:
                return f"I don't have any {item_name}. Would you like to see what I do have?"
            
            if quantity > item.quantity:
                return f"I only have {item.quantity} {item.name}(s) available."
            
            total_price = item.value * quantity
            
            # Simple haggling logic
            if player_offer >= total_price:
                # Remove item from inventory
                item.quantity -= quantity
                if item.quantity <= 0:
                    self.inventory.remove(item)
                
                # Positive relationship boost
                self._adjust_relationship("player", relationship_boost=5)
                
                return f"Excellent! {quantity} {item.name}(s) for {player_offer} gold. Pleasure doing business with you!"
            
            elif player_offer >= total_price * 0.8:  # 80% of asking price
                # Accept lower offer sometimes
                item.quantity -= quantity
                if item.quantity <= 0:
                    self.inventory.remove(item)
                
                self._adjust_relationship("player", relationship_boost=2)
                
                return f"You drive a hard bargain, but I'll accept {player_offer} gold for {quantity} {item.name}(s)."
            
            else:
                # Reject low offer
                return f"Ha! {player_offer} gold for {quantity} {item.name}(s)? That's worth at least {total_price} gold. Make me a serious offer!"
        
        @self.agent.function(
            name="show_available_quests",
            description="Show available quests from this character"
        )
        async def show_available_quests() -> str:
            """Show available quests."""
            if not self.available_quests:
                return "I don't have any tasks for you right now. Check back later!"
            
            quests_text = "I could use your help with a few things:\n\n"
            
            for quest in self.available_quests:
                if quest.status == QuestStatus.AVAILABLE:
                    quests_text += f"**{quest.title}**\n"
                    quests_text += f"{quest.description}\n"
                    quests_text += f"Objectives:\n"
                    for obj in quest.objectives:
                        quests_text += f"  • {obj}\n"
                    
                    if quest.rewards:
                        quests_text += f"Rewards:\n"
                        for reward in quest.rewards:
                            quests_text += f"  • {reward.name}"
                            if reward.quantity > 1:
                                quests_text += f" x{reward.quantity}"
                            quests_text += "\n"
                    quests_text += "\n"
            
            return quests_text
        
        @self.agent.function(
            name="accept_quest",
            description="Accept a quest from this character",
            parameters={
                "quest_title": {"type": "string", "description": "Title of the quest to accept"}
            }
        )
        async def accept_quest(quest_title: str) -> str:
            """Accept a quest."""
            quest = None
            for q in self.available_quests:
                if quest_title.lower() in q.title.lower() and q.status == QuestStatus.AVAILABLE:
                    quest = q
                    break
            
            if not quest:
                return f"I don't have a quest called '{quest_title}' available."
            
            quest.status = QuestStatus.ACTIVE
            self._adjust_relationship("player", relationship_boost=10)
            
            return f"Excellent! You've accepted '{quest.title}'. {quest.description} Remember: {', '.join(quest.objectives)}. Good luck!"
        
        @self.agent.function(
            name="get_local_information",
            description="Get information about the local area, people, or events"
        )
        async def get_local_information() -> str:
            """Provide local information and gossip."""
            info_categories = [
                f"Location info: {self.world_lore['locations'].get(self.location, 'Nothing special about this place.')}",
                f"Recent events: {random.choice(self.world_lore['current_events'])}",
                f"Weather: The {self.game_state.weather} weather has been {self._get_weather_opinion()}",
                f"Time context: It's {self.game_state.time_of_day}, so {self._get_time_context()}"
            ]
            
            selected_info = random.choice(info_categories)
            
            # Add character-specific perspective
            if self.character_type == CharacterType.MERCHANT:
                selected_info += " This affects business, you know."
            elif self.character_type == CharacterType.GUARD:
                selected_info += " I keep an eye on such things for security."
            elif self.character_type == CharacterType.INNKEEPER:
                selected_info += " That's what the travelers have been saying."
            
            return selected_info
        
        @self.agent.function(
            name="change_mood",
            description="Change character mood based on interaction",
            parameters={
                "new_mood": {"type": "string", "enum": ["happy", "neutral", "angry", "sad", "excited", "fearful", "suspicious"], "description": "New mood to set"},
                "reason": {"type": "string", "description": "Reason for mood change"}
            }
        )
        async def change_mood(new_mood: str, reason: str) -> str:
            """Change character mood."""
            old_mood = self.current_mood
            self.current_mood = Mood(new_mood)
            
            mood_responses = {
                Mood.HAPPY: "I'm feeling quite cheerful now!",
                Mood.ANGRY: "*scowls* I'm not pleased about this.",
                Mood.SAD: "*sighs heavily* This saddens me.",
                Mood.EXCITED: "*eyes light up* How exciting!",
                Mood.FEARFUL: "*looks around nervously* This worries me.",
                Mood.SUSPICIOUS: "*narrows eyes* I'm not so sure about this."
            }
            
            response = mood_responses.get(self.current_mood, "My mood has changed.")
            
            print(f"🎭 {self.personality.name} mood: {old_mood.value} → {new_mood} ({reason})")
            
            return response
    
    def _adjust_relationship(self, player_id: str, relationship_boost: int = 0, trust_boost: int = 0, fear_boost: int = 0):
        """Adjust relationship with player."""
        if player_id not in self.player_relationships:
            self.player_relationships[player_id] = PlayerRelationship(player_id=player_id)
        
        rel = self.player_relationships[player_id]
        rel.relationship_level = max(-100, min(100, rel.relationship_level + relationship_boost))
        rel.trust_level = max(0, min(100, rel.trust_level + trust_boost))
        rel.fear_level = max(0, min(100, rel.fear_level + fear_boost))
        rel.interactions_count += 1
        rel.last_interaction = datetime.now()
    
    def _get_weather_opinion(self) -> str:
        """Get character's opinion on weather."""
        weather_opinions = {
            "clear": "perfect for business",
            "rainy": "keeping people indoors",
            "stormy": "quite dangerous for travel",
            "foggy": "mysterious and unsettling"
        }
        return weather_opinions.get(self.game_state.weather, "unremarkable")
    
    def _get_time_context(self) -> str:
        """Get time-appropriate context."""
        time_contexts = {
            "morning": "most people are starting their day",
            "afternoon": "business is usually busy",
            "evening": "people are winding down",
            "night": "only night owls are still about"
        }
        return time_contexts.get(self.game_state.time_of_day, "time passes")
    
    async def start_interaction(self):
        """Start interaction with the NPC."""
        if not self.agent:
            await self.setup()
        
        # Generate contextual greeting
        greeting = self._generate_greeting()
        
        print(f"\n🎮 {self.personality.name} ({self.character_type.value.title()})")
        print(f"   Location: {self.location}")
        print(f"   Mood: {self.current_mood.value}")
        print(f"   Time: {self.game_state.time_of_day}")
        print(f"\n{greeting}")
        print(f"\n   Available interactions:")
        print(f"   • 'What do you have for sale?' (if merchant)")
        print(f"   • 'Do you have any work for me?' (if quest giver)")
        print(f"   • 'Tell me about this place'")
        print(f"   • 'What's the local news?'")
        print(f"   Press Ctrl+C to leave\n")
        
        try:
            await self.agent.start_conversation()
        except KeyboardInterrupt:
            print(f"\n👋 {self._generate_farewell()}")
    
    def _generate_greeting(self) -> str:
        """Generate contextual greeting."""
        greetings_by_type = {
            CharacterType.MERCHANT: [
                "Welcome to my shop! Looking for anything in particular?",
                "Greetings, traveler! I've got the finest goods in town.",
                "Come, come! Don't be shy - I have treasures from distant lands!"
            ],
            CharacterType.GUARD: [
                "Halt! State your business in the city.",
                "Keep moving, citizen. Nothing to see here.",
                "Everything peaceful today. How can I help you?"
            ],
            CharacterType.INNKEEPER: [
                "Welcome to the Prancing Pony! Room or meal?",
                "Come in, come in! You look like you could use a hot meal.",
                "Greetings! What brings you to our humble establishment?"
            ],
            CharacterType.WIZARD: [
                "Ah, another seeker of knowledge approaches...",
                "The arcane energies brought you here, didn't they?",
                "Welcome, young one. What wisdom do you seek?"
            ],
            CharacterType.BLACKSMITH: [
                "*looks up from anvil* Need something forged?",
                "Welcome to the finest smithy in the realm!",
                "*wipes sweat* What brings you to my forge?"
            ]
        }
        
        base_greetings = greetings_by_type.get(self.character_type, ["Hello there!"])
        greeting = random.choice(base_greetings)
        
        # Add mood modifier
        if self.current_mood == Mood.HAPPY:
            greeting = "*smiles warmly* " + greeting
        elif self.current_mood == Mood.ANGRY:
            greeting = "*gruffly* " + greeting
        elif self.current_mood == Mood.SUSPICIOUS:
            greeting = "*eyes you warily* " + greeting
        
        return greeting
    
    def _generate_farewell(self) -> str:
        """Generate contextual farewell."""
        farewells = [
            "Safe travels, friend!",
            "May fortune favor you!",
            "Come back anytime!",
            "Until we meet again!",
            "Farewell, adventurer!"
        ]
        
        return f"{self.personality.name}: {random.choice(farewells)}"
    
    async def demo_mode(self):
        """Run NPC demonstration."""
        if not self.agent:
            await self.setup()
        
        print(f"\n🎭 {self.personality.name} Demo")
        print("=" * 60)
        
        # Demo interactions based on character type
        if self.character_type == CharacterType.MERCHANT:
            demo_interactions = [
                ("Greeting", "Hello!"),
                ("Show Inventory", "What do you have for sale?"),
                ("Trade Attempt", "I'll give you 20 gold for a healing potion"),
                ("Local Info", "What's the news around here?"),
                ("Farewell", "Thanks for your time!")
            ]
        elif self.character_type == CharacterType.QUEST_GIVER:
            demo_interactions = [
                ("Greeting", "Greetings!"),
                ("Quest Inquiry", "Do you have any work for me?"),
                ("Quest Details", "Tell me more about the goblin problem"),
                ("Accept Quest", "I'll take on the goblin problem quest"),
                ("Farewell", "I'll get started on that quest!")
            ]
        else:
            demo_interactions = [
                ("Greeting", "Hello there!"),
                ("Local Info", "Tell me about this place"),
                ("Character Info", "What's your story?"),
                ("Current Events", "What's happening around here?"),
                ("Farewell", "Nice talking with you!")
            ]
        
        for step_name, user_input in demo_interactions:
            print(f"\n📝 {step_name}")
            print(f"Player: {user_input}")
            
            response = await self.agent.process_text(user_input)
            print(f"{self.personality.name}: {response}")
            
            await asyncio.sleep(1)
        
        print(f"\n✅ {self.personality.name} demo completed!")


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Gaming NPC Example")
    parser.add_argument("--character", choices=["merchant", "guard", "innkeeper", "wizard", "blacksmith", "quest_giver"],
                       default="merchant", help="Type of NPC character")
    parser.add_argument("--location", default="tavern", help="Starting location")
    parser.add_argument("--mode", choices=["live", "demo"], default="demo",
                       help="Run mode: live interaction or demo")
    
    args = parser.parse_args()
    
    print("🎮 Gaming NPC Example")
    print("=" * 50)
    
    npc = GamingNPC(args.character, args.location)
    
    if args.mode == "live":
        await npc.start_interaction()
    else:  # demo mode
        await npc.demo_mode()


if __name__ == "__main__":
    # Check for required environment variables
    required_vars = {
        "OPENAI_API_KEY": "OpenAI API key for LLM and STT",
        "ELEVENLABS_API_KEY": "ElevenLabs API key for character voices"
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