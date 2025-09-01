"""
Live coaching shell with terminal output and optional TTS support.
"""

import logging
import queue
import threading
import time
from typing import List, Optional, Set
from datetime import datetime

from ..gsi.client import GSIClient
from ..gsi.schema import GSISnapshot
from .rules import ComboCoach, CoachSuggestion
from .rl_bridge import CombinedRLCoach

logger = logging.getLogger(__name__)


class TTSHandler:
    """Text-to-speech handler with fallback options."""
    
    def __init__(self, enabled: bool = True, engine: str = "auto"):
        self.enabled = enabled
        self.engine = engine
        self.tts_available = False
        self.tts_engine = None
        
        if enabled:
            self._init_tts()
    
    def _init_tts(self):
        """Initialize TTS engine."""
        try:
            # Try pyttsx3 first (cross-platform)
            import pyttsx3
            self.tts_engine = pyttsx3.init()
            
            # Configure speech rate (slower for gaming)
            rate = self.tts_engine.getProperty('rate')
            self.tts_engine.setProperty('rate', max(150, rate - 50))
            
            self.tts_available = True
            logger.info("TTS initialized with pyttsx3")
            
        except ImportError:
            logger.warning("pyttsx3 not available, trying system TTS")
            self._init_system_tts()
        except Exception as e:
            logger.warning(f"Failed to initialize pyttsx3: {e}")
            self._init_system_tts()
    
    def _init_system_tts(self):
        """Initialize system-specific TTS."""
        import platform
        system = platform.system().lower()
        
        try:
            if system == "darwin":  # macOS
                self.engine = "say"
                self.tts_available = True
                logger.info("TTS initialized with macOS 'say'")
            elif system == "linux":
                # Try espeak or spd-say
                import subprocess
                try:
                    subprocess.run(["espeak", "--version"], capture_output=True, check=True)
                    self.engine = "espeak"
                    self.tts_available = True
                    logger.info("TTS initialized with espeak")
                except (subprocess.CalledProcessError, FileNotFoundError):
                    logger.warning("No suitable TTS engine found on Linux")
            elif system == "windows":
                # Windows has built-in SAPI
                self.engine = "sapi"
                self.tts_available = True
                logger.info("TTS initialized with Windows SAPI")
        except Exception as e:
            logger.warning(f"Failed to initialize system TTS: {e}")
    
    def speak(self, text: str, priority: int = 1):
        """
        Speak text using TTS.
        
        Args:
            text: Text to speak
            priority: Priority level (3=urgent, 2=normal, 1=low)
        """
        if not self.enabled or not self.tts_available:
            return
        
        try:
            # Filter out very frequent messages for lower priorities
            if priority < 3 and self._is_spam_message(text):
                return
            
            if self.tts_engine:  # pyttsx3
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            else:  # System TTS
                self._speak_system(text)
                
        except Exception as e:
            logger.warning(f"TTS failed: {e}")
    
    def _speak_system(self, text: str):
        """Use system TTS command."""
        import subprocess
        
        if self.engine == "say":  # macOS
            subprocess.run(["say", text], capture_output=True)
        elif self.engine == "espeak":  # Linux
            subprocess.run(["espeak", text], capture_output=True)
        elif self.engine == "sapi":  # Windows
            # Use PowerShell for Windows TTS
            ps_command = f'Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak("{text}")'
            subprocess.run(["powershell", "-Command", ps_command], capture_output=True)
    
    def _is_spam_message(self, text: str) -> bool:
        """Check if message is repetitive/spam."""
        # Simple spam detection - could be improved
        spam_phrases = ["excellent", "great", "good job"]
        return any(phrase in text.lower() for phrase in spam_phrases)


class SuggestionBuffer:
    """Buffer coaching suggestions to avoid spam."""
    
    def __init__(self, max_age: float = 5.0, max_per_type: int = 1):
        self.max_age = max_age
        self.max_per_type = max_per_type
        self.recent_suggestions = []
        self.type_counts = {}
    
    def should_show(self, suggestion: CoachSuggestion) -> bool:
        """Check if suggestion should be shown."""
        current_time = time.time()
        
        # Clean old suggestions
        self._cleanup_old_suggestions(current_time)
        
        # Check type frequency
        type_count = self.type_counts.get(suggestion.type, 0)
        if type_count >= self.max_per_type:
            return False
        
        # Check for similar recent suggestions
        for recent in self.recent_suggestions:
            if (recent['type'] == suggestion.type and 
                self._similarity(recent['message'], suggestion.message) > 0.7):
                return False
        
        return True
    
    def add(self, suggestion: CoachSuggestion):
        """Add suggestion to buffer."""
        current_time = time.time()
        
        self.recent_suggestions.append({
            'type': suggestion.type,
            'message': suggestion.message,
            'time': current_time,
            'priority': suggestion.priority
        })
        
        self.type_counts[suggestion.type] = self.type_counts.get(suggestion.type, 0) + 1
    
    def _cleanup_old_suggestions(self, current_time: float):
        """Remove old suggestions from buffer."""
        cutoff_time = current_time - self.max_age
        
        # Filter out old suggestions
        old_suggestions = [s for s in self.recent_suggestions if s['time'] < cutoff_time]
        self.recent_suggestions = [s for s in self.recent_suggestions if s['time'] >= cutoff_time]
        
        # Update type counts
        for old_suggestion in old_suggestions:
            old_type = old_suggestion['type']
            if old_type in self.type_counts:
                self.type_counts[old_type] = max(0, self.type_counts[old_type] - 1)
    
    def _similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity measure."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / max(union, 1)


class LiveCoachShell:
    """Live coaching shell with terminal output and TTS."""
    
    def __init__(self, coach_type: str = "rules", policy_path: str = None,
                 tts_enabled: bool = False, quiet_mode: bool = False,
                 gsi_port: int = 3000):
        
        self.coach_type = coach_type
        self.tts_enabled = tts_enabled
        self.quiet_mode = quiet_mode
        self.running = False
        
        # Create coach
        if coach_type == "rules":
            self.coach = ComboCoach()
        elif coach_type == "rl":
            self.coach = CombinedRLCoach(policy_path)
        elif coach_type == "combined":
            # TODO: Implement combined coach
            self.coach = CombinedRLCoach(policy_path)
        else:
            raise ValueError(f"Unknown coach type: {coach_type}")
        
        # Create GSI client
        self.gsi_client = GSIClient(port=gsi_port)
        self.gsi_client.add_callback(self._handle_gsi_update)
        
        # TTS handler
        self.tts = TTSHandler(enabled=tts_enabled)
        
        # Suggestion buffering
        self.suggestion_buffer = SuggestionBuffer(max_age=3.0)
        
        # Statistics
        self.stats = {
            "suggestions_shown": 0,
            "suggestions_spoken": 0,
            "snapshots_processed": 0,
            "start_time": None
        }
        
        # Console state
        self.last_status_line = ""
        
    def _handle_gsi_update(self, snapshot: GSISnapshot):
        """Handle incoming GSI updates."""
        try:
            self.stats["snapshots_processed"] += 1
            
            # Get coaching suggestions
            suggestions = self.coach.analyze(snapshot)
            
            # Process and display suggestions
            for suggestion in suggestions:
                if self.suggestion_buffer.should_show(suggestion):
                    self._display_suggestion(suggestion, snapshot)
                    self.suggestion_buffer.add(suggestion)
            
            # Update status line
            self._update_status(snapshot)
            
        except Exception as e:
            logger.error(f"Error handling GSI update: {e}")
    
    def _display_suggestion(self, suggestion: CoachSuggestion, snapshot: GSISnapshot):
        """Display a coaching suggestion."""
        # Format timestamp
        game_time = snapshot.map_state.game_time if snapshot.map_state else 0
        minutes = int(game_time // 60)
        seconds = int(game_time % 60)
        
        # Priority indicators
        priority_str = "🔥" if suggestion.priority == 3 else "⚠️" if suggestion.priority == 2 else "ℹ️"
        
        # Console output
        if not self.quiet_mode:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] [{minutes:02d}:{seconds:02d}] {priority_str} {suggestion.type.upper()}: {suggestion.message}")
        
        # TTS output for high priority suggestions
        if self.tts_enabled and suggestion.priority >= 2:
            self.tts.speak(suggestion.message, suggestion.priority)
            self.stats["suggestions_spoken"] += 1
        
        self.stats["suggestions_shown"] += 1
    
    def _update_status(self, snapshot: GSISnapshot):
        """Update status line."""
        if self.quiet_mode:
            return
        
        try:
            # Build status info
            status_parts = []
            
            if snapshot.hero:
                status_parts.append(f"Hero: {snapshot.hero.name}")
                status_parts.append(f"L{snapshot.hero.level}")
                status_parts.append(f"HP: {snapshot.hero.health_percent}%")
                status_parts.append(f"MP: {snapshot.hero.mana_percent}%")
            
            if snapshot.player:
                status_parts.append(f"CS: {snapshot.player.last_hits}")
                status_parts.append(f"D: {snapshot.player.denies}")
                status_parts.append(f"Gold: {snapshot.player.gold}")
                status_parts.append(f"GPM: {snapshot.player.gpm}")
            
            if snapshot.map_state:
                game_time = snapshot.map_state.game_time
                minutes = int(game_time // 60)
                seconds = int(game_time % 60)
                status_parts.append(f"Time: {minutes:02d}:{seconds:02d}")
            
            # Update status line (overwrite previous)
            status_line = " | ".join(status_parts)
            if status_line != self.last_status_line:
                print(f"\r{status_line}", end="", flush=True)
                self.last_status_line = status_line
            
        except Exception as e:
            logger.warning(f"Error updating status: {e}")
    
    def start(self):
        """Start the live coaching shell."""
        if self.running:
            return
        
        self.stats["start_time"] = time.time()
        self.running = True
        
        try:
            # Start GSI client
            self.gsi_client.start()
            
            print(f"🎮 Live Coach Started!")
            print(f"Coach: {self.coach_type}")
            print(f"TTS: {'ON' if self.tts_enabled else 'OFF'}")
            print(f"Quiet: {'ON' if self.quiet_mode else 'OFF'}")
            print(f"GSI Port: {self.gsi_client.port}")
            print()
            print("Start Dota 2 with: -gamestateintegration")
            print("Press Ctrl+C to stop")
            print("=" * 50)
            print()
            
            # Main loop
            while self.running:
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n\nStopping coach...")
        except Exception as e:
            logger.error(f"Error in coaching shell: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the coaching shell."""
        if not self.running:
            return
        
        self.running = False
        
        try:
            self.gsi_client.stop()
            
            # Print final stats
            if self.stats["start_time"]:
                duration = time.time() - self.stats["start_time"]
                print(f"\n📊 Session Stats:")
                print(f"Duration: {duration:.0f}s")
                print(f"Snapshots: {self.stats['snapshots_processed']}")
                print(f"Suggestions: {self.stats['suggestions_shown']}")
                print(f"TTS: {self.stats['suggestions_spoken']}")
            
        except Exception as e:
            logger.error(f"Error stopping coach: {e}")


def main():
    """Main entry point for live coaching."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Live Dota 2 coaching shell")
    parser.add_argument("--coach", choices=["rules", "rl", "combined"], default="rules",
                       help="Coaching system to use")
    parser.add_argument("--policy", type=str, help="Path to RL policy file")
    parser.add_argument("--tts", action="store_true", help="Enable text-to-speech")
    parser.add_argument("--quiet", action="store_true", help="Quiet mode (minimal output)")
    parser.add_argument("--port", type=int, default=3000, help="GSI port")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and start coach
    coach_shell = LiveCoachShell(
        coach_type=args.coach,
        policy_path=args.policy,
        tts_enabled=args.tts,
        quiet_mode=args.quiet,
        gsi_port=args.port
    )
    
    coach_shell.start()


if __name__ == "__main__":
    main()