"""
Simple voice output using macOS built-in text-to-speech.
"""

import subprocess
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def speak(text: str, voice: Optional[str] = None, rate: Optional[int] = None) -> bool:
    """
    Speak text using macOS built-in 'say' command.
    
    Args:
        text: Text to speak
        voice: Voice name (e.g., 'Alex', 'Samantha'). None uses system default.
        rate: Speaking rate in words per minute. None uses system default.
        
    Returns:
        True if speech command executed successfully, False otherwise
    """
    if not text.strip():
        return False
    
    # Build command
    cmd = ["say"]
    
    if voice:
        cmd.extend(["-v", voice])
    
    if rate:
        cmd.extend(["-r", str(rate)])
    
    cmd.append(text)
    
    try:
        # Run asynchronously so it doesn't block
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        logger.debug(f"Speaking: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        return True
        
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        logger.error(f"Failed to speak text: {e}")
        return False


def list_voices() -> list:
    """
    Get list of available voices on macOS.
    
    Returns:
        List of voice names, empty list if command fails
    """
    try:
        result = subprocess.run(["say", "-v", "?"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            # Parse output like "Alex             en_US    # Most people recognize me by my voice."
            voices = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    voice_name = line.split()[0]
                    voices.append(voice_name)
            return voices
        else:
            logger.warning("Failed to list voices")
            return []
            
    except (subprocess.SubprocessError, subprocess.TimeoutExpired) as e:
        logger.error(f"Error listing voices: {e}")
        return []


def is_speaking() -> bool:
    """
    Check if the say command is currently speaking.
    
    Returns:
        True if currently speaking, False otherwise
    """
    try:
        # Check if any 'say' processes are running
        result = subprocess.run(
            ["pgrep", "-f", "say"], 
            capture_output=True, 
            timeout=2
        )
        return result.returncode == 0
        
    except (subprocess.SubprocessError, subprocess.TimeoutExpired):
        return False


def stop_speaking():
    """Stop any currently running speech."""
    try:
        subprocess.run(["pkill", "-f", "say"], timeout=2)
        
    except (subprocess.SubprocessError, subprocess.TimeoutExpired):
        pass  # Fail silently