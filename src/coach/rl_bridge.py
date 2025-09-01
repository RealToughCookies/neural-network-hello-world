"""
Bridge between RL last-hit policy and GSI data for attack timing suggestions.
"""

import logging
import torch
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path

from ..gsi.schema import GSISnapshot, CreepState
from .rules import CoachSuggestion

logger = logging.getLogger(__name__)


class LastHitRLBridge:
    """Bridge RL policy with GSI state for last-hit timing."""
    
    def __init__(self, policy_path: Optional[str] = None):
        self.policy_path = policy_path
        self.policy = None
        self.device = torch.device("cpu")
        
        # State tracking
        self.last_prediction_time = 0
        self.prediction_interval = 0.1  # 100ms
        
        # Load policy if path provided
        if policy_path:
            self.load_policy(policy_path)
    
    def load_policy(self, policy_path: str):
        """Load trained last-hit policy."""
        try:
            policy_file = Path(policy_path)
            if not policy_file.exists():
                logger.warning(f"Policy file not found: {policy_path}")
                return
            
            # Load checkpoint (assuming it's a saved model state_dict)
            checkpoint = torch.load(policy_path, map_location=self.device, weights_only=False)
            
            # Import policy class (would need to match your actual policy class)
            from src.rl.checkpoint import MultiHeadPolicy
            
            # Create policy with expected dimensions (adjust based on your actual obs dims)
            obs_dims = {"good": 12, "adv": 12}  # Placeholder - adjust to match your training
            n_act = 5
            
            self.policy = MultiHeadPolicy(obs_dims, n_act)
            
            # Load state dict 
            if isinstance(checkpoint, dict) and "model_state" in checkpoint:
                self.policy.load_state_dict(checkpoint["model_state"])
            else:
                self.policy.load_state_dict(checkpoint)
            
            self.policy.eval()
            logger.info(f"Loaded RL policy from {policy_path}")
            
        except Exception as e:
            logger.error(f"Failed to load RL policy: {e}")
            self.policy = None
    
    def gsi_to_rl_features(self, snapshot: GSISnapshot) -> Optional[np.ndarray]:
        """Convert GSI snapshot to RL policy input features."""
        if not snapshot.hero or not snapshot.player or not snapshot.map_state:
            return None
        
        try:
            # Extract relevant features for last-hit timing
            # This should match the feature extraction used during RL training
            
            features = []
            
            # Hero state
            features.extend([
                snapshot.hero.health / max(snapshot.hero.max_health, 1),  # HP ratio
                snapshot.hero.mana / max(snapshot.hero.max_mana, 1),     # Mana ratio
                snapshot.hero.level / 25.0,                              # Level (normalized)
            ])
            
            # Game time (normalized)
            game_time_minutes = snapshot.map_state.game_time / 60.0
            features.append(min(game_time_minutes / 60.0, 1.0))  # Normalize to 1 hour
            
            # Player stats
            features.extend([
                snapshot.player.last_hits / 100.0,  # CS (normalized)
                snapshot.player.denies / 50.0,      # Denies (normalized)  
                snapshot.player.gold / 10000.0,     # Gold (normalized)
            ])
            
            # Day/night cycle
            features.append(1.0 if snapshot.map_state.daytime else 0.0)
            
            # Pad or truncate to expected size (12 features to match training)
            while len(features) < 12:
                features.append(0.0)
            features = features[:12]
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Error converting GSI to RL features: {e}")
            return None
    
    def predict_action(self, features: np.ndarray, role: str = "good") -> Tuple[int, float]:
        """
        Predict action using RL policy.
        
        Returns:
            Tuple of (action, confidence) where action is 0-4 and confidence is 0-1
        """
        if self.policy is None:
            return 0, 0.0
        
        try:
            with torch.no_grad():
                # Convert to torch tensor
                obs_tensor = torch.tensor(features).unsqueeze(0).to(self.device)
                
                # Get action probabilities
                logits = self.policy.pi[role](obs_tensor)
                probs = torch.softmax(logits, dim=-1)
                
                # Get predicted action and confidence
                action = torch.argmax(probs, dim=-1).item()
                confidence = torch.max(probs).item()
                
                return action, confidence
                
        except Exception as e:
            logger.error(f"Error in RL prediction: {e}")
            return 0, 0.0
    
    def analyze_attack_timing(self, snapshot: GSISnapshot) -> List[CoachSuggestion]:
        """Analyze attack timing using RL policy."""
        suggestions = []
        
        if self.policy is None:
            return suggestions
        
        import time
        current_time = time.time()
        
        # Rate limit predictions
        if current_time - self.last_prediction_time < self.prediction_interval:
            return suggestions
        
        # Convert GSI to features
        features = self.gsi_to_rl_features(snapshot)
        if features is None:
            return suggestions
        
        try:
            # Get RL policy prediction
            action, confidence = self.predict_action(features, role="good")
            
            # Action mapping (adjust based on your actual action space)
            # 0: no-op, 1: left, 2: right, 3: down, 4: up
            # For last-hitting, we might interpret specific actions as "attack now"
            
            if action == 4 and confidence > 0.7:  # High confidence "attack" action
                suggestions.append(CoachSuggestion(
                    type="last_hit",
                    message=f"RL suggests: Attack now! (confidence: {confidence:.2f})",
                    priority=3,
                    confidence=confidence,
                    timestamp=snapshot.timestamp,
                    context={
                        "rl_action": action,
                        "rl_confidence": confidence,
                        "features": features.tolist()
                    }
                ))
            elif action == 3 and confidence > 0.6:  # Medium confidence positioning
                suggestions.append(CoachSuggestion(
                    type="positioning", 
                    message=f"RL suggests: Adjust position (confidence: {confidence:.2f})",
                    priority=2,
                    confidence=confidence,
                    timestamp=snapshot.timestamp,
                    context={
                        "rl_action": action,
                        "rl_confidence": confidence
                    }
                ))
            
            self.last_prediction_time = current_time
            
        except Exception as e:
            logger.error(f"Error in RL attack timing analysis: {e}")
        
        return suggestions


class CreepHealthTracker:
    """Track creep health and predict last-hit timing windows."""
    
    def __init__(self):
        self.creep_history = {}  # Track creep health over time
        self.damage_estimation = 60  # Estimated hero damage (rough)
        
    def update_creeps(self, snapshot: GSISnapshot):
        """Update creep health tracking."""
        if not hasattr(snapshot, 'creeps') or not snapshot.creeps:
            return
        
        current_time = snapshot.timestamp
        
        # Track health changes for each creep
        for creep_id, creep in snapshot.creeps.items():
            if creep_id not in self.creep_history:
                self.creep_history[creep_id] = []
            
            self.creep_history[creep_id].append({
                'time': current_time,
                'health': creep.health,
                'max_health': creep.max_health
            })
            
            # Keep only recent history (last 10 seconds)
            self.creep_history[creep_id] = [
                entry for entry in self.creep_history[creep_id] 
                if self._time_diff(current_time, entry['time']) < 10.0
            ]
    
    def predict_lasthit_window(self, creep: CreepState) -> Optional[float]:
        """Predict when creep will be in last-hit range."""
        if creep.health <= self.damage_estimation:
            return 0.0  # Ready now
        
        # Simple linear extrapolation based on health decay
        # In practice, you'd want more sophisticated prediction
        health_above_threshold = creep.health - self.damage_estimation
        
        # Assume roughly 20 damage per second from other creeps
        decay_rate = 20.0
        
        if decay_rate > 0:
            time_to_lasthit = health_above_threshold / decay_rate
            return time_to_lasthit
        
        return None
    
    def _time_diff(self, time1: str, time2: str) -> float:
        """Calculate time difference between ISO timestamps."""
        from datetime import datetime
        try:
            dt1 = datetime.fromisoformat(time1.replace('Z', '+00:00'))
            dt2 = datetime.fromisoformat(time2.replace('Z', '+00:00'))
            return abs((dt1 - dt2).total_seconds())
        except:
            return 0.0
    
    def get_lasthit_suggestions(self, snapshot: GSISnapshot) -> List[CoachSuggestion]:
        """Get suggestions based on creep health tracking."""
        suggestions = []
        
        if not hasattr(snapshot, 'creeps') or not snapshot.creeps:
            return suggestions
        
        # Find creeps ready for last-hitting
        ready_creeps = []
        for creep_id, creep in snapshot.creeps.items():
            if creep.team != snapshot.hero.team_id:  # Enemy creeps only
                lasthit_time = self.predict_lasthit_window(creep)
                if lasthit_time is not None and lasthit_time < 2.0:  # Within 2 seconds
                    ready_creeps.append((creep, lasthit_time))
        
        if ready_creeps:
            # Sort by urgency (soonest first)
            ready_creeps.sort(key=lambda x: x[1])
            creep, time_left = ready_creeps[0]
            
            if time_left < 0.5:
                priority = 3  # Urgent
                message = f"Last-hit NOW! {creep.name} ({creep.health} HP)"
            else:
                priority = 2
                message = f"Prepare last-hit: {creep.name} in {time_left:.1f}s ({creep.health} HP)"
            
            suggestions.append(CoachSuggestion(
                type="last_hit",
                message=message,
                priority=priority,
                confidence=0.8,
                timestamp=snapshot.timestamp,
                context={
                    "creep_name": creep.name,
                    "creep_health": creep.health,
                    "time_to_lasthit": time_left
                }
            ))
        
        return suggestions


class CombinedRLCoach:
    """Combined RL + rule-based coaching system."""
    
    def __init__(self, policy_path: Optional[str] = None):
        self.rl_bridge = LastHitRLBridge(policy_path)
        self.creep_tracker = CreepHealthTracker()
        
    def analyze(self, snapshot: GSISnapshot) -> List[CoachSuggestion]:
        """Analyze using both RL and rule-based approaches."""
        suggestions = []
        
        try:
            # Update creep tracking
            self.creep_tracker.update_creeps(snapshot)
            
            # Get RL-based suggestions
            rl_suggestions = self.rl_bridge.analyze_attack_timing(snapshot)
            suggestions.extend(rl_suggestions)
            
            # Get creep health-based suggestions
            creep_suggestions = self.creep_tracker.get_lasthit_suggestions(snapshot)
            suggestions.extend(creep_suggestions)
            
        except Exception as e:
            logger.error(f"Error in combined RL coach analysis: {e}")
        
        # Sort by priority and confidence
        suggestions.sort(key=lambda x: (x.priority, x.confidence), reverse=True)
        
        return suggestions