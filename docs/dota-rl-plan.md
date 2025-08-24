# Dota-Class Reinforcement Learning Research Plan

## What OpenAI Five Accomplished

OpenAI Five demonstrated that self-play reinforcement learning could master complex, multi-agent strategy games at a superhuman level. Key achievements:

**Technical Approach:**
- **Self-Play PPO**: Proximal Policy Optimization with agents learning against copies of themselves
- **LSTM Memory**: Recurrent networks to handle partial observability and long-term planning
- **No Human Data**: Pure RL without imitation learning or human demonstrations
- **Custom Dota API**: Direct game state access and action execution

**Scale & Infrastructure:**
- **Training Time**: ~10 months of continuous training
- **Compute**: ~180 in-game years per day on "Rapid" infrastructure
- **Hardware**: ~256 GPUs + ~128,000 CPU cores at peak
- **Outcome**: Defeated Team OG (reigning world champions) at The International 2019

**Sources:**
- [OpenAI Five Blog](https://openai.com/research/openai-five)
- [Dota 2 with Large Scale Deep RL Paper](https://arxiv.org/abs/1912.06680)

## Our Approach & Constraints

**Limitations:**
- Dota 2's game API is closed and unavailable for research
- OpenAI Five's full infrastructure is not reproducible at individual scale
- We cannot directly replicate the exact training environment

**Alternative Strategy:**
- **Primary Testbed**: [Google Research Football](https://github.com/google-research/football) - open-source, multi-agent, strategic
- **Backup Options**: [MicroRTS](https://github.com/Farama-Foundation/MicroRTS-Py) and [PettingZoo](https://pettingzoo.farama.org/) environments
- **Dota Integration**: Use [OpenDota API](https://docs.opendota.com/) for replay analysis, draft prediction, and item recommendation (imitation learning tasks)

**Core Techniques to Implement:**
- Self-play PPO with opponent policy freezing/mixing
- LSTM for temporal reasoning and partial observability
- Curriculum learning and reward shaping
- Multi-process rollout collection and vectorized training

## Research Roadmap

### Phase 1: PPO Self-Play Baseline
- Implement PPO with self-play on Google Research Football
- Start with stateless policies, migrate to LSTM
- Basic evaluation scripts and training metrics
- Target: Stable learning curve and basic strategic behavior

### Phase 2: Curriculum & Reward Engineering
- Design curriculum from simple scenarios to full matches
- Implement reward shaping for strategic objectives
- Multi-process rollout collection for sample efficiency
- Target: Consistent improvement over built-in AI opponents

### Phase 3: Scaling & Infrastructure
- Vectorized environment batching for throughput
- Optional integration with Ray RLlib for distributed training
- Hyperparameter tuning and architecture search
- Target: Training stability at larger scales

### Phase 4: Dota Integration
- OpenDota replay dataset pipeline for imitation learning
- Supervised learning heads for draft prediction and item builds
- Offline RL exploration on replay trajectories
- Target: Transferable insights from Dota professional play