"""
traffic_ppo.py
--------------
PPO Agent tối ưu cho bài toán điều khiển đèn giao thông.
Đã tinh chỉnh: learning rate, entropy, GAE, reward normalization.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


def _orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
        super().__init__()
        self.net = nn.Sequential(
            _orthogonal_init(nn.Linear(state_size, hidden_size), gain=np.sqrt(2)),
            nn.Tanh(),
            _orthogonal_init(nn.Linear(hidden_size, hidden_size), gain=np.sqrt(2)),
            nn.Tanh(),
            _orthogonal_init(nn.Linear(hidden_size, action_size), gain=0.01),
        )

    def forward(self, x):
        return self.net(x)


class Critic(nn.Module):
    def __init__(self, state_size, hidden_size=256):
        super().__init__()
        self.net = nn.Sequential(
            _orthogonal_init(nn.Linear(state_size, hidden_size), gain=np.sqrt(2)),
            nn.Tanh(),
            _orthogonal_init(nn.Linear(hidden_size, hidden_size), gain=np.sqrt(2)),
            nn.Tanh(),
            _orthogonal_init(nn.Linear(hidden_size, 1), gain=1.0),
        )

    def forward(self, x):
        return self.net(x)


class PPOAgent:
    def __init__(
        self,
        state_size,
        action_size,
        hidden_size=256,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        entropy_coef=0.010,
        value_coef=0.5,
        max_grad_norm=0.5,
        rollout_steps=128,
        mini_batch_size=64,
        epochs=8,
        lr_decay_steps=300_000,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_steps = rollout_steps
        self.mini_batch_size = mini_batch_size
        self.epochs = epochs
        self.total_steps = 0
        self.lr_decay_steps = lr_decay_steps
        self.base_lr = learning_rate

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(state_size, action_size, hidden_size).to(self.device)
        self.critic = Critic(state_size, hidden_size).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate, eps=1e-5)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate, eps=1e-5)

        self._reset_buffer()

    def _reset_buffer(self):
        self.states: list = []
        self.actions: list = []
        self.log_probs: list = []
        self.values: list = []
        self.rewards: list = []
        self.dones: list = []

    def _decay_lr(self):
        """Linear LR decay"""
        frac = max(0.0, 1.0 - self.total_steps / self.lr_decay_steps)
        lr = self.base_lr * frac + 1e-6
        for opt in (self.actor_optimizer, self.critic_optimizer):
            for pg in opt.param_groups:
                pg["lr"] = lr

    def select_action(self, state, training=True):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.actor(state_t)
            value = self.critic(state_t)
        dist = Categorical(logits=logits)
        action = dist.sample() if training else torch.argmax(dist.probs, dim=-1)
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item(), value.item()

    def evaluate_action(self, state, action):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.actor(state_t)
            value = self.critic(state_t)
        dist = Categorical(logits=logits)
        action_t = torch.tensor([action], device=self.device)
        log_prob = dist.log_prob(action_t)
        return log_prob.item(), value.item()

    def get_value(self, state):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            value = self.critic(state_t)
        return value.item()

    def store_transition(self, state, action, log_prob, value, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
        self.total_steps += 1

    def ready(self):
        return len(self.states) >= self.rollout_steps

    def _compute_gae(self, last_value):
        """Tính GAE trên reward đã được shaping ở môi trường."""
        T = len(self.rewards)
        advantages = np.zeros(T, dtype=np.float32)
        values_arr = np.array(self.values + [last_value], dtype=np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            mask = 0.0 if self.dones[t] else 1.0
            delta = self.rewards[t] + self.gamma * values_arr[t + 1] * mask - values_arr[t]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages[t] = gae

        returns = advantages + np.array(self.values, dtype=np.float32)

        return advantages, returns

    def update(self, last_value=0.0):
        if not self.states:
            return None

        self._decay_lr()
        advantages, returns = self._compute_gae(last_value)

        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        old_values = torch.FloatTensor(self.values).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)
        advantages_t = torch.FloatTensor(advantages).to(self.device)

        # Normalize advantages
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        batch_size = states.size(0)
        indices = np.arange(batch_size)

        for _ in range(self.epochs):
            np.random.shuffle(indices)
            for start in range(0, batch_size, self.mini_batch_size):
                end = min(start + self.mini_batch_size, batch_size)
                mb_idx = indices[start:end]

                mb_states = states[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_lp = old_log_probs[mb_idx]
                mb_returns = returns_t[mb_idx]
                mb_adv = advantages_t[mb_idx]
                mb_old_values = old_values[mb_idx]

                # Actor loss
                logits = self.actor(mb_states)
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - mb_old_lp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                actor_loss = policy_loss - self.entropy_coef * entropy

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # Critic loss với clipping
                new_values = self.critic(mb_states).squeeze(-1)
                v_clipped = mb_old_values + torch.clamp(
                    new_values - mb_old_values, -self.clip_eps, self.clip_eps
                )
                value_loss1 = (mb_returns - new_values).pow(2)
                value_loss2 = (mb_returns - v_clipped).pow(2)
                value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()

                critic_loss = self.value_coef * value_loss
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

        self._reset_buffer()
        return True

    def save(self, filepath):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "total_steps": self.total_steps,
        }, filepath)

    def load(self, filepath):
        try:
            ckpt = torch.load(filepath, map_location=self.device)
            self.actor.load_state_dict(ckpt["actor"])
            self.critic.load_state_dict(ckpt["critic"])
            self.actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
            self.critic_optimizer.load_state_dict(ckpt["critic_optimizer"])
            self.total_steps = ckpt.get("total_steps", 0)
            return True
        except Exception:
            return False
