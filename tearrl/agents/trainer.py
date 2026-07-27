"""Production shared-weight PPO trainer for the frozen TEAR-RL conditions.

One transition is one agent's own decision. The other role's intervening turn
is inside the dynamics. Trajectories are partitioned by (episode, agent,
own-decision timestep) before shared-weight optimization.

The frozen protocol names a scalar cost for C2/C3 but never defines its
construction. Those conditions require a caller-supplied negative-cost
provider; C3/C3-IM also require the calibrated PID target. No default is
invented. C1, C4, C5, and C6 are fully specified here.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Sequence

import numpy as np
import torch
from torch import nn

from ..conscience import CH_DW_OTHER, ResponsibilityState
from ..spec import D_MAX, N_ACTIONS, Verb
from ..witness.targets import ACTION_DESCRIPTORS, NORM_DWELFARE, WITNESS_DIM
from ..world import TearWorld
from .inputs import PolicyInput, policy_input
from .networks import C3IMPolicy, Critic, PolicyNet, Selector, c3im_equalized_width
from .ppo import (
    MINIBATCH, TRANSITION_BUDGET, GradientDiagnostics, assign_flat_grad,
    clipped_surrogate, compute_gae, conscience_reward, entropy_bonus, normalize,
    pcgrad,
)

ROLLOUT_BATCH = 16_384
SMOKE_LIMIT = 2_000
PID_KP, PID_KI, PID_KD, PID_LAMBDA_MAX = 0.5, 0.05, 0.0, 10.0


class ProtocolDecisionRequired(RuntimeError):
    """A frozen document leaves a scientific choice undefined."""


class Condition(str, Enum):
    C1, C2, C3, C3_IM, C4, C5, C6 = "C1", "C2", "C3", "C3-IM", "C4", "C5", "C6"

    @property
    def uses_witness(self) -> bool:
        return self in {self.C3_IM, self.C4, self.C5, self.C6}

    @property
    def carries_responsibility(self) -> bool:
        return self in {self.C5, self.C6}

    @property
    def uses_conscience_stream(self) -> bool:
        return self in {self.C3, self.C3_IM, self.C5, self.C6}

    @property
    def needs_scalar_cost_provider(self) -> bool:
        return self in {self.C2, self.C3, self.C3_IM}


@dataclass(frozen=True, slots=True)
class TrainerConfig:
    learning_rate: float = 3e-4
    entropy_beta: float = 0.003
    epochs: int = 2
    minibatch: int = MINIBATCH
    rollout_batch: int = ROLLOUT_BATCH
    transition_budget: int = TRANSITION_BUDGET
    seed: int = 0
    device: str = "cpu"
    pid_target: float | None = None

    def __post_init__(self) -> None:
        if self.learning_rate not in {1e-4, 3e-4}:
            raise ValueError("learning_rate must be a frozen B.4 grid value")
        if self.entropy_beta not in {0.003, 0.01}:
            raise ValueError("entropy_beta must be a frozen B.4 grid value")
        if self.epochs not in {2, 4}:
            raise ValueError("epochs must be a frozen B.4 grid value")
        if min(self.minibatch, self.rollout_batch, self.transition_budget) <= 0:
            raise ValueError("batch sizes and transition budget must be positive")
        if self.pid_target is not None and self.pid_target < 0:
            raise ValueError("pid_target must be non-negative")


@dataclass(slots=True)
class PIDController:
    target: float
    value: float = 0.0
    integral: float = 0.0
    previous_error: float = 0.0

    def update(self, observed_cost: float) -> float:
        if not np.isfinite(observed_cost):
            raise ValueError("observed PID cost must be finite")
        error = observed_cost - self.target
        self.integral += error
        derivative = error - self.previous_error
        self.value = float(np.clip(
            self.value + PID_KP * error + PID_KI * self.integral + PID_KD * derivative,
            0.0, PID_LAMBDA_MAX,
        ))
        self.previous_error = error
        return self.value


@dataclass(slots=True)
class Transition:
    episode: int
    agent: int
    own_t: int
    obs: np.ndarray
    c: np.ndarray
    q_other: np.ndarray
    witness: np.ndarray
    mask: np.ndarray
    action: int
    old_logprob: float
    reward_task: float
    reward_conscience: float
    value_task: float
    value_conscience: float
    next_value_task: float = 0.0
    next_value_conscience: float = 0.0
    terminated: bool = False
    truncated: bool = False

    @property
    def transition_id(self) -> tuple[int, int, int]:
        return self.episode, self.agent, self.own_t

    def finish(self, *, next_value_task: float, next_value_conscience: float,
               terminated: bool, truncated: bool) -> None:
        if terminated and truncated:
            raise ValueError("transition cannot be both terminated and truncated")
        self.next_value_task = float(next_value_task)
        self.next_value_conscience = float(next_value_conscience)
        self.terminated, self.truncated = bool(terminated), bool(truncated)


@dataclass(slots=True)
class PreparedBatch:
    rows: tuple[Transition, ...]
    ids: tuple[tuple[int, int, int], ...]
    obs: torch.Tensor
    c: torch.Tensor
    q_other: torch.Tensor
    witness: torch.Tensor
    mask: torch.Tensor
    action: torch.Tensor
    old_logprob: torch.Tensor
    advantage_task: torch.Tensor
    return_task: torch.Tensor
    advantage_conscience: torch.Tensor | None
    return_conscience: torch.Tensor | None
    mean_cost: float

    def __len__(self) -> int:
        return len(self.rows)


@dataclass(slots=True)
class UpdateReport:
    transitions: int
    actor_loss: float
    task_critic_loss: float
    conscience_critic_loss: float
    entropy: float
    transition_visits: dict[tuple[int, int, int], int]
    gradient_diagnostics: list[GradientDiagnostics] = field(default_factory=list)
    lagrange_multiplier: float | None = None

    @property
    def finite(self) -> bool:
        return bool(np.isfinite([
            self.actor_loss, self.task_critic_loss,
            self.conscience_critic_loss, self.entropy,
        ]).all())


@dataclass(slots=True)
class SmokeReport:
    condition: str
    seed: int
    transitions: int
    per_agent: dict[int, int]
    parameters_changed: bool
    update: UpdateReport


ScalarCostProvider = Callable[[np.ndarray, ResponsibilityState, Verb], float]


def carry_conscience_reward(responsibility: ResponsibilityState,
                             prediction: np.ndarray,
                             action: Verb) -> tuple[float, float]:
    """Adopted Amendment 002 timing: restore, charge carry debt, then add harm."""
    prediction = np.asarray(prediction, dtype=np.float64)
    benefit = max(0.0, float(prediction[CH_DW_OTHER]))
    restored = benefit if Verb(action) is Verb.REPAIR else 0.0
    d_carry = float(np.clip(responsibility.d - restored, 0.0, D_MAX))
    return conscience_reward(prediction, d_carry), d_carry


class ConditionRuntime:
    """Networks, frozen instrument, optimizers, and sealed policy path."""

    def __init__(self, condition: Condition | str, config: TrainerConfig,
                 witness: nn.Module | None = None,
                 scalar_cost_provider: ScalarCostProvider | None = None) -> None:
        self.condition, self.config = Condition(condition), config
        self.device = torch.device(config.device)
        torch.manual_seed(config.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(config.seed)
        self.scalar_cost_provider = scalar_cost_provider
        if self.condition.uses_witness and witness is None:
            raise ValueError(f"{self.condition.value} requires the frozen witness")
        if self.condition.needs_scalar_cost_provider and scalar_cost_provider is None:
            raise ProtocolDecisionRequired(
                f"{self.condition.value} requires an explicit scalar-cost provider; "
                "the frozen protocol does not define C-bar's construction"
            )
        if self.condition in {Condition.C3, Condition.C3_IM} and config.pid_target is None:
            raise ProtocolDecisionRequired(
                f"{self.condition.value} requires the calibrated PID target"
            )
        self.witness = witness
        if witness is not None:
            witness.to(self.device).eval()
            for parameter in witness.parameters():
                parameter.requires_grad_(False)
        hidden = c3im_equalized_width()
        if self.condition in {Condition.C1, Condition.C2, Condition.C3}:
            self.actor: nn.Module = PolicyNet(hidden)
        elif self.condition is Condition.C3_IM:
            self.actor = C3IMPolicy(hidden)
        else:
            self.actor = Selector()
        self.actor.to(self.device)
        use_c = self.condition.carries_responsibility
        self.task_critic = Critic(use_c=use_c).to(self.device)
        self.conscience_critic = (
            Critic(use_c=use_c).to(self.device)
            if self.condition.uses_conscience_stream else None
        )
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=config.learning_rate)
        self.task_critic_optimizer = torch.optim.Adam(
            self.task_critic.parameters(), lr=config.learning_rate)
        self.conscience_critic_optimizer = (
            torch.optim.Adam(self.conscience_critic.parameters(), lr=config.learning_rate)
            if self.conscience_critic is not None else None
        )
        self.pid = (PIDController(float(config.pid_target))
                    if self.condition in {Condition.C3, Condition.C3_IM} else None)
        self._sampling = torch.Generator(device="cpu").manual_seed(config.seed)
        self._shuffling = torch.Generator(device="cpu").manual_seed(config.seed ^ 0x7EA4)

    def _effective_c(self, sanitized: PolicyInput) -> np.ndarray:
        return (sanitized.c.as_array() if self.condition.carries_responsibility
                else np.zeros(4, dtype=np.float32))

    @torch.no_grad()
    def witness_matrix(self, sanitized: PolicyInput) -> np.ndarray:
        if not isinstance(sanitized, PolicyInput):
            raise TypeError("trainer policy path accepts PolicyInput only")
        if self.witness is None:
            return np.zeros((N_ACTIONS, WITNESS_DIM), dtype=np.float32)
        obs = torch.tensor(sanitized.obs, dtype=torch.float32, device=self.device)
        rows = obs.unsqueeze(0).expand(N_ACTIONS, -1)
        descriptors = torch.tensor(ACTION_DESCRIPTORS, device=self.device)
        prediction = self.witness.predict(torch.cat([rows, descriptors], dim=1))
        return prediction.cpu().numpy().astype(np.float32, copy=False)

    def _tensorize(self, sanitized: PolicyInput, witness: np.ndarray):
        return (
            torch.tensor([sanitized.obs], dtype=torch.float32, device=self.device),
            torch.tensor(self._effective_c(sanitized)[None, :], device=self.device),
            torch.tensor([sanitized.q_other], dtype=torch.float32, device=self.device),
            torch.tensor(witness[None], dtype=torch.float32, device=self.device),
            torch.tensor([sanitized.mask], dtype=torch.bool, device=self.device),
        )

    def logits(self, sanitized: PolicyInput,
               witness: np.ndarray | None = None) -> torch.Tensor:
        if not isinstance(sanitized, PolicyInput):
            raise TypeError("trainer policy path accepts PolicyInput only")
        matrix = self.witness_matrix(sanitized) if witness is None else witness
        obs, c, q, matrix_t, mask = self._tensorize(sanitized, matrix)
        if self.condition in {Condition.C1, Condition.C2, Condition.C3}:
            return self.actor(obs, q, mask)
        if self.condition is Condition.C3_IM:
            return self.actor(obs, matrix_t, q, mask)
        return self.actor(matrix_t, c, q, mask)

    @torch.no_grad()
    def _critic_values(self, sanitized: PolicyInput) -> tuple[float, float]:
        obs = torch.tensor([sanitized.obs], dtype=torch.float32, device=self.device)
        c = torch.tensor(self._effective_c(sanitized)[None, :], device=self.device)
        task = float(self.task_critic(obs, c if self.task_critic.use_c else None)[0])
        conscience = 0.0 if self.conscience_critic is None else float(
            self.conscience_critic(obs, c if self.conscience_critic.use_c else None)[0])
        return task, conscience

    @torch.no_grad()
    def decide(self, sanitized: PolicyInput) -> tuple[int, float, float, float, np.ndarray]:
        if not isinstance(sanitized, PolicyInput):
            raise TypeError("trainer policy path accepts PolicyInput only")
        matrix = self.witness_matrix(sanitized)
        logits = self.logits(sanitized, matrix)[0]
        probabilities = torch.softmax(logits, dim=-1).cpu()
        action = int(torch.multinomial(probabilities, 1, generator=self._sampling)[0])
        logprob = float(torch.log_softmax(logits, dim=-1)[action])
        task_value, conscience_value = self._critic_values(sanitized)
        return action, logprob, task_value, conscience_value, matrix

    def reward_conscience(self, prediction: np.ndarray,
                           responsibility: ResponsibilityState,
                           action: Verb) -> float:
        if self.condition in {Condition.C1, Condition.C4}:
            return 0.0
        if self.condition in {Condition.C5, Condition.C6}:
            return carry_conscience_reward(responsibility, prediction, action)[0]
        assert self.scalar_cost_provider is not None
        reward = float(self.scalar_cost_provider(
            prediction, responsibility.clone(), action))
        if not np.isfinite(reward) or reward > 0.0:
            raise ValueError("scalar-cost provider must return finite non-positive reward")
        return reward

    def batch_logits(self, obs: torch.Tensor, c: torch.Tensor,
                     q_other: torch.Tensor, witness: torch.Tensor,
                     mask: torch.Tensor) -> torch.Tensor:
        if self.condition in {Condition.C1, Condition.C2, Condition.C3}:
            return self.actor(obs, q_other, mask)
        if self.condition is Condition.C3_IM:
            return self.actor(obs, witness, q_other, mask)
        return self.actor(witness, c, q_other, mask)

class RolloutCollector:
    """Exact-budget collector with per-agent terminal/truncation provenance."""

    def __init__(self, runtime: ConditionRuntime) -> None:
        self.runtime = runtime

    def collect(self, transitions: int, seed: int | None = None) -> list[Transition]:
        if transitions <= 0:
            raise ValueError("transition request must be positive")
        seed = self.runtime.config.seed if seed is None else seed
        quotas = {0: (transitions + 1) // 2, 1: transitions // 2}
        completed: dict[int, list[Transition]] = {0: [], 1: []}
        episode = 0

        while len(completed[0]) + len(completed[1]) < transitions:
            world = TearWorld(seed=seed + episode * 1_000_003)
            responsibility = {0: ResponsibilityState(), 1: ResponsibilityState()}
            last_other_welfare = {
                0: world.state.agents[1].welfare,
                1: world.state.agents[0].welfare,
            }
            own_clock = {0: 0, 1: 0}
            pending: dict[int, Transition | None] = {0: None, 1: None}

            while not world.state.done:
                actor = world.state.actor_index
                c_for_policy = (responsibility[actor]
                                if self.runtime.condition.carries_responsibility
                                else None)
                sanitized = policy_input(world, actor=actor, c=c_for_policy)

                previous = pending[actor]
                if previous is not None:
                    next_task, next_conscience = self.runtime._critic_values(sanitized)
                    stops_here = len(completed[actor]) + 1 >= quotas[actor]
                    previous.finish(
                        next_value_task=next_task,
                        next_value_conscience=next_conscience,
                        terminated=False,
                        truncated=stops_here,
                    )
                    completed[actor].append(previous)
                    pending[actor] = None

                if all(len(completed[i]) >= quotas[i] for i in (0, 1)):
                    break

                should_record = len(completed[actor]) < quotas[actor]
                action_index, logprob, value_task, value_c, matrix = (
                    self.runtime.decide(sanitized)
                )
                action = Verb(action_index)
                before_other = world.state.agents[1 - actor].welfare
                result = world.step(action)
                prediction = matrix[action_index]
                reward_c = self.runtime.reward_conscience(
                    prediction, responsibility[actor], action)
                intervening_change = (
                    before_other - last_other_welfare[actor]
                ) / NORM_DWELFARE
                responsibility[actor].update(prediction, action, intervening_change)
                last_other_welfare[actor] = world.state.agents[1 - actor].welfare

                if should_record:
                    pending[actor] = Transition(
                        episode=episode,
                        agent=actor,
                        own_t=own_clock[actor],
                        obs=np.asarray(sanitized.obs, dtype=np.float32),
                        c=self.runtime._effective_c(sanitized),
                        q_other=np.asarray(sanitized.q_other, dtype=np.float32),
                        witness=matrix.copy(),
                        mask=np.asarray(sanitized.mask, dtype=bool),
                        action=action_index,
                        old_logprob=logprob,
                        reward_task=float(result.reward_task),
                        reward_conscience=reward_c,
                        value_task=value_task,
                        value_conscience=value_c,
                    )
                own_clock[actor] += 1

                if result.done:
                    for index in (0, 1):
                        tail = pending[index]
                        if tail is None:
                            continue
                        tail.finish(
                            next_value_task=0.0,
                            next_value_conscience=0.0,
                            terminated=True,
                            truncated=False,
                        )
                        completed[index].append(tail)
                        pending[index] = None
            episode += 1

        rows = completed[0] + completed[1]
        rows.sort(key=lambda row: row.transition_id)
        if len(rows) != transitions:
            raise AssertionError(
                f"collector produced {len(rows)} rows for budget {transitions}")
        ids = [row.transition_id for row in rows]
        if len(set(ids)) != len(ids):
            raise AssertionError("duplicate transition IDs in rollout")
        if any(not (row.terminated or row.truncated) for row in self._tails(rows)):
            raise AssertionError("every trajectory tail needs explicit provenance")
        return rows

    @staticmethod
    def _tails(rows: Sequence[Transition]) -> list[Transition]:
        grouped: dict[tuple[int, int], list[Transition]] = defaultdict(list)
        for row in rows:
            grouped[(row.episode, row.agent)].append(row)
        return [max(group, key=lambda row: row.own_t)
                for group in grouped.values()]


def _group_rows(rows: Sequence[Transition]) -> dict[tuple[int, int], list[Transition]]:
    grouped: dict[tuple[int, int], list[Transition]] = defaultdict(list)
    for row in rows:
        grouped[(row.episode, row.agent)].append(row)
    for group in grouped.values():
        group.sort(key=lambda row: row.own_t)
    return grouped


def prepare_batch(runtime: ConditionRuntime,
                  rows: Sequence[Transition]) -> PreparedBatch:
    """Compute per-agent GAE, then normalize streams once over the merged batch."""
    if not rows:
        raise ValueError("cannot prepare empty rollout")
    ids = [row.transition_id for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("transition IDs must be unique before optimization")

    task_adv: dict[tuple[int, int, int], float] = {}
    task_ret: dict[tuple[int, int, int], float] = {}
    conscience_adv: dict[tuple[int, int, int], float] = {}
    conscience_ret: dict[tuple[int, int, int], float] = {}

    for group in _group_rows(rows).values():
        for earlier, later in zip(group, group[1:]):
            if earlier.terminated or earlier.truncated:
                raise ValueError("trajectory continues after terminal/truncated row")
            if later.own_t != earlier.own_t + 1:
                raise ValueError("own-decision clock has a trajectory gap")
            if not np.isclose(earlier.next_value_task, later.value_task, atol=1e-5):
                raise ValueError("task next-value provenance mismatch")
            if (runtime.condition.uses_conscience_stream and not np.isclose(
                    earlier.next_value_conscience, later.value_conscience,
                    atol=1e-5)):
                raise ValueError("conscience next-value provenance mismatch")
        if not (group[-1].terminated or group[-1].truncated):
            raise ValueError("trajectory tail lacks provenance")

        terminated = np.asarray([row.terminated for row in group], dtype=bool)
        rewards = np.asarray([
            row.reward_task + row.reward_conscience
            if runtime.condition is Condition.C2 else row.reward_task
            for row in group
        ], dtype=np.float64)
        values = np.asarray([row.value_task for row in group], dtype=np.float64)
        last_task = group[-1].next_value_task if group[-1].truncated else 0.0
        adv, returns = compute_gae(
            rewards, values, terminated, last_value=last_task)
        for row, advantage, returned in zip(group, adv, returns):
            task_adv[row.transition_id] = float(advantage)
            task_ret[row.transition_id] = float(returned)

        if runtime.condition.uses_conscience_stream:
            rewards_c = np.asarray(
                [row.reward_conscience for row in group], dtype=np.float64)
            values_c = np.asarray(
                [row.value_conscience for row in group], dtype=np.float64)
            last_c = (group[-1].next_value_conscience
                      if group[-1].truncated else 0.0)
            adv_c, returns_c = compute_gae(
                rewards_c, values_c, terminated, last_value=last_c)
            for row, advantage, returned in zip(group, adv_c, returns_c):
                conscience_adv[row.transition_id] = float(advantage)
                conscience_ret[row.transition_id] = float(returned)

    device, ordered = runtime.device, tuple(rows)
    advantage_task = normalize(torch.tensor(
        [task_adv[row.transition_id] for row in ordered], device=device)).float()
    return_task = torch.tensor(
        [task_ret[row.transition_id] for row in ordered],
        dtype=torch.float32, device=device)
    advantage_c_tensor = return_c_tensor = None
    if runtime.condition.uses_conscience_stream:
        advantage_c_tensor = normalize(torch.tensor(
            [conscience_adv[row.transition_id] for row in ordered],
            device=device)).float()
        return_c_tensor = torch.tensor(
            [conscience_ret[row.transition_id] for row in ordered],
            dtype=torch.float32, device=device)

    return PreparedBatch(
        rows=ordered,
        ids=tuple(row.transition_id for row in ordered),
        obs=torch.tensor(np.stack([row.obs for row in ordered]), device=device),
        c=torch.tensor(np.stack([row.c for row in ordered]), device=device),
        q_other=torch.tensor(np.stack([row.q_other for row in ordered]), device=device),
        witness=torch.tensor(np.stack([row.witness for row in ordered]), device=device),
        mask=torch.tensor(np.stack([row.mask for row in ordered]), device=device),
        action=torch.tensor([row.action for row in ordered],
                            dtype=torch.long, device=device),
        old_logprob=torch.tensor([row.old_logprob for row in ordered],
                                 dtype=torch.float32, device=device),
        advantage_task=advantage_task,
        return_task=return_task,
        advantage_conscience=advantage_c_tensor,
        return_conscience=return_c_tensor,
        mean_cost=float(np.mean([-row.reward_conscience for row in ordered])),
    )

def _flat_autograd(outputs: torch.Tensor,
                   parameters: Sequence[nn.Parameter],
                   retain_graph: bool) -> torch.Tensor:
    gradients = torch.autograd.grad(
        outputs, parameters, retain_graph=retain_graph, allow_unused=False)
    return torch.cat([gradient.reshape(-1) for gradient in gradients])


def _critic_step(critic: Critic, optimizer: torch.optim.Optimizer,
                 obs: torch.Tensor, c: torch.Tensor,
                 returns: torch.Tensor) -> float:
    optimizer.zero_grad(set_to_none=True)
    predicted = critic(obs, c if critic.use_c else None)
    loss = torch.mean((predicted - returns) ** 2)
    if not torch.isfinite(loss):
        raise FloatingPointError("critic loss is non-finite")
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def update(runtime: ConditionRuntime, batch: PreparedBatch) -> UpdateReport:
    """Run PPO epochs; every ID appears exactly once per epoch, including tails."""
    n = len(batch)
    visits: Counter[tuple[int, int, int]] = Counter()
    actor_losses, task_losses = [], []
    conscience_losses, entropies = [], []
    diagnostics: list[GradientDiagnostics] = []
    actor_parameters = tuple(runtime.actor.parameters())

    for _epoch in range(runtime.config.epochs):
        order = torch.randperm(n, generator=runtime._shuffling).tolist()
        for start in range(0, n, runtime.config.minibatch):
            indices_cpu = order[start:start + runtime.config.minibatch]
            indices = torch.tensor(
                indices_cpu, dtype=torch.long, device=runtime.device)
            for index in indices_cpu:
                visits[batch.ids[index]] += 1

            obs, c = batch.obs[indices], batch.c[indices]
            q, witness = batch.q_other[indices], batch.witness[indices]
            mask, actions = batch.mask[indices], batch.action[indices]
            logits = runtime.batch_logits(obs, c, q, witness, mask)
            log_probs = torch.log_softmax(logits, dim=-1).gather(
                1, actions.unsqueeze(1)).squeeze(1)
            entropy = entropy_bonus(logits, mask)
            j_task = clipped_surrogate(
                log_probs, batch.old_logprob[indices],
                batch.advantage_task[indices]
            ) + runtime.config.entropy_beta * entropy

            runtime.actor_optimizer.zero_grad(set_to_none=True)
            if runtime.condition.uses_conscience_stream:
                assert batch.advantage_conscience is not None
                j_conscience = clipped_surrogate(
                    log_probs, batch.old_logprob[indices],
                    batch.advantage_conscience[indices])
                if runtime.condition is Condition.C6:
                    g_task = _flat_autograd(
                        j_task, actor_parameters, retain_graph=True)
                    g_conscience = _flat_autograd(
                        j_conscience, actor_parameters, retain_graph=False)
                    projected, diagnostic = pcgrad(g_task, g_conscience)
                    diagnostics.append(diagnostic)
                    # Adam minimizes; registered g_T/g_C are ascent directions.
                    assign_flat_grad(
                        actor_parameters, -(g_conscience + projected))
                    actor_loss = -(j_task.detach() + j_conscience.detach())
                else:
                    multiplier = (runtime.pid.value
                                  if runtime.condition in {
                                      Condition.C3, Condition.C3_IM
                                  } else 1.0)
                    actor_loss = -(j_task + multiplier * j_conscience)
                    actor_loss.backward()
            else:
                actor_loss = -j_task
                actor_loss.backward()

            if not torch.isfinite(actor_loss):
                raise FloatingPointError("actor loss is non-finite")
            if any(parameter.grad is None or
                   not torch.isfinite(parameter.grad).all()
                   for parameter in actor_parameters):
                raise FloatingPointError("actor gradient is missing/non-finite")
            runtime.actor_optimizer.step()

            task_loss = _critic_step(
                runtime.task_critic, runtime.task_critic_optimizer,
                obs, c, batch.return_task[indices])
            conscience_loss = 0.0
            if runtime.conscience_critic is not None:
                assert runtime.conscience_critic_optimizer is not None
                assert batch.return_conscience is not None
                conscience_loss = _critic_step(
                    runtime.conscience_critic,
                    runtime.conscience_critic_optimizer,
                    obs, c, batch.return_conscience[indices])

            actor_losses.append(float(actor_loss.detach()))
            task_losses.append(task_loss)
            conscience_losses.append(conscience_loss)
            entropies.append(float(entropy.detach()))

    expected = runtime.config.epochs
    if set(visits) != set(batch.ids) or any(
            count != expected for count in visits.values()):
        raise AssertionError("transition-ID accounting failed across PPO epochs")
    lagrange = None if runtime.pid is None else runtime.pid.update(batch.mean_cost)
    report = UpdateReport(
        transitions=n,
        actor_loss=float(np.mean(actor_losses)),
        task_critic_loss=float(np.mean(task_losses)),
        conscience_critic_loss=float(np.mean(conscience_losses)),
        entropy=float(np.mean(entropies)),
        transition_visits=dict(visits),
        gradient_diagnostics=diagnostics,
        lagrange_multiplier=lagrange,
    )
    if not report.finite:
        raise FloatingPointError("update report contains non-finite values")
    return report


def parameter_vector(module: nn.Module) -> torch.Tensor:
    return torch.cat([
        parameter.detach().reshape(-1).cpu()
        for parameter in module.parameters()
    ])


def run_smoke(runtime: ConditionRuntime, transitions: int = 64,
              seed: int | None = None) -> SmokeReport:
    if transitions > SMOKE_LIMIT:
        raise ValueError(f"smoke runs are capped at {SMOKE_LIMIT} transitions")
    before = parameter_vector(runtime.actor).clone()
    rows = RolloutCollector(runtime).collect(transitions, seed=seed)
    report = update(runtime, prepare_batch(runtime, rows))
    changed = not torch.equal(before, parameter_vector(runtime.actor))
    if not changed:
        raise AssertionError("smoke optimizer did not change actor parameters")
    return SmokeReport(
        condition=runtime.condition.value,
        seed=runtime.config.seed if seed is None else seed,
        transitions=len(rows),
        per_agent=dict(Counter(row.agent for row in rows)),
        parameters_changed=changed,
        update=report,
    )
