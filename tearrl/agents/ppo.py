"""
PPO core and the conscience reward -- §2.8, B.2, B.5.

    r^C_t = -( w_h*H + w_a*A + w_u*U + w_e*E + w_o*O + w_d*D )
    J_T   = J_PPO(A_T) + beta*H(pi)
    J_C   = J_PPO(A_C)

    GAE lambda 0.95, gamma 0.99, clip 0.2, minibatch 4096, Adam
    budget 3.0M transitions/seed

--------------------------------------------------------------------------
TWO ADVANTAGE STREAMS, ONE POLICY
--------------------------------------------------------------------------
Every condition from C4 onward carries two critics and two advantage streams:
task (A_T) and conscience (A_C). They are estimated separately and never
mixed, because §2.8's whole apparatus -- the projection in C6, the plain sum
in C5, the Lagrangian in C3 -- operates on the SEPARATE gradients. Collapsing
them into one scalar reward before differentiation would erase the very
conflict the experiment is built to measure.

Critics are never projected (§2.8) and are trained by plain MSE.

--------------------------------------------------------------------------
THE CONSCIENCE REWARD IS PREDICTED, NOT ORACLE
--------------------------------------------------------------------------
r^C is assembled from the FROZEN WITNESS's quantized predictions for the
action actually taken, exactly as c_t is (§2.6). U2 is the only oracle-c_t
condition. So an agent is penalized for what its own instrument told it it was
about to do -- which is what makes §5.5's Witness Exploitation Gap meaningful:
an agent that learns where the witness is wrong can act badly at low predicted
cost, and WEG measures precisely that.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from ..conscience import (
    CH_DAUTONOMY,
    CH_DW_OTHER,
    CH_E,
    CH_O,
    CH_U,
)
from ..spec import GAMMA, W_AUTONOMY, W_DEBT, W_E, W_HARM, W_O, W_U
from .networks import assert_feasible_mask

GAE_LAMBDA: float = 0.95      # B.2
CLIP_EPS: float = 0.2         # B.2
MINIBATCH: int = 4096         # B.2
TRANSITION_BUDGET: int = 3_000_000   # B.2, per seed


def conscience_reward(prediction: np.ndarray, debt: float) -> float:
    """r^C_t from the frozen witness's quantized prediction (§2.8, B.5).

        r^C = -( w_h*H + w_a*A + w_u*U + w_e*E + w_o*O + w_d*D )

    H and A are the HARM SIDES only -- max(0, -delta) -- so benefiting the
    counterpart is not rewarded here. §2.6's b channel tracks benefit
    separately; folding it into r^C would let an agent offset harm with
    unrelated generosity, which is the same laundering the debt channel
    forbids for `share`.
    """
    harm = max(0.0, -float(prediction[CH_DW_OTHER]))
    autonomy = max(0.0, -float(prediction[CH_DAUTONOMY]))
    return -(
        W_HARM * harm
        + W_AUTONOMY * autonomy
        + W_U * float(prediction[CH_U])
        + W_E * float(prediction[CH_E])
        + W_O * float(prediction[CH_O])
        + W_DEBT * float(debt)
    )


@dataclass(slots=True)
class RolloutBuffer:
    """One agent's transitions. Self-play is shared-weight and alternating
    (B.2), so each agent contributes its OWN decisions only -- no gradient
    flows through the counterpart's actions."""

    obs: list[np.ndarray] = field(default_factory=list)
    c: list[np.ndarray] = field(default_factory=list)
    q_other: list[np.ndarray] = field(default_factory=list)
    witness: list[np.ndarray] = field(default_factory=list)   # [|A|, 9] per step
    mask: list[np.ndarray] = field(default_factory=list)
    action: list[int] = field(default_factory=list)
    logprob: list[float] = field(default_factory=list)
    reward_task: list[float] = field(default_factory=list)
    reward_conscience: list[float] = field(default_factory=list)
    value_task: list[float] = field(default_factory=list)
    value_conscience: list[float] = field(default_factory=list)
    done: list[bool] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.action)

    def clear(self) -> None:
        for name in self.__slots__:
            getattr(self, name).clear()

    def add(self, **kwargs) -> None:
        for name, value in kwargs.items():
            getattr(self, name).append(value)


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float = GAMMA,
    lam: float = GAE_LAMBDA,
    last_value: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generalized advantage estimation (B.2). Returns (advantages, returns).

    `dones` marks the END of an episode for THIS agent. TearWorld has a fixed
    32-step horizon and no terminal reward (§3.3), so bootstrapping past the
    boundary would inject value that the environment never pays.
    """
    n = len(rewards)
    advantages = np.zeros(n, dtype=np.float64)
    running = 0.0
    next_value = last_value

    for t in reversed(range(n)):
        non_terminal = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * next_value * non_terminal - values[t]
        running = delta + gamma * lam * non_terminal * running
        advantages[t] = running
        next_value = values[t]

    return advantages, advantages + values


def normalize(advantages: torch.Tensor) -> torch.Tensor:
    """Per-batch advantage normalization. Applied to each stream SEPARATELY:
    task and conscience advantages live on different scales, and jointly
    normalizing them would silently reweight the §2.8 combination."""
    if advantages.numel() < 2:
        # std() of a singleton is NaN under Bessel correction, which would
        # silently poison an entire update. A single transition carries no
        # dispersion to normalize by; centre it and move on.
        return advantages - advantages.mean()
    return (advantages - advantages.mean()) / (advantages.std() + 1e-8)


def clipped_surrogate(
    logprob: torch.Tensor,
    old_logprob: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float = CLIP_EPS,
) -> torch.Tensor:
    """J_PPO(A) under the ASCENT convention (§2.8).

    Returns a quantity to be MAXIMIZED. The protocol is written throughout in
    ascent form -- `theta <- theta + eta*Adam(...)` -- and mixing in a
    descent-convention loss would flip the sign of the PCGrad projection in C6
    without any obvious symptom.
    """
    ratio = torch.exp(logprob - old_logprob)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    return torch.min(unclipped, clipped).mean()


def entropy_bonus(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """H(pi) over feasible actions only (§2.8's beta term).

    Masked actions are untrained (§2.4); counting them would make entropy a
    function of how many verbs happen to be legal rather than of the policy's
    indecision.
    """
    assert_feasible_mask(mask)
    masked = logits.masked_fill(~mask, float("-inf"))
    log_probs = torch.log_softmax(masked, dim=-1)

    # NaN TRAP -- do not "simplify" this back.
    # Masked entries are -inf, so probs*log_probs forms 0 * -inf = NaN. Hiding
    # that behind torch.where AFTER the multiply fixes only the forward value:
    # autograd still differentiates the product, torch.where sends 0 into the
    # discarded branch, 0 * NaN = NaN, and because log_softmax couples every
    # entry the NaN lands on the FEASIBLE logits too. Confirmed empirically:
    # gradient [nan, nan, 0.0, 0.0] on a 2-of-4 mask, with a finite forward
    # entropy of 0.6628 -- which is exactly why a forward-only test passed.
    # The fix is to neutralize -inf BEFORE it is ever multiplied.
    zeros = torch.zeros_like(log_probs)
    safe_log = torch.where(mask, log_probs, zeros)
    probs = torch.where(mask, log_probs.exp(), zeros)
    return -(probs * safe_log).sum(dim=-1).mean()


@dataclass(slots=True)
class GradientDiagnostics:
    """§2.8 mandatory logging: cos(g_T, g_C), conflict fraction, removed norm.

    These are not optional telemetry. §6.1's interpretation table reads the
    conflict structure directly, and a C6 that never encounters g_T . g_C < 0
    has had its projection do nothing -- which would make it C5 wearing a
    different label.
    """

    cosine: float = 0.0
    conflicted: bool = False
    removed_norm: float = 0.0
    task_norm: float = 0.0
    conscience_norm: float = 0.0


def flat_grad(parameters) -> torch.Tensor:
    return torch.cat(
        [
            (p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
            for p in parameters
        ]
    )


def assign_flat_grad(parameters, flat: torch.Tensor) -> None:
    offset = 0
    for p in parameters:
        n = p.numel()
        p.grad = flat[offset : offset + n].view_as(p).clone()
        offset += n


def pcgrad(
    g_task: torch.Tensor, g_conscience: torch.Tensor
) -> tuple[torch.Tensor, GradientDiagnostics]:
    """C6's protected update (§2.8):

        if g_T . g_C < 0:  g_T <- g_T - proj_{g_C}(g_T)

    Only the TASK gradient is projected. The asymmetry is the entire point:
    conscience learning is protected from conflicting task gradients, not the
    reverse. A symmetric projection would let the task objective erode the
    conscience signal exactly when they disagree -- which is when it matters.
    """
    dot = float(torch.dot(g_task, g_conscience))
    task_norm = float(g_task.norm())
    conscience_norm = float(g_conscience.norm())

    diagnostics = GradientDiagnostics(
        cosine=dot / (task_norm * conscience_norm + 1e-12),
        conflicted=dot < 0.0,
        task_norm=task_norm,
        conscience_norm=conscience_norm,
    )

    if dot >= 0.0 or conscience_norm == 0.0 or not np.isfinite(conscience_norm):
        return g_task, diagnostics

    # SCALE INVARIANCE. The projection is mathematically invariant to rescaling
    # g_C, but an ABSOLUTE cutoff on ||g_C||^2 is not: with a 1e-12 guard,
    # ||g_C||=0.9e-6 left the task gradient untouched while 1.1e-6 removed
    # almost all of it -- a harmless rescaling flipping the mechanism on and
    # off. Projecting onto the UNIT vector removes the squared norm from the
    # denominator entirely, so behaviour depends on g_C's DIRECTION only.
    unit = g_conscience / conscience_norm
    projection = torch.dot(g_task, unit) * unit
    projected = g_task - projection
    diagnostics.removed_norm = float(projection.norm())
    return projected, diagnostics
