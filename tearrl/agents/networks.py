"""
Condition architectures -- B.1, §2.5.

    Selector f_theta (C4-C6) : MLP [9 + 4 (c_t) + 4 (q_other)] -> 64 -> 64 -> 1
                               shared across candidates; rows permuted
    C3-IM policy             : MLP [dim(o^pi) + 9|A| + 4 (q_other)]
                               -> widths equalizing parameters with C6
    Critics V_T, V_C         : [dim(o^pi) + 4 (c_t where applicable)]
                               -> 128 -> 128 -> 1;  no identity input

--------------------------------------------------------------------------
q_other APPEARS EXACTLY ONCE  (B.6 frozen assertion)
--------------------------------------------------------------------------
o^pi = o \\ {q_self, q_other} carries no identity at all (§2.1, correction 1),
so appending q_other once here is the ONLY place it enters. That is asserted
structurally by `assert_identity_enters_once` rather than trusted.

--------------------------------------------------------------------------
PERMUTATION EQUIVARIANCE (§2.5)
--------------------------------------------------------------------------
The selector scores each candidate with the SAME network and no action-ID
embedding, bias or positional encoding, so its output is equivariant to row
order by construction. q_other is the single declared bottleneck exception.
An action's score can therefore depend only on its predicted consequences,
c_t and who the counterpart is -- never on which verb index it happens to be.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..conscience import C_DIM
from ..spec import N_ACTIONS, Q_DIM
from ..state import OBS_DIM
from ..witness.targets import WITNESS_DIM

SELECTOR_HIDDEN: int = 64          # B.1
CRITIC_HIDDEN: int = 128           # B.1

SELECTOR_INPUT_DIM: int = WITNESS_DIM + C_DIM + Q_DIM          # 9 + 4 + 4 = 17
C3IM_INPUT_DIM: int = OBS_DIM + WITNESS_DIM * N_ACTIONS + Q_DIM  # o^pi + 9|A| + 4


def parameter_budget(input_dim: int, hidden: int, output_dim: int) -> int:
    """Parameters of an MLP input -> hidden -> hidden -> output (with biases)."""
    return (
        (input_dim * hidden + hidden)
        + (hidden * hidden + hidden)
        + (hidden * output_dim + output_dim)
    )


def equalizing_width(input_dim: int, output_dim: int, target: int) -> int:
    """Hidden width whose two-layer MLP comes closest to `target` parameters.

    B.1 requires C3-IM's widths to equalize parameters with the C6 machinery,
    config-asserted. Solving

        h^2 + h*(input + output + 2) + output = target

    and rounding to the nearer feasible integer gives that width.
    """
    b = input_dim + output_dim + 2
    discriminant = b * b - 4 * (output_dim - target)
    if discriminant < 0:
        return 1
    root = (-b + math.sqrt(discriminant)) / 2
    candidates = [max(1, int(math.floor(root))), max(1, int(math.ceil(root)))]
    return min(candidates, key=lambda h: abs(parameter_budget(input_dim, h, output_dim) - target))


class EmptyActionMask(ValueError):
    """Raised when a state offers no feasible action.

    A.3 gives `wait` no precondition, so a world-generated mask is never empty.
    That is an invariant of the environment, not of the tensor boundary: a
    hand-built probe, a batching bug or a corrupted restore can still deliver
    an all-false row, and the failure mode is silent -- all logits become -inf,
    softmax returns NaN, and the NaN reaches the optimizer looking like a
    learning-rate problem. Failing loudly here converts that into a stack trace.
    """


def assert_feasible_mask(mask: torch.Tensor) -> None:
    """Every row must offer at least one feasible action."""
    if mask.numel() == 0 or not bool(mask.any(dim=-1).all()):
        empty = (~mask.any(dim=-1)).nonzero().flatten().tolist()
        raise EmptyActionMask(
            f"action mask has no feasible action in row(s) {empty[:8]}; "
            "A.3 guarantees `wait` is always legal, so this mask did not come "
            "from the environment"
        )


def _mlp(input_dim: int, hidden: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden),
        nn.Tanh(),
        nn.Linear(hidden, hidden),
        nn.Tanh(),
        nn.Linear(hidden, output_dim),
    )


class Selector(nn.Module):
    """C4-C6 actor (§2.5, B.1).

        l_a = f_theta( yhat(a), c_t, q_other );  pi(a|o) = softmax over feasible

    One scorer applied to every candidate row. c_t is zeroed where a condition
    does not carry it (C4), keeping the input width identical across C4-C6 so
    the conditions differ in information, not in shape.
    """

    def __init__(self, hidden: int = SELECTOR_HIDDEN) -> None:
        super().__init__()
        self.net = _mlp(SELECTOR_INPUT_DIM, hidden, 1)

    def forward(
        self,
        witness_out: torch.Tensor,   # [B, |A|, 9]
        c: torch.Tensor,             # [B, 4]
        q_other: torch.Tensor,       # [B, 4]
        mask: torch.Tensor,          # [B, |A|] bool
    ) -> torch.Tensor:
        assert_feasible_mask(mask)
        batch, n_actions, _ = witness_out.shape
        c_rows = c.unsqueeze(1).expand(batch, n_actions, C_DIM)
        q_rows = q_other.unsqueeze(1).expand(batch, n_actions, Q_DIM)
        rows = torch.cat([witness_out, c_rows, q_rows], dim=-1)
        logits = self.net(rows).squeeze(-1)
        return logits.masked_fill(~mask, float("-inf"))


class PolicyNet(nn.Module):
    """C1 / C2 / C3 actor.

    B.1 does not specify these architectures -- only C3-IM's, which is pinned
    to C6 by parameter equalization. IMPLEMENTATION CHOICE, disclosed: C1-C3
    use the SAME hidden width as C3-IM, so that C3 vs C3-IM isolates exactly
    one variable (whether the witness output is in the input) rather than
    confounding it with capacity.
    """

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.input_dim = OBS_DIM + Q_DIM
        self.net = _mlp(self.input_dim, hidden, N_ACTIONS)

    def forward(
        self, obs: torch.Tensor, q_other: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        assert_feasible_mask(mask)
        logits = self.net(torch.cat([obs, q_other], dim=-1))
        return logits.masked_fill(~mask, float("-inf"))


class C3IMPolicy(nn.Module):
    """C3-IM: the frozen headline comparator (§4, B.1).

    Receives the identical frozen-witness output C6 sees -- flattened over all
    candidate actions -- plus o^pi and q_other, with widths equalizing
    parameters against the C6 selector. C6's advantage over it is therefore
    neither information nor capacity, but the protected conscience update
    (§2.8) and the responsibility state.
    """

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.input_dim = C3IM_INPUT_DIM
        self.net = _mlp(self.input_dim, hidden, N_ACTIONS)

    def forward(
        self,
        obs: torch.Tensor,           # [B, OBS_DIM]
        witness_out: torch.Tensor,   # [B, |A|, 9]
        q_other: torch.Tensor,       # [B, 4]
        mask: torch.Tensor,          # [B, |A|] bool
    ) -> torch.Tensor:
        assert_feasible_mask(mask)
        # REPLACE the infeasible witness rows before flattening -- do not
        # multiply by the mask.
        #
        # §2.4 masks infeasible actions and leaves them UNTRAINED, so those nine
        # channels carry undefined values. Flattening them raw let arbitrary
        # garbage in rows the agent cannot even select move the FEASIBLE action
        # probabilities -- measured at 6.3% on a random C3-IM. Since C3-IM is
        # the frozen headline comparator for G2, its behaviour would have partly
        # depended on noise. C6's selector never has this exposure: it scores
        # each row independently and masks infeasible rows out of the softmax.
        #
        # Multiplication is NOT sufficient: 0 * NaN = NaN and 0 * inf = NaN, so
        # a non-finite value in an untrained row would survive the gate and
        # poison every feasible logit. Same defect class as the entropy NaN.
        gated = witness_out.masked_fill(~mask.unsqueeze(-1), 0.0)
        flat = gated.reshape(gated.shape[0], -1)
        logits = self.net(torch.cat([obs, flat, q_other], dim=-1))
        return logits.masked_fill(~mask, float("-inf"))


class Critic(nn.Module):
    """V_T and V_C (B.1).

    'no identity input (q is causally inert)' -- q affects no transition in
    A.3, so a value function that consumed it could only overfit episode
    labels. Critics are never projected (§2.8).
    """

    def __init__(self, use_c: bool, hidden: int = CRITIC_HIDDEN) -> None:
        super().__init__()
        self.use_c = use_c
        self.input_dim = OBS_DIM + (C_DIM if use_c else 0)
        self.net = _mlp(self.input_dim, hidden, 1)

    def forward(self, obs: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
        x = torch.cat([obs, c], dim=-1) if self.use_c else obs
        return self.net(x).squeeze(-1)


# ----------------------------------------------------------------------
# config assertions (B.1, B.6)
# ----------------------------------------------------------------------


def c6_actor_parameters(hidden: int = SELECTOR_HIDDEN) -> int:
    """Learned ACTOR parameters of the C6 machinery.

    The witness is frozen and shared identically by C3-IM, C4, C5 and C6
    (§2.3(4)), so its parameters are not part of any condition's learned
    budget; c_t is a deterministic recurrence with no parameters (§2.6). The
    selector is therefore the whole learned actor.
    """
    return parameter_budget(SELECTOR_INPUT_DIM, hidden, 1)


def c3im_equalized_width() -> int:
    return equalizing_width(C3IM_INPUT_DIM, N_ACTIONS, c6_actor_parameters())


def assert_parameter_equalization(tolerance: float = 0.05) -> tuple[int, int, int]:
    """B.1: 'widths equalizing parameters with C6 machinery (config-asserted)'.

    Returns (width, c3im_params, c6_params).
    """
    width = c3im_equalized_width()
    c3im = parameter_budget(C3IM_INPUT_DIM, width, N_ACTIONS)
    c6 = c6_actor_parameters()
    relative = abs(c3im - c6) / c6
    assert relative <= tolerance, (
        f"C3-IM ({c3im:,}) and C6 ({c6:,}) actor parameters differ by "
        f"{relative:.1%}, above the {tolerance:.0%} equalization tolerance"
    )
    return width, c3im, c6


def assert_identity_enters_once(module: nn.Module) -> None:
    """B.6 #3: q_other appears exactly once in the policy input vector.

    Structural check: the base observation o^pi is identity-free by
    construction (§2.1), so the declared q-slot is the sole occurrence and the
    input width must equal base + exactly one Q_DIM block.
    """
    if isinstance(module, C3IMPolicy):
        expected = OBS_DIM + WITNESS_DIM * N_ACTIONS + Q_DIM
    elif isinstance(module, PolicyNet):
        expected = OBS_DIM + Q_DIM
    elif isinstance(module, Selector):
        expected = WITNESS_DIM + C_DIM + Q_DIM
    else:  # pragma: no cover
        raise TypeError(f"no identity contract declared for {type(module).__name__}")

    actual = module.net[0].in_features
    assert actual == expected, (
        f"{type(module).__name__} input width {actual} != {expected}; "
        "q_other does not appear exactly once"
    )


if __name__ == "__main__":  # pragma: no cover
    width, c3im, c6 = assert_parameter_equalization()
    print(f"C6 selector input        : {SELECTOR_INPUT_DIM}")
    print(f"C3-IM policy input       : {C3IM_INPUT_DIM}")
    print(f"equalizing hidden width  : {width}")
    print(f"C3-IM actor parameters   : {c3im:,}")
    print(f"C6 actor parameters      : {c6:,}")
    print(f"relative difference      : {abs(c3im - c6) / c6:.2%}")
