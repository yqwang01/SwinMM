"""View operations."""

from typing import Callable, Sequence, Tuple

import torch
import numpy as np

ViewType = int

# Input format: [B, C, X, Y, Z, ...]
VIEW_TRANSFORMS = {
    0: lambda x: x,
    1: lambda x: x.permute(0, 1, 3, 2, 4),
    2: lambda x: x.permute(0, 1, 4, 3, 2),
}


def get_view_transform(
        view_src: ViewType,
        view_dst: ViewType) -> Callable[[torch.Tensor], torch.Tensor]:
    """Gets transform function from view src to view dst."""

    def transform(x: torch.Tensor) -> torch.Tensor:
        x_view_0 = VIEW_TRANSFORMS[view_src](x)
        return VIEW_TRANSFORMS[view_dst](x_view_0).contiguous()

    return transform


def view_inverse(xs: Sequence[torch.Tensor],
                 views: Sequence[ViewType]) -> Sequence[torch.Tensor]:
    """Transforms data back to origin view."""
    return [get_view_transform(0, view)(x) for x, view in zip(xs, views)]


def view_rand(
        x: torch.Tensor,
        num_samples: int = 2
) -> Tuple[Sequence[torch.Tensor], Sequence[ViewType]]:
    """Samples different transforms of data."""
    if num_samples > len(VIEW_TRANSFORMS):
        raise ValueError('Duplicate samples.')
    view_dsts = np.random.permutation(
        len(VIEW_TRANSFORMS))[:num_samples].tolist()
    return [get_view_transform(0, view)(x) for view in view_dsts], view_dsts
