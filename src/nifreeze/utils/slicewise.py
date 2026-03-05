# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright The NiPreps Developers <nipreps@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# We support and encourage derived works from this project, please read
# about our expectations at
#
#     https://www.nipreps.org/community/licensing/
#
"""Utilities for slice-level operations in slice-to-volume registration.

Functions for extracting and reinserting slices from 3D volumes and for
composing per-excitation transforms with volume-level transforms.

"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def extract_slice_group(
    volume: npt.NDArray,
    slice_indices: list[int],
    axis: int = 2,
) -> npt.NDArray:
    """Extract a sparse 3D volume containing only the requested slices.

    All slices *not* in ``slice_indices`` are zeroed out.  The returned
    array has the same shape as *volume*.

    Parameters
    ----------
    volume : :obj:`~numpy.ndarray`
        A 3D array (X, Y, Z).
    slice_indices : :obj:`list` of :obj:`int`
        Indices of slices to keep along *axis*.
    axis : :obj:`int`
        Slice-encoding axis (default ``2`` = Z).

    Returns
    -------
    :obj:`~numpy.ndarray`
        A copy of *volume* with only the selected slices non-zero.

    Examples
    --------
    >>> vol = np.ones((2, 2, 4))
    >>> sparse = extract_slice_group(vol, [0, 2], axis=2)
    >>> float(sparse[:, :, 1].sum())
    0.0
    >>> float(sparse[:, :, 0].sum())
    4.0

    """
    out = np.zeros_like(volume)
    slicing: list[slice | list[int]] = [slice(None)] * volume.ndim
    slicing[axis] = slice_indices
    out[tuple(slicing)] = volume[tuple(slicing)]
    return out


def slice_mask(
    shape: tuple[int, ...],
    slice_indices: list[int],
    axis: int = 2,
    dtype: npt.DTypeLike = np.uint8,
) -> npt.NDArray:
    """Create a binary mask that is non-zero only at the given slices.

    Parameters
    ----------
    shape : :obj:`tuple` of :obj:`int`
        Spatial shape of the volume (e.g., ``(X, Y, Z)``).
    slice_indices : :obj:`list` of :obj:`int`
        Indices of slices to mark as ``1`` along *axis*.
    axis : :obj:`int`
        Slice-encoding axis (default ``2``).
    dtype : numpy dtype
        Output dtype (default :obj:`~numpy.uint8`).

    Returns
    -------
    :obj:`~numpy.ndarray`
        Binary mask with the same shape as *shape*.

    Examples
    --------
    >>> m = slice_mask((2, 2, 4), [1, 3], axis=2)
    >>> m.shape
    (2, 2, 4)
    >>> int(m[:, :, 0].sum())
    0
    >>> int(m[:, :, 1].sum())
    4

    """
    mask = np.zeros(shape, dtype=dtype)
    slicing: list[slice | list[int]] = [slice(None)] * len(shape)
    slicing[axis] = slice_indices
    mask[tuple(slicing)] = 1
    return mask


def reinsert_slices(
    target: npt.NDArray,
    data: npt.NDArray,
    slice_indices: list[int],
    axis: int = 2,
) -> None:
    """Insert resampled slice data back into a full 3D volume (in-place).

    Parameters
    ----------
    target : :obj:`~numpy.ndarray`
        The destination 3D array to update (modified in-place).
    data : :obj:`~numpy.ndarray`
        Source 3D array containing the resampled data.  Only the slices at
        *slice_indices* are read from this array.
    slice_indices : :obj:`list` of :obj:`int`
        Which slices along *axis* to copy from *data* into *target*.
    axis : :obj:`int`
        Slice-encoding axis (default ``2``).

    Examples
    --------
    >>> tgt = np.zeros((2, 2, 4))
    >>> src = np.ones((2, 2, 4)) * 5
    >>> reinsert_slices(tgt, src, [1, 3], axis=2)
    >>> float(tgt[:, :, 1].sum())
    20.0
    >>> float(tgt[:, :, 0].sum())
    0.0

    """
    slicing: list[slice | list[int]] = [slice(None)] * target.ndim
    slicing[axis] = slice_indices
    idx = tuple(slicing)
    target[idx] = data[idx]


def compose_slice_transforms(
    volume_affine: npt.NDArray,
    excitation_affines: npt.NDArray,
    excitation_groups: list[list[int]],
    n_slices: int,
) -> npt.NDArray:
    """Compose per-excitation and volume-level transforms into per-slice transforms.

    The effective transform for a physical slice belonging to excitation *k* is::

        T_eff = excitation_affines[k] @ volume_affine

    Parameters
    ----------
    volume_affine : :obj:`~numpy.ndarray`
        A single 4x4 volume-level affine.
    excitation_affines : :obj:`~numpy.ndarray`
        Per-excitation deviations, shape ``(N_excitations, 4, 4)``.
    excitation_groups : :obj:`list` of :obj:`list` of :obj:`int`
        Mapping from excitation index to physical slice indices.
    n_slices : :obj:`int`
        Total number of physical slices.

    Returns
    -------
    :obj:`~numpy.ndarray`
        Array of shape ``(n_slices, 4, 4)`` with one affine per physical slice.

    Examples
    --------
    >>> vol_aff = np.eye(4) * 2; vol_aff[3, 3] = 1
    >>> exc_aff = np.stack([np.eye(4), np.eye(4)])
    >>> groups = [[0, 2], [1, 3]]
    >>> result = compose_slice_transforms(vol_aff, exc_aff, groups, n_slices=4)
    >>> result.shape
    (4, 4, 4)
    >>> np.allclose(result[0], vol_aff)
    True
    >>> np.allclose(result[2], vol_aff)
    True

    """
    per_slice = np.empty((n_slices, 4, 4), dtype=np.float64)
    for exc_idx, slice_list in enumerate(excitation_groups):
        composed = excitation_affines[exc_idx] @ volume_affine
        for s in slice_list:
            per_slice[s] = composed
    return per_slice
