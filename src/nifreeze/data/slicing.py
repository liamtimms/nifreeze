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
"""Slice acquisition metadata for slice-to-volume registration.

This module provides :class:`SliceAcquisition`, an attrs-based descriptor
of how slices within each volume of a 4D dataset were acquired.  It tracks
the multiband (SMS) factor, the slice excitation ordering, and the
slice-encoding axis, and exposes helpers that map between physical slice
indices and *excitation* indices (groups of simultaneously acquired slices).

The single-band case (``multiband_factor=1``) is the degenerate default —
every excitation corresponds to exactly one slice.

Examples
--------
>>> sa = SliceAcquisition(n_slices=6, multiband_factor=2)
>>> sa.n_excitations
3
>>> sa.excitation_groups
[[0, 3], [1, 4], [2, 5]]
>>> sa.slice_to_excitation(4)
1
>>> sa.excitation_to_slices(0)
[0, 3]

"""

from __future__ import annotations

import attrs
import numpy as np
import numpy.typing as npt

MULTIBAND_FACTOR_ERROR_MSG = "multiband_factor ({mb}) must evenly divide n_slices ({n})."
"""Error raised when multiband_factor does not evenly divide n_slices."""

SLICE_ORDER_LENGTH_ERROR_MSG = "slice_order length ({so_len}) must equal n_excitations ({n_exc})."
"""Error raised when slice_order has wrong length."""

SLICE_ORDER_VALUES_ERROR_MSG = "slice_order must be a permutation of 0..n_excitations-1."
"""Error raised when slice_order values are not a valid permutation."""


def _validate_multiband_factor(
    inst: SliceAcquisition,
    attr: attrs.Attribute,
    value: int,
) -> None:
    """Ensure multiband_factor evenly divides n_slices."""
    if value < 1:
        raise ValueError("multiband_factor must be >= 1.")
    if inst.n_slices % value != 0:
        raise ValueError(MULTIBAND_FACTOR_ERROR_MSG.format(mb=value, n=inst.n_slices))


def _validate_slice_order(
    inst: SliceAcquisition,
    attr: attrs.Attribute,
    value: npt.NDArray[np.intp] | None,
) -> None:
    """Ensure slice_order is a valid permutation of 0..n_excitations-1."""
    if value is None:
        return

    n_exc = inst.n_slices // inst.multiband_factor
    if len(value) != n_exc:
        raise ValueError(SLICE_ORDER_LENGTH_ERROR_MSG.format(so_len=len(value), n_exc=n_exc))
    if set(value.tolist()) != set(range(n_exc)):
        raise ValueError(SLICE_ORDER_VALUES_ERROR_MSG)


@attrs.define(slots=True)
class SliceAcquisition:
    """Describe slice acquisition geometry for one volume of a 4D dataset.

    Parameters
    ----------
    n_slices : :obj:`int`
        Total number of slices along the slice-encoding axis.
    multiband_factor : :obj:`int`
        Simultaneous multi-slice (SMS / multiband) factor.  ``1`` means
        conventional single-band acquisition.  Must evenly divide
        *n_slices*.
    slice_order : :obj:`~numpy.ndarray` or ``None``
        A 1-D integer array of length *n_excitations* giving the temporal
        order of excitations.  ``slice_order[t]`` is the excitation index
        fired at the *t*-th RF pulse.  If ``None``, ascending order
        (``0, 1, 2, …``) is assumed.
    slice_axis : :obj:`int`
        Spatial axis along which slices are stacked (0=X, 1=Y, **2=Z**).

    """

    n_slices: int = attrs.field()
    """Total number of physical slices per volume."""

    multiband_factor: int = attrs.field(default=1, validator=_validate_multiband_factor)
    """Simultaneous multi-slice (SMS) factor."""

    slice_order: npt.NDArray[np.intp] | None = attrs.field(
        default=None, validator=_validate_slice_order
    )
    """Temporal order of excitations (length ``n_excitations``).

    ``slice_order[t]`` is the excitation index fired at the *t*-th RF pulse.
    ``None`` means ascending order ``(0, 1, 2, …)``.
    """

    slice_axis: int = attrs.field(default=2)
    """Spatial axis of slice encoding (default 2 = Z)."""

    # ---- derived properties --------------------------------------------------

    @property
    def n_excitations(self) -> int:
        """Number of RF excitations (slice groups) per volume.

        Examples
        --------
        >>> SliceAcquisition(n_slices=60, multiband_factor=3).n_excitations
        20

        """
        return self.n_slices // self.multiband_factor

    @property
    def excitation_groups(self) -> list[list[int]]:
        """Map each excitation index to its simultaneously acquired slice indices.

        For a multiband factor *M* and *N* total slices the *k*-th excitation
        acquires slices ``[k, k + N/M, k + 2*N/M, …]`` (i.e., slices
        separated by ``n_excitations`` positions along the slice axis).

        Returns
        -------
        :obj:`list` of :obj:`list` of :obj:`int`
            ``groups[k]`` contains the slice indices acquired during excitation *k*.

        Examples
        --------
        >>> SliceAcquisition(n_slices=6, multiband_factor=2).excitation_groups
        [[0, 3], [1, 4], [2, 5]]
        >>> SliceAcquisition(n_slices=4, multiband_factor=1).excitation_groups
        [[0], [1], [2], [3]]

        """
        n_exc = self.n_excitations
        return [list(range(k, self.n_slices, n_exc)) for k in range(n_exc)]

    def slice_to_excitation(self, slice_idx: int) -> int:
        """Return the excitation index that acquired *slice_idx*.

        Parameters
        ----------
        slice_idx : :obj:`int`
            Physical slice index (0-based).

        Returns
        -------
        :obj:`int`
            Excitation index.

        Examples
        --------
        >>> sa = SliceAcquisition(n_slices=6, multiband_factor=2)
        >>> sa.slice_to_excitation(0)
        0
        >>> sa.slice_to_excitation(3)
        0
        >>> sa.slice_to_excitation(4)
        1

        """
        return slice_idx % self.n_excitations

    def excitation_to_slices(self, exc_idx: int) -> list[int]:
        """Return the slice indices acquired during excitation *exc_idx*.

        Parameters
        ----------
        exc_idx : :obj:`int`
            Excitation index (0-based).

        Returns
        -------
        :obj:`list` of :obj:`int`
            Slice indices simultaneously acquired.

        Examples
        --------
        >>> SliceAcquisition(n_slices=6, multiband_factor=2).excitation_to_slices(0)
        [0, 3]

        """
        return self.excitation_groups[exc_idx]

    def temporal_order(self) -> npt.NDArray[np.intp]:
        """Return excitation indices in temporal acquisition order.

        If ``slice_order`` is ``None`` returns ``np.arange(n_excitations)``.

        Returns
        -------
        :obj:`~numpy.ndarray`
            1-D integer array of length ``n_excitations``.

        Examples
        --------
        >>> SliceAcquisition(n_slices=4).temporal_order()
        array([0, 1, 2, 3])
        >>> sa = SliceAcquisition(
        ...     n_slices=4,
        ...     slice_order=np.array([0, 2, 1, 3]),
        ... )
        >>> sa.temporal_order()
        array([0, 2, 1, 3])

        """
        if self.slice_order is not None:
            return self.slice_order.copy()
        return np.arange(self.n_excitations, dtype=np.intp)
