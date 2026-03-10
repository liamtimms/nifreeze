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
"""EDDY-style slice-level outlier detection and replacement.

Implements the ``--repol`` algorithm from FSL EDDY
(:footcite:p:`andersson_incorporating_2016`): for each volume, the mean
residual (observed − predicted) is computed per slice within the brain mask.
Slices whose mean residual deviates by more than *N* standard deviations
from the distribution across slices are flagged and replaced with the
model prediction.

References
----------
.. footbibliography::
"""

from __future__ import annotations

import attrs
import numpy as np

DEFAULT_OL_NSTD: float = 4.0
"""Default number of standard deviations for outlier detection."""

DEFAULT_OL_NVOX: int = 250
"""Minimum number of brain voxels for a slice to be considered."""


@attrs.define
class OutlierConfig:
    """Configuration for slice-level outlier detection.

    Parameters
    ----------
    enabled : :obj:`bool`
        Whether outlier detection/replacement is active (``--repol``).
    nstd : :obj:`float`
        Number of standard deviations for outlier threshold (``--ol_nstd``).
    nvox : :obj:`int`
        Minimum brain voxels per slice for it to be considered (``--ol_nvox``).
    detect_positive : :obj:`bool`
        Also detect signal *increases* (``--ol_pos``).
    detect_squared : :obj:`bool`
        Detect using squared residuals (``--ol_sqr``).
    slice_axis : :obj:`int`
        Axis along which slices are defined (default 2 = axial).

    """

    enabled: bool = False
    nstd: float = DEFAULT_OL_NSTD
    nvox: int = DEFAULT_OL_NVOX
    detect_positive: bool = False
    detect_squared: bool = False
    slice_axis: int = 2


def compute_slice_residuals(
    observed: np.ndarray,
    predicted: np.ndarray,
    brainmask: np.ndarray | None,
    slice_axis: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-slice mean residual within the brain mask.

    Parameters
    ----------
    observed : :obj:`~numpy.ndarray`
        The observed 3D volume.
    predicted : :obj:`~numpy.ndarray`
        The model-predicted 3D volume.
    brainmask : :obj:`~numpy.ndarray` or :obj:`None`
        Boolean brain mask.  If ``None``, all voxels are used.
    slice_axis : :obj:`int`
        The slice-encoding axis (default 2).

    Returns
    -------
    mean_residuals : :obj:`~numpy.ndarray`
        Mean of ``(observed - predicted)`` per slice, shape ``(n_slices,)``.
    n_brain_voxels : :obj:`~numpy.ndarray`
        Number of brain voxels per slice, shape ``(n_slices,)``.

    """
    residual = observed.astype(np.float64) - predicted.astype(np.float64)
    n_slices = observed.shape[slice_axis]
    mean_residuals = np.zeros(n_slices, dtype=np.float64)
    n_brain_voxels = np.zeros(n_slices, dtype=np.int64)

    for s in range(n_slices):
        slicing = [slice(None)] * observed.ndim
        slicing[slice_axis] = s
        sl = tuple(slicing)

        if brainmask is not None:
            mask_sl = brainmask[sl].astype(bool)
            n_vox = int(mask_sl.sum())
            n_brain_voxels[s] = n_vox
            if n_vox > 0:
                mean_residuals[s] = residual[sl][mask_sl].mean()
        else:
            n_vox = int(np.prod(observed[sl].shape))
            n_brain_voxels[s] = n_vox
            mean_residuals[s] = residual[sl].mean()

    return mean_residuals, n_brain_voxels


def detect_outlier_slices(
    observed: np.ndarray,
    predicted: np.ndarray,
    brainmask: np.ndarray | None,
    config: OutlierConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect outlier slices by comparing observed vs predicted signal.

    For each slice, the mean residual ``(observed - predicted)`` within the
    brain mask is computed.  Slices whose mean residual deviates by more than
    ``config.nstd`` standard deviations from the distribution across slices are
    flagged.

    Parameters
    ----------
    observed : :obj:`~numpy.ndarray`
        Observed 3D volume.
    predicted : :obj:`~numpy.ndarray`
        Model-predicted 3D volume.
    brainmask : :obj:`~numpy.ndarray` or :obj:`None`
        Boolean brain mask.
    config : :obj:`OutlierConfig`
        Detection parameters.

    Returns
    -------
    outlier_mask : :obj:`~numpy.ndarray`
        Boolean array of shape ``(n_slices,)``.  ``True`` marks an outlier.
    nstdev_values : :obj:`~numpy.ndarray`
        Number of standard deviations each slice departs from the mean,
        shape ``(n_slices,)``.  When only negative-outlier detection is
        active, positive values indicate signal *below* prediction
        (dropout).  When ``detect_squared`` is enabled, values are
        absolute magnitudes.

    """
    mean_res, n_vox = compute_slice_residuals(
        observed, predicted, brainmask, slice_axis=config.slice_axis,
    )
    n_slices = len(mean_res)

    # Only consider slices with enough brain voxels
    valid = n_vox >= config.nvox
    if valid.sum() < 2:
        return np.zeros(n_slices, dtype=bool), np.zeros(n_slices, dtype=np.float64)

    # Statistics over valid slices
    mu = mean_res[valid].mean()
    sigma = mean_res[valid].std(ddof=1)

    nstdev_values = np.zeros(n_slices, dtype=np.float64)
    outlier_mask = np.zeros(n_slices, dtype=bool)

    if sigma == 0:
        return outlier_mask, nstdev_values

    # Compute how many std each slice deviates (negative = dropout)
    nstdev_values[valid] = (mu - mean_res[valid]) / sigma

    # Flag negative outliers (signal dropout): mean_res << mu
    outlier_mask[valid] = nstdev_values[valid] > config.nstd

    # Optionally flag positive outliers (signal increase)
    if config.detect_positive:
        outlier_mask[valid] |= nstdev_values[valid] < -config.nstd

    # Optionally detect using squared residuals
    if config.detect_squared:
        sq_mask, sq_nstd = _detect_squared_outliers(
            observed, predicted, brainmask, config, valid,
        )
        outlier_mask |= sq_mask
        # Keep the larger absolute deviation for reporting
        abs_sq = np.abs(sq_nstd)
        abs_main = np.abs(nstdev_values)
        nstdev_values = np.where(abs_sq > abs_main, abs_sq, abs_main)

    return outlier_mask, nstdev_values


def _detect_squared_outliers(
    observed: np.ndarray,
    predicted: np.ndarray,
    brainmask: np.ndarray | None,
    config: OutlierConfig,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect outliers using mean squared residuals per slice."""
    residual = observed.astype(np.float64) - predicted.astype(np.float64)
    sq_residual = residual ** 2
    n_slices = observed.shape[config.slice_axis]
    mean_sq = np.zeros(n_slices, dtype=np.float64)

    for s in range(n_slices):
        slicing = [slice(None)] * observed.ndim
        slicing[config.slice_axis] = s
        sl = tuple(slicing)

        if brainmask is not None:
            mask_sl = brainmask[sl].astype(bool)
            if mask_sl.sum() > 0:
                mean_sq[s] = sq_residual[sl][mask_sl].mean()
        else:
            mean_sq[s] = sq_residual[sl].mean()

    mu_sq = mean_sq[valid].mean()
    sigma_sq = mean_sq[valid].std(ddof=1)

    sq_nstd = np.zeros(n_slices, dtype=np.float64)
    sq_mask = np.zeros(n_slices, dtype=bool)

    if sigma_sq > 0:
        sq_nstd[valid] = (mean_sq[valid] - mu_sq) / sigma_sq
        sq_mask[valid] = sq_nstd[valid] > config.nstd

    return sq_mask, sq_nstd


def replace_outlier_slices(
    observed: np.ndarray,
    predicted: np.ndarray,
    outlier_mask: np.ndarray,
    slice_axis: int = 2,
) -> np.ndarray:
    """Replace outlier slices with model predictions.

    Parameters
    ----------
    observed : :obj:`~numpy.ndarray`
        Observed 3D volume.
    predicted : :obj:`~numpy.ndarray`
        Model-predicted 3D volume.
    outlier_mask : :obj:`~numpy.ndarray`
        Boolean array of shape ``(n_slices,)``.
    slice_axis : :obj:`int`
        Slice-encoding axis (default 2).

    Returns
    -------
    :obj:`~numpy.ndarray`
        Copy of *observed* with outlier slices replaced by *predicted*.

    """
    result = observed.copy()
    for s in np.where(outlier_mask)[0]:
        slicing = [slice(None)] * observed.ndim
        slicing[slice_axis] = s
        sl = tuple(slicing)
        result[sl] = predicted[sl]
    return result
