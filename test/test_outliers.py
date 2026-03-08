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
"""Tests for :mod:`nifreeze.analysis.outliers`."""

import numpy as np

from nifreeze.analysis.outliers import (
    OutlierConfig,
    compute_slice_residuals,
    detect_outlier_slices,
    replace_outlier_slices,
)


class TestComputeSliceResiduals:
    """Test per-slice residual computation."""

    def test_identical_volumes(self):
        """Residuals should be zero when observed == predicted."""
        vol = np.ones((4, 4, 8))
        mean_res, n_vox = compute_slice_residuals(vol, vol, brainmask=None)
        assert mean_res.shape == (8,)
        np.testing.assert_array_almost_equal(mean_res, 0.0)
        assert (n_vox == 16).all()  # 4*4

    def test_with_brainmask(self):
        """Only brain voxels should contribute."""
        observed = np.ones((4, 4, 4))
        predicted = np.zeros((4, 4, 4))
        mask = np.zeros((4, 4, 4), dtype=bool)
        mask[0, 0, :] = True  # Only one voxel per slice

        mean_res, n_vox = compute_slice_residuals(observed, predicted, mask)
        assert (n_vox == 1).all()
        np.testing.assert_array_almost_equal(mean_res, 1.0)

    def test_axis_0(self):
        """Residuals along axis 0."""
        vol = np.ones((6, 3, 3))
        pred = np.zeros((6, 3, 3))
        mean_res, n_vox = compute_slice_residuals(vol, pred, None, slice_axis=0)
        assert mean_res.shape == (6,)
        np.testing.assert_array_almost_equal(mean_res, 1.0)


class TestDetectOutlierSlices:
    """Test outlier slice detection."""

    def test_no_outliers_clean_data(self):
        """Clean data should produce no outlier flags."""
        rng = np.random.default_rng(42)
        observed = rng.random((4, 4, 10)) + 100  # High baseline
        predicted = observed + rng.normal(0, 0.01, observed.shape)  # Tiny residuals
        config = OutlierConfig(enabled=True, nstd=4.0, nvox=1)

        ol_mask, nstd = detect_outlier_slices(observed, predicted, None, config)
        assert ol_mask.shape == (10,)
        assert not ol_mask.any(), "No slices should be flagged in clean data"

    def test_detect_dropout_slice(self):
        """A slice with signal dropout should be flagged."""
        rng = np.random.default_rng(42)
        observed = np.ones((8, 8, 20)) * 100.0
        predicted = np.ones((8, 8, 20)) * 100.0

        # Add small normal noise to create a distribution
        observed += rng.normal(0, 0.5, observed.shape)

        # Create a severe dropout on slice 5
        observed[:, :, 5] = 0.0

        config = OutlierConfig(enabled=True, nstd=3.0, nvox=1)
        ol_mask, nstd = detect_outlier_slices(observed, predicted, None, config)

        assert ol_mask[5], "Dropout slice should be flagged"
        # Other slices should not be flagged
        non_dropout = np.delete(np.arange(20), 5)
        assert not ol_mask[non_dropout].any()

    def test_nvox_filter(self):
        """Slices with fewer than nvox brain voxels should be skipped."""
        observed = np.ones((4, 4, 6)) * 100.0
        predicted = np.ones((4, 4, 6)) * 100.0
        observed[:, :, 3] = 0.0  # Dropout

        # Mask with very few voxels
        mask = np.zeros((4, 4, 6), dtype=bool)
        mask[0, 0, :] = True  # Only 1 voxel per slice

        config = OutlierConfig(enabled=True, nstd=3.0, nvox=10)
        ol_mask, _ = detect_outlier_slices(observed, predicted, mask, config)

        # With nvox=10, no slice has enough voxels → no detection
        assert not ol_mask.any()

    def test_positive_outlier(self):
        """Positive outliers detected only with detect_positive=True."""
        rng = np.random.default_rng(42)
        observed = np.ones((8, 8, 20)) * 100.0
        predicted = np.ones((8, 8, 20)) * 100.0
        observed += rng.normal(0, 0.5, observed.shape)

        # Slice 3 has abnormally HIGH signal (positive outlier)
        observed[:, :, 3] = 500.0

        # Without detect_positive, should not be flagged
        config_neg = OutlierConfig(enabled=True, nstd=3.0, nvox=1)
        ol_mask_neg, _ = detect_outlier_slices(observed, predicted, None, config_neg)
        assert not ol_mask_neg[3], "Positive outlier not flagged by default"

        # With detect_positive, should be flagged
        config_pos = OutlierConfig(
            enabled=True, nstd=3.0, nvox=1, detect_positive=True,
        )
        ol_mask_pos, _ = detect_outlier_slices(observed, predicted, None, config_pos)
        assert ol_mask_pos[3], "Positive outlier should be flagged"

    def test_nstd_threshold_sensitivity(self):
        """Lower threshold should flag more slices."""
        rng = np.random.default_rng(42)
        observed = np.ones((8, 8, 20)) * 100.0
        predicted = np.ones((8, 8, 20)) * 100.0
        observed += rng.normal(0, 0.5, observed.shape)

        # Moderate dropout on slice 7
        observed[:, :, 7] *= 0.5

        config_strict = OutlierConfig(enabled=True, nstd=10.0, nvox=1)
        config_lenient = OutlierConfig(enabled=True, nstd=2.0, nvox=1)

        ol_strict, _ = detect_outlier_slices(observed, predicted, None, config_strict)
        ol_lenient, _ = detect_outlier_slices(
            observed, predicted, None, config_lenient,
        )

        assert ol_lenient.sum() >= ol_strict.sum()

    def test_squared_residuals(self):
        """Squared residual detection catches artifacts without mean change."""
        rng = np.random.default_rng(42)
        observed = np.ones((8, 8, 20)) * 100.0
        predicted = np.ones((8, 8, 20)) * 100.0
        observed += rng.normal(0, 0.1, observed.shape)

        # Slice 10: large variance but mean ~0 (half voxels +50, half -50)
        noise = np.ones((8, 8)) * 50.0
        noise[::2, :] = -50.0
        observed[:, :, 10] = predicted[:, :, 10] + noise

        config_sq = OutlierConfig(
            enabled=True, nstd=3.0, nvox=1, detect_squared=True,
        )
        ol_mask, _ = detect_outlier_slices(observed, predicted, None, config_sq)
        assert ol_mask[10], "High-variance slice should be flagged with ol_sqr"


class TestReplaceOutlierSlices:
    """Test slice replacement."""

    def test_replacement(self):
        """Flagged slices replaced, clean slices preserved."""
        observed = np.ones((4, 4, 6)) * 10.0
        predicted = np.ones((4, 4, 6)) * 99.0
        outlier_mask = np.array([False, False, True, False, True, False])

        result = replace_outlier_slices(observed, predicted, outlier_mask)

        # Replaced slices should equal predicted
        np.testing.assert_array_equal(result[:, :, 2], 99.0)
        np.testing.assert_array_equal(result[:, :, 4], 99.0)
        # Clean slices should equal observed
        np.testing.assert_array_equal(result[:, :, 0], 10.0)
        np.testing.assert_array_equal(result[:, :, 1], 10.0)
        np.testing.assert_array_equal(result[:, :, 3], 10.0)

    def test_no_outliers_returns_copy(self):
        """With no outliers, result should equal observed (as a copy)."""
        observed = np.ones((3, 3, 4))
        predicted = np.zeros((3, 3, 4))
        outlier_mask = np.zeros(4, dtype=bool)

        result = replace_outlier_slices(observed, predicted, outlier_mask)
        np.testing.assert_array_equal(result, observed)
        # Should be a copy, not the same array
        assert result is not observed

    def test_does_not_modify_input(self):
        """Input arrays should not be modified."""
        observed = np.ones((3, 3, 4)) * 5.0
        predicted = np.ones((3, 3, 4)) * 99.0
        original = observed.copy()
        outlier_mask = np.array([True, False, False, False])

        replace_outlier_slices(observed, predicted, outlier_mask)
        np.testing.assert_array_equal(observed, original)

    def test_axis_0(self):
        """Replacement along axis 0."""
        observed = np.ones((6, 3, 3)) * 10.0
        predicted = np.ones((6, 3, 3)) * 99.0
        outlier_mask = np.array([False, True, False, False, False, False])

        result = replace_outlier_slices(
            observed, predicted, outlier_mask, slice_axis=0,
        )
        np.testing.assert_array_equal(result[1], 99.0)
        np.testing.assert_array_equal(result[0], 10.0)


class TestOutlierConfig:
    """Test OutlierConfig defaults and construction."""

    def test_defaults(self):
        cfg = OutlierConfig()
        assert cfg.enabled is False
        assert cfg.nstd == 4.0
        assert cfg.nvox == 250
        assert cfg.detect_positive is False
        assert cfg.detect_squared is False
        assert cfg.slice_axis == 2

    def test_custom(self):
        cfg = OutlierConfig(
            enabled=True, nstd=3.0, nvox=100,
            detect_positive=True, detect_squared=True, slice_axis=0,
        )
        assert cfg.enabled is True
        assert cfg.nstd == 3.0
        assert cfg.slice_axis == 0
