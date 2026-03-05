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
"""Tests for :mod:`nifreeze.utils.slicewise`."""

import numpy as np

from nifreeze.utils.slicewise import (
    compose_slice_transforms,
    extract_slice_group,
    reinsert_slices,
    slice_mask,
)


class TestExtractSliceGroup:
    """Test slice extraction from 3D volumes."""

    def test_basic_extraction(self):
        vol = np.ones((3, 3, 6))
        sparse = extract_slice_group(vol, [0, 3], axis=2)
        # Selected slices should be preserved
        np.testing.assert_array_equal(sparse[:, :, 0], 1.0)
        np.testing.assert_array_equal(sparse[:, :, 3], 1.0)
        # Other slices should be zero
        np.testing.assert_array_equal(sparse[:, :, 1], 0.0)
        np.testing.assert_array_equal(sparse[:, :, 2], 0.0)

    def test_preserves_shape(self):
        vol = np.random.randn(4, 5, 6)
        sparse = extract_slice_group(vol, [1], axis=2)
        assert sparse.shape == vol.shape

    def test_does_not_modify_input(self):
        vol = np.ones((2, 2, 4))
        original = vol.copy()
        extract_slice_group(vol, [0], axis=2)
        np.testing.assert_array_equal(vol, original)

    def test_axis_0(self):
        vol = np.ones((4, 3, 3))
        sparse = extract_slice_group(vol, [1, 3], axis=0)
        assert sparse[0].sum() == 0
        assert sparse[1].sum() == 9


class TestSliceMask:
    """Test binary slice mask creation."""

    def test_basic_mask(self):
        m = slice_mask((3, 3, 6), [0, 5], axis=2)
        assert m.shape == (3, 3, 6)
        assert m[:, :, 0].sum() == 9
        assert m[:, :, 5].sum() == 9
        assert m[:, :, 1].sum() == 0

    def test_dtype(self):
        m = slice_mask((2, 2, 4), [0], dtype=np.float32)
        assert m.dtype == np.float32

    def test_different_axis(self):
        m = slice_mask((4, 3, 3), [0, 2], axis=0)
        assert m[0].sum() == 9
        assert m[1].sum() == 0
        assert m[2].sum() == 9


class TestReinsertSlices:
    """Test slice reinsertion into 3D volumes."""

    def test_basic_reinsertion(self):
        target = np.zeros((2, 2, 4))
        source = np.ones((2, 2, 4)) * 7.0
        reinsert_slices(target, source, [1, 3], axis=2)
        np.testing.assert_array_equal(target[:, :, 1], 7.0)
        np.testing.assert_array_equal(target[:, :, 3], 7.0)
        np.testing.assert_array_equal(target[:, :, 0], 0.0)
        np.testing.assert_array_equal(target[:, :, 2], 0.0)

    def test_modifies_in_place(self):
        target = np.zeros((2, 2, 4))
        source = np.ones((2, 2, 4))
        reinsert_slices(target, source, [0], axis=2)
        assert target[:, :, 0].sum() == 4


class TestComposeSliceTransforms:
    """Test V2V + S2V transform composition."""

    def test_identity_composition(self):
        vol_aff = np.eye(4)
        exc_aff = np.stack([np.eye(4), np.eye(4), np.eye(4)])
        groups = [[0, 3], [1, 4], [2, 5]]
        result = compose_slice_transforms(vol_aff, exc_aff, groups, n_slices=6)
        assert result.shape == (6, 4, 4)
        for i in range(6):
            np.testing.assert_array_almost_equal(result[i], np.eye(4))

    def test_volume_transform_propagates(self):
        vol_aff = np.eye(4)
        vol_aff[:3, 3] = [10, 20, 30]  # translation
        exc_aff = np.stack([np.eye(4)])  # identity excitation
        groups = [[0]]
        result = compose_slice_transforms(vol_aff, exc_aff, groups, n_slices=1)
        np.testing.assert_array_almost_equal(result[0], vol_aff)

    def test_excitation_deviation_composed(self):
        vol_aff = np.eye(4)
        exc_dev = np.eye(4)
        exc_dev[0, 3] = 5.0  # 5mm x-translation deviation
        exc_aff = np.stack([exc_dev, np.eye(4)])
        groups = [[0, 2], [1, 3]]
        result = compose_slice_transforms(vol_aff, exc_aff, groups, n_slices=4)
        # Slices 0,2 should have the deviation
        assert result[0, 0, 3] == 5.0
        assert result[2, 0, 3] == 5.0
        # Slices 1,3 should be identity
        np.testing.assert_array_almost_equal(result[1], np.eye(4))

    def test_multiband_slices_share_transform(self):
        vol_aff = np.eye(4)
        exc_dev = np.eye(4)
        exc_dev[1, 3] = 3.0
        exc_aff = np.stack([exc_dev])
        groups = [[0, 1, 2]]  # All slices in one excitation (MB=3)
        result = compose_slice_transforms(vol_aff, exc_aff, groups, n_slices=3)
        for i in range(3):
            np.testing.assert_array_almost_equal(result[i], exc_dev)
