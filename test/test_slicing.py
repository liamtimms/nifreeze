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
"""Tests for :mod:`nifreeze.data.slicing`."""

import numpy as np
import pytest

from nifreeze.data.slicing import (
    SliceAcquisition,
)


class TestSliceAcquisitionCreation:
    """Test SliceAcquisition construction and validation."""

    def test_single_band_default(self):
        sa = SliceAcquisition(n_slices=60)
        assert sa.multiband_factor == 1
        assert sa.n_excitations == 60
        assert sa.slice_axis == 2
        assert sa.slice_order is None

    def test_multiband(self):
        sa = SliceAcquisition(n_slices=60, multiband_factor=3)
        assert sa.n_excitations == 20

    def test_multiband_factor_must_divide_n_slices(self):
        with pytest.raises(ValueError, match="must evenly divide"):
            SliceAcquisition(n_slices=7, multiband_factor=3)

    def test_multiband_factor_must_be_positive(self):
        with pytest.raises(ValueError, match="must be >= 1"):
            SliceAcquisition(n_slices=6, multiband_factor=0)

    def test_slice_order_valid(self):
        order = np.array([2, 0, 1, 3])
        sa = SliceAcquisition(n_slices=4, slice_order=order)
        np.testing.assert_array_equal(sa.slice_order, order)

    def test_slice_order_wrong_length(self):
        with pytest.raises(ValueError, match="slice_order length"):
            SliceAcquisition(n_slices=4, slice_order=np.array([0, 1]))

    def test_slice_order_not_permutation(self):
        with pytest.raises(ValueError, match="permutation"):
            SliceAcquisition(n_slices=4, slice_order=np.array([0, 0, 1, 2]))

    def test_slice_order_with_multiband(self):
        # 6 slices, MB=2 → 3 excitations, order must have 3 elements
        sa = SliceAcquisition(n_slices=6, multiband_factor=2, slice_order=np.array([2, 0, 1]))
        assert sa.n_excitations == 3


class TestExcitationGroups:
    """Test excitation group mapping."""

    def test_single_band_groups(self):
        sa = SliceAcquisition(n_slices=4)
        assert sa.excitation_groups == [[0], [1], [2], [3]]

    def test_multiband_2(self):
        sa = SliceAcquisition(n_slices=6, multiband_factor=2)
        groups = sa.excitation_groups
        assert len(groups) == 3
        assert groups[0] == [0, 3]
        assert groups[1] == [1, 4]
        assert groups[2] == [2, 5]

    def test_multiband_3(self):
        sa = SliceAcquisition(n_slices=9, multiband_factor=3)
        groups = sa.excitation_groups
        assert len(groups) == 3
        assert groups[0] == [0, 3, 6]
        assert groups[1] == [1, 4, 7]
        assert groups[2] == [2, 5, 8]


class TestSliceExcitationMapping:
    """Test bidirectional slice ↔ excitation mapping."""

    def test_slice_to_excitation_single_band(self):
        sa = SliceAcquisition(n_slices=4)
        for i in range(4):
            assert sa.slice_to_excitation(i) == i

    def test_slice_to_excitation_multiband(self):
        sa = SliceAcquisition(n_slices=6, multiband_factor=2)
        # Slices 0,3 → excitation 0
        assert sa.slice_to_excitation(0) == 0
        assert sa.slice_to_excitation(3) == 0
        # Slices 1,4 → excitation 1
        assert sa.slice_to_excitation(1) == 1
        assert sa.slice_to_excitation(4) == 1

    def test_excitation_to_slices(self):
        sa = SliceAcquisition(n_slices=6, multiband_factor=2)
        assert sa.excitation_to_slices(0) == [0, 3]
        assert sa.excitation_to_slices(1) == [1, 4]
        assert sa.excitation_to_slices(2) == [2, 5]


class TestTemporalOrder:
    """Test temporal ordering of excitations."""

    def test_default_ascending(self):
        sa = SliceAcquisition(n_slices=4)
        np.testing.assert_array_equal(sa.temporal_order(), np.arange(4))

    def test_custom_order(self):
        order = np.array([0, 2, 1, 3])
        sa = SliceAcquisition(n_slices=4, slice_order=order)
        np.testing.assert_array_equal(sa.temporal_order(), order)

    def test_temporal_order_returns_copy(self):
        order = np.array([1, 0, 2])
        sa = SliceAcquisition(n_slices=3, slice_order=order)
        result = sa.temporal_order()
        result[0] = 999
        # Original should be unmodified
        assert sa.slice_order[0] == 1
