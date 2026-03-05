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
"""Tests for slice-to-volume infrastructure on BaseDataset."""

from tempfile import TemporaryDirectory

import numpy as np
import pytest

from nifreeze.data.base import BaseDataset
from nifreeze.data.slicing import SliceAcquisition


def _make_dataset(shape=(8, 8, 6, 3), mb=1, with_slice_acq=True):
    """Create a minimal BaseDataset with optional slice acquisition metadata."""
    sa = None
    if with_slice_acq:
        sa = SliceAcquisition(n_slices=shape[2], multiband_factor=mb)
    return BaseDataset(
        dataobj=np.random.randn(*shape).astype(np.float32),
        affine=np.eye(4),
        slice_acquisition=sa,
    )


class TestSetSliceTransform:
    """Test per-excitation transform storage on BaseDataset."""

    def test_set_slice_transform_initializes(self):
        ds = _make_dataset(mb=2)
        assert ds.slice_motion_affines is None
        ds.set_slice_transform(0, 0, np.eye(4) * 2)
        assert ds.slice_motion_affines is not None
        # Shape: (n_vols, n_excitations, 4, 4)
        assert ds.slice_motion_affines.shape == (3, 3, 4, 4)

    def test_set_slice_transform_stores_correctly(self):
        ds = _make_dataset(mb=2)
        custom = np.eye(4)
        custom[0, 3] = 42.0
        ds.set_slice_transform(1, 2, custom)
        np.testing.assert_array_equal(ds.slice_motion_affines[1, 2], custom)
        # Other entries should be identity
        np.testing.assert_array_equal(ds.slice_motion_affines[0, 0], np.eye(4))

    def test_set_slice_transform_without_acquisition_raises(self):
        ds = _make_dataset(with_slice_acq=False)
        with pytest.raises(ValueError, match="slice_acquisition"):
            ds.set_slice_transform(0, 0, np.eye(4))


class TestHDF5RoundTrip:
    """Test HDF5 serialization of slice acquisition metadata."""

    def test_roundtrip_without_slice_acquisition(self):
        ds = _make_dataset(with_slice_acq=False)
        with TemporaryDirectory() as tmp:
            path = f"{tmp}/test.h5"
            ds.to_filename(path)
            loaded = BaseDataset.from_filename(path)
        assert loaded.slice_acquisition is None
        assert loaded.slice_motion_affines is None

    def test_roundtrip_with_slice_acquisition(self):
        ds = _make_dataset(mb=2)
        with TemporaryDirectory() as tmp:
            path = f"{tmp}/test.h5"
            ds.to_filename(path)
            loaded = BaseDataset.from_filename(path)
        assert loaded.slice_acquisition is not None
        assert loaded.slice_acquisition.n_slices == 6
        assert loaded.slice_acquisition.multiband_factor == 2
        assert loaded.slice_acquisition.slice_axis == 2
        assert loaded.slice_acquisition.n_excitations == 3

    def test_roundtrip_with_slice_order(self):
        sa = SliceAcquisition(n_slices=6, multiband_factor=2, slice_order=np.array([2, 0, 1]))
        ds = BaseDataset(
            dataobj=np.random.randn(8, 8, 6, 3).astype(np.float32),
            affine=np.eye(4),
            slice_acquisition=sa,
        )
        with TemporaryDirectory() as tmp:
            path = f"{tmp}/test.h5"
            ds.to_filename(path)
            loaded = BaseDataset.from_filename(path)
        np.testing.assert_array_equal(loaded.slice_acquisition.slice_order, np.array([2, 0, 1]))

    def test_roundtrip_with_slice_motion_affines(self):
        ds = _make_dataset(mb=2)
        custom = np.eye(4)
        custom[0, 3] = 7.5
        ds.set_slice_transform(0, 1, custom)
        with TemporaryDirectory() as tmp:
            path = f"{tmp}/test.h5"
            ds.to_filename(path)
            loaded = BaseDataset.from_filename(path)
        np.testing.assert_array_almost_equal(loaded.slice_motion_affines[0, 1], custom)
