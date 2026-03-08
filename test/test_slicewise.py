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
"""Tests for :mod:`nifreeze.utils.slicewise` and SVR infrastructure."""

import nibabel as nb
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


class TestPrepareSvrMask:
    """Test SVR slice mask NIfTI creation."""

    def test_mb2_mask(self, tmp_path):
        """Mask for MB2 excitation selects exactly 2 non-contiguous slices."""
        from nifreeze.registration.ants import _prepare_svr_mask

        shape = (4, 4, 8)
        affine = np.eye(4)
        # MB2: excitation 0 acquires slices [0, 4]
        mask_path = _prepare_svr_mask(
            shape, affine, [0, 4], exc_idx=0, vol_idx=0, dirname=tmp_path
        )
        assert mask_path.exists()
        img = nb.load(mask_path)
        data = np.asanyarray(img.dataobj)
        assert data.shape == shape
        # Selected slices should be 1
        assert data[:, :, 0].sum() == 16  # 4*4
        assert data[:, :, 4].sum() == 16
        # Other slices should be 0
        assert data[:, :, 1].sum() == 0
        assert data[:, :, 2].sum() == 0

    def test_mb3_mask(self, tmp_path):
        """Mask for MB3 excitation selects exactly 3 non-contiguous slices."""
        from nifreeze.registration.ants import _prepare_svr_mask

        shape = (3, 3, 9)
        affine = np.eye(4)
        # MB3: excitation 1 acquires slices [1, 4, 7]
        mask_path = _prepare_svr_mask(
            shape, affine, [1, 4, 7], exc_idx=1, vol_idx=0, dirname=tmp_path
        )
        data = np.asanyarray(nb.load(mask_path).dataobj)
        assert data[:, :, 1].sum() == 9
        assert data[:, :, 4].sum() == 9
        assert data[:, :, 7].sum() == 9
        assert data[:, :, 0].sum() == 0

    def test_custom_axis(self, tmp_path):
        """Slice mask along axis=0."""
        from nifreeze.registration.ants import _prepare_svr_mask

        shape = (6, 3, 3)
        affine = np.eye(4)
        mask_path = _prepare_svr_mask(
            shape, affine, [0, 3], exc_idx=0, vol_idx=0, dirname=tmp_path, axis=0
        )
        data = np.asanyarray(nb.load(mask_path).dataobj)
        assert data[0, :, :].sum() == 9
        assert data[3, :, :].sum() == 9
        assert data[1, :, :].sum() == 0


class TestCorrectedGradients:
    """Test DWI.corrected_gradients property."""

    def test_no_motion_returns_original(self):
        """Without motion_affines, corrected_gradients == gradients."""
        from nifreeze.data.dmri import DWI

        n_vols = 10
        rng = np.random.default_rng(42)
        bvecs = rng.standard_normal((n_vols, 3))
        bvecs /= np.linalg.norm(bvecs, axis=1, keepdims=True)
        gradients = np.column_stack([bvecs, np.full(n_vols, 1000.0)])

        dwi = DWI(
            dataobj=rng.random((4, 4, 4, n_vols)).astype(np.float32),
            affine=np.eye(4),
            gradients=gradients,
        )
        np.testing.assert_array_equal(dwi.corrected_gradients, dwi.gradients)

    def test_identity_motion_returns_original(self):
        """Identity motion_affines produce unchanged gradients."""
        from nifreeze.data.dmri import DWI

        n_vols = 10
        rng = np.random.default_rng(43)
        bvecs = rng.standard_normal((n_vols, 3))
        bvecs /= np.linalg.norm(bvecs, axis=1, keepdims=True)
        gradients = np.column_stack([bvecs, np.full(n_vols, 1000.0)])

        dwi = DWI(
            dataobj=rng.random((4, 4, 4, n_vols)).astype(np.float32),
            affine=np.eye(4),
            gradients=gradients,
        )
        dwi.motion_affines = np.repeat(np.eye(4)[None, ...], len(dwi), axis=0)
        np.testing.assert_array_almost_equal(
            dwi.corrected_gradients, dwi.gradients
        )

    def test_rotation_changes_bvecs(self):
        """A non-trivial rotation should change the b-vectors."""
        from nifreeze.data.dmri import DWI

        n_vols = 7
        # Create distinct gradient directions (need >= 6 DW orientations)
        rng = np.random.default_rng(42)
        bvecs = rng.standard_normal((n_vols, 3))
        bvecs /= np.linalg.norm(bvecs, axis=1, keepdims=True)
        gradients = np.column_stack([bvecs, np.full(n_vols, 1000.0)])

        dwi = DWI(
            dataobj=rng.random((4, 4, 4, n_vols)).astype(np.float32),
            affine=np.eye(4),
            gradients=gradients,
        )

        # Apply a 90-degree rotation around z for volume 0
        rot = np.eye(4)
        rot[0, 0] = 0
        rot[0, 1] = -1
        rot[1, 0] = 1
        rot[1, 1] = 0
        dwi.motion_affines = np.repeat(np.eye(4)[None, ...], len(dwi), axis=0)
        dwi.motion_affines[0] = rot

        corrected = dwi.corrected_gradients
        # Volume 0 should have changed bvec
        assert not np.allclose(corrected[0, :3], dwi.gradients[0, :3])
        # Volume 1 should be unchanged (identity)
        np.testing.assert_array_almost_equal(
            corrected[1, :3], dwi.gradients[1, :3]
        )
        # bvals should be unchanged for all volumes
        np.testing.assert_array_almost_equal(
            corrected[:, -1], dwi.gradients[:, -1]
        )


class TestGetextraUsesCorrectedGradients:
    """Test that _getextra returns corrected gradients."""

    def test_getextra_with_motion(self):
        """_getextra should use corrected_gradients, not raw gradients."""
        from nifreeze.data.dmri import DWI

        n_vols = 7
        rng = np.random.default_rng(99)
        bvecs = rng.standard_normal((n_vols, 3))
        bvecs /= np.linalg.norm(bvecs, axis=1, keepdims=True)
        gradients = np.column_stack([bvecs, np.full(n_vols, 1000.0)])

        dwi = DWI(
            dataobj=rng.random((4, 4, 4, n_vols)).astype(np.float32),
            affine=np.eye(4),
            gradients=gradients,
        )

        # Set a non-trivial motion for volume 0
        rot = np.eye(4)
        rot[0, 0] = 0
        rot[0, 1] = -1
        rot[1, 0] = 1
        rot[1, 1] = 0
        dwi.motion_affines = np.repeat(np.eye(4)[None, ...], len(dwi), axis=0)
        dwi.motion_affines[0] = rot

        # Access via __getitem__ which calls _getextra
        _, _, grad_out = dwi[0]
        expected = dwi.corrected_gradients[0]
        np.testing.assert_array_almost_equal(grad_out, expected)
