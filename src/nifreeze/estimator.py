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
"""Orchestrates model and registration in volume-to-volume artifact estimation."""

from __future__ import annotations

from importlib.resources import files
from os import cpu_count
from pathlib import Path
from tempfile import TemporaryDirectory
from timeit import default_timer as timer
from typing import TypeVar

import nibabel as nb
import nitransforms as nt
import numpy as np
from tqdm import tqdm
from typing_extensions import Self

from nifreeze.analysis.outliers import (
    OutlierConfig,
    detect_outlier_slices,
    replace_outlier_slices,
)
from nifreeze.data.base import BaseDataset
from nifreeze.data.pet import PET
from nifreeze.model.base import BaseModel, ModelFactory
from nifreeze.model.pet import BSplinePETModel
from nifreeze.registration.ants import (
    Registration,
    _prepare_registration_data,
    _prepare_svr_mask,
    _run_registration,
    _run_slice_registration,
    _to_nifti,
)
from nifreeze.utils import iterators

DatasetT = TypeVar("DatasetT", bound=BaseDataset)

DEFAULT_CHUNK_SIZE: int = int(1e6)
FIT_MSG = "Fit&predict"
REG_MSG = "Realign"
SVR_MSG = "SVR"


class Filter:
    """Alters an input data object (e.g., downsampling)."""

    def run(self, dataset: DatasetT, **kwargs) -> DatasetT:
        """
        Trigger execution of the designated filter.

        Parameters
        ----------
        dataset : :obj:`~nifreeze.data.base.BaseDataset`
            The input dataset this estimator operates on.

        Returns
        -------
        dataset : :obj:`~nifreeze.data.base.BaseDataset`
            The dataset, after filtering.

        """
        return dataset


class Estimator:
    """Orchestrates components for a single estimation step."""

    __slots__ = (
        "_model", "_single_fit", "_strategy", "_prev",
        "_model_kwargs", "_align_kwargs", "_outlier_config",
    )

    def __init__(
        self,
        model: BaseModel | str,
        strategy: str = "random",
        prev: Estimator | Filter | None = None,
        model_kwargs: dict | None = None,
        single_fit: bool = False,
        outlier_config: OutlierConfig | None = None,
        **kwargs,
    ):
        self._model = model
        self._prev = prev
        self._strategy = strategy
        self._single_fit = single_fit
        self._model_kwargs = model_kwargs or {}
        self._align_kwargs = kwargs or {}
        self._outlier_config = outlier_config

    def _init_model(
        self, dataset: BaseDataset, chunk_size: int
    ) -> BaseModel:
        """Instantiate or return the estimator's model."""
        if isinstance(self._model, str):
            if self._model.endswith("dti"):
                self._model_kwargs["step"] = chunk_size
            return ModelFactory.init(
                model=self._model,
                dataset=dataset,
                **self._model_kwargs,
            )
        return self._model

    def run(self, dataset: DatasetT, **kwargs) -> Self:
        """
        Trigger execution of the workflow this estimator belongs.

        Parameters
        ----------
        dataset : :obj:`~nifreeze.data.base.BaseDataset`
            The input dataset this estimator operates on.

        Returns
        -------
        :obj:`~nifreeze.estimator.Estimator`
            The estimator, after fitting.

        """
        if self._prev is not None:
            result = self._prev.run(dataset, **kwargs)
            if isinstance(self._prev, Filter):
                dataset = result  # type: ignore[assignment]

        n_jobs = kwargs.pop("n_jobs", None) or min(cpu_count() or 1, 8)
        n_threads = kwargs.pop("omp_nthreads", None) or ((cpu_count() or 2) - 1)

        num_voxels = dataset.brainmask.sum() if dataset.brainmask is not None else dataset.size3d
        chunk_size = DEFAULT_CHUNK_SIZE * (n_threads or 1)

        # SVR-specific parameters
        n_iter = kwargs.pop("n_iter", 1)
        s2v_niter = kwargs.pop("s2v_niter", 0)

        # Determine whether SVR is possible (slice_acquisition must be set)
        has_svr = (
            s2v_niter > 0
            and getattr(dataset, "slice_acquisition", None) is not None
        )

        # Prepare iterator
        iterfunc = getattr(iterators, f"{self._strategy}_iterator")
        iter_kwargs = {
            "size": len(dataset),
            "bvals": kwargs.pop("bvals", None),
            "uptake": kwargs.pop("uptake", None),
            "seed": kwargs.get("seed", None),
            "round_decimals": kwargs.pop("round_decimals", iterators.DEFAULT_ROUND_DECIMALS),
        }

        # Initialize model
        model = self._init_model(dataset, chunk_size)

        # Prepare fit/predict keyword arguments
        fit_pred_kwargs = {
            "n_jobs": n_jobs,
            "omp_nthreads": n_threads,
        }
        if model.__class__.__name__ == "DTIModel":
            fit_pred_kwargs["step"] = chunk_size

        print(f"Dataset size: {num_voxels}x{len(dataset)}.")
        print(f"Parallel execution: {fit_pred_kwargs}.")
        print(f"Model: {model}.")
        if has_svr:
            sa = dataset.slice_acquisition
            print(
                f"SVR enabled: {sa.n_excitations} excitations/vol, "
                f"MB{sa.multiband_factor}, {s2v_niter} inner iter(s)."
            )

        # Initialize outlier maps if detection is enabled
        ol_config = self._outlier_config
        if ol_config is not None and ol_config.enabled:
            n_slices = dataset.dataobj.shape[ol_config.slice_axis]
            dataset.outlier_map = np.zeros(
                (n_slices, len(dataset)), dtype=bool,
            )
            dataset.outlier_nstdev_map = np.zeros(
                (n_slices, len(dataset)), dtype=np.float64,
            )
            print(
                f"Outlier detection enabled: nstd={ol_config.nstd}, "
                f"nvox={ol_config.nvox}."
            )

        if self._single_fit:
            print("Fitting 'single' model started ...")
            start = timer()
            model.fit_predict(None, **fit_pred_kwargs)
            print(f"Fitting 'single' model finished, elapsed {timer() - start}s.")

        kwargs["num_threads"] = n_threads
        kwargs = self._align_kwargs | kwargs

        dataset_length = len(dataset)

        for outer_iter in range(n_iter):
            if n_iter > 1:
                print(f"--- Outer iteration {outer_iter + 1}/{n_iter} ---")

            # Re-generate the iterator for each outer pass
            index_iter = iterfunc(**iter_kwargs)

            _run_lovo_pass(
                dataset=dataset,
                model=model,
                index_iter=index_iter,
                dataset_length=dataset_length,
                fit_pred_kwargs=fit_pred_kwargs,
                has_svr=has_svr,
                s2v_niter=s2v_niter,
                outlier_config=ol_config,
                **kwargs,
            )

            # Between outer iterations: apply corrections so model trains on
            # motion-corrected data with rotated gradients.
            if outer_iter < n_iter - 1:
                print("Applying corrections (resampling + gradient rotation)...")
                dataset.apply_corrections()
                # Re-init model so it picks up corrected dataset
                model = self._init_model(dataset, chunk_size)

        return self


def _run_lovo_pass(
    dataset: BaseDataset,
    model: BaseModel,
    index_iter,
    dataset_length: int,
    fit_pred_kwargs: dict,
    has_svr: bool = False,
    s2v_niter: int = 0,
    outlier_config: OutlierConfig | None = None,
    **kwargs,
) -> None:
    """Execute one full LOVO pass (V2V + optional SVR) over all volumes."""
    with TemporaryDirectory() as tmp_dir:
        print(f"Processing in <{tmp_dir}>")
        ptmp_dir = Path(tmp_dir)

        bmask_path = None
        if dataset.brainmask is not None:
            bmask_path = ptmp_dir / "brainmask.nii.gz"
            nb.Nifti1Image(
                dataset.brainmask.astype(np.uint8), dataset.affine, None
            ).to_filename(bmask_path)

        with tqdm(total=dataset_length, unit="vols.") as pbar:
            for i in index_iter:
                pbar.set_description_str(f"{FIT_MSG: <16} vol. <{i}>")

                # fit the model
                predicted = model.fit_predict(  # type: ignore[union-attr]
                    i,
                    **fit_pred_kwargs,
                )

                # Outlier detection and replacement (before registration)
                if (
                    outlier_config is not None
                    and outlier_config.enabled
                    and predicted is not None
                ):
                    observed_vol = dataset[i][0]
                    ol_mask, ol_nstd = detect_outlier_slices(
                        observed_vol, predicted, dataset.brainmask,
                        outlier_config,
                    )
                    dataset.outlier_map[:, i] = ol_mask
                    dataset.outlier_nstdev_map[:, i] = ol_nstd
                    if ol_mask.any():
                        dataset.dataobj[..., i] = replace_outlier_slices(
                            observed_vol, predicted, ol_mask,
                            slice_axis=outlier_config.slice_axis,
                        )

                # prepare data for running ANTs
                predicted_path, volume_path, init_path = (
                    _prepare_registration_data(
                        dataset[i][0],
                        predicted,
                        dataset.affine,
                        i,
                        ptmp_dir,
                        kwargs.pop("clip", "both"),
                    )
                )

                pbar.set_description_str(f"{REG_MSG: <16} vol. <{i}>")

                xform = _run_registration(
                    predicted_path,
                    volume_path,
                    i,
                    ptmp_dir,
                    init_affine=init_path,
                    fixedmask_path=bmask_path,
                    output_transform_prefix=f"ants-{i:05d}",
                    **kwargs,
                )

                dataset.set_transform(i, xform.matrix)

                # ---- SVR inner loop ----
                if has_svr:
                    _run_svr_loop(
                        dataset=dataset,
                        model=model,
                        vol_idx=i,
                        predicted_path=predicted_path,
                        volume_path=volume_path,
                        bmask_path=bmask_path,
                        dirname=ptmp_dir,
                        s2v_niter=s2v_niter,
                        pbar=pbar,
                        **kwargs,
                    )

                pbar.update()


def _run_svr_loop(
    dataset: BaseDataset,
    model: BaseModel,
    vol_idx: int,
    predicted_path: Path,
    volume_path: Path,
    bmask_path: Path | None,
    dirname: Path,
    s2v_niter: int = 1,
    pbar: tqdm | None = None,
    **kwargs,
) -> None:
    """
    Run the slice-to-volume (SVR) inner loop for a single volume.

    For each excitation group within the volume, a per-excitation predicted
    volume is synthesised at the effective (rotated) gradient direction, and
    the excitation's slices are registered to the prediction using dual masks
    (brain mask on the fixed image, slice mask on the moving image).

    Parameters
    ----------
    dataset : :obj:`~nifreeze.data.base.BaseDataset`
        The dataset (must have ``slice_acquisition`` set).
    model : :obj:`~nifreeze.model.base.BaseModel`
        The **already-fitted** model (after :meth:`fit_predict` for this volume).
    vol_idx : :obj:`int`
        Index of the volume being processed.
    predicted_path : :obj:`~pathlib.Path`
        Path to the V2V-level predicted NIfTI (used as fallback if the model
        does not support :meth:`predict_at`).
    volume_path : :obj:`~pathlib.Path`
        Path to the actual (moving) volume NIfTI.
    bmask_path : :obj:`~pathlib.Path` or ``None``
        Path to the brain mask NIfTI (fixed mask).
    dirname : :obj:`~pathlib.Path`
        Working directory for ANTs outputs.
    s2v_niter : :obj:`int`
        Number of SVR inner iterations.
    pbar : :obj:`~tqdm.tqdm` or ``None``
        Progress bar for status updates.
    **kwargs
        Forwarded to :func:`_run_slice_registration` (seed, num_threads, …).

    """
    sa = dataset.slice_acquisition
    exc_groups = sa.excitation_groups
    t_order = sa.temporal_order()
    v2v_xform = dataset.motion_affines[vol_idx]

    # Check whether the model supports predict_at (DWI models do)
    has_predict_at = hasattr(model, "predict_at") and callable(
        getattr(model, "predict_at", None)
    )
    # DWI datasets carry gradient information
    has_gradients = hasattr(dataset, "gradients") and dataset.gradients is not None

    for _s2v_it in range(s2v_niter):
        for exc_idx in t_order:
            if pbar is not None:
                pbar.set_description_str(
                    f"{SVR_MSG: <16} vol.<{vol_idx}> exc.<{exc_idx}>"
                )

            slice_indices = exc_groups[exc_idx]

            # --- 1. Compute per-excitation predicted volume ---
            exc_predicted_path = predicted_path  # fallback: V2V prediction

            if has_predict_at and has_gradients:
                # Compose S2V × V2V to get effective rotation for this excitation
                if dataset.slice_motion_affines is not None:
                    s2v_xfm = dataset.slice_motion_affines[vol_idx, exc_idx]
                    composed_xfm = s2v_xfm @ v2v_xform
                else:
                    composed_xfm = v2v_xform

                from nifreeze.data.dmri.utils import transform_fsl_bvec

                rotated_bvec = transform_fsl_bvec(
                    dataset.gradients[vol_idx, :3],
                    composed_xfm,
                    dataset.affine,
                    invert=True,
                )
                rotated_gradient = np.append(
                    rotated_bvec, dataset.gradients[vol_idx, -1]
                )

                # Predict at the excitation's effective gradient direction
                predicted_exc = model.predict_at(rotated_gradient)

                exc_predicted_path = (
                    dirname
                    / f"predicted_v{vol_idx:05d}_e{exc_idx:03d}.nii.gz"
                )
                _to_nifti(
                    predicted_exc,
                    dataset.affine,
                    exc_predicted_path,
                    clip=True,
                )

            # --- 2. Create moving-space slice mask ---
            slicemask_path = _prepare_svr_mask(
                dataset.dataobj.shape[:3],
                dataset.affine,
                slice_indices,
                exc_idx,
                vol_idx,
                dirname,
                axis=sa.slice_axis,
            )

            # --- 3. Register ---
            exc_xform = _run_slice_registration(
                exc_predicted_path,
                volume_path,
                vol_idx=vol_idx,
                exc_idx=exc_idx,
                dirname=dirname,
                fixedmask_path=bmask_path,
                movingmask_path=slicemask_path,
                excitation_time=None,  # reserved for future temporal prediction
                **kwargs,
            )

            # --- 4. Store per-excitation transform ---
            dataset.set_slice_transform(vol_idx, exc_idx, exc_xform.matrix)


class PETMotionEstimator:
    """Estimates motion within PET imaging data aligned with generic Estimator workflow."""

    def __init__(self, align_kwargs: dict | None = None, strategy: str = "lofo"):
        self.align_kwargs = align_kwargs or {}
        self.strategy = strategy

    def run(self, pet_dataset: PET, omp_nthreads: int | None = None) -> list:
        n_frames = len(pet_dataset)
        frame_indices = np.arange(n_frames).astype(int)

        if omp_nthreads:
            self.align_kwargs["num_threads"] = omp_nthreads

        affine_matrices = []

        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            for idx in tqdm(frame_indices, desc="Estimating PET motion"):
                (train_data, train_times), (test_data, test_time) = pet_dataset.lofo_split(idx)

                if train_times is None:
                    raise ValueError(
                        f"train_times is None at index {idx}, check midframe initialization."
                    )

                # Build a temporary dataset excluding the test frame
                train_dataset = PET(
                    dataobj=train_data,
                    affine=pet_dataset.affine,
                    brainmask=pet_dataset.brainmask,
                    midframe=train_times,
                    total_duration=pet_dataset.total_duration,
                )

                # Instantiate the PET model explicitly
                model = BSplinePETModel(dataset=train_dataset)

                # Fit the model once on the training dataset
                model.fit_predict(None)

                # Predict the reference volume at the test frame's timepoint
                predicted = model.fit_predict(idx)

                fixed_image_path = tmp_path / f"fixed_frame_{idx:03d}.nii.gz"
                moving_image_path = tmp_path / f"moving_frame_{idx:03d}.nii.gz"

                fixed_img = nb.Nifti1Image(predicted, pet_dataset.affine)
                moving_img = nb.Nifti1Image(test_data, pet_dataset.affine)

                moving_img = nb.as_closest_canonical(moving_img, enforce_diag=True)

                fixed_img.to_filename(fixed_image_path)
                moving_img.to_filename(moving_image_path)

                registration_config = files("nifreeze.registration.config").joinpath(
                    "pet-to-pet_level1.json"
                )

                registration = Registration(
                    from_file=registration_config,
                    fixed_image=str(fixed_image_path),
                    moving_image=str(moving_image_path),
                    output_warped_image=True,
                    output_transform_prefix=f"ants_{idx:03d}",
                    **self.align_kwargs,
                )

                try:
                    result = registration.run(cwd=str(tmp_path))
                    if result.outputs.forward_transforms:
                        transform = nt.io.itk.ITKLinearTransform.from_filename(
                            result.outputs.forward_transforms[0]
                        )
                        matrix = transform.to_ras(
                            reference=str(fixed_image_path), moving=str(moving_image_path)
                        )
                        affine_matrices.append(matrix)
                    else:
                        affine_matrices.append(np.eye(4))
                        print(f"No transforms produced for index {idx}")
                except Exception as e:
                    affine_matrices.append(np.eye(4))
                    print(f"Failed to process frame {idx} due to {e}")

        return affine_matrices
