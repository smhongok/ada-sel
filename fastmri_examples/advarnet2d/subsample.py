"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch

from fastmri.data.subsample import MaskFunc, RandomMaskFunc, EquiSpacedMaskFunc, EquispacedMaskFractionFunc, temp_seed

import contextlib
@contextlib.contextmanager

def create_mask_for_mask_type(
    mask_type_str: str,
    center_fractions: Sequence[float],
    accelerations: Sequence[int],
) -> MaskFunc:
    """
    Creates a mask of the specified type.

    Args:
        center_fractions: What fraction of the center of k-space to include.
        accelerations: What accelerations to apply.
        skip_low_freqs: Whether to skip already sampled low-frequency lines
            for the purposes of determining where equispaced lines should be.
            Set this `True` to guarantee the same number of sampled lines for
            all masks with a given (acceleration, center_fraction) setting.

    Returns:
        A mask func for the target mask type.
    """
    if mask_type_str == "adaptive":
        return CenterMaskFractionFunc(center_fractions, accelerations)        
    elif mask_type_str == "random":
        return RandomMaskFunc(center_fractions, accelerations)
    elif mask_type_str == "equispaced":
        return EquiSpacedMaskFunc(center_fractions, accelerations)
    elif mask_type_str == "equispaced_fraction":
        return EquispacedMaskFractionFunc(center_fractions, accelerations)
    else:
        raise ValueError(f"{mask_type_str} not supported")

class CenterMaskFractionFunc(MaskFunc):
    """
    Equispaced mask with strictly exact acceleration matching.

    The mask selects a subset of columns from the input k-space data. If the
    k-space data has N columns, the mask picks out:
        1. N_low_freqs = (N * center_fraction) columns in the center
           corresponding to low-frequencies.
        2. The other columns are selected with equal spacing at a proportion
           that reaches the desired acceleration rate taking into consideration
           the number of low frequencies. This ensures that the expected number
           of columns selected is equal to (N / acceleration)

    It is possible to use multiple center_fractions and accelerations, in which
    case one possible (center_fraction, acceleration) is chosen uniformly at
    random each time the EquispacedMaskFunc object is called.

    Note that this function may not give equispaced samples (documented in
    https://github.com/facebookresearch/fastMRI/issues/54), which will require
    modifications to standard GRAPPA approaches. Nonetheless, this aspect of
    the function has been preserved to match the public multicoil data.
    """

    def __init__(
        self,
        center_fractions: Sequence[float],
        accelerations: Sequence[int], # Does Nothing
    ):
        """
        Args:
            center_fractions: Fraction of low-frequency columns to be retained.
                If multiple values are provided, then one of these numbers is
                chosen uniformly each time.
            accelerations: Amount of under-sampling. This should have the same
                length as center_fractions. If multiple values are provided,
                then one of these is chosen uniformly each time.
            skip_low_freqs: Whether to skip already sampled low-frequency lines
                for the purposes of determining where equispaced lines should
                be. Set this `True` to guarantee the same number of sampled
                lines for all masks with a given (acceleration,
                center_fraction) setting.
        """
        super().__init__(center_fractions, accelerations)

    def calculate_acceleration_mask(
        self,
        num_cols: int,
        acceleration: int,
        offset: Optional[int],
        num_low_frequencies: int,
    ) -> np.ndarray:
        """
        Produce mask for non-central acceleration lines.

        Args:
            num_cols: Number of columns of k-space (2D subsampling).
            acceleration: Desired acceleration rate.
            offset: Offset from 0 to begin masking. If no offset is specified,
                then one is selected randomly.
            num_low_frequencies: Number of low frequencies. Used to adjust mask
                to exactly match the target acceleration.

        Returns:
            A mask for the high spatial frequencies of k-space.
        """
        mask = np.zeros(num_cols)
        
        #         pad = (num_cols - num_low_frequencies + 1) // 2

        #         # determine acceleration rate by adjusting for the number of low frequencies
        #         adjusted_accel = (acceleration * (num_low_frequencies - num_cols)) / (
        #             num_low_frequencies * acceleration - num_cols
        #         )
        #         offset = self.rng.randint(0, round(adjusted_accel) - 1)

        #         # Select samples from the remaining columns
        #         accel_samples = np.arange(
        #             offset, num_cols - num_low_frequencies - 1, adjusted_accel
        #         )
        #         accel_samples = np.around(accel_samples).astype(int)

        #         skip = (
        #             num_low_frequencies  # Skip low freq AND optionally lines right next to it
        #         )
        #         for sample in accel_samples:
        #             if sample < pad:
        #                 mask[sample] = True
        #             else:  # sample is further than center, so skip low_freqs
        #                 mask[int(sample + skip)] = True

        return mask


class Mask2DFunc:
    """
    An object for 2D sampling masks.

    This crates a 2D sampling mask that densely samples the center while
    subsampling outer k-space regions based on the undersampling factor.

    When called, ``Mask2DFunc`` uses internal functions create mask by 1)
    creating a mask for the k-space center, 2) create a mask outside of the
    k-space center, and 3) combining them into a total mask. The internals are
    handled by ``sample_mask``, which calls ``calculate_center_mask`` for (1)
    and ``calculate_acceleration_mask`` for (2). The combination is executed
    in the ``Mask2DFunc`` ``__call__`` function.

    If you would like to implement a new mask, simply subclass ``Mask2DFunc``
    and overwrite the ``sample_mask`` logic. See examples in ``RandomMask2DFunc``.
    """

    def __init__(
        self,
        center_fractions: Sequence[float],
        accelerations: Sequence[int],
        allow_any_combination: bool = False,
        seed: Optional[int] = None,
    ):
        """
        Args:
            center_fractions: Fraction of low-frequency columns and low-frequency rows to be retained.
                If multiple values are provided, then one of these numbers is
                chosen uniformly each time.
            accelerations: Amount of under-sampling. This should have the same
                length as center_fractions. If multiple values are provided,
                then one of these is chosen uniformly each time.
            allow_any_combination: Whether to allow cross combinations of
                elements from ``center_fractions`` and ``accelerations``.
            seed: Seed for starting the internal random number generator of the
                ``Mask2DFunc``.
        """
        if len(center_fractions) != len(accelerations) and not allow_any_combination:
            raise ValueError(
                "Number of center fractions should match number of accelerations "
                "if allow_any_combination is False."
            )

        self.center_fractions = center_fractions
        self.accelerations = accelerations
        self.allow_any_combination = allow_any_combination
        self.rng = np.random.RandomState(seed)

    def __call__(
        self,
        shape: Sequence[int],
        offset: Optional[int] = None,
        seed: Optional[Union[int, Tuple[int, ...]]] = None,
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Sample and return a k-space mask.

        Args:
            shape: Shape of k-space.
            offset: Offset from 0 to begin mask (for equispaced masks). If no
                offset is given, then one is selected randomly.
            seed: Seed for random number generator for reproducibility.

        Returns:
            A 2-tuple containing 1) the 2D k-space mask and 2) the number of
            center frequency lines.
        """
        if len(shape) < 3:
            raise ValueError("Shape should have 3 or more dimensions")

        with temp_seed(self.rng, seed):
            center_mask, accel_mask, num_low_frequencies = self.sample_mask(
                shape, offset
            )

        # combine masks together
        return torch.max(center_mask, accel_mask), num_low_frequencies

    def sample_mask(
        self,
        shape: Sequence[int],
        offset: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
        """
        Sample a new 2D k-space mask.

        This function samples and returns two components of a k-space mask: 1)
        the center mask (e.g., for sensitivity map calculation) and 2) the
        acceleration mask (for the edge of k-space). Both of these masks, as
        well as the integer of low frequency samples, are returned.

        Args:
            shape: Shape of the k-space to subsample.
            offset: Offset from 0 to begin mask (for equispaced masks).

        Returns:
            A 3-tuple contaiing 1) the mask for the center of k-space, 2) the
            mask for the high frequencies of k-space, and 3) the integer count
            of low frequency samples.
        """
        num_rows, num_cols = shape[-3], shape[-2]
        center_fraction, acceleration = self.choose_acceleration()
        num_low_frequencies = (round(num_rows * center_fraction), round(num_cols * center_fraction))
        center_mask = self.reshape_mask(
            self.calculate_center_mask(shape, num_low_frequencies), shape
        ) # 2D mask
        acceleration_mask = self.reshape_mask(
            self.calculate_acceleration_mask(
                num_rows, num_cols, acceleration, offset, num_low_frequencies
            ),
            shape,
        )

        return center_mask, acceleration_mask, num_low_frequencies

    def reshape_mask(self, mask: np.ndarray, shape: Sequence[int]) -> torch.Tensor:
        """Reshape mask to desired output shape."""
        num_rows, num_cols = shape[-3], shape[-2]
        mask_shape = [1 for _ in shape]
        mask_shape[-3] = num_rows
        mask_shape[-2] = num_cols        

        return torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))

    def calculate_acceleration_mask(
        self,
        num_rows: int,
        num_cols: int,
        acceleration: int,
        offset: Optional[int],
        num_low_frequencies: Tuple[int, int],
    ) -> np.ndarray:
        """
        Produce mask for non-central acceleration lines.

        Args:
            num_rows: Number of rows of k-space.
            num_cols: Number of columns of k-space.            
            acceleration: Desired acceleration rate.
            offset: Offset from 0 to begin masking (for equispaced masks).
            num_low_frequencies: Integer count of low-frequency lines sampled.

        Returns:
            A mask for the high spatial frequencies of k-space.
        """
        raise NotImplementedError

    def calculate_center_mask(
        self, shape: Sequence[int], num_low_freqs: Tuple[int, int]
    ) -> np.ndarray:
        """
        Build center mask based on number of low frequencies.

        Args:
            shape: Shape of k-space to mask.
            num_low_freqs: Number of low-frequency lines to sample. (row, col)

        Returns:
            A 2D mask for hte low spatial frequencies of k-space.
        """
        num_rows, num_cols = shape[-3], shape[-2]
        mask = np.zeros((num_rows, num_cols), dtype=np.float32)
        pad_row = (num_rows - num_low_freqs[0] + 1) // 2
        pad_col = (num_cols - num_low_freqs[1] + 1) // 2        
        mask[pad_row : pad_row + num_low_freqs[0], pad_col : pad_col + num_low_freqs[1]] = 1
        assert mask.sum() == num_low_freqs[0] * num_low_freqs[1]

        return mask

    def choose_acceleration(self):
        """Choose acceleration based on class parameters."""
        if self.allow_any_combination:
            return self.rng.choice(self.center_fractions), self.rng.choice(
                self.accelerations
            )
        else:
            choice = self.rng.randint(len(self.center_fractions))
            return self.center_fractions[choice], self.accelerations[choice]

def create_mask2d_for_mask_type(
    mask_type_str: str,
    center_fractions: Sequence[float],
    accelerations: Sequence[int],
    vmap_target_path: str,
    budget: int,
    power_vmap: float,
    sortvmap_fraction: float
) -> Mask2DFunc:
    """
    Creates a 2-dim mask of the specified type.

    Args:
        center_fractions: What fraction of the center of k-space to include.
        accelerations: What accelerations to apply.

    Returns:
        A mask func for the target mask type.
    """      
    if mask_type_str == "random":
        return RandomMask2DFunc(center_fractions, accelerations)
    elif mask_type_str == "equispaced_fraction":
        return EquispacedMask2DFractionFunc(center_fractions, accelerations)
    elif mask_type_str == "adaptive" or mask_type_str == "loupe":
        return CenterMask2DFractionFunc(center_fractions, accelerations)
    elif mask_type_str == "designate":
        return DesignateMask2DFunc(center_fractions, accelerations, vmap_target_path, budget, power_vmap, sortvmap_fraction)
    else:
        raise ValueError(f"{mask_type_str} not supported")
 
def create_budget_for_acquisition(
    crop_size: Sequence[Tuple[int, int]],
    center_fractions: Sequence[float],
    accelerations: Sequence[int],
):
    """
    return budget for acquisition
    """    
    budget = int(
        crop_size[0] * crop_size[1] * (1 / accelerations - center_fractions ** 2)
    )
    
    return budget

def calc_num_sense_lines(
    crop_size: Sequence[Tuple[int, int]],
    center_fractions: Sequence[float],
):
    """
    return num_sense_lines
    """

    num_sense_lines = (int(crop_size[0] * center_fractions), int(crop_size[1] * center_fractions))
    
    return num_sense_lines
    
 
class RandomMask2DFunc(Mask2DFunc):
    """
    Creates a random 2-dim sub-sampling mask of a given shape.

    The mask selects a subset point from the input k-space data. If the
    k-space data has N columns and M rows, the mask picks out:
        1. N_low_freqs = (N * center_fraction) columns and (M * center_fraction) rows in the center
           corresponding to low-frequencies.
        2. The other columns are selected uniformly at random with a
        probability equal to: 
        prob = (N * M / acceleration - N_low_freqs) / (N * M - N_low_freqs). 
        This ensures that the expected number of points selected is equal to (N * M / acceleration).

    It is possible to use multiple center_fractions and accelerations, in which
    case one possible (center_fraction, acceleration) is chosen uniformly at
    random each time the ``RandomMaskFunc`` object is called.

    For example, if accelerations = [4, 8] and center_fractions = [0.08, 0.04],
    then there is a 50% probability that 4-fold acceleration with 8% center
    fraction is selected and a 50% probability that 8-fold acceleration with 4%
    center fraction is selected.
    """

    def calculate_acceleration_mask(
        self,
        num_rows: int,
        num_cols: int,
        acceleration: int,
        offset: Optional[int],
        num_low_frequencies: Tuple[int, int],
    ) -> np.ndarray:
        prob = (num_rows * num_cols / acceleration - num_low_frequencies[0] * num_low_frequencies[1]) / (
            num_cols * num_rows - num_low_frequencies[0] * num_low_frequencies[1]
        )

        return self.rng.uniform(size=(num_rows, num_cols)) < prob
    
class DesignateMask2DFunc(Mask2DFunc):
    """
    Creates a Designate Mask 2-dim sub-sampling mask of a given shape.

    The mask selects a subset point from the input k-space data. If the
    k-space data has N columns and M rows, the mask picks out:
        1. N_low_freqs = (N * center_fraction) columns and (M * center_fraction) rows in the center
           corresponding to low-frequencies.
        2. The other columns are selected uniformly at random with a
        probability equal to: 
        prob = (N * M / acceleration - N_low_freqs) / (N * M - N_low_freqs). 
        This ensures that the expected number of points selected is equal to (N * M / acceleration).

    It is possible to use multiple center_fractions and accelerations, in which
    case one possible (center_fraction, acceleration) is chosen uniformly at
    random each time the ``RandomMaskFunc`` object is called.

    For example, if accelerations = [4, 8] and center_fractions = [0.08, 0.04],
    then there is a 50% probability that 4-fold acceleration with 8% center
    fraction is selected and a 50% probability that 8-fold acceleration with 4%
    center fraction is selected.
    """
    def __init__(
        self,
        center_fractions: Sequence[float],
        accelerations: Sequence[int],
        vmap_target_path: str,
        budget: int,
        power_vmap: float,
        softvmap_fraction: float,
        allow_any_combination: bool = False,
        seed: Optional[int] = None,
    ):
        if len(center_fractions) != len(accelerations) and not allow_any_combination:
            raise ValueError(
                "Number of center fractions should match number of accelerations "
                "if allow_any_combination is False."
            )

        self.center_fractions = center_fractions
        self.accelerations = accelerations
        self.vmap_target_path = vmap_target_path
        self.budget = budget
        self.power_vmap = power_vmap
        self.softvmap_fraction = softvmap_fraction
        self.allow_any_combination = allow_any_combination
        self.rng = np.random.RandomState(seed)
        
    def calculate_acceleration_mask(
        self,
        num_rows: int,
        num_cols: int,
        acceleration: int,
        offset: Optional[Tuple[int, int]],
        num_low_frequencies: Tuple[int, int],
    ) -> np.ndarray:
        """
        Produce 2D mask for non-central acceleration lines.

        Args:
            num_rows: Number of rows of k-space (2D subsampling).
            num_cols: Number of columns of k-space (2D subsampling).            
            acceleration: Desired acceleration rate.
            offset: Offset from 0 to begin masking. If no offset is specified,
                then one is selected randomly.
            num_low_frequencies: Number of low frequencies. Used to adjust mask
                to exactly match the target acceleration.

        Returns:
            A mask for the high spatial frequencies of k-space.
        """
        vmap = np.load(self.vmap_target_path)
        
        #### preprocess vmap
        vmap = np.power(vmap, self.power_vmap)
        ####

        center_mask = np.zeros((num_rows, num_cols), dtype=bool)
        pad_row = (num_rows - num_low_frequencies[0] + 1) // 2
        pad_col = (num_cols - num_low_frequencies[1] + 1) // 2 
        center_mask[pad_row : pad_row + num_low_frequencies[0], pad_col : pad_col + num_low_frequencies[1]] = 1
        assert center_mask.sum() == num_low_frequencies[0] * num_low_frequencies[1]
        
        vmap_mask = vmap * ~center_mask
        vmap_flatten = vmap_mask.flatten()   
        
        sorted_budget = int(self.budget * self.softvmap_fraction)
        random_budget = self.budget - sorted_budget
        
        vmap_hf_flatten = vmap_flatten.copy()
        if sorted_budget > 0:
            sorted_idx = np.argsort(vmap_flatten)[-sorted_budget:]
            vmap_hf_flatten[sorted_idx] = 0
        
        vmap_hf_flatten = vmap_hf_flatten / vmap_hf_flatten.sum()
        
        # subsampling
        np.random.seed(42)
        sampled_idx = np.random.choice(vmap_hf_flatten.shape[0], random_budget, replace=False, p=vmap_hf_flatten)
        mask = np.zeros_like(vmap_hf_flatten)
        if sorted_budget > 0:
            mask[sorted_idx] = 1
        mask[sampled_idx] = 1
        mask = mask.reshape((num_rows, num_cols))
        
        return mask
    
class EquispacedMask2DFractionFunc(Mask2DFunc):
    """
    2-dim Equispaced mask with approximate acceleration matching.

    The mask selects a subset of columns from the input k-space data. If the
    k-space data has N columns and M rows, the mask picks out:
        1. N_low_freqs = (N * center_fraction) columns and (M * center_fraction) rows in the center
           corresponding to low-frequencies.
        2. The other columns are selected with equal spacing at a proportion
           that reaches the desired acceleration rate taking into consideration
           the number of low frequencies. This ensures that the expected number
           of columns selected is equal to (N / sqrt(acceleration)) and the expected number
           of rows selected is equal to (M / sqrt(acceleration)).

    It is possible to use multiple center_fractions and accelerations, in which
    case one possible (center_fraction, acceleration) is chosen uniformly at
    random each time the EquispacedMaskFunc object is called.

    Note that this function may not give equispaced samples (documented in
    https://github.com/facebookresearch/fastMRI/issues/54), which will require
    modifications to standard GRAPPA approaches. Nonetheless, this aspect of
    the function has been preserved to match the public multicoil data.
    """

    def calculate_acceleration_mask(
        self,
        num_rows: int,
        num_cols: int,
        acceleration: int,
        offset: Optional[Tuple[int, int]],
        num_low_frequencies: int,
    ) -> np.ndarray:
        """
        Produce 2D mask for non-central acceleration lines.

        Args:
            num_rows: Number of rows of k-space (2D subsampling).
            num_cols: Number of columns of k-space (2D subsampling).            
            acceleration: Desired acceleration rate.
            offset: Offset from 0 to begin masking. If no offset is specified,
                then one is selected randomly.
            num_low_frequencies: Number of low frequencies. Used to adjust mask
                to exactly match the target acceleration.

        Returns:
            A mask for the high spatial frequencies of k-space.
        """
        # determine acceleration rate by adjusting for the number of low frequencies
        adjusted_accel_row = np.sqrt(
            (num_rows / num_cols) * (num_rows * num_cols - num_low_frequencies[0] * num_low_frequencies[1]) * acceleration / (
                num_rows * num_cols - num_low_frequencies[0] * num_low_frequencies[1] * acceleration
            )
        )
        adjusted_accel_col = np.sqrt(
            (num_cols / num_rows) * (num_rows * num_cols - num_low_frequencies[0] * num_low_frequencies[1]) * acceleration / (
                num_rows * num_cols - num_low_frequencies[0] * num_low_frequencies[1] * acceleration
            )
        )

        if offset is None:
            offset = self.rng.randint(0, high=round(adjusted_accel_row)), self.rng.randint(0, high=round(adjusted_accel_col))

        mask = np.zeros((num_rows, num_cols))
        accel_samples_row = np.around(np.arange(offset[0], num_rows - 1, adjusted_accel_row)).astype(np.uint)
        accel_samples_col = np.around(np.arange(offset[1], num_cols - 1, adjusted_accel_col)).astype(np.uint) 
        
        accel_samples = np.meshgrid(accel_samples_row, accel_samples_col)
        mask[accel_samples[0], accel_samples[1]] = 1.0
        
        return mask
    
class CenterMask2DFractionFunc(Mask2DFunc):
    """
    Produce zero 2D mask for non-central acceleration lines.
    (for AdVarNet2D)
    """

    def calculate_acceleration_mask(
        self,
        num_rows: int,
        num_cols: int,
        acceleration: int,
        offset: Optional[Tuple[int, int]],
        num_low_frequencies: int,
    ) -> np.ndarray:
        """
        Produce zero 2D mask for non-central acceleration lines.

        Args:
            num_rows: Number of rows of k-space (2D subsampling).
            num_cols: Number of columns of k-space (2D subsampling).            
            acceleration: Desired acceleration rate.
            offset: Offset from 0 to begin masking. If no offset is specified,
                then one is selected randomly.
            num_low_frequencies: Number of low frequencies. Used to adjust mask
                to exactly match the target acceleration.

        Returns:
            A zero mask for the high spatial frequencies of k-space.
        """
        mask = np.zeros((num_rows, num_cols))
        
        return mask
     
