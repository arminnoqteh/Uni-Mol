from __future__ import absolute_import, division, print_function

import numpy as np
import torch
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold
from rdkit.Chem.Scaffolds import MurckoScaffold
from ..utils import logger


def generate_scaffold(smiles, include_chirality=False):
    """
    Obtain Bemis-Murcko scaffold from smiles

    Args:
        smiles: smiles sequence
        include_chirality: Default=False

    Return:
        the scaffold of the given smiles.
    """
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(
            smiles=smiles, includeChirality=include_chirality
        )
    except Exception as e:
        print(e)
        return None
    return scaffold


# TODO (armin) add fracs to config file and read from there


class Splitter(object):
    """
    The Splitter class is responsible for splitting a dataset into train and test sets
    based on the specified method.
    """

    def __init__(
        self,
        method="random",
        kfold=5,
        seed=42,
        frac_train=0.8,
        frac_val=0.1,
        frac_test=0.1,
        **params,
    ):
        """
        Initializes the Splitter with a specified split method and random seed.

        Args:
            method (str): The method for splitting the dataset ('random', 'scaffold', 'group', 'stratified', 'select')
            kfold (int): Number of folds for cross-validation
            seed (int): Random seed for reproducibility
            frac_train (float): Fraction of data for training (for scaffold split)
            frac_val (float): Fraction of data for validation (for scaffold split)
            frac_test (float): Fraction of data for testing (for scaffold split)
        """
        self.method = method
        self.n_splits = kfold
        self.seed = seed
        self.frac_train = frac_train
        self.frac_val = frac_val
        self.frac_test = frac_test
        self.splitter = self._init_split()

    def _init_split(self):
        """
        Initializes the actual splitter object based on the specified method.

        Returns:
            The initialized splitter object.
        Raises:
            ValueError: If an unknown splitting method is specified.
        """
        if self.n_splits == 1:
            return None
        if self.method == "random":
            splitter = KFold(
                n_splits=self.n_splits, shuffle=True, random_state=self.seed
            )
        elif self.method == "group":
            splitter = GroupKFold(n_splits=self.n_splits)
        elif self.method == "stratified":
            splitter = StratifiedKFold(
                n_splits=self.n_splits, shuffle=True, random_state=self.seed
            )
        elif self.method == "select":
            splitter = GroupKFold(n_splits=self.n_splits)
        elif self.method == "scaffold":
            return None  # Scaffold split uses its own logic
        else:
            raise ValueError(
                f"Unknown splitter method: {self.n_splits}fold - {self.method}"
            )

        return splitter

    def _scaffold_split(self, smiles):
        """
        Performs scaffold-based splitting of the dataset.

        Args:
            smiles: List of SMILES strings

        Returns:
            Tuple of (train_idx, valid_idx, test_idx) as torch tensors
        """
        N = len(smiles)
        failed_scaffolds = []

        # Create dict of the form {scaffold_i: [idx1, idx....]}
        all_scaffolds = {}
        for i in range(N):
            scaffold = generate_scaffold(smiles[i])
            if scaffold is None:
                failed_scaffolds.append(smiles[i])
                continue
            if scaffold not in all_scaffolds:
                all_scaffolds[scaffold] = [i]
            else:
                all_scaffolds[scaffold].append(i)

        N = N - len(failed_scaffolds)

        # Sort from largest to smallest sets
        all_scaffolds = {key: sorted(value) for key, value in all_scaffolds.items()}
        all_scaffold_sets = [
            scaffold_set
            for (scaffold, scaffold_set) in sorted(
                all_scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True
            )
        ]

        # Get train, valid test indices
        train_cutoff = self.frac_train * N
        valid_cutoff = (self.frac_train + self.frac_val) * N
        train_idx, valid_idx, test_idx = [], [], []

        for scaffold_set in all_scaffold_sets:
            if len(train_idx) + len(scaffold_set) > train_cutoff:
                if len(train_idx) + len(valid_idx) + len(scaffold_set) > valid_cutoff:
                    test_idx.extend(scaffold_set)
                else:
                    valid_idx.extend(scaffold_set)
            else:
                train_idx.extend(scaffold_set)

        return (
            torch.tensor(train_idx, dtype=torch.long),
            torch.tensor(valid_idx, dtype=torch.long),
            torch.tensor(test_idx, dtype=torch.long),
        )

    def split(self, smiles, target=None, group=None, scaffolds=None, **params):
        """
        Splits the dataset into train and test sets based on the initialized method.

        Args:
            smiles: List of SMILES strings
            target: (optional) Target labels for stratified splitting
            group: (optional) Group labels for group-based splitting
            scaffolds: (optional) Pre-computed scaffolds for scaffold splitting

        Returns:
            An iterator yielding train and test set indices for each fold.
        """
        if self.n_splits == 1:
            logger.warning(
                "Only one fold is used for training, no splitting is performed."
            )
            return [(np.arange(len(smiles)), ())]

        if self.method == "scaffold":
            train_idx, valid_idx, test_idx = self._scaffold_split(smiles)
            # Convert to numpy for consistency with other methods
            return [(train_idx.numpy(), valid_idx.numpy())]
        elif self.method == "random":
            self.skf = self.splitter.split(smiles)
        elif self.method == "group":
            self.skf = self.splitter.split(smiles, target, group)
        elif self.method == "stratified":
            self.skf = self.splitter.split(smiles, group)
        elif self.method == "select":
            unique_groups = np.unique(group)
            if len(unique_groups) == self.n_splits:
                split_folds = []
                for unique_group in unique_groups:
                    train_idx = np.where(group != unique_group)[0]
                    test_idx = np.where(group == unique_group)[0]
                    split_folds.append((train_idx, test_idx))
                self.split_folds = split_folds
                return self.split_folds
            else:
                logger.error(
                    "The number of unique groups is not equal to the number of splits."
                )
                exit(1)
        else:
            logger.error(f"Unknown splitter method: {self.method}")
            exit(1)

        self.split_folds = list(self.skf)
        return self.split_folds
