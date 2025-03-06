# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function

import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import joblib
from torch.utils.data import Dataset
import numpy as np
from ..utils import logger
from .unimol import UniMolModel
from .unimolv2 import UniMolV2Model
from .loss import GHMC_Loss, FocalLossWithLogits, myCrossEntropyLoss, MAEwithNan


NNMODEL_REGISTER = {
    "unimolv1": UniMolModel,
    "unimolv2": UniMolV2Model,
}

LOSS_RREGISTER = {
    "classification": myCrossEntropyLoss,
    "multiclass": myCrossEntropyLoss,
    "regression": nn.MSELoss(),
    "multilabel_classification": {
        "bce": nn.BCEWithLogitsLoss(),
        "ghm": GHMC_Loss(bins=10, alpha=0.5),
        "focal": FocalLossWithLogits,
    },
    "multilabel_regression": MAEwithNan,
}
ACTIVATION_FN = {
    # predict prob shape should be (N, K), especially for binary classification, K equals to 1.
    "classification": lambda x: F.softmax(x, dim=-1)[:, 1:],
    # softmax is used for multiclass classification
    "multiclass": lambda x: F.softmax(x, dim=-1),
    "regression": lambda x: x,
    # sigmoid is used for multilabel classification
    "multilabel_classification": lambda x: F.sigmoid(x),
    # no activation function is used for multilabel regression
    "multilabel_regression": lambda x: x,
}
OUTPUT_DIM = {
    "classification": 2,
    "regression": 1,
}


class NNModel(object):
    """A :class:`NNModel` class is responsible for initializing the model"""

    def __init__(self, data, trainer, **params):
        """
        Initializes the neural network model with the given data and parameters.

        :param data: (dict) Contains the dataset information, including features and target scaling.
        :param trainer: (object) An instance of a training class, responsible for managing training processes.
        :param params: Various additional parameters used for model configuration.

        The model is configured based on the task type and specific parameters provided.
        """
        self.data = data
        self.num_classes = self.data["num_classes"]
        self.target_scaler = self.data["target_scaler"]
        self.features = data["unimol_input"]
        self.model_name = params.get("model_name", "unimolv1")
        self.data_type = params.get("data_type", "molecule")
        self.loss_key = params.get("loss_key", None)
        self.trainer = trainer
        # self.splitter = self.trainer.splitter
        self.model_params = params.copy()
        self.task = params["task"]
        if self.task in OUTPUT_DIM:
            self.model_params["output_dim"] = OUTPUT_DIM[self.task]
        elif self.task == "multiclass":
            self.model_params["output_dim"] = self.data["multiclass_cnt"]
        else:
            self.model_params["output_dim"] = self.num_classes
        self.model_params["device"] = self.trainer.device
        self.cv = dict()
        self.metrics = self.trainer.metrics
        if self.task == "multilabel_classification":
            if self.loss_key is None:
                self.loss_key = "focal"
            self.loss_func = LOSS_RREGISTER[self.task][self.loss_key]
        else:
            self.loss_func = LOSS_RREGISTER[self.task]
        self.activation_fn = ACTIVATION_FN[self.task]
        self.save_path = self.trainer.save_path
        self.trainer.set_seed(self.trainer.seed)
        self.model = self._init_model(**self.model_params)

    def _init_model(self, model_name, **params):
        """
        Initializes the neural network model based on the provided model name and parameters.

        :param model_name: (str) The name of the model to initialize.
        :param params: Additional parameters for model configuration.

        :return: An instance of the specified neural network model.
        :raises ValueError: If the model name is not recognized.
        """
        freeze_layers = params.get("freeze_layers", None)
        freeze_layers_reversed = params.get("freeze_layers_reversed", False)
        if model_name in NNMODEL_REGISTER:
            model = NNMODEL_REGISTER[model_name](**params)
            if isinstance(freeze_layers, str):
                freeze_layers = freeze_layers.replace(" ", "").split(",")
            if isinstance(freeze_layers, list):
                for layer_name, layer_param in model.named_parameters():
                    should_freeze = any(
                        layer_name.startswith(freeze_layer)
                        for freeze_layer in freeze_layers
                    )
                    layer_param.requires_grad = not (
                        freeze_layers_reversed ^ should_freeze
                    )
        else:
            raise ValueError("Unknown model: {}".format(self.model_name))
        return model

    def collect_data(self, X, y, idx):
        """
        Collects and formats the training or validation data.

        :param X: (np.ndarray or dict) The input features, either as a numpy array or a dictionary of tensors.
        :param y: (np.ndarray) The target values as a numpy array.
        :param idx: Indices to select the specific data samples.

        :return: A tuple containing processed input data and target values.
        :raises ValueError: If X is neither a numpy array nor a dictionary.
        """
        assert isinstance(y, np.ndarray), "y must be numpy array"
        if isinstance(X, np.ndarray):
            return torch.from_numpy(X[idx]).float(), torch.from_numpy(y[idx])
        elif isinstance(X, list):
            return {k: v[idx] for k, v in X.items()}, torch.from_numpy(y[idx])
        else:
            raise ValueError("X must be numpy array or dict")

    def run(self):
        """
        Run the model training and evaluation process using the three-way split.

        This modified version handles the train/validation/test splits.
        """
        split_nfolds = self.data["split_nfolds"]

        cv_pred = []
        cv_true = []
        for fold, (train_idx, valid_idx, test_idx) in enumerate(split_nfolds):
            logger.info(
                f"Fold {fold}: Training with {len(train_idx)} samples, validating with {len(valid_idx)} samples, testing with {len(test_idx)} samples"
            )

            train_dataset = self.get_dataset(train_idx)
            valid_dataset = self.get_dataset(valid_idx)
            test_dataset = self.get_dataset(test_idx) if len(test_idx) > 0 else None

            # Initialize model for this fold
            model = self._init_model(**self.model_params)

            # Train and validate the model
            y_preds = self.trainer.fit_predict(
                model=model,
                train_dataset=train_dataset,
                valid_dataset=valid_dataset,
                loss_func=self.loss_func,
                activation_fn=self.activation_fn,
                dump_dir=self.save_path,
                fold=fold,
                target_scaler=self.data["target_scaler"],
                feature_name=None,
            )

            # Add validation predictions to cross-validation results
            valid_indices = (
                valid_idx.cpu().numpy()
                if isinstance(valid_idx, torch.Tensor)
                else valid_idx
            )
            cv_pred.append((valid_indices, y_preds))
            cv_true.append(self.data["target"][valid_indices])

            # If a test set exists, evaluate on it
            if test_dataset is not None and len(test_idx) > 0:
                test_indices = (
                    test_idx.cpu().numpy()
                    if isinstance(test_idx, torch.Tensor)
                    else test_idx
                )
                test_preds, _, _ = self.trainer.predict(
                    model=model,
                    dataset=test_dataset,
                    loss_func=self.loss_func,
                    activation_fn=self.activation_fn,
                    dump_dir=self.save_path,
                    fold=fold,
                    target_scaler=self.data["target_scaler"],
                    epoch=0,
                    load_model=True,
                    feature_name=None,
                )

                # Store test predictions
                if "test_pred" not in self.cv:
                    self.cv["test_pred"] = np.zeros(
                        (len(self.data["target"]), test_preds.shape[1])
                    )
                    self.cv["test_indices"] = np.array([])

                self.cv["test_pred"][test_indices] = test_preds
                self.cv["test_indices"] = np.append(
                    self.cv["test_indices"], test_indices
                )

        # Process validation predictions
        pred_raw = []
        for indices, preds in cv_pred:
            for idx, pred in zip(indices, preds):
                pred_raw.append((idx, pred))

        pred_raw.sort(key=lambda x: x[0])
        indices, preds = zip(*pred_raw)

        preds = np.vstack(preds)
        indices = np.array(indices)

        # Reorder predictions to match original data order
        pred = np.zeros((len(self.data["target"]), preds.shape[1]))
        pred[indices] = preds

        self.cv["pred"] = pred
        return self.cv

    def evaluate(self, trainer, model_dir, test_data=None):
        """
        Evaluate the model on the test dataset.

        :param trainer: The Trainer instance
        :param model_dir: Directory containing saved model weights
        :param test_data: Optional test data to evaluate on
        :return: Test predictions
        """
        if test_data is not None:
            # Using provided test data
            test_dataset = self.get_test_dataset(test_data)
        elif "test_indices" in self.cv and len(self.cv["test_indices"]) > 0:
            # Using test indices from the split
            test_indices = self.cv["test_indices"].astype(int)
            test_dataset = self.get_dataset(test_indices)
        else:
            # If no specific test data, use all data
            test_dataset = self.get_dataset(np.arange(len(self.data["target"])))

        # Use the model from the first fold for evaluation
        model = self.init_new_model()

        # Load the model weights
        model_path = os.path.join(model_dir, f"model_0.pth")
        if os.path.exists(model_path):
            model_dict = torch.load(model_path, map_location=trainer.device)[
                "model_state_dict"
            ]
            model.load_state_dict(model_dict)
            logger.info(f"Loaded model from {model_path}")
        else:
            logger.warning(f"Model file {model_path} not found, using untrained model")

        # Predict on test dataset
        test_preds, _, _ = trainer.predict(
            model=model,
            dataset=test_dataset,
            loss_func=self.loss_func,
            activation_fn=self.activation_fn,
            dump_dir=model_dir,
            fold=0,
            target_scaler=self.data["target_scaler"],
            epoch=0,
            load_model=False,  # Already loaded above
            feature_name=None,
        )

        self.cv["test_pred"] = test_preds
        return test_preds

    def dump(self, data, dir, name):
        """
        Saves the specified data to a file.

        :param data: The data to be saved.
        :param dir: (str) The directory where the data will be saved.
        :param name: (str) The name of the file to save the data.
        """
        path = os.path.join(dir, name)
        if not os.path.exists(dir):
            os.makedirs(dir)
        joblib.dump(data, path)

    def count_parameters(self, model):
        """
        Counts the number of trainable parameters in the model.

        :param model: The model whose parameters are to be counted.

        :return: (int) The number of trainable parameters.
        """
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Add these methods to the NNModel class in models.py

    def get_dataset(self, indices):
        """
        Create a dataset from the given indices.

        :param indices: Indices of samples to include in the dataset
        :return: Dataset object for the specified indices
        """
        if isinstance(indices, torch.Tensor):
            indices = indices.cpu().numpy()

        # Get the relevant data for the indices
        if "unimol_input" in self.data:
            inputs = [self.data["unimol_input"][i] for i in indices]
            targets = self.data["target"][indices]

            from torch.utils.data import Dataset

            # Create dataset
            class MolDataset(Dataset):
                def __init__(self, inputs, targets):
                    self.inputs = inputs
                    self.targets = targets

                def __len__(self):
                    return len(self.inputs)

                def __getitem__(self, idx):
                    return self.inputs[idx], self.targets[idx]

            return MolDataset(inputs, targets)
        else:
            # Handle other dataset types if needed
            raise ValueError(
                "Unsupported data format. 'unimol_input' not found in data dictionary."
            )

    def get_test_dataset(self, test_data):
        """
        Create a dataset from external test data.

        :param test_data: External test data
        :return: Dataset object for the test data
        """
        if "unimol_input" in test_data:
            inputs = test_data["unimol_input"]
            if "target" in test_data:
                targets = test_data["target"]
            else:
                # If no targets available, use dummy values
                targets = np.zeros((len(inputs), self.output_dim))

            from torch.utils.data import Dataset

            # Create dataset
            class MolDataset(Dataset):
                def __init__(self, inputs, targets):
                    self.inputs = inputs
                    self.targets = targets

                def __len__(self):
                    return len(self.inputs)

                def __getitem__(self, idx):
                    return self.inputs[idx], self.targets[idx]

            return MolDataset(inputs, targets)
        else:
            # Handle other dataset types if needed
            raise ValueError(
                "Unsupported test data format. 'unimol_input' not found in test data dictionary."
            )

    def load_best_model(self, path):
        """
        Loads the best model weights from the specified path.
        """
        try:
            model_dict = torch.load(path, map_location=self.trainer.device)[
                "model_state_dict"
            ]
            self.model.load_state_dict(model_dict)
            logger.info(f"Loaded model from {path}")
        except Exception as e:
            logger.warning(f"Model file {path} not found, using untrained model: {e}")


def NNDataset(data, label=None):
    """
    Creates a dataset suitable for use with PyTorch models.

    :param data: The input data.
    :param label: Optional labels corresponding to the input data.

    :return: An instance of TorchDataset.
    """
    return TorchDataset(data, label)


class TorchDataset(Dataset):
    """
    A custom dataset class for PyTorch that handles data and labels. This class is compatible with PyTorch's Dataset interface
    and can be used with a DataLoader for efficient batch processing. It's designed to work with both numpy arrays and PyTorch tensors.
    """

    def __init__(self, data, label=None):
        """
        Initializes the dataset with data and labels.

        :param data: The input data.
        :param label: The target labels for the input data.
        """
        self.data = data
        self.label = label if label is not None else np.zeros((len(data), 1))

    def __getitem__(self, idx):
        """
        Retrieves the data item and its corresponding label at the specified index.

        :param idx: (int) The index of the data item to retrieve.

        :return: A tuple containing the data item and its label.
        """
        return self.data[idx], self.label[idx]

    def __len__(self):
        """
        Returns the total number of items in the dataset.

        :return: (int) The size of the dataset.
        """
        return len(self.data)
