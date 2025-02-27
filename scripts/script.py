import os
import urllib.request
import argparse
import pandas as pd
from rdkit import Chem
import gdown
import yaml
from pathlib import Path
import sys
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from unimol_tools.unimol_tools import MolTrain, MolPredict
from unimol_tools.unimol_tools.data.split import scaffold_split


def load_dataset_config(dataset_name):
    """Load dataset configuration from YAML file."""
    config_path = Path(f"configs/{dataset_name}.yaml")
    if not config_path.exists():
        raise ValueError(
            f"Configuration file for dataset {dataset_name} not found at {config_path}"
        )

    with open(config_path) as f:
        return yaml.safe_load(f)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train a molecular property prediction model"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Name of the dataset (corresponding to a YAML config file)",
    )
    parser.add_argument(
        "--data_dir", type=str, default="data", help="Directory to store extracted data"
    )
    parser.add_argument(
        "--epochs", type=int, help="Number of training epochs (overrides config)"
    )
    parser.add_argument(
        "--batch_size", type=int, help="Batch size for training (overrides config)"
    )
    parser.add_argument(
        "--model_size",
        type=str,
        choices=["84m", "450m"],
        help="Model size to use (overrides config)",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["random", "group", "scaffold"],
        help="Data split method (overrides config)",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        help="List of target properties to predict (overrides config)",
    )
    return parser.parse_args()


def validate_targets(requested_targets, available_targets):
    """Validate that requested targets are available in the dataset."""
    invalid_targets = [t for t in requested_targets if t not in available_targets]
    if invalid_targets:
        raise ValueError(
            f"The following requested targets are not available: {invalid_targets}\n"
            f"Available targets are: {available_targets}"
        )


def main():
    args = parse_arguments()

    # Load dataset configuration
    config = load_dataset_config(args.dataset)

    # Merge command line arguments with config defaults
    epochs = args.epochs or config["recommended_parameters"]["epochs"]
    batch_size = args.batch_size or config["recommended_parameters"]["batch_size"]
    model_size = args.model_size or config["recommended_parameters"]["model_size"]
    split = args.split or config["recommended_parameters"]["split"]
    targets = args.targets or config["default_targets"]

    # Validate targets
    validate_targets(targets, config["available_targets"])

    # Create paths using dataset name
    structure_file = f"{config['name']}.{config['file_format']['structure']}"
    properties_file = f"{config['name']}.{config['file_format']['structure']}.{config['file_format']['properties']}"
    structure_path = os.path.join(args.data_dir, structure_file)
    properties_path = os.path.join(args.data_dir, properties_file)
    compressed_file = f"{config['name']}.tar.gz"

    # Download and extract data if needed
    if (
        not os.path.exists(args.data_dir)
        or not os.path.exists(structure_path)
        or not os.path.exists(properties_path)
    ):

        if not os.path.exists(args.data_dir):
            os.makedirs(args.data_dir)

        gdown.download(config["url"], compressed_file, quiet=False)
        os.system(f"tar -xzvf {compressed_file} -C {args.data_dir}")

        # Delete the compressed file after extraction
        if os.path.exists(compressed_file):
            os.remove(compressed_file)

    # Process molecular data
    mol_coords = []
    smiles = []
    atoms_list = []

    with Chem.SDMolSupplier(structure_path) as suppl:
        for mol in suppl:
            if mol is not None:
                atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
                atoms_list.append(atoms)
                smiles.append(Chem.MolToSmiles(mol, isomericSmiles=True))
                mol_coords.append(mol.GetConformer().GetPositions())

    df = pd.read_csv(properties_path)

    # train_val_idx = torch.cat([train_idx, valid_idx])
    # train_idx, valid_idx, test_idx = scaffold_split(smiles)

    # train_data_dict = {
    #     "smiles": [smiles[i] for i in train_val_idx],
    #     "atoms": [atoms_list[i] for i in train_val_idx],
    #     "coordinates": [mol_coords[i] for i in train_val_idx],
    # }

    data_dict = {
        "smiles": smiles,
        "atoms": atoms_list,
        "coordinates": mol_coords,
    }

    for target in targets:
        # train_data_dict[target] = df.iloc[train_val_idx][target].values
        data_dict[target] = df[target]

    # Initialize and train model
    clf = MolTrain(
        task="regression",
        data_type="molecule",
        epochs=epochs,
        batch_size=batch_size,
        metrics="mae",
        model_name="unimolv2",
        model_size=model_size,
        split=split,
        smiles_col="smiles",
        target_cols=targets,
        kfold=1,
    )

    res = clf.fit(data=data_dict)

    predictor = MolPredict(load_model="exp")
    preds = predictor.predict()

    return res, preds

    # clf = MolPredict(load_model='../exp')
    # res = clf.predict(data=data_dict)


if __name__ == "__main__":
    main()
