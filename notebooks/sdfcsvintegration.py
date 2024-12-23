from rdkit import Chem
import pandas as pd


def process_sdf_and_csv(sdf_file_path, csv_file_path, output_csv_path):
    """
    Process SDF file and CSV file to add coordinates column

    Args:
        sdf_file_path (str): Path to the input SDF file
        csv_file_path (str): Path to the input CSV file with SMILES
        output_csv_path (str): Path where the output CSV will be saved
    """

    # Create dictionary to store molecule coordinates
    mol_coords = []
    smiles = []
    with Chem.SDMolSupplier(sdf_file_path) as suppl:
        for mol in suppl:
            if mol is not None:
                # Get the SMILES
                smiles.append(Chem.MolToSmiles(mol, isomericSmiles=True))
                mol_coords.append(mol.GetConformer().GetPositions())

    df = pd.read_csv(csv_file_path)
    df["coordinates"] = mol_coords
    df["smiles"] = smiles

    # Save the updated CSV
    df.to_csv(output_csv_path, index=False)

    return df


# Example usage
if __name__ == "__main__":
    sdf_file = "data/gdb9-3d-opt/gdb9.sdf"
    csv_file = "data/gdb9-3d-opt/gdb9.sdf.csv"
    output_file = "output_with_coordinates.csv"

    try:
        result_df = process_sdf_and_csv(sdf_file, csv_file, output_file)
        print(f"Successfully processed files. Output saved to {output_file}")
        print("\nFirst few rows of the processed data:")
        print(result_df.head())
    except Exception as e:
        print(f"An error occurred: {str(e)}")
