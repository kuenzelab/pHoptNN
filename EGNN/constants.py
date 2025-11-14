# labels from PDB files
ALL_LABELS_BACKBONE = ['C', 'CA', 'N', 'O', 'OXT']

ALL_ATOM_LABELS = ['C', 'CA', 'CB', 'CD', 'CD1', 'CD2', 'CE', 'CE1', 'CE2', 'CE3',
 'CG', 'CG1', 'CG2', 'CH2', 'CZ', 'CZ2', 'CZ3', 'N', 'ND1', 'ND2', 'NE', 'NE1',
 'NE2', 'NH1', 'NH2', 'NZ', 'O', 'OD1', 'OD2', 'OE1', 'OE2', 'OG', 'OG1', 'OH',
 'OXT', 'SD', 'SG']

DICT_AA_SELECTION = {
    'ALA': [],
    'ARG': ['CA', 'NH1', 'NH2', 'NE'],  # guanidin group
    'ASN': ['CA', 'CG', 'OD1', 'ND2'],  # carbamide group
    'ASP': ['CA', 'CG', 'OD1', 'OD2'],  # carboxy group
    'CYS': ['CA', 'CB', 'SG'],  #thiol group
    'GLN': ['CA', 'CD', 'OE1', 'NE2'],  # carbamide group
    'GLU': ['CA', 'CD', 'OE1', 'OE2'],  # carboxy group
    'GLY': [],
    'HIS': ['CA', 'ND1', 'CE1', 'NE2'],  # imidazol
    'ILE': [],
    'LEU': [],
    'LYS': ['CA', 'CD', 'CE', 'NZ'],  # terminal amine
    'MET': [],
    'PHE': ['CA', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'],  # phenyl group
    'PRO': [],
    'SER': ['CA', 'CB', 'OG'],  # hydroxy group
    'THR': ['CA', 'CB', 'CG2', 'OG1'],  # hydroxy group
    'TRP': ['CA', 'CG', 'CD1', 'CD2', 'CE2', 'CE3', 'NE1', 'CH2', 'CZ2', 'CZ3'],  # indol ring
    'TYR': ['CA', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'OH', 'CZ'], # phenyl group
    'VAL': [],
    'UNK': []
}

DICT_AA_RESIDUE_EDGES = {
    'ALA': [],
    'ARG': [('CB', 'CG'), ('CG', 'CD'), ('CD', 'NE'), ('NE', 'CZ'), ('CZ', 'NH1'), ('CZ', 'NH2')],
    'ASN': [('CB', 'CG'), ('CG', 'OD1'), ('CG', 'ND2')],
    'ASP': [('CB', 'CG'), ('CG', 'OD1'), ('CG', 'OD2')],
    'CYS': [('CB', 'SG')],
    'GLN': [('CB', 'CG'), ('CG', 'CD'), ('CD', 'OE1'), ('CD', 'NE2')],
    'GLU': [('CB', 'CG'), ('CG', 'CD'), ('CD', 'OE1'), ('CD', 'OE2')],
    'GLY': [],
    'HIS': [('CB', 'CG'), ('CG', 'ND1'), ('CG', 'CD2'), ('ND1', 'CE1'), ('CD2', 'NE2'), ('CE1', 'NE2')],
    'ILE': [('CB', 'CG1'), ('CB', 'CG2'), ('CG1', 'CD1')],
    'LEU': [('CB', 'CG'), ('CG', 'CD1'), ('CG', 'CD2')],
    'LYS': [('CB', 'CG'), ('CG', 'CD'), ('CD', 'CE'), ('CE', 'NZ')],
    'MET': [('CB', 'CG'), ('CG', 'SD'), ('SD', 'CE')],
    'PHE': [('CB', 'CG'), ('CG', 'CD1'), ('CG', 'CD2'), ('CD1', 'CE1'), ('CD2', 'CE2'), ('CE1', 'CZ'), ('CE2', 'CZ')],
    'PRO': [('CB', 'CG'), ('CG', 'CD'), ('CD', 'N')],
    'SER': [('CB', 'OG')],
    'THR': [('CB', 'OG1'), ('CB', 'CG2')],
    'TRP': [('CB', 'CG'), ('CG', 'CD1'), ('CG', 'CD2'), ('CD1', 'NE1'), ('NE1', 'CE2'), ('CE2', 'CZ2'), ('CD2', 'CE2'), ('CD2', 'CE3'), ('CE3', 'CZ3'), ('CZ3', 'CH2'), ('CH2', 'CZ2')],
    'TYR': [('CB', 'CG'), ('CG', 'CD1'), ('CG', 'CD2'), ('CD1', 'CE1'), ('CD2', 'CE2'), ('CE1', 'CZ'), ('CE2', 'CZ'), ('CZ', 'OH')],
    'VAL': [('CB', 'CG1'), ('CB', 'CG2')],
    'UNK': []
}

SELECTED_ATOMS = [atom for atoms in DICT_AA_SELECTION.values() for atom in atoms] + ['CB']
SELECTED_ATOMS = sorted(list(set(SELECTED_ATOMS)))
