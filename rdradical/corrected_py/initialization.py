import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem
from rdkit.Chem.rdchem import ChiralType, BondType, BondDir, BondStereo

from rdchiral.chiral import template_atom_could_have_been_tetra
from rdchiral.utils import PLEVEL
from rdchiral.bonds import enumerate_possible_cistrans_defs, bond_dirs_by_mapnum, \
    get_atoms_across_double_bonds

BondDirOpposite = {AllChem.BondDir.ENDUPRIGHT: AllChem.BondDir.ENDDOWNRIGHT,
                   AllChem.BondDir.ENDDOWNRIGHT: AllChem.BondDir.ENDUPRIGHT}
                   
RDKIT_SMILES_PARSER_PARAMS = Chem.SmilesParserParams()


def str_to_mol(string: str, explicit_hydrogens: bool = True) -> Chem.Mol:

    if string.startswith('InChI'):
        mol = Chem.MolFromInchi(string, removeHs=not explicit_hydrogens)
    else:
        # Set params here so we don't remove hydrogens with atom mapping
        RDKIT_SMILES_PARSER_PARAMS.removeHs = not explicit_hydrogens
        mol = Chem.MolFromSmiles(string, RDKIT_SMILES_PARSER_PARAMS)

    if explicit_hydrogens:
        return mol
        #return Chem.AddHs(mol)
    else:
        return Chem.RemoveHs(mol)

class rdradicalReaction(object):
    '''Class to store everything that should be pre-computed for a reaction. This
    makes library application much faster, since we can pre-do a lot of work
    instead of doing it for every mol-template pair

    Attributes:
        reaction_smarts (str): reaction SMARTS string
        rxn (rdkit.Chem.rdChemReactions.ChemicalReaction): RDKit reaction object.
            Generated from `reaction_smarts` using `initialize_rxn_from_smarts`
        template_r: Reaction reactant template fragments
        template_p: Reaction product template fragments
        atoms_rt_map (dict): Dictionary mapping from atom map number to RDKit Atom for reactants
        atoms_pt_map (dict): Dictionary mapping from atom map number to RDKit Atom for products
        atoms_rt_idx_to_map (dict): Dictionary mapping from atom idx to RDKit Atom for reactants
        atoms_pt_idx_to_map (dict): Dictionary mapping from atom idx to RDKit Atom for products

    Args:
        reaction_smarts (str): Reaction SMARTS string
    '''
    def __init__(self, reaction_smarts, rad_rt_dic, rad_pt_dic):
        # Keep smarts, useful for reporting
        self.reaction_smarts = reaction_smarts

        # Initialize - assigns stereochemistry and fills in missing rct map numbers
        self.rxn = initialize_rxn_from_smarts(reaction_smarts, rad_rt_dic, rad_pt_dic)
        
        # Generate dic: atom map number -> which reactant belongs to
        self.multi_rt = False
        if self.rxn.GetNumReactantTemplates() > 1:
            self.multi_rt = True
            self.atomMapToReactantMap={}
            for ri in range(self.rxn.GetNumReactantTemplates()):
                rt = self.rxn.GetReactantTemplate(ri)
                for atom in rt.GetAtoms():
                    self.atomMapToReactantMap[atom.GetAtomMapNum()] = ri

        # Combine template fragments so we can play around with mapnums
        self.template_r, self.template_p = get_template_frags_from_rxn(self.rxn)

        # Define molAtomMapNumber->atom dictionary for template rct and prd
        self.atoms_rt_map = {a.GetAtomMapNum(): a \
            for a in self.template_r.GetAtoms() if a.GetAtomMapNum()}
        self.atoms_pt_map = {a.GetAtomMapNum(): a \
            for a in self.template_p.GetAtoms() if a.GetAtomMapNum()}

        # Back-up the mapping for the reaction
        self.atoms_rt_idx_to_map = {a.GetIdx(): a.GetAtomMapNum()
            for a in self.template_r.GetAtoms()}
        self.atoms_pt_idx_to_map = {a.GetIdx(): a.GetAtomMapNum()
            for a in self.template_p.GetAtoms()}

        # Check consistency (this should not be necessary...)
        if any(self.atoms_rt_map[i].GetAtomicNum() != self.atoms_pt_map[i].GetAtomicNum() \
                for i in self.atoms_rt_map if i in self.atoms_pt_map):
            raise ValueError('Atomic identity should not change in a reaction!')

        # Call template_atom_could_have_been_tetra to pre-assign value to atom
        [template_atom_could_have_been_tetra(a) for a in self.template_r.GetAtoms()]
        [template_atom_could_have_been_tetra(a) for a in self.template_p.GetAtoms()]

        # Pre-list chiral double bonds (for copying back into outcomes/matching)
        self.rt_bond_dirs_by_mapnum = bond_dirs_by_mapnum(self.template_r)
        self.pt_bond_dirs_by_mapnum = bond_dirs_by_mapnum(self.template_p)

        # Enumerate possible cis/trans...
        self.required_rt_bond_defs, self.required_bond_defs_coreatoms = \
            enumerate_possible_cistrans_defs(self.template_r)

    def reset(self):
        '''Reset atom map numbers for template fragment atoms'''
        for (idx, mapnum) in self.atoms_rt_idx_to_map.items():
            self.template_r.GetAtomWithIdx(idx).SetAtomMapNum(mapnum)
        for (idx, mapnum) in self.atoms_pt_idx_to_map.items():
            self.template_p.GetAtomWithIdx(idx).SetAtomMapNum(mapnum)

class rdradicalReactants(object):
    '''Class to store everything that should be pre-computed for a reactant mol
    so that library application is faster

    Attributes:
        reactant_smiles (str): Reactant SMILES string
        reactants (rdkit.Chem.rdchem.Mol): RDKit Molecule create from `initialize_reactants_from_smiles`
        atoms_r (dict): Dictionary mapping from atom map number to atom in `reactants` Molecule
        idx_to_mapnum (callable): callable function that takes idx and returns atom map number
        reactants_achiral (rdkit.Chem.rdchem.Mol): achiral version of `reactants`
        bonds_by_mapnum (list): List of reactant bonds
            (int, int, rdkit.Chem.rdchem.Bond)
        bond_dirs_by_mapnum (dict): Dictionary mapping from atom map number tuples to BondDir
        atoms_across_double_bonds (list): List of cis/trans specifications from `get_atoms_across_double_bonds`

    Args:
        reactant_smiles (str): Reactant SMILES string
    '''
    def __init__(self, reactant_smiles):
        # Keep original smiles, useful for reporting
        self.reactant_smiles = reactant_smiles

        # Initialize into RDKit mol
        self.reactants, self.reactants_list = initialize_reactants_from_smiles(reactant_smiles)
        self.multi_reactants = False
        if self.reactants_list is not None:
            self.multi_reactants = True

        # Set mapnum->atom dictionary
        # all reactant atoms must be mapped after initialization, so this is safe
        self.atoms_r = {a.GetAtomMapNum(): a for a in self.reactants.GetAtoms()}
        self.idx_to_mapnum = lambda idx: self.reactants.GetAtomWithIdx(idx).GetAtomMapNum()

        # Create copy of molecule without chiral information, used with
        # RDKit's naive runReactants
        self.reactants_achiral, self.reactants_achiral_list = initialize_reactants_from_smiles(reactant_smiles)
        [a.SetChiralTag(ChiralType.CHI_UNSPECIFIED) for a in self.reactants_achiral.GetAtoms()]
        [(b.SetStereo(BondStereo.STEREONONE), b.SetBondDir(BondDir.NONE)) \
            for b in self.reactants_achiral.GetBonds()]
        if self.multi_reactants == True:
            [a.SetChiralTag(ChiralType.CHI_UNSPECIFIED) for react in self.reactants_achiral_list \
                    for a in react.GetAtoms()]
            [(b.SetStereo(BondStereo.STEREONONE), b.SetBondDir(BondDir.NONE)) \
                    for react in self.reactants_achiral_list for b in react.GetBonds()]

        # Pre-list reactant bonds (for stitching broken products)
        self.bonds_by_mapnum = [
            (b.GetBeginAtom().GetAtomMapNum(), b.GetEndAtom().GetAtomMapNum(), b) \
            for b in self.reactants.GetBonds()
        ]
        # Pre-list chiral double bonds (for copying back into outcomes/matching)
        self.bond_dirs_by_mapnum = {}
        for (i, j, b) in self.bonds_by_mapnum:
            if b.GetBondDir() != BondDir.NONE:
                print('**********this is just a flage in line 159**********')
                self.bond_dirs_by_mapnum[(i, j)] = b.GetBondDir()
                self.bond_dirs_by_mapnum[(j, i)] = BondDirOpposite[b.GetBondDir()]

        # Get atoms across double bonds defined by mapnum
        self.atoms_across_double_bonds = get_atoms_across_double_bonds(self.reactants)


def initialize_rxn_from_smarts(reaction_smarts, rad_rt_dic, rad_pt_dic):
    '''Initialize RDKit reaction object from SMARTS string

    Args:
        reaction_smarts (str): Reaction SMARTS string
        rad_rt_dic(dic) str: int

    Returns:
        rdkit.Chem.rdChemReactions.ChemicalReaction: RDKit reaction object
    '''
    # Initialize reaction
    rxn = AllChem.ReactionFromSmarts(reaction_smarts)
    # set radical elec
    if rad_rt_dic:
        for react in rxn.GetReactants():
            for a in react.GetAtoms():
                if str(a.GetAtomMapNum()) in rad_rt_dic.keys():
                    a.SetNumRadicalElectrons(int(rad_rt_dic[str(a.GetAtomMapNum())]))
    if rad_pt_dic:
        for prod in rxn.GetProducts():
            for a in prod.GetAtoms():
                if str(a.GetAtomMapNum()) in rad_pt_dic.keys():
                    a.SetNumRadicalElectrons(int(rad_pt_dic[str(a.GetAtomMapNum())]))
    rxn.Initialize()
    if rxn.Validate()[1] != 0:
        raise ValueError('validation failed')
    if PLEVEL >= 2: print('Validated rxn without errors')


    # Figure out if there are unnecessary atom map numbers (that are not balanced)
    # e.g., leaving groups for retrosynthetic templates. This is because additional
    # atom map numbers in the input SMARTS template may conflict with the atom map
    # numbers of the molecules themselves
    prd_maps = [a.GetAtomMapNum() for prd in rxn.GetProducts() for a in prd.GetAtoms() if a.GetAtomMapNum()]

    unmapped = 700
    for rct in rxn.GetReactants():
        rct.UpdatePropertyCache(strict=False)
        Chem.AssignStereochemistry(rct)
        # Fill in atom map numbers for unmapped atoms
        for a in rct.GetAtoms():
            if not a.GetAtomMapNum() or a.GetAtomMapNum() not in prd_maps:
                a.SetAtomMapNum(unmapped)
                unmapped += 1
    if PLEVEL >= 2: print('Added {} map nums to unmapped reactants'.format(unmapped-700))
    if unmapped > 800:
        raise ValueError('Why do you have so many unmapped atoms in the template reactants?')

    return rxn

def initialize_reactants_from_smiles(reactant_smiles):
    '''Initialize RDKit molecule from SMILES string

    Args:
        reactant_smiles (str): Reactant SMILES string

    Returns:
        rdkit.Chem.rdchem.Mol: RDKit molecule
    '''
    # Initialize reactants
    reactants = str_to_mol(reactant_smiles)
    #print(reactant_smiles)
    Chem.AssignStereochemistry(reactants, flagPossibleStereoCenters=True)
    reactants.UpdatePropertyCache(strict=False)
    # To have the product atoms match reactant atoms, we
    # need to populate the map number field, since this field
    # gets copied over during the reaction via reactant_atom_idx.
    # Check the initial map number firstly:
    HasMapNumber = True
    for atom in reactants.GetAtoms():
        if atom.GetAtomMapNum():
            continue
        else:
            HasMapNumber = False
            break
    if HasMapNumber == False:
        [a.SetAtomMapNum(i+1) for (i, a) in enumerate(reactants.GetAtoms())]
    if PLEVEL >= 2: print('Initialized reactants, assigned map numbers, stereochem, flagpossiblestereocenters')
    react_smiles = Chem.MolToSmiles(reactants)
    if '.' in react_smiles:
        react_smiles_list = reactant_smiles.split('.')
        reactants_list = [str_to_mol(reactant_smile) for reactant_smile in react_smiles_list]
        return reactants, reactants_list
    return reactants, None

def get_template_frags_from_rxn(rxn):
    '''Get template fragments from RDKit reaction object

    Args:
        rxn (rdkit.Chem.rdChemReactions.ChemicalReaction): RDKit reaction object

    Returns:
        (rdkit.Chem.rdchem.Mol, rdkit.Chem.rdchem.Mol): tuple of fragment molecules
    '''
    # Copy reaction template so we can play around with map numbers
    for i, rct in enumerate(rxn.GetReactants()):
        if i == 0:
            template_r = rct
        else:
            template_r = AllChem.CombineMols(template_r, rct)
    for i, prd in enumerate(rxn.GetProducts()):
        if i == 0:
            template_p = prd
        else:
            template_p = AllChem.CombineMols(template_p, prd)
    return template_r, template_p