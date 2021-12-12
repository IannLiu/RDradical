from __future__ import print_function

PLEVEL = 0
def vprint(level, txt, *args):
    if PLEVEL >= level:
        print(txt.format(*args))

def parity4(data):
    '''
    Thanks to http://www.dalkescientific.com/writings/diary/archive/2016/08/15/fragment_parity_calculation.html
    '''
    if data[0] < data[1]:
        if data[2] < data[3]:
            if data[0] < data[2]:
                if data[1] < data[2]:
                    return 0 # (0, 1, 2, 3) 
                else:
                    if data[1] < data[3]:
                        return 1 # (0, 2, 1, 3) 
                    else:
                        return 0 # (0, 3, 1, 2) 
            else:
                if data[0] < data[3]:
                    if data[1] < data[3]:
                        return 0 # (1, 2, 0, 3) 
                    else:
                        return 1 # (1, 3, 0, 2) 
                else:
                    return 0 # (2, 3, 0, 1) 
        else:
            if data[0] < data[3]:
                if data[1] < data[2]:
                    if data[1] < data[3]:
                        return 1 # (0, 1, 3, 2) 
                    else:
                        return 0 # (0, 2, 3, 1) 
                else:
                    return 1 # (0, 3, 2, 1) 
            else:
                if data[0] < data[2]:
                    if data[1] < data[2]:
                        return 1 # (1, 2, 3, 0) 
                    else:
                        return 0 # (1, 3, 2, 0) 
                else:
                    return 1 # (2, 3, 1, 0) 
    else:
        if data[2] < data[3]:
            if data[0] < data[3]:
                if data[0] < data[2]:
                    return 1 # (1, 0, 2, 3) 
                else:
                    if data[1] < data[2]:
                        return 0 # (2, 0, 1, 3) 
                    else:
                        return 1 # (2, 1, 0, 3) 
            else:
                if data[1] < data[2]:
                    return 1 # (3, 0, 1, 2) 
                else:
                    if data[1] < data[3]:
                        return 0 # (3, 1, 0, 2) 
                    else:
                        return 1 # (3, 2, 0, 1) 
        else:
            if data[0] < data[2]:
                if data[0] < data[3]:
                    return 0 # (1, 0, 3, 2) 
                else:
                    if data[1] < data[3]:
                        return 1 # (2, 0, 3, 1) 
                    else:
                        return 0 # (2, 1, 3, 0) 
            else:
                if data[1] < data[2]:
                    if data[1] < data[3]:
                        return 0 # (3, 0, 2, 1) 
                    else:
                        return 1 # (3, 1, 2, 0) 
                else:
                    return 0 # (3, 2, 1, 0) 

def bond_to_label(bond):
    '''This function takes an RDKit bond and creates a label describing
    the most important attributes
    
    Args:
        bond (rdkit.Chem.rdchem.Bond): RDKit bond object

    Returns:
        str: String representing most important attributes of bond
    '''
    
    a1_label = str(bond.GetBeginAtom().GetAtomicNum())
    a2_label = str(bond.GetEndAtom().GetAtomicNum())
    if bond.GetBeginAtom().GetAtomMapNum():
        a1_label += str(bond.GetBeginAtom().GetAtomMapNum())
    if bond.GetEndAtom().GetAtomMapNum():
        a2_label += str(bond.GetEndAtom().GetAtomMapNum())
    atoms = sorted([a1_label, a2_label])

    return '{}{}{}'.format(atoms[0], bond.GetSmarts(), atoms[1])

def nb_H_Num(atom):
    nb_Hs = 0
    for nb in atom.GetNeighbors():
        if nb.GetSymbol() == 'H':
            nb_Hs = nb_Hs + 1
    return(nb_Hs)
    
def atoms_are_different(atom1, atom2):
    '''Compares two RDKit atoms based on basic properties
    
    Args:
        atom1 (rdkit.Chem.rdchem.Atom): First atom to compare
        atom2 (rdkit.Chem.rdchem.Atom): Second atom to compare

    Returns:
        bool: Whether the two atoms are different
    '''

    if atom1.GetAtomicNum() != atom2.GetAtomicNum(): return True # must be true for atom mapping
    # Because of explicit Hs, we must count Hs manually
    atom1_Hs = nb_H_Num(atom1)
    atom2_Hs = nb_H_Num(atom2)
    if atom1_Hs != atom2_Hs: return True
    if atom1.GetFormalCharge() != atom2.GetFormalCharge(): return True
    # if Hs number is same, the atom degrees are independent on whether or not Hs are explicit in the graph
    if atom1.GetDegree() != atom2.GetDegree(): return True
    #if atom1.IsInRing() != atom2.IsInRing(): return True # do not want to check this!
    # e.g., in macrocycle formation, don't want the template to include the entire ring structure
    if atom1.GetNumRadicalElectrons() != atom2.GetNumRadicalElectrons(): return True
    if atom1.GetIsAromatic() != atom2.GetIsAromatic(): return True 

    # Check bonds and nearest neighbor identity
    bonds1 = sorted([bond_to_label(bond) for bond in atom1.GetBonds()]) 
    bonds2 = sorted([bond_to_label(bond) for bond in atom2.GetBonds()]) 
    if bonds1 != bonds2: return True

    return False