import pytest

from chemprop.utils import make_mol


@pytest.fixture(params=[True, False])
def use_duck_typing(request):
    return request.param


def test_no_keep_h(use_duck_typing):
    mol = make_mol("[H]C", keep_h=False, use_duck_typing=use_duck_typing)
    assert mol.GetNumAtoms() == 1


def test_keep_h(use_duck_typing):
    mol = make_mol("[H]C", keep_h=True, use_duck_typing=use_duck_typing)
    assert mol.GetNumAtoms() == 2


def test_add_h(use_duck_typing):
    mol = make_mol("[H]C", add_h=True, use_duck_typing=use_duck_typing)
    assert mol.GetNumAtoms() == 5


def test_no_reorder_atoms(use_duck_typing):
    mol = make_mol("[CH3:2][OH:1]", reorder_atoms=False, use_duck_typing=use_duck_typing)
    assert mol.GetAtomWithIdx(0).GetSymbol() == "C"


def test_reorder_atoms(use_duck_typing):
    mol = make_mol("[CH3:2][OH:1]", reorder_atoms=True, use_duck_typing=use_duck_typing)
    assert mol.GetAtomWithIdx(0).GetSymbol() == "O"


def test_reorder_atoms_no_atom_map(use_duck_typing):
    mol = make_mol("CCO", reorder_atoms=False, use_duck_typing=use_duck_typing)
    reordered_mol = make_mol("CCO", reorder_atoms=True, use_duck_typing=use_duck_typing)
    assert all(
        [
            mol.GetAtomWithIdx(i).GetSymbol() == reordered_mol.GetAtomWithIdx(i).GetSymbol()
            for i in range(mol.GetNumAtoms())
        ]
    )


def test_make_mol_invalid_smiles(use_duck_typing):
    with pytest.raises(RuntimeError):
        make_mol("chemprop", use_duck_typing=use_duck_typing)
