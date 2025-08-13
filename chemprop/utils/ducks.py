import cppyy
import pathlib
import os
import sys


def setup_rdkit_cpp():
    """
    Setup RDKit C++ for packages with rdkit-dev dependency.
    Assumes RDKit headers and libraries are available in the current environment.
    """
    prefix = pathlib.Path(sys.prefix)
    if os.name == "nt":
        prefix = prefix / "Library"

    include_path = prefix / "include" / "rdkit"
    if not include_path.exists():
        raise RuntimeError(f"Include path not found: {include_path}")
    cppyy.add_include_path(str(include_path))

    lib_path = prefix / "lib"
    if not lib_path.exists():
        raise RuntimeError(f"Library path not found: {lib_path}")
    cppyy.add_library_path(str(lib_path))

    cppyy.load_library("libRDKitGraphMol")
    cppyy.load_library("libRDKitSmilesParse")
    cppyy.include("GraphMol/RDKitBase.h")
    cppyy.include("GraphMol/SmilesParse/SmilesParse.h")


setup_rdkit_cpp()


cppyy.cppdef(
    """
    #include <string>
    #include <memory>
    #include <exception>
    #include <vector>
    #include <algorithm>
    #include <numeric>

    struct AtomData {
        std::string symbol;
        int atomic_num;
        int total_degree;
        int formal_charge;
        int total_num_hs;
        int chiral_tag;
        int hybridization;
        bool is_aromatic;
        double mass;
    };

    struct BondData {
        int begin_atom_idx;
        int end_atom_idx;
        int bond_type;
        bool is_aromatic;
        bool is_conjugated;  
        bool in_ring;
        int stereo;
    };

    struct MoleculeData {
        std::vector<AtomData> atom_data;
        std::vector<BondData> bond_data;
        std::string error_message;
        bool success;
    };

    MoleculeData extract_molecule_data(
        const std::string& smiles,
        bool keep_h = false,
        bool add_h = false,
        bool ignore_stereo = false,
        bool reorder_atoms = false
    ) {
        MoleculeData result;
        result.success = false;
        result.error_message = "";

        try {
            RDKit::SmilesParserParams parser_params;
            parser_params.removeHs = !keep_h;
            
            std::unique_ptr<RDKit::RWMol> mol(RDKit::SmilesToMol(smiles, parser_params));
            if (!mol) {
                result.error_message = "Failed to parse SMILES";
                return result;
            }

            int num_atoms = mol->getNumAtoms();
            int num_bonds = mol->getNumBonds();

            // Add hydrogens if requested
            if (add_h) {
                RDKit::MolOps::addHs(*mol);
            }

            // Remove stereochemistry if requested
            if (ignore_stereo) {
                for (auto atom : mol->atoms()) {
                    atom->setChiralTag(RDKit::Atom::CHI_UNSPECIFIED);
                }
                for (auto bond : mol->bonds()) {
                    bond->setStereo(RDKit::Bond::STEREONONE);
                }
            }

            if (reorder_atoms) {
                std::vector<unsigned int> indices(num_atoms);
                std::iota(indices.begin(), indices.end(), 0);
                std::sort(
                    indices.begin(),
                    indices.end(),
                    [&mol](unsigned int a, unsigned int b) {
                        int amap_a = mol->getAtomWithIdx(a)->getAtomMapNum();
                        int amap_b = mol->getAtomWithIdx(b)->getAtomMapNum();
                        return amap_a < amap_b;
                    }
                );
                
                RDKit::ROMol* reordered_mol = RDKit::MolOps::renumberAtoms(*mol, indices);
                mol.reset(static_cast<RDKit::RWMol*>(reordered_mol));
            }

            RDKit::RingInfo* ring_info = mol->getRingInfo();
            if (!ring_info->isSssrOrBetter()) {
                RDKit::MolOps::findSSSR(*mol);
            }

            result.atom_data.reserve(num_atoms);
            result.bond_data.reserve(num_bonds);

            for (const auto& atom : mol->atoms()) {
                AtomData &atom_data = result.atom_data.emplace_back();
                atom_data.symbol = atom->getSymbol();
                atom_data.atomic_num = atom->getAtomicNum();
                atom_data.total_degree = atom->getTotalDegree();
                atom_data.formal_charge = atom->getFormalCharge();
                atom_data.total_num_hs = atom->getTotalNumHs();
                atom_data.chiral_tag = static_cast<int>(atom->getChiralTag());
                atom_data.hybridization = static_cast<int>(atom->getHybridization());
                atom_data.is_aromatic = atom->getIsAromatic();
                atom_data.mass = atom->getMass();
            }

            for (const auto& bond : mol->bonds()) {
                BondData &bond_data = result.bond_data.emplace_back();
                bond_data.begin_atom_idx = bond->getBeginAtomIdx();
                bond_data.end_atom_idx = bond->getEndAtomIdx();
                bond_data.bond_type = static_cast<int>(bond->getBondType());
                bond_data.is_aromatic = bond->getIsAromatic();
                bond_data.is_conjugated = bond->getIsConjugated();
                bond_data.in_ring = ring_info->numBondRings(bond->getIdx()) != 0;
                bond_data.stereo = static_cast<int>(bond->getStereo());
            }
            
            result.success = true;
        
        } catch (const std::exception& e) {
            result.error_message = e.what();
        } catch (...) {
            result.error_message = "Unknown error";
        }
        
        return result;
    }
    """
)


class DuckAtom:
    """
    A lightweight, duck-typed representation of an RDKit Atom, backed by AtomData from C++.

    Provides a minimal interface for atom features, mimicking RDKit's Atom API for use in
    featurization and graph construction without requiring a full RDKit molecule object.
    """
    def __init__(
        self,
        index: int,
        atom_data: cppyy.gbl.AtomData,
    ):
        self.index = index
        self.atom_data = atom_data

    def __hash__(self):
        return self.atom_data.hash

    def GetIdx(self) -> int:
        return self.index

    def GetSymbol(self) -> str:
        return self.atom_data.symbol

    def GetAtomicNum(self) -> int:
        return self.atom_data.atomic_num

    def GetTotalDegree(self) -> int:
        return self.atom_data.total_degree

    def GetFormalCharge(self) -> int:
        return self.atom_data.formal_charge

    def GetChiralTag(self) -> int:
        return self.atom_data.chiral_tag

    def GetTotalNumHs(self) -> int:
        return self.atom_data.total_num_hs

    def GetHybridization(self) -> int:
        return self.atom_data.hybridization

    def GetIsAromatic(self) -> bool:
        return self.atom_data.is_aromatic

    def GetMass(self) -> float:
        return self.atom_data.mass


class DuckBond:
    """
    A lightweight, duck-typed representation of an RDKit Bond, backed by BondData from C++.

    Provides a minimal interface for bond features, mimicking RDKit's Bond API for use in
    featurization and graph construction without requiring a full RDKit molecule object.
    """
    def __init__(
        self,
        index: int,
        atoms: list[DuckAtom],
        bond_data: cppyy.gbl.BondData
    ):
        self.index = index
        self.atoms = atoms
        self.bond_data = bond_data

    def __hash__(self):
        return self.bond_data.hash

    def GetIdx(self) -> int:
        return self.index

    def GetBeginAtomIdx(self) -> int:
        return self.bond_data.begin_atom_idx

    def GetEndAtomIdx(self) -> int:
        return self.bond_data.end_atom_idx

    def GetBeginAtom(self) -> DuckAtom:
        return self.atoms[self.bond_data.begin_atom_idx]

    def GetEndAtom(self) -> DuckAtom:
        return self.atoms[self.bond_data.end_atom_idx]

    def GetBondType(self) -> int:
        return self.bond_data.bond_type

    def GetIsConjugated(self) -> bool:
        return self.bond_data.is_conjugated

    def IsInRing(self) -> bool:
        return self.bond_data.in_ring

    def GetStereo(self) -> int:
        return self.bond_data.stereo


class DuckMol:
    """
    A lightweight, duck-typed representation of an RDKit molecule, backed by MoleculeData from C++.

    Provides a minimal interface for molecule features, mimicking RDKit's Molecule API for use in
    featurization and graph construction without requiring a full RDKit molecule object.
    """
    def __init__(self, mol_data: cppyy.gbl.MoleculeData):
        self.mol_data = mol_data
        self.atoms = [
            DuckAtom(index, data)
            for index, data in enumerate(self.mol_data.atom_data)
        ]

        self.bonds = [
            DuckBond(index, self.atoms, data)
            for index, data in enumerate(self.mol_data.bond_data)
        ]

    def GetAtomWithIdx(self, idx: int) -> DuckAtom:
        return self.atoms[idx]

    def GetBondWithIdx(self, idx: int) -> DuckBond:
        return self.bonds[idx]

    def GetAtoms(self) -> list[DuckAtom]:
        return self.atoms

    def GetBonds(self) -> list[DuckBond]:
        return self.bonds

    def GetNumAtoms(self) -> int:
        return len(self.atoms)

    def GetNumBonds(self) -> int:
        return len(self.bonds)


def make_duck_mol(
    smi: str,
    keep_h: bool = False,
    add_h: bool = False,
    ignore_stereo: bool = False,
    reorder_atoms: bool = False,
) -> DuckMol | None:
    mol_data = cppyy.gbl.extract_molecule_data(smi, keep_h, add_h, ignore_stereo, reorder_atoms)
    if not mol_data.success:
        raise RuntimeError(mol_data.error_message)
    return DuckMol(mol_data)
