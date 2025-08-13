from abc import abstractmethod
from collections.abc import Hashable, Sized
from typing import Callable, Generic, Sequence, Type, TypeVar

import numpy as np

from chemprop.data.molgraph import MolGraph

S = TypeVar("S")
T = TypeVar("T")


class Featurizer(Generic[S, T]):
    """A :class:`Featurizer` featurizes inputs of type ``S`` into outputs of type ``T``."""

    @abstractmethod
    def __call__(self, input: S | None, *args, **kwargs) -> T:
        """Featurize an input."""


class VectorFeaturizer(Featurizer[S, np.ndarray], Sized):
    ...


class GraphFeaturizer(Featurizer[S, MolGraph]):
    @property
    @abstractmethod
    def shape(self) -> tuple[int, int]:
        ...


class OneHotFeaturizer(VectorFeaturizer[S]):
    """Create a one-hot encoding from an input object.

    This class uses a getter function to extract an attribute from an input object and converts
    the attribute into a one-hot encoded vector.

    Parameters
    ----------
    getter : Callable[[S], Hashable]
        A function that retrieves the attribute from an input object.
    choices : Sequence[Hashable]
        A sequence of unique possible values.
    padding : bool, default=False
        If True, adds an extra dimension to handle unknown values.

    Raises
    ------
    ValueError
        If the provided `choices` are not unique.

    Example
    -------
    >>> from rdkit import Chem
    >>> symbol_featurizer = OneHotFeaturizer[Chem.Atom](
    ...     getter=lambda atom: atom.GetSymbol(),
    ...     choices=["C", "N", "O"],
    ...     padding=True,
    ... )
    >>> mol = Chem.MolFromSmiles("C(O)N")
    >>> atom = mol.GetAtomWithIdx(0)
    >>> symbol_featurizer(atom)
    array([1., 0., 0., 0.])
    >>> symbol_featurizer.to_string(atom)
    '1000'

    >>> bond_type_featurizer = OneHotFeaturizer[Chem.Bond](
    ...     getter=lambda bond: bond.GetBondType(),
    ...     choices=[Chem.BondType.SINGLE, Chem.BondType.DOUBLE],
    ... )
    >>> bond = mol.GetBondWithIdx(0)
    >>> bond_type_featurizer(bond)
    array([1., 0.])
    >>> bond_type_featurizer.to_string(bond)
    '10'

    """

    def __init__(
        self, getter: Callable[[S], Hashable], choices: Sequence[Hashable], padding: bool = False
    ):
        self.getter = getter
        self.choices = choices
        self.padding = padding
        if len(self.choices) != len(set(self.choices)):
            raise ValueError("choices must be unique")
        self._size = len(self.choices) + int(self.padding)
        self._index_lookup = {choice: i for i, choice in enumerate(self.choices)}
        self._num_choices = len(self.choices)

    def __len__(self) -> int:
        """Return the length of the feature vector."""
        return self._size

    def __call__(self, input: S | None) -> np.ndarray:
        """Encode an input object as a one-hot vector.

        Parameters
        ----------
        input : S | None
            The input object.

        Returns
        -------
        np.ndarray
            One-hot encoded vector.

        """
        vector = np.zeros(self._size, dtype=float)
        if input is None:
            return vector
        self._set(input, vector)
        return vector

    def _set(self, input: S, vector: np.ndarray) -> None:
        """Set the hot bit of a one-hot encoded feature vector for the given input.

        Parameters
        ----------
        input : S
            The input object.
        vector : np.ndarray
            The one-hot encoded vector to set the hot bit of.
        """
        index = self._index_lookup.get(self.getter(input), self._num_choices)
        if self.padding or index != self._num_choices:
            vector[index] = 1

    def to_string(self, input: S | None) -> str:
        """Return a string representation of the feature encoding.

        Parameters
        ----------
        input : S | None
            The input entity. If None, returns a string of zeros.

        Returns
        -------
        str
            The string encoding of the feature vector.
        """
        return "".join(str(int(x)) for x in self(input))


class ValueFeaturizer(VectorFeaturizer[S]):
    """Extract a raw value from an input object.

    This class uses a getter function to extract an attribute from an input object and converts
    the attribute into a single-element vector.

    Parameters
    ----------
    getter : Callable[[S], bool | int | float]
        A function that extracts the attribute to be encoded from an input object.
    dtype : bool | int | float
        The data type of the output vector.

    Example
    -------
    >>> from rdkit import Chem
    >>> mass_featurizer = ValueFeaturizer[Chem.Atom](
    ...     getter=lambda atom: atom.GetMass(), dtype=float
    ... )
    >>> mol = Chem.MolFromSmiles("C(O)N")
    >>> atom = mol.GetAtomWithIdx(0)
    >>> mass_featurizer(atom)
    array([12.011])
    >>> mass_featurizer.to_string(atom)
    '12.011'

    """

    def __init__(self, getter: Callable[[S], bool | int | float], dtype: Type[bool | int | float]):
        self.getter = getter
        self.dtype = dtype

    def __len__(self) -> int:
        """Return the length of the feature vector."""
        return 1

    def __call__(self, input: S | None) -> np.ndarray:
        """Encode a raw value as a vector.

        Parameters
        ----------
        input : S | None
            The input object. If None, returns a vector with a zero value.

        Returns
        -------
        np.ndarray
            A vector with the raw extracted value.

        """
        if input is None:
            return np.zeros(1, dtype=float)
        return np.array([self.getter(input)], dtype=float)

    def _set(self, input: S, vector: np.ndarray) -> None:
        """Set the value of a feature vector for the given input.

        Parameters
        ----------
        input : S
            The input object.
        vector : np.ndarray
            The feature vector to set the value of.
        """
        vector[0] = self.getter(input)

    def to_string(self, input: S | None, decimals: int = 3) -> str:
        """Return a string representation of the feature encoding.

        Parameters
        ----------
        input : S | None
            The input entity. If None, returns '0'.
        decimals : int, default=3
            Number of decimals (only relevant if the feature is a float value).

        Returns
        -------
        str
            The string encoding of the feature vector.
        """
        x = 0 if input is None else self.getter(input)
        if issubclass(self.dtype, float):
            return f"{x:.{decimals}f}"
        return str(int(x))


class MultiHotFeaturizer(VectorFeaturizer[S]):
    """A vector featurizer that concatenates multiple subfeaturizers.

    Parameters
    ----------
    *subfeats : Subfeaturizer
        The subfeatures to concatenate.
    prepend_null_bit : bool, default=False
        If True, prepends a bit to the feature vector to indicate that the input is
        None.

    Example
    -------
    >>> from rdkit import Chem
    >>> symbol_featurizer = OneHotFeaturizer[Chem.Atom](
    ...     getter=lambda atom: atom.GetSymbol(),
    ...     choices=["C", "N", "O"],
    ...     padding=True,
    ... )
    >>> mass_featurizer = ValueFeaturizer[Chem.Atom](
    ...     getter=lambda atom: 0.01 * atom.GetMass(), dtype=float
    ... )
    >>> featurizer = MultiHotFeaturizer[Chem.Atom](
    ...     symbol_featurizer, mass_featurizer, prepend_null_bit=True
    ... )
    >>> mol = Chem.MolFromSmiles("C(O)N")
    >>> atom = mol.GetAtomWithIdx(0)
    >>> featurizer.to_string(atom)
    '0 1000 0.120'
    >>> featurizer.to_string(None)
    '1 0000 0.000'
    """

    def __init__(
        self, *subfeats: OneHotFeaturizer[S] | ValueFeaturizer[S], prepend_null_bit: bool = False
    ):
        self.subfeats = subfeats
        self.prepend_null_bit = prepend_null_bit
        self._subfeat_sizes = list(map(len, subfeats))
        self._size = sum(self._subfeat_sizes) + int(prepend_null_bit)

    def __len__(self):
        return self._size

    def __call__(self, input: S | None) -> np.ndarray:
        x = np.zeros(self._size, dtype=float)
        if input is None:
            if self.prepend_null_bit:
                x[0] = 1
            return x
        start = self.prepend_null_bit
        for subfeat, size in zip(self.subfeats, self._subfeat_sizes):
            end = start + size
            subfeat._set(input, x[start:end])
            start = end
        return x

    def to_string(self, input: S | None, decimals: int = 3) -> str:
        """Return a string representation of the concatenated subfeatures.

        Parameters
        ----------
        input : S | None
            The input object.
        decimals : int, default=3
            The number of decimal places to round float-valued features to.

        Returns
        -------
        str
            The string encoding of the concatenated subfeatures, with spaces separating each
            subfeature.

        """
        strings = [
            f.to_string(input, decimals) if isinstance(f, ValueFeaturizer) else f.to_string(input)
            for f in self.subfeats
        ]
        if self.prepend_null_bit:
            strings = ["1" if input is None else "0"] + strings
        return " ".join(strings)
