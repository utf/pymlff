"""Core Configuration and MLAB objects."""

import warnings
from copy import deepcopy
from dataclasses import dataclass

import numpy as np


@dataclass
class Configuration:
    """
    A single ab initio calculation.

    Parameters
    ----------
    name : str
        The name of the system.
    atom_types_numbers: dict
        Mapping of atom symbol to number of atoms.
    ctifor : float
        The machine learning error tolerance when the sample was collected.
    lattice : array
        The lattice parameters as (3, 3) array.
    coords : array
        The cartesian coordinates as a (n, 3) array.
    forces : array
        The forces as a (n, 3) array.
    total_energy : float
        The total energy of the system.
    stress_diagonal : array
        The diagonal part of the stress tensor (xx, yy, zz).
    stress_off_diagonal : array
        The off diagonal part of the stress tensor (xz, yz, zx).
    """

    name: str
    atom_types_numbers: dict[str, int]
    ctifor: float
    lattice: np.ndarray
    coords: np.ndarray
    forces: np.ndarray
    total_energy: float
    stress_diagonal: np.ndarray
    stress_off_diagonal: np.ndarray

    @property
    def num_atom_types(self) -> int:
        """Overall number of atom types (elements) in the system."""
        return len(self.atom_types_numbers)

    @property
    def num_atoms(self) -> int:
        """Total number of atoms in the structure."""
        return sum(self.atom_types_numbers.values())

    @property
    def atom_types(self) -> list[str]:
        """Atom types (elements) in the system."""
        return list(self.atom_types_numbers.keys())


@dataclass
class MLAB:
    """
    Main class representing the ML_AB file.

    Parameters
    ----------
    configurations : list of Configurations
        A list of configuration objects.
    basis_set : dict
        The basis set for the trained force field. Is formatted as a dictionary mapping
        the elemental species to the basis set. The basis set is specified as a list of
        [configuration_index, atom_index], that gives which atoms are included in the
        training for the model. Note that the configuration and atom index are 1-indexed
        as required by VASP.
    atomic_mass : dict
        The atomic mass for each atom type.
    reference_energy : dict
        The reference energy for each atom type.
    version : str
        The version of the ML_AB file format.
    """

    configurations: list[Configuration]
    basis_set: dict[str, list[tuple[int, int]]]
    atomic_mass: dict[str, float]
    reference_energy: dict[str, float]
    version: str

    def __post_init__(self):
        """Ensure configurations, reference energies and masses are consistent."""
        if set(self.atom_types) != set(self.basis_set.keys()):
            raise ValueError("Basis set is missing some atom types in configurations")

        if set(self.atom_types) != set(self.atomic_mass.keys()):
            raise ValueError(
                "Atomic masses are missing some atom types in configurations"
            )

        if set(self.atom_types) != set(self.reference_energy.keys()):
            raise ValueError(
                "Atomic masses are missing some atom types in configurations"
            )

    @property
    def atom_types(self) -> list[str]:
        """Sorted list of all atom types across all configurations."""
        return list(
            sorted(set([el for c in self.configurations for el in c.atom_types]))
        )

    @property
    def num_configurations(self) -> int:
        """Total number of configurations."""
        return len(self.configurations)

    @property
    def max_num_atom_types(self) -> int:
        """Maximum number of atom types across all configurations."""
        return max([c.num_atom_types for c in self.configurations])

    @property
    def max_num_atoms(self) -> int:
        """Maximum number of total atoms across all configurations."""
        return max([c.num_atoms for c in self.configurations])

    @property
    def max_num_atoms_per_type(self) -> int:
        """Maximum number of atoms (for one type) across all configurations."""
        return max([max(c.atom_types_numbers.values()) for c in self.configurations])

    @property
    def num_basis_set_per_type(self) -> list[int]:
        """Overall number of samples in the basis set for each atom type."""
        return [len(self.basis_set[x]) for x in self.atom_types]

    @property
    def reference_energy_per_type(self) -> list[float]:
        """List of reference energies for each atom type."""
        return [self.reference_energy[x] for x in self.atom_types]

    @property
    def atomic_mass_per_type(self) -> list[float]:
        """Atomic mass for each atom type."""
        return [self.atomic_mass[x] for x in self.atom_types]

    @classmethod
    def from_file(cls, filename):
        """
        Create an MLAB object from an ML_AB file.

        Parameters
        ----------
        filename
            Path to an ML_AB file

        Returns
        -------
        MLAB
            An MLAB object.
        """
        from pymlff.io import read_ml_ab_file

        return read_ml_ab_file(filename)

    def filter_configurations(self, filter_func):
        """
        Filter the configurations and return a new MLAB object.

        Parameters
        ----------
        filter_func : func
            A function to filter the configurations. The function should accept two
            arguments, the index of the configuration in the configurations list and a
            configuration itself. If the filter function returns True, the configuration
            will be kept, if False, the configuration will be removed. For example,
            to filter configurations with less than 256 atoms:

            .. code-block:: yaml

                mlab.filter_configurations(lambda i, c: c.num_atoms > 500)

            The basis set will be automatically regenerated to account for the filtered
            configurations.

        Returns
        -------
        MLAB
            A new MLAB object.
        """
        new_configurations = []
        config_mapping = {}  # mapping from old_idx: new_idx  (VASP uses 1-indexing)
        for i, configuration in enumerate(self.configurations):
            if filter_func(i, configuration):
                new_configurations.append(configuration)
                config_mapping[i + 1] = len(new_configurations)

        new_atom_types = list(
            sorted(set([el for c in new_configurations for el in c.atom_types]))
        )

        new_basis_set = {}
        for atom_type in new_atom_types:
            new_basis_set[atom_type] = [
                (config_mapping[c_idx], a_idx)
                for (c_idx, a_idx) in self.basis_set[atom_type]
                if c_idx in config_mapping
            ]
        new_atomic_mass = {k: self.atomic_mass[k] for k in new_atom_types}
        new_reference_energy = {k: self.reference_energy[k] for k in new_atom_types}

        return MLAB(
            configurations=new_configurations,
            basis_set=new_basis_set,
            atomic_mass=new_atomic_mass,
            reference_energy=new_reference_energy,
            version=self.version,
        )

    def __add__(self, ml_ab2):
        """Combine two MLAB objects."""
        if not isinstance(ml_ab2, MLAB):
            raise ValueError("Can only add MLAB objects to other MLAB objects.")

        for el, weight in ml_ab2.atomic_mass.items():
            if (
                el in self.atomic_mass
                and abs(self.atomic_mass[el] - ml_ab2.atomic_mass[el]) > 1e-4
            ):
                raise ValueError("Atomic masses do not match between MLAB objects.")

        for el, weight in ml_ab2.reference_energy.items():
            if (
                el in self.reference_energy
                and abs(self.reference_energy[el] - ml_ab2.reference_energy[el]) > 1e-4
            ):
                raise ValueError(
                    "Reference energies do not match between MLAB objects."
                )

        if self.version != ml_ab2.version:
            warnings.warn(
                "Versions do not match, this may have unintended consequences."
            )

        new_configurations = deepcopy(self.configurations)
        new_atomic_mass = deepcopy(self.atomic_mass)
        new_reference_energy = deepcopy(self.reference_energy)
        new_basis_set = deepcopy(self.basis_set)

        new_configurations.extend(ml_ab2.configurations)
        new_atomic_mass.update(ml_ab2.atomic_mass)
        new_reference_energy.update(ml_ab2.reference_energy)

        # reconstruct new basis set (i.e., update configuration idx of new samples)
        last_idx = len(self.configurations)
        for (config_idx, atom_idx) in ml_ab2.basis_set:
            new_basis_set.append((config_idx + last_idx, atom_idx))

        return MLAB(
            configurations=new_configurations,
            basis_set=new_basis_set,
            atomic_mass=new_atomic_mass,
            reference_energy=new_reference_energy,
            version=self.version,
        )

    def to_string(self) -> str:
        """Get a string representation of the MLAB object."""
        from pymlff.io import ml_ab_to_string

        return ml_ab_to_string(self)

    def write_file(self, filename):
        """
        Write MLAB object to a file. The written file can be used directly by VASP.

        Parameters
        ----------
        filename
            A filename.
        """
        with open(filename, "w") as f:
            f.write(self.to_string())
