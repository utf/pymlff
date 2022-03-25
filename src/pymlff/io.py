"""Input and output functions."""

from itertools import chain

import numpy as np

_LATTICE_FMT_STR = (
    "{:.16f}  {:.16f}  {:.16f}\n{:.16f}  {:.16f}  {:.16f}\n{:.16f}  {:.16f}  {:.16f}"
)
_BASIS_FMT_STR = """**************************************************
     Basis set for {}
--------------------------------------------------
{}"""
_CONFIGURATION_FMT_STR = """**************************************************
     Configuration num.    {}
==================================================
     System name
--------------------------------------------------
     {}
==================================================
     The number of atom types
--------------------------------------------------
       {}
==================================================
     The number of atoms
--------------------------------------------------
         {}
**************************************************
     Atom types and atom numbers
--------------------------------------------------
{}
==================================================
     CTIFOR
--------------------------------------------------
   {:.14f}
==================================================
     Primitive lattice vectors (ang.)
--------------------------------------------------
{}
==================================================
     Atomic positions (ang.)
--------------------------------------------------
{}
==================================================
     Total energy (eV)
--------------------------------------------------
  {:.14f}
==================================================
     Forces (eV ang.^-1)
--------------------------------------------------
{}
==================================================
     Stress (kbar)
--------------------------------------------------
     XX YY ZZ
--------------------------------------------------
 {}
--------------------------------------------------
     XY YZ ZX
--------------------------------------------------
 {}"""
_ML_AB_FMT_STR = """ {}
**************************************************
     The number of configurations
--------------------------------------------------
        {}
**************************************************
     The maximum number of atom type
--------------------------------------------------
       {}
**************************************************
     The atom types in the data file
--------------------------------------------------
     {}
**************************************************
     The maximum number of atoms per system
--------------------------------------------------
            {}
**************************************************
     The maximum number of atoms per atom type
--------------------------------------------------
            {}
**************************************************
     Reference atomic energy (eV)
--------------------------------------------------
   {}
**************************************************
     Atomic mass
--------------------------------------------------
   {}
**************************************************
     The numbers of basis sets per atom type
--------------------------------------------------
       {}
{}
{}
"""


def read_ml_ab_file(filename):
    """
    Read an MLAB file.

    Parameters
    ----------
    filename
        Path to an MLAB file.

    Returns
    -------
    MLAB
        An MLAB object.
    """
    from pymlff.core import MLAB

    configurations = []
    configuration_data = None
    header_data = []
    header = None
    with open(filename) as file:
        for i, line in enumerate(file):
            if "Configuration num." in line:
                if configuration_data:
                    configurations.append(_parse_config(configuration_data))
                elif configuration_data is None:
                    header = _parse_header(header_data)
                configuration_data = []
            elif configuration_data is None:
                header_data.append(line.rstrip())
            else:
                configuration_data.append(line.rstrip())

        # end of file; parse final config
        configurations.append(_parse_config(configuration_data))

    return MLAB(
        configurations=configurations,
        basis_set=header["basis_set"],
        atomic_mass=header["atomic_mass"],
        reference_energy=header["reference_energy"],
        version=header["version"],
    )


def ml_ab_to_string(ml_ab):
    """
    Convert an MLAB object to a string representation.

    Parameters
    ----------
    ml_ab : MLAB
        An MLAB object.

    Returns
    -------
    str
        The string representation of an MLAB object.
    """
    ref_energy_fmt = "  ".join(map("{:.14f} ".format, ml_ab.reference_energy_per_type))
    atomic_mass_fmt = "  ".join(map("{:.14f} ".format, ml_ab.atomic_mass_per_type))
    basis_number_fmt = "  ".join(map(str, ml_ab.num_basis_set_per_type))
    basis_fmt = "\n".join(
        [
            _BASIS_FMT_STR.format(
                x, "\n".join(["    {}   {}".format(*i) for i in ml_ab.basis_set[x]])
            )
            for x in ml_ab.atom_types
        ]
    )
    configuration_fmt = "\n".join(
        [
            _CONFIGURATION_FMT_STR.format(
                i,
                c.name,
                c.num_atom_types,
                c.num_atoms,
                "\n".join(
                    ["    {}   {}".format(*i) for i in c.atom_types_numbers.items()]
                ),
                c.ctifor,
                _LATTICE_FMT_STR.format(*chain.from_iterable(c.lattice)),
                "\n".join(["{:.16f}   {:.16f}   {:.16f}".format(*i) for i in c.coords]),
                c.total_energy,
                "\n".join(["{:.16f}   {:.16f}   {:.16f}".format(*i) for i in c.forces]),
                "{:.16f}   {:.16f}   {:.16f}".format(*c.stress_diagonal),
                "{:.16f}   {:.16f}   {:.16f}".format(*c.stress_off_diagonal),
            )
            for i, c in enumerate(ml_ab.configurations)
        ]
    )

    return _ML_AB_FMT_STR.format(
        ml_ab.version,
        ml_ab.num_configurations,
        ml_ab.max_num_atom_types,
        " ".join(ml_ab.atom_types),
        ml_ab.max_num_atoms,
        ml_ab.max_num_atoms_per_type,
        ref_energy_fmt,
        atomic_mass_fmt,
        basis_number_fmt,
        basis_fmt,
        configuration_fmt,
    )


def _parse_config(config):
    from pymlff.core import Configuration

    name = config[3]
    natom_types = int(config[7])
    natoms = int(config[11])
    atom_types_numbers = {
        x.split()[0]: int(x.split()[1]) for x in config[15 : 15 + natom_types]
    }
    ctifor = float(config[18 + natom_types])
    lattice = [
        list(map(float, x.split())) for x in config[22 + natom_types : 25 + natom_types]
    ]
    coords = [
        list(map(float, x.split()))
        for x in config[28 + natom_types : 28 + natom_types + natoms]
    ]
    total_energy = float(config[31 + natom_types + natoms])
    forces = [
        list(map(float, x.split()))
        for x in config[35 + natom_types + natoms : 35 + natom_types + natoms * 2]
    ]
    stress_diagonal = list(map(float, config[40 + natom_types + natoms * 2].split()))
    stress_off_diagonal = list(
        map(float, config[44 + natom_types + natoms * 2].split())
    )

    return Configuration(
        name=name,
        atom_types_numbers=atom_types_numbers,
        ctifor=ctifor,
        lattice=np.array(lattice),
        coords=np.array(coords),
        forces=np.array(forces),
        total_energy=total_energy,
        stress_diagonal=np.array(stress_diagonal),
        stress_off_diagonal=np.array(stress_off_diagonal),
    )


def _parse_header(header):
    version = header[0]
    atom_types = header[12].split()
    reference_energy = list(map(float, header[24].split()))
    atomic_mass = list(map(float, header[28].split()))
    num_basis = list(map(int, header[32].split()))
    basis_set = {}

    for i, nbasis in enumerate(num_basis):
        c = sum(num_basis[:i]) + 3 * i
        basis_set[atom_types[i]] = [
            list(map(int, x.split())) for x in header[36 + c : 36 + nbasis + c]
        ]

    return {
        "version": version,
        "atom_types": atom_types,
        "reference_energy": dict(zip(atom_types, reference_energy)),
        "atomic_mass": dict(zip(atom_types, atomic_mass)),
        "num_basis": num_basis,
        "basis_set": basis_set,
    }
