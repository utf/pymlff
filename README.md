# pymlff

A lightweight Python 3.7+ package for reading and writing VASP ML_AB files.

## Installation

Clone the repository and install using

```bash
pip install .
```

Pymlff depends on numpy and click.

## Examples

### Command line utility

Pymlff includes a basic command line utility for merging ML_AB.

```bash
mlff merge ML_AB1 ML_AB2 ML_AB_NEW
```

More than 2 ML_AB files can be combined at once, e.g.

```bash
mlff merge ML_AB1 ML_AB2 ML_AB3 ML_AB4 ML_AB_NEW
```

Pymlff also includes a command line utility for converting ML_AB files to extended xyz (extxyz) files. The last argument is the units to use for converting the VASP kbar units. Valid units are eV/A^3 and kbar.

```bash
# Convert ML_AB to extxyz
mlff write-extxyz ML_AB ML_AB.xyz

# Convert ML_AB to extxyz and convert the units of stress from kbar to eV/A^3 (applies a negative sign)
mlff write-extxyz ML_AB ML_AB.xyz --stress-unit=eV/A^3
```

### Python API

More functionality is available through the Python API.

#### Reading and writing ML_AB files

```python
from pymlff import MLAB

# load an ML_AB file
ab = MLAB.from_file("ML_AB")

# write an ML_AB file
ab.write_file("ML_AB_NEW")
```

#### Filtering configurations

Configurations can be filtered based on a filter function. The function should accept two
arguments, the index of the configuration in the configurations list and a configuration
itself. If the filter function returns True, the configuration  will be kept, if False,
the configuration will be removed. For example, to filter configurations with less than
256 atoms:

```python
new_ab = ab.filter_configurations(lambda i, config: config.num_atoms >= 256)
```

See the [Configuration](https://github.com/utf/pymlff/blob/97f972f9f955c145fb43c2cc74c71fabeac523fb/src/pymlff/core.py#L11) object for the available attributes to use for filtering.

#### Combining ML_AB files

MLAB objects can be combined using the Python `+` operator. For example,

```python
from pymlff import MLAB

# load two ML_AB files
ab1 = MLAB.from_file("ML_AB1")
ab2 = MLAB.from_file("ML_AB2")

new_ab = ab1 + ab2
```

#### Writing to extxyz

MLAB objects can be written to the extxyz format using the `write_extxyz` method.

```python
from pymlff import MLAB

# load an ML_AB file
ab = MLAB.from_file("ML_AB")

# write an extxyz file
ab.write_extxyz("ML_AB.xyz", "eV/A^3")
```
