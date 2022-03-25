# pymlff

A lightweight Python 3.7+ package for reading and writing VASP ML_AB files.

## Installation

Clone the repository and install using

```bash
pip install .
```

The only dependency is numpy.

## Examples

### Reading and writing ML_AB files

```python
from pymlff import MLAB

# load an ML_AB file
ab = MLAB.from_file("ML_AB")

# write an ML_AB file
ab.write_file("ML_AB_NEW")
```

### Filtering configurations

Configurations can be filtered based on a filter function. The function should accept two
arguments, the index of the configuration in the configurations list and a configuration
itself. If the filter function returns True, the configuration  will be kept, if False,
the configuration will be removed. For example, to filter configurations with less than
256 atoms:

```python
new_ab = ab.filter_configurations(lambda i, config: config.num_atoms >= 256)
```

See the `Configuration` object for the available attributes to use for filtering.

### Combining ML_AB files

MLAB objects can be combined using the Python `+` operator. For example,

```python
from pymlff import MLAB

# load two ML_AB files
ab1 = MLAB.from_file("ML_AB1")
ab2 = MLAB.from_file("ML_AB2")

new_ab = ab1 + ab2
```
