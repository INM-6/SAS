<!-- 
SAS - Assessing the similarity of real matrices with arbitrary shape.
Copyright (C) 2024 Forschungszentrum Juelich GmbH, IAS-6

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with
this program. If not, see <https://www.gnu.org/licenses/>.

SPDX-License-Identifier: GPL-3.0-or-later
-->

# SAS: Singular Angle Similarity
Method for comparing the singular angle similarity of arbitrary matrices of the same shape, based on singular value decomposition.

## Installation
```bash
pip install -e .
```

## Example use

### Comparing single matrices

```python
import numpy as np
import sas

# parameters
dim = 10

# create matrices
matrix_a = np.random.normal(0, 1, (dim, dim))
matrix_b = np.random.normal(0, 1, (dim, dim))

# calculate similarity
similarity = sas.compare(matrix_a, matrix_b)
```

### Calculating the similarity across instances

```python
import numpy as np
import sas

# parameters
dim = 10
reps = 10

# create matrices
matrices = {'normal_block_1': [np.random.normal(0, 1, (dim, dim)) for _ in range(reps)],
			'normal_block_2': [np.random.normal(0, 1, (dim, dim)) for _ in range(reps)],
			'normal_noblock': [np.random.normal(0, 1, (dim, dim)) for _ in range(reps)]}
for rep in range(reps):
	matrices['normal_block_1'][rep][:dim//2, :dim//2] += np.random.normal(2, 1, (dim//2, dim//2))
	matrices['normal_block_2'][rep][:dim//2, :dim//2] += np.random.normal(2, 1, (dim//2, dim//2))

# calculate self- and cross-similarity
self_similarity = [sas.compare(matrices['normal_block_1'][i], matrices['normal_block_2'][i]) for i in range(reps)]
cross_similarity = [sas.compare(matrices['normal_block_1'][i], matrices['normal_noblock'][i]) for i in range(reps)]

# calculate effect size between similarity distributions
# effect size > 1 indicates statistical separability
effect_size = sas.effect_size(self_similarity, cross_similarity)
```

## Contact

[Jasper Albers](mailto:j.albers@fz-juelich.de?subject=SAS)