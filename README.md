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

## Properties of SAS

- SAS attains values between $0$ and $1$ where higher values imply greater similarity.
- SAS is invariant under actions of identical orthogonal maps from the left or the right on the compared matrices.
- SAS is invariant under transposition of both matrices; this includes the consistent permutation of rows and columns as a special case.
- SAS is invariant under scaling with a positive factor; in particular, SAS $= 1$ for $M_b = c_{1} M_a$ where $c_1 \in \mathbb{R}^+$.
- SAS is zero if scaled with a negative factor; in particular, SAS $= 0$ for $M_b = c_2 M_a$ where $c_{2} \in \mathbb{R}^-$.

The formal derivation of the measure and its properties is presented [here](link_to_paper). In particular, the linked manuscript provides cases for which SAS detects similarity where traditional measures such as the Frobenius norm or the cosine similarity fail.

We recommend to use the `match_values` method of SAS, which is selected by default. When multiple singular values are close in magnitude and noise is present, it may be beneficial to select pairs of vectors based on their alignment rather than based on their singular value. Possible applications of this method, in this implementation called `match_vectors`, are discussed in the publication linked above.

## Installation
```bash
pip install .
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

Here, we show that two generative models of matrices can be distinguished from each other: a normally distributed matrix with a block in its upper left quarter is more similar to another instantiation of the same matrix type than to a normally distributed matrix with no blocks.

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
