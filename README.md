# jacobi-mpi-solver
Parallel implementation of the Jacobi iterative method for solving systems of linear algebraic equations (Ax = b) using MPI.

# Jacobi MPI Solver

Parallel implementation of the Jacobi iterative method for solving systems of linear algebraic equations (Ax = b) using MPI.

Developed for CMPE-523 Parallel and Distributed Programming course.

## Features

- Sequential solver for baseline measurements (T1)
- Parallel MPI solver with row-wise decomposition
- Automatic SpeedUp, Efficiency, and Amdahl's Law calculations
- Solution verification against known exact solutions
- Support for multi-node distributed clusters

## Performance Results

Tested on a 10-node cluster across two networks:

| Matrix Size | Processors | SpeedUp | Efficiency |
|-------------|------------|---------|------------|
| N=10000 | p=2 | 4.36x | 217.8% |
| N=10000 | p=4 | 4.47x | 111.7% |
| N=10000 | p=6 | 5.52x | 92.0% |
| N=10000 | p=8 | 5.96x | 74.5% |
| N=15000 | p=2 | 1.98x | 99.2% |

Super-linear speedup (efficiency > 100%) observed due to cache effects.

## Requirements

- Python 3.8+
- NumPy
- mpi4py
- OpenMPI 4.x

## Installation

```bash
# Install dependencies
pip install numpy mpi4py

# Clone repository
git clone https://github.com/YOUR_USERNAME/jacobi-mpi-solver.git
cd jacobi-mpi-solver
```

## Usage

### Sequential (Baseline)

```bash
# Run with specific matrix size
python3 jacobi_sequential.py 10000

# Output includes T1 time for speedup calculations
```

### Parallel (Single Machine)

```bash
# 4 processors
mpiexec -n 4 python3 jacobi_parallel.py 10000

# With T1 for performance metrics
mpiexec -n 4 python3 jacobi_parallel.py 10000 5.6458
```

### Parallel (Distributed Cluster)

```bash
# Using host list
mpiexec -n 8 --host server1,server2,server3,server4,server5,server6,server7,server8 \
    python3 jacobi_parallel.py 10000 5.6458
```

## Algorithm

The Jacobi iteration formula:

```
x_i^(k+1) = (1/a_ii) * [b_i - sum(a_ij * x_j^(k)) for j != i]
```

### Parallelization Strategy

1. Matrix A is divided by rows among p processors
2. Each processor computes N/p rows independently
3. MPI_Allgatherv exchanges updated values after each iteration
4. Convergence check: ||x_new - x|| < tolerance

## Performance Metrics

- **SpeedUp**: S(p) = T1 / Tp
- **Efficiency**: E(p) = S(p) / p
- **Sequential Fraction**: f = (Tp - T1/p) / (T1 * (1 - 1/p))

## Project Structure

```
jacobi-mpi-solver/
├── jacobi_sequential.py    # Sequential implementation
├── jacobi_parallel.py      # Parallel MPI implementation
└── README.md
```

## Author

Ercikti Burak

## License

MIT License
