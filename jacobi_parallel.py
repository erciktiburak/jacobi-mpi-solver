#!/usr/bin/env python3

from mpi4py import MPI
import numpy as np
import time
import sys
import socket

# MPI Initialization
comm = MPI.COMM_WORLD
rank = comm.Get_rank()      # Process ID (0 to size-1)
size = comm.Get_size()      # Total number of processes
hostname = socket.gethostname()


def generate_diagonally_dominant_matrix(N, seed=42):

    np.random.seed(seed)
    A = np.random.uniform(-10, 10, (N, N))
    
    for i in range(N):
        row_sum = np.sum(np.abs(A[i, :])) - np.abs(A[i, i])
        A[i, i] = row_sum + np.random.uniform(1, 10)
    
    return A


def generate_known_solution_system(N, seed=42):

    np.random.seed(seed)
    A = generate_diagonally_dominant_matrix(N, seed)
    x_exact = np.random.uniform(1, 10, N)
    b = np.dot(A, x_exact)
    
    return A, b, x_exact


def compute_row_distribution(N, num_procs):

    base_rows = N // num_procs
    extra = N % num_procs
    
    counts = []
    displacements = []
    current_displacement = 0
    
    for p in range(num_procs):
        rows_for_p = base_rows + (1 if p < extra else 0)
        counts.append(rows_for_p)
        displacements.append(current_displacement)
        current_displacement += rows_for_p
    
    return counts, displacements


def jacobi_parallel_optimized(A_local, b_local, N, row_counts, row_displs,
                               max_iterations=10000, tolerance=1e-10, verbose=False):

    local_rows = row_counts[rank]
    start_row = row_displs[rank]
    
    # Initialize global x vector
    x = np.zeros(N, dtype=np.float64)
    
    # Local portion of new x
    x_local_new = np.zeros(local_rows, dtype=np.float64)
    
    # Extract diagonal elements for local rows
    local_diag = np.array([A_local[i, start_row + i] for i in range(local_rows)], dtype=np.float64)
    
    # Create R_local = A_local with zeros on diagonal positions
    R_local = A_local.copy()
    for i in range(local_rows):
        R_local[i, start_row + i] = 0.0
    
    converged = False
    iterations = 0
    
    # Prepare for Allgatherv
    recvcounts = np.array(row_counts, dtype=np.int32)
    recvdispls = np.array(row_displs, dtype=np.int32)
    
    for k in range(max_iterations):
        iterations = k + 1
        
        # Vectorized computation of local rows
        x_local_new = (b_local - np.dot(R_local, x)) / local_diag
        
        # Gather all results from all processors
        x_new = np.zeros(N, dtype=np.float64)
        comm.Allgatherv(x_local_new, [x_new, recvcounts, recvdispls, MPI.DOUBLE])
        
        # Check convergence
        diff_norm = np.linalg.norm(x_new - x)
        
        if verbose and rank == 0 and (k + 1) % 100 == 0:
            print(f"  Iteration {k+1}: ||x_new - x|| = {diff_norm:.2e}")
        
        x = x_new.copy()
        
        if diff_norm < tolerance:
            converged = True
            break
    
    return x, iterations, converged


def verify_solution(A, x, b, x_exact=None):

    if rank != 0:
        return None
    
    results = {}
    
    # Residual: ||Ax - b||
    residual = np.dot(A, x) - b
    results['residual_norm'] = np.linalg.norm(residual)
    results['relative_residual'] = results['residual_norm'] / np.linalg.norm(b)
    
    # Error compared to exact solution
    if x_exact is not None:
        error = x - x_exact
        results['error_norm'] = np.linalg.norm(error)
        results['relative_error'] = results['error_norm'] / np.linalg.norm(x_exact)
    
    return results


def print_performance_metrics(T1, Tp, p):

    speedup = T1 / Tp
    efficiency = speedup / p
    
    # Sequential fraction (Amdahl's Law)
    if p > 1:
        f = (Tp - T1/p) / (T1 * (1 - 1/p))
    else:
        f = 0
    
    # Amdahl's maximum speedup
    if f > 0 and f < 1:
        max_speedup = 1 / f
    else:
        max_speedup = float('inf')
    
    print("\n" + "=" * 70)
    print("PERFORMANCE METRICS")
    print("=" * 70)
    print(f"T1 (sequential time):     {T1:.4f} seconds")
    print(f"Tp (parallel time):       {Tp:.4f} seconds")
    print(f"Number of processors:     {p}")
    print("-" * 70)
    print(f"SpeedUp S(p):             {speedup:.2f}x")
    print(f"Efficiency E(p):          {efficiency*100:.1f}%")
    print(f"Ideal SpeedUp:            {p}x")
    print("-" * 70)
    print(f"Sequential Fraction (f):  {f:.4f} ({f*100:.1f}%)")
    if f > 0 and f < 1:
        print(f"Amdahl Max SpeedUp:       {max_speedup:.2f}x")
    print("-" * 70)
    
    # Performance interpretation
    print("Interpretation:")
    if efficiency > 1.0:
        print("  * Super-linear speedup (cache effects)")
    elif efficiency >= 0.8:
        print("  * Excellent parallel efficiency")
    elif efficiency >= 0.5:
        print("  * Good parallel efficiency")
    elif efficiency >= 0.3:
        print("  * Moderate efficiency (overhead exists)")
    else:
        print("  * Low efficiency (communication overhead)")
    
    if f < 0:
        print("  * Negative f indicates super-linear speedup (cache benefits)")
    elif f < 0.1:
        print("  * Low f: algorithm is highly parallelizable")
    elif f < 0.3:
        print("  * Moderate f: some sequential overhead")
    else:
        print("  * High f: significant communication overhead")
    
    print("=" * 70)


def main():
    # Main function to run parallel Jacobi solver.
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        try:
            N = int(sys.argv[1])
        except ValueError:
            if rank == 0:
                print(f"Invalid matrix size: {sys.argv[1]}. Using default N=1000")
            N = 1000
    else:
        N = 1000
    
    # Print header (rank 0 only)
    if rank == 0:
        print("\n" + "=" * 70)
        print("JACOBI ITERATION - PARALLEL MPI SOLVER")
        print("=" * 70)
        print(f"\nConfiguration:")
        print(f"  Matrix size (N):     {N}")
        print(f"  Processors (p):      {size}")
        print(f"  Tolerance:           1e-10")
        print(f"  Max iterations:      10000")
        print()
    
    # Print process info
    comm.Barrier()
    print(f"  Process {rank}/{size} on {hostname}")
    comm.Barrier()
    
    if rank == 0:
        print()
    
    # Compute row distribution
    row_counts, row_displs = compute_row_distribution(N, size)
    
    if rank == 0:
        print("Row Distribution:")
        for p in range(size):
            print(f"  Process {p}: rows {row_displs[p]} to {row_displs[p] + row_counts[p] - 1} ({row_counts[p]} rows)")
        print()
    
    # Phase 1: Generate and distribute matrix
    if rank == 0:
        print("-" * 70)
        print("Phase 1: Matrix Generation and Distribution")
        gen_start = time.time()
        A, b, x_exact = generate_known_solution_system(N, seed=42)
        gen_time = time.time() - gen_start
        print(f"  Matrix generation time: {gen_time:.4f} seconds")
    else:
        A = None
        b = None
        x_exact = None
    
    # Broadcast b vector
    if rank == 0:
        b_full = b.copy()
    else:
        b_full = np.zeros(N, dtype=np.float64)
    comm.Bcast(b_full, root=0)
    
    # Scatter matrix rows
    local_rows = row_counts[rank]
    A_local = np.zeros((local_rows, N), dtype=np.float64)
    b_local = np.zeros(local_rows, dtype=np.float64)
    
    if rank == 0:
        sendbuf_A = A.flatten()
        sendbuf_b = b
        sendcounts_A = [c * N for c in row_counts]
        senddispls_A = [d * N for d in row_displs]
    else:
        sendbuf_A = None
        sendbuf_b = None
        sendcounts_A = None
        senddispls_A = None
    
    # Scatter A
    recvbuf_A = np.zeros(local_rows * N, dtype=np.float64)
    sendcounts_A = comm.bcast(sendcounts_A, root=0)
    senddispls_A = comm.bcast(senddispls_A, root=0)
    comm.Scatterv([sendbuf_A, sendcounts_A, senddispls_A, MPI.DOUBLE], recvbuf_A, root=0)
    A_local = recvbuf_A.reshape((local_rows, N))
    
    # Scatter b
    sendcounts_b = comm.bcast(row_counts, root=0)
    senddispls_b = comm.bcast(row_displs, root=0)
    comm.Scatterv([sendbuf_b, sendcounts_b, senddispls_b, MPI.DOUBLE], b_local, root=0)
    
    comm.Barrier()
    if rank == 0:
        print("  Data distributed to all processes.")
    
    # Phase 2: Solve using parallel Jacobi
    if rank == 0:
        print("\n" + "-" * 70)
        print("Phase 2: Parallel Jacobi Iteration")
    
    comm.Barrier()
    solve_start = time.time()
    
    x_solution, iterations, converged = jacobi_parallel_optimized(
        A_local, b_local, N, row_counts, row_displs,
        max_iterations=10000, tolerance=1e-10, verbose=(rank == 0 and N <= 2000)
    )
    
    comm.Barrier()
    solve_time = time.time() - solve_start
    
    # Phase 3: Verify and report results (rank 0 only)
    if rank == 0:
        print("\n" + "-" * 70)
        print("Phase 3: Results")
        print(f"  Converged:            {converged}")
        print(f"  Iterations:           {iterations}")
        print(f"  Parallel time (Tp):   {solve_time:.6f} seconds")
        
        print("\n" + "-" * 70)
        print("Phase 4: Verification")
        verification = verify_solution(A, x_solution, b, x_exact)
        print(f"  Residual norm:        {verification['residual_norm']:.2e}")
        print(f"  Error norm:           {verification['error_norm']:.2e}")
        print(f"  Relative error:       {verification['relative_error']:.2e}")
        
        if verification['relative_error'] < 1e-6:
            print(f"  Solution is correct (relative error < 1e-6)")
        else:
            print(f"  Solution may have issues")
        
        # Performance metrics (if T1 provided)
        if len(sys.argv) > 2:
            try:
                T1 = float(sys.argv[2])
                print_performance_metrics(T1, solve_time, size)
            except ValueError:
                print("\nTo see performance metrics, run with T1 value:")
                print(f"  mpiexec -n {size} python3 jacobi_parallel.py {N} <T1>")
        else:
            print("\n" + "=" * 70)
            print("RESULT SUMMARY")
            print("=" * 70)
            print(f"Matrix size (N):        {N}")
            print(f"Number of processes:    {size}")
            print(f"Parallel time (Tp):     {solve_time:.6f} seconds")
            print(f"Iterations:             {iterations}")
            print(f"Converged:              {converged}")
            print("=" * 70)
            print("\nTo calculate SpeedUp and Efficiency, run with T1 value:")
            print(f"  mpiexec -n {size} python3 jacobi_parallel.py {N} <T1>")


if __name__ == "__main__":
    main()
