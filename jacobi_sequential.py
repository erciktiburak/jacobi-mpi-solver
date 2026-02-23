#!/usr/bin/env python3

import numpy as np
import time
import sys


def generate_diagonally_dominant_matrix(N, seed=42):
  
    np.random.seed(seed)
    
    # Generate random matrix with values between -10 and 10
    A = np.random.uniform(-10, 10, (N, N))
    
    # Make it diagonally dominant
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


def jacobi_sequential_optimized(A, b, max_iterations=10000, tolerance=1e-10, verbose=False):

    N = len(b)
    x = np.zeros(N)
    
    # Extract diagonal and create matrix without diagonal
    diag = np.diag(A).copy()
    
    if np.any(diag == 0):
        raise ValueError("Matrix has zero diagonal element(s). Jacobi method cannot be applied.")
    
    # Create matrix R = A with zeros on diagonal
    R = A.copy()
    np.fill_diagonal(R, 0)
    
    converged = False
    iterations = 0
    
    for k in range(max_iterations):
        iterations = k + 1
        
        # Vectorized Jacobi iteration
        x_new = (b - np.dot(R, x)) / diag
        
        # Check convergence
        diff_norm = np.linalg.norm(x_new - x)
        
        if verbose and (k + 1) % 100 == 0:
            residual = np.linalg.norm(np.dot(A, x_new) - b)
            print(f"  Iteration {k+1}: ||x_new - x|| = {diff_norm:.2e}, ||Ax - b|| = {residual:.2e}")
        
        x = x_new
        
        if diff_norm < tolerance:
            converged = True
            break
    
    residual_norm = np.linalg.norm(np.dot(A, x) - b)
    
    return x, iterations, converged, residual_norm


def verify_solution(A, x, b, x_exact=None):

    results = {}
    
    # Residual: ||Ax - b||
    residual = np.dot(A, x) - b
    results['residual_norm'] = np.linalg.norm(residual)
    results['residual_max'] = np.max(np.abs(residual))
    results['relative_residual'] = results['residual_norm'] / np.linalg.norm(b)
    
    # Error compared to exact solution
    if x_exact is not None:
        error = x - x_exact
        results['error_norm'] = np.linalg.norm(error)
        results['error_max'] = np.max(np.abs(error))
        results['relative_error'] = results['error_norm'] / np.linalg.norm(x_exact)
    
    return results


def run_test(N, seed=42, verbose=True):

    print(f"Test Case: N = {N}")
    print("=" * 60)
    
    # Generate system
    print(f"\n1) Generating {N}x{N} diagonally dominant matrix...")
    gen_start = time.time()
    A, b, x_exact = generate_known_solution_system(N, seed)
    gen_time = time.time() - gen_start
    print(f"   Matrix generation time: {gen_time:.4f} seconds")
    
    # Solve using optimized Jacobi
    print(f"\n2) Solving using Jacobi iteration...")
    solve_start = time.time()
    x, iterations, converged, residual = jacobi_sequential_optimized(
        A, b, max_iterations=10000, tolerance=1e-10, verbose=verbose
    )
    solve_time = time.time() - solve_start
    
    # Results
    print(f"\n3) Results:")
    print(f"   Converged: {converged}")
    print(f"   Iterations: {iterations}")
    print(f"   Solve time (T1): {solve_time:.4f} seconds")
    
    # Verification
    print(f"\n4) Verification:")
    verification = verify_solution(A, x, b, x_exact)
    print(f"   Residual norm: {verification['residual_norm']:.2e}")
    print(f"   Error norm: {verification['error_norm']:.2e}")
    print(f"   Relative error: {verification['relative_error']:.2e}")
    
    if verification['relative_error'] < 1e-6:
        print(f"   Solution is correct (relative error < 1e-6)")
    else:
        print(f"   Solution may have issues (relative error >= 1e-6)")
    
    print("=" * 60)
    
    return {
        'N': N,
        'generation_time': gen_time,
        'solve_time': solve_time,
        'iterations': iterations,
        'converged': converged,
        'residual_norm': verification['residual_norm'],
        'error_norm': verification.get('error_norm', None),
        'relative_error': verification.get('relative_error', None)
    }


def main():
    """Main function to run sequential Jacobi solver."""
    print("\n" + "=" * 60)
    print("JACOBI ITERATION - SEQUENTIAL SOLVER")
    print("=" * 60)
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        try:
            N = int(sys.argv[1])
            test_sizes = [N]
        except ValueError:
            print(f"Invalid argument: {sys.argv[1]}. Using default test sizes.")
            test_sizes = [1000, 5000, 10000]
    else:
        test_sizes = [1000, 5000, 10000]
    
    print(f"\nTest sizes: {test_sizes}")
    print(f"Tolerance: 1e-10")
    print(f"Max iterations: 10000")
    
    # Run tests
    all_results = []
    for N in test_sizes:
        print("\n")
        result = run_test(N, seed=42, verbose=(N <= 500))
        all_results.append(result)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY - Sequential Times (T1)")
    print("=" * 60)
    print(f"{'N':>10} | {'T1 (s)':>10} | {'Iterations':>10} | {'Residual':>12}")
    print("-" * 50)
    for r in all_results:
        print(f"{r['N']:>10} | {r['solve_time']:>10.4f} | {r['iterations']:>10} | {r['residual_norm']:>12.2e}")
    
    print("\nSequential tests completed.")
    print("Use T1 values for speedup calculations with parallel version.")
    print("=" * 60)


if __name__ == "__main__":
    main()
