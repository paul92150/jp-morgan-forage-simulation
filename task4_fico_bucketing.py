"""
FICO Score Bucketing – Two Approaches

This script implements two methods for bucketing FICO scores to maximize default prediction accuracy via log-likelihood:

1. Dynamic Programming (APPROACH = 'dp')
   - Guarantees the **optimal solution**.
   - But takes ~10 minutes for 10,000 data points and 5 buckets.
   - Becomes **impractical** for larger datasets or more buckets due to O(N² x B) complexity.

2. Hybrid Optuna-Guided Top-K Search (APPROACH = 'hybrid')
   - Uses **smart Bayesian search** (Optuna) and dynamic programming structure.
   - Finds a nearly optimal solution within seconds.
   - Scales easily to large datasets and high bucket counts.

Set `APPROACH = 'dp'` or `APPROACH = 'hybrid'` below to choose which method to run.

Author: Paul Lemaire
"""

import pandas as pd
import numpy as np
import optuna
import matplotlib.pyplot as plt

# === Configuration ===
APPROACH = 'hybrid'  # 'dp' or 'hybrid'

# === Log-likelihood helper ===
def compute_log_likelihood(ki, ni):
    if ni == 0 or ki == 0 or ki == ni:
        return 0
    pi = ki / ni
    return ki * np.log(pi) + (ni - ki) * np.log(1 - pi)

# === Dynamic Programming Approach ===
def create_buckets(fico_scores, defaults, n_buckets):
    sorted_indices = np.argsort(fico_scores)
    fico_scores = fico_scores[sorted_indices]
    defaults = defaults[sorted_indices]
    N = len(fico_scores)

    cum_defaults = np.cumsum(defaults)
    cum_count = np.arange(1, N + 1)

    def get_bucket_stats(start, end):
        total = cum_count[end] - (cum_count[start - 1] if start > 0 else 0)
        defaults_in_bucket = cum_defaults[end] - (cum_defaults[start - 1] if start > 0 else 0)
        return defaults_in_bucket, total

    dp = np.full((N, n_buckets), -np.inf)
    backtrack = np.zeros((N, n_buckets), dtype=int)

    for i in range(N):
        ki, ni = get_bucket_stats(0, i)
        dp[i, 0] = compute_log_likelihood(ki, ni)
        backtrack[i, 0] = -1

    for k in range(1, n_buckets):
        for i in range(N):
            for j in range(k - 1, i):
                ki, ni = get_bucket_stats(j + 1, i)
                score = dp[j, k - 1] + compute_log_likelihood(ki, ni)
                if score > dp[i, k]:
                    dp[i, k] = score
                    backtrack[i, k] = j

    boundaries = []
    idx = N - 1
    for k in range(n_buckets - 1, -1, -1):
        idx = backtrack[idx, k]
        if idx != -1:
            boundaries.append(fico_scores[idx])

    return sorted(boundaries)

# === Bucket Assignment ===
def assign_bucket(fico_score, boundaries):
    for i, boundary in enumerate(boundaries):
        if fico_score <= boundary:
            return i
    return len(boundaries)

# === Log-likelihood for any cut configuration ===
def score_buckets_log_likelihood(boundaries, scores, defaults):
    fico_min = int(np.min(scores))
    fico_max = int(np.max(scores))
    boundaries = sorted(boundaries)
    prev = fico_min
    total_ll = 0
    for b in boundaries:
        mask = (scores >= prev) & (scores <= b)
        ni = np.sum(mask)
        ki = np.sum(defaults[mask])
        total_ll += compute_log_likelihood(ki, ni)
        prev = b + 1
    mask = (scores >= prev) & (scores <= fico_max)
    ni = np.sum(mask)
    ki = np.sum(defaults[mask])
    total_ll += compute_log_likelihood(ki, ni)
    return total_ll

# === Hybrid Top-K Guided Search ===
def hybrid_topk_guided_search(fico_scores, defaults, max_buckets=5, top_k=500, optuna_trials=20):
    results = {1: [([], score_buckets_log_likelihood([], fico_scores, defaults))]}
    fico_min = int(np.min(fico_scores))
    fico_max = int(np.max(fico_scores))

    for b in range(2, max_buckets + 1):
        print(f"\n🔍 Building bucket {b} from previous {b - 1} buckets...")
        candidates = []
        for prev_cuts, _ in results[b - 1]:
            def objective(trial):
                new_cut = trial.suggest_int("cut", fico_min + 1, fico_max - 1)
                if new_cut in prev_cuts:
                    raise optuna.exceptions.TrialPruned()
                full_cuts = sorted(prev_cuts + [new_cut])
                return -score_buckets_log_likelihood(full_cuts, fico_scores, defaults)

            study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
            study.optimize(objective, n_trials=optuna_trials, show_progress_bar=False)

            best_cut = study.best_params["cut"]
            best_score = -study.best_value
            full_set = sorted(prev_cuts + [best_cut])
            candidates.append((full_set, best_score))

        top_candidates = sorted(candidates, key=lambda x: -x[1])[:top_k]
        results[b] = top_candidates

    return results

# === Main execution ===
if __name__ == "__main__":
    df = pd.read_csv("data/Task 3 and 4_Loan_Data.csv")
    fico_scores = df['fico_score'].values
    defaults = df['default'].values

    if APPROACH == 'dp':
        print("\n=== Exact Dynamic Programming ===")
        boundaries_dp = create_buckets(fico_scores, defaults, n_buckets=5)
        print(f"Boundaries (DP): {boundaries_dp}")

        df['bucket_dp'] = df['fico_score'].apply(lambda x: assign_bucket(x, boundaries_dp))
        bucket_default_rate_dp = df.groupby('bucket_dp')['default'].mean()

        plt.figure(figsize=(10, 5))
        bucket_default_rate_dp.plot(kind='bar')
        plt.title("Default Rate per Bucket (DP)")
        plt.xlabel("Bucket")
        plt.ylabel("Default Rate")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    elif APPROACH == 'hybrid':
        print("\n=== Hybrid Optuna + Top-K ===")
        results_hybrid = hybrid_topk_guided_search(fico_scores, defaults, max_buckets=5, top_k=200, optuna_trials=200)
        best_hybrid = max(results_hybrid[5], key=lambda x: x[1])
        print(f"Best Boundaries (Hybrid): {best_hybrid[0]}")
        print(f"Best Log-Likelihood Score (Hybrid): {best_hybrid[1]}")

        scores = [max(x[1] for x in results_hybrid[b]) for b in sorted(results_hybrid)]
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, 6), scores, marker='o')
        plt.title("Best Log-Likelihood vs Number of Buckets (Hybrid)")
        plt.xlabel("Number of Buckets")
        plt.ylabel("Log-Likelihood")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    else:
        raise ValueError("APPROACH must be 'dp' or 'hybrid'")