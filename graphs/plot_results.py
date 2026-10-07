import matplotlib.pyplot as plt

implementations = ['Sequential', 'OpenMP (8T)', 'MPI (4 VMs)', 'CUDA (GPU)']
times = [244.12, 30.83, 92.98, 0.165]
speedups = [1.00, 7.92, 2.63, 1479.48]

# 1. Execution Time Plot (Log Scale)
plt.figure(figsize=(8, 5))
plt.bar(implementations, times, color=['#4C72B0', '#55A868', '#C44E52', '#8172D5'])
plt.yscale('log')
plt.ylabel('Execution Time (seconds) - Log Scale')
plt.title('4000x4000 Matrix Multiplication Execution Time')
plt.grid(axis='y', linestyle='--', alpha=0.7)
for i, v in enumerate(times):
    plt.text(i, v * 1.15, f"{v}s", ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('graphs/execution_time.png', dpi=300)
plt.close()

# 2. Speedup Comparison Plot
plt.figure(figsize=(8, 5))
plt.bar(implementations, speedups, color=['#4C72B0', '#55A868', '#C44E52', '#8172D5'])
plt.yscale('log')
plt.ylabel('Speedup Factor vs Sequential (Log Scale)')
plt.title('Speedup Achieved Across Computing Models')
plt.grid(axis='y', linestyle='--', alpha=0.7)
for i, v in enumerate(speedups):
    plt.text(i, v * 1.15, f"{v}x", ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('graphs/speedup_comparison.png', dpi=300)
plt.close()

print("Graphs generated successfully.")