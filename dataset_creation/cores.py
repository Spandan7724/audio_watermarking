import os
import multiprocessing

# Get number of logical and physical CPU cores
logical_cores = os.cpu_count()  # Includes hyper-threading
physical_cores = multiprocessing.cpu_count()  # Physical cores only

print(f"Logical CPU Cores: {logical_cores}")
print(f"Physical CPU Cores: {physical_cores}")
