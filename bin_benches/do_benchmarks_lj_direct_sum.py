import subprocess
import time

def average_runtime(executable, iterations=1000):
    total_time = 0.0

    for i in range(iterations):
        start_time = time.time()

        # Run the executable
        subprocess.run(executable, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        end_time = time.time()
        total_time += (end_time - start_time)

        if (i + 1) % 10 == 0:
            print(f"Completed {i + 1} iterations")

    average_time = total_time / iterations
    return average_time

if __name__ == "__main__":
    import sys
    executables = ["./lj_sum_4096_ForLoop_1MCycles", "./lj_sum_4096_axisIterMut_1MCycles"]
    iterations = 100
    for exe in executables:
        average_time = average_runtime(exe, iterations)
        print(f"Average runtime of executable {exe} over {iterations} runs: {average_time:.6f} seconds")

        