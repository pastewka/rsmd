import matplotlib.pyplot as plt
import json
import re


def plotMenchmarkResultLjDirectSummation():
    folder = "input_files/"
    time = []
    atoms = []
    with open(folder + "benchmark_lj_direct_summation.json") as benchmark_file:
        for bench_name in json.load(benchmark_file):
            with open(
                "target/criterion/different_sized_lj_clusters/lj_direct_summation/input_file_"
                + bench_name
                + "/new/estimates.json"
            ) as measurement_file:
                measurement_content = measurement_file.read()
                json_data = json.loads(measurement_content)
                time.append(
                    json_data["median"]["point_estimate"] / 1e6
                )  # convert ns to ms
                match = re.search(r"(\d+)\.xyz$", bench_name)
                atoms.append(match.group(1))
    print(time)
    print(atoms)

    plt.plot(time, atoms, marker="x", label="Rust")
    time = [
        8.6319e-02,
        1.02854,
        5.83044,
        20.6786,
        54.5749,
        119.574,
        261.331,
        526.098,
        984.54,
        1752.07,
        2961.46,
        4802.24,
        7433.54,
        11358.3,
        14850.3,
        16547.2,
    ]

    plt.plot(time, atoms, marker="x", label="C++")
    plt.title("LJ Direct Summation Benchmark")
    plt.xlabel("elapsed time in ms")
    plt.ylabel("number of atoms")
    plt.legend()
    plt.grid()
    plt.show()


plotMenchmarkResultLjDirectSummation()
