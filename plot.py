import matplotlib.pyplot as plt
import json
import re
from sys import argv

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

    plt.plot(time, atoms, marker="x", label="Rust")
    t = [
        9.43e-07,
        8.186e-06,
        4.6262e-05,
        0.000177433,
        0.000531137,
        0.00135132,
        0.00321028,
        0.00678484,
        0.0120352,
        0.015849,
        0.0268553,
        0.0419334,
        0.0661737,
        0.0934785,
        0.110992,
        0.130721
    ]
    time = [i*1000 for i in t]
    print(time)

    plt.plot(time, atoms, marker="x", label="C++")
    plt.title("LJ Direct Summation Benchmark")
    plt.xlabel("elapsed time in ms")
    plt.ylabel("number of atoms")
    plt.legend()
    plt.grid()
    if len(argv) >= 2:
        if argv[1] == "p" or argv[1] == "plot" or argv[1] == "-p" or argv[1] == "-plot":
            plt.show()
    plt.savefig("docs/LJ_Direct_Summation_Benchmark_Rust_Vs_C++",format="png")



plotMenchmarkResultLjDirectSummation()
