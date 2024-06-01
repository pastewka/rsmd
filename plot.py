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
                "target/criterion/different_sized_lj_clusters/lj_direct_summation/input_files_"
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
        1.027e-06,
        7.093e-06,
        4.0439e-05,
        0.000154782,
        0.000469344,
        0.00112891,
        0.00265009,
        0.00519953,
        0.00989274,
        0.0151726,
        0.0232958,
        0.0351333,
        0.0514191,
        0.0812369,
        0.100645,
        0.121801
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
    plt.savefig("docs/LJ_Direct_Summation_Benchmark_Rust_Vs_C++.png",format="png")



plotMenchmarkResultLjDirectSummation()
