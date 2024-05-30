use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use rsmd::md_implementation::{self, atoms::Atoms};
use std::fs;

fn criterion_benchmark(c: &mut Criterion) {
    const EPSILON: f64 = 0.7;
    const SIGMA: f64 = 0.3;

    let folder = "input_files/";
    let json_file_path = folder.to_owned() + "benchmark_lj_direct_summation.json";
    let content = fs::read_to_string(&json_file_path).expect("JSON file \"benchmark_lj_direct_summation.json\" could not be loaded to benchmark with the specified input files.");
    let json: serde_json::Value =
        serde_json::from_str(&content).expect("JSON was not well-formatted");

    let input_files = json.as_array().expect(
        &("The JSON file \"".to_owned()
            + &json_file_path
            + "\" doesn't contain a valid JSON array with the filenames of the input files."),
    );

    for json_path in input_files {
        let path = folder.to_owned() + json_path.as_str().unwrap();
        if !fs::metadata(&path).is_ok() {
            panic!("input file \"{}\" doesn't exist!", path);
        }
    }

    let mut group = c.benchmark_group("different_sized_lj_clusters");
    for f in input_files {
        let path = folder.to_owned() + f.as_str().unwrap();
        group.bench_function(
            BenchmarkId::new("lj_direct_summation", "input_file_".to_owned() + &path),
            |b| {
                b.iter_batched_ref(
                    || -> Atoms { md_implementation::xyz::read_xyz(path.to_string()).unwrap() },
                    |v| black_box(v.lj_direct_summation(Some(EPSILON), Some(SIGMA))),
                    BatchSize::SmallInput,
                )
            },
        );
    }
    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
