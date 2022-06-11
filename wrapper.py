from subprocess import check_output
from argparse import ArgumentParser
from pathlib import Path

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("cpu_binary", type=Path)
    parser.add_argument("gpu_binary", type=Path)

    parser.add_argument("sigma", type=float)
    parser.add_argument("thr_high", type=float)
    parser.add_argument("thr_low", type=float)

    parser.add_argument("in_image", type=Path)
    parser.add_argument("out_image", type=Path)

    parser.add_argument("--n_runs", type=int, default=1)

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    single_thread_save_path = args.out_image.parent / f"{args.out_image.stem}_cpu.png"
    openmp_save_path = args.out_image.parent / f"{args.out_image.stem}_openmp.png"
    cuda_save_path = args.out_image.parent / f"{args.out_image.stem}_cuda.png"

    single_thread_cmd = f"OMP_NUM_THREADS=1 {args.cpu_binary} {args.sigma} {args.thr_high} {args.thr_low} {args.in_image} " \
                        f"{single_thread_save_path} {args.n_runs}"
    openmp_cmd = f"{args.cpu_binary} {args.sigma} {args.thr_high} {args.thr_low} {args.in_image} " \
                        f"{openmp_save_path} {args.n_runs}"
    cuda_cmd = f"{args.gpu_binary} {args.sigma} {args.thr_high} {args.thr_low} {args.in_image} " \
                      f"{cuda_save_path} {args.n_runs}"

    print(f"Running all executables for {args.n_runs} iterations")
    print(f"Running single threaded filter\n{single_thread_cmd}")
    single_thread_output = check_output(single_thread_cmd, shell=True).decode().split("\n")[-2]
    print(f"Running openmp filter\n{openmp_cmd}")
    openmp_output = check_output(openmp_cmd, shell=True).decode().split("\n")[-2]
    print(f"Running cuda filter\n{cuda_cmd}")
    cuda_output = check_output(cuda_cmd, shell=True).decode().split("\n")[-2]

    print(40*"=")
    single_thread = list(map(int, single_thread_output.split()))
    print(f"Single threaded process took {single_thread[1]} milliseconds on average")
    openmp = list(map(int, openmp_output.split()))
    print(f"OpenMP process took {openmp[1]} milliseconds on average")
    cuda = list(map(int, cuda_output.split()))
    print(f"CUDA process took {cuda[0]} milliseconds (without memory access time) on average")
    print(f"CUDA process took {cuda[1]} milliseconds (with memory access time) on average")
    print(f"CUDA process copying time took {cuda[1] - cuda[0]} milliseconds on average")

