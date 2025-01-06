# cuDSSを用いたCSR形式疎行列のLU分解と方程式の解法
- このリポジトリは、CSR（Compressed Sparse Row）形式の複素数疎行列データを入力として、cuDSSライブラリを使用したLU分解と線形方程式の解法を実現するC++実装を提供する。
- 直接法の各計算フェーズでの計算時間を計測しており、ベンチマークが取れるようになっている。

# References
- [NVIDIA cuDSS Documentation](https://docs.nvidia.com/cuda/cudss/index.html)

# 機能
- cuDSS（CUDA）を使用した疎行列のLU分解
- 大規模な線形方程式を効率的に解くことが可能
- GPUを活用した高性能な計算をサポート

# ビルド環境
- CUDA Toolkit：version 12.0以降
- cuDSS：疎行列演算用, 0.4.0以降
- C++コンパイラ：C++17以降に対応（例：GCC、Clang、MSVCなど）
- CMake：version 3.19以降

# ビルド方法
- build
```bash
cmake -B build
cmake --build build
```
- 実行
```bash
./build/main
```

# 入力形式
- csr形式のデータ、つまりrow pointer, column index, valuesの3つのデータをそれぞれテキストファイルで用意しておく。
- row pointer: row_ptr.datとして保存
```dat
0
2
3
(続く)
```

- column index: col_idx.datとして保存
```dat
0
1
1
(続く)
```

- values: values.datとして保存
    - 複素数データとして入力
    - 各成分をreal_part imag_partと一行づつ記述
```dat
1. 0.
2. 1.
0. 2.
(続く)
```
