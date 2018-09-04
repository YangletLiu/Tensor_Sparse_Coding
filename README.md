We are implemting the following paper：

[AAAI] F. Jiang, X.-Y. Liu, H. Lu, R. Shen. Efficient multi-dimensional tensor sparse coding using t-linear combinations. AAAI, 2018

Phase I: Tensor sparse coding in Python 

1 Run starrt.py to load the origin dataset (a 3D tensor), we add noise.

  Then through cofficience learning and dictionary learning. We get a reconstruct tensor with high PSNR.

2 The resulted picture is:

Tensor_Sparse_Coding/python sparse coding/result/sparsecoding.png

Phase II　Tensor sparse coding in CUDA

1. Optimized cuTensor library for the low-tubal-rank tensor models.

2. Tensor sparse coding baased on the cuTensor library.



