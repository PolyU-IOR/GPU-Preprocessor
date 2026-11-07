# Source attribution

GPU-Preprocessor-QP originally contained the Julia GPU QP presolver extracted
from [GPU-Presolver-development](https://github.com/ljw2024polyu/GPU-Presolver-development).
The C11/CUDA implementation rewrites the model, sparse transformation engine,
reductions, recovery, MPS I/O, command-line tools, example and regression suite
without a Julia runtime or wrapper. Prior implementation history remains in the source QP repository.

The original-space KKT convention follows
[HPR-QP](https://github.com/PolyU-IOR/HPR-QP).

Original copyright and MIT terms are retained in [LICENSE](LICENSE).

The LP module in `lp/` continues the implementation in
[PolyU-IOR/GPU-Preprocessor](https://github.com/PolyU-IOR/GPU-Preprocessor/tree/274a543520ef62e1c7a1f12d1c8989c8122a6869),
commit `274a543520ef62e1c7a1f12d1c8989c8122a6869`. Its source copyright
headers and [AUTHORS](AUTHORS) are retained. The combined LP/QP project extends
the build integration, command-line interface and documentation.
