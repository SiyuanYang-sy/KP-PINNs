# KP-PINNs: Kernel Packet Accelerated Physics Informed Neural Networks

This repository contains the official PyTorch implementation of the paper:

**KP-PINNs: Kernel Packet Accelerated Physics Informed Neural Networks** 📄 [Read the paper](https://arxiv.org/abs/2506.08563)

Accepted to *IJCAI 2025 (34th International Joint Conference on Artificial Intelligence)*  

---

## 📝 Abstract

Differential equations are involved in modeling many engineering problems. Many efforts have been devoted to solving differential equations. Due to the flexibility of neural networks, Physics Informed Neural Networks (PINNs) have recently been proposed to solve complex differential equations and have demonstrated superior performance in many applications. While the L2 loss function is usually a default choice in PINNs, it has been shown that the corresponding numerical solution is incorrect and unstable for some complex equations. In this work, we propose a new PINNs framework named Kernel Packet accelerated PINNs (KP-PINNs), which gives a new expression of the loss function using the reproducing kernel Hilbert space (RKHS) norm and uses the Kernel Packet (KP) method to accelerate the computation. Theoretical results show that KP-PINNs can be stable across various differential equations. Numerical experiments illustrate that KP-PINNs can solve differential equations effectively and efficiently. This framework provides a promising direction for improving the stability and accuracy of PINNs-based solvers in scientific computing.

---

## ⚙️ Environment
```
This project was tested under the following configuration:

- Python: 3.11.9
- NumPy: 1.24.3
- PyTorch: 2.3.1  
- CUDA Toolkit: 12.1

Additional packages (used in training or visualization):

- pandas, scipy, matplotlib
```

---

## 📁 Code Structure
```
KP-PINNs/
├── Stiff equation/
│ ├── Stiff_Inference_KP.py                                  → Forward problem
│ ├── Stiff_Identification_KP.py                             → Inverse problem
│ ├── KP_compute_APhi.py                                     → KP method to compute loss
│ └── stiff_solution.mat                                     → True solution data
│
├── Helmholtz equation/
│ ├── Helmholtz_Inference_KP.py                              → Forward problem
│ ├── Helmholtz_Identification_KP.py                         → Inverse problem
│ ├── KP_compute_APhi.py                                     → KP method to compute loss
│ └── helmholtz_solution.mat                                 → True solution data
│
├── Linear Quadratic Gaussian (LQG) equation/
│ ├── LQG_Inference_KP.py                                    → Forward problem
│ ├── LQG_Identification_KP.py                               → Inverse problem
│ ├── KP_compute_APhi.py                                     → KP method to compute loss
│ ├── LQG_bc.npz, LQG_solution.npz                           → True solution data
│
├── Navier-Stokes (NS) equation/
│ ├── NS_Inference_KP.py                                     → Forward problem
│ ├── NS_Identification_KP.py                                → Inverse problem
│ ├── KP_compute_APhi.py                                     → KP method to compute loss
│ ├── NS_ic.csv, 2s_NS_bc.csv, 2s_noise_free_NS_in_col.csv   → True solution data
```

---

## 📚 Citation
```
@misc{yang2025kppinnskernelpacketaccelerated,
      title={KP-PINNs: Kernel Packet Accelerated Physics Informed Neural Networks}, 
      author={Siyuan Yang and Cheng Song and Zhilu Lai and Wenjia Wang},
      year={2025},
      eprint={2506.08563},
      archivePrefix={arXiv},
      primaryClass={cs.CE},
      url={https://arxiv.org/abs/2506.08563}, 
}
```

