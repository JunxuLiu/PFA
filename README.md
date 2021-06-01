# PFA



<!-- GETTING STARTED -->
## Getting Started

This is the source code of paper "Projected Federated Averaging with Heterogeneous Differential Privacy"(under review).

### Prerequisites

The essential packages for deploying the project:

* Tensorflow 2.x
  ```sh
  pip install tensorflow-gpu
  ```

* Tensorflow Privacy
  ```sh
  pip install tensorflow-privacy
  ```
  or
  ```sh
  git clone https://github.com/tensorflow/privacy
  ```
 
### Installation

* Clone the repo
   ```sh
   git clone https://github.com/Emory-AIMS/PFA.git
   ```

<!-- USAGE EXAMPLES -->
## Usage

Note that we omit the basic arguments such as `dataset`, `model`, `lr`, etc. And default Values have been set for these arguments.

* NP-FedAvg algorithm:
  ```python
  python main.py --fedavg True
  ```
* FedAvg with HDP algorithm:
  ```python
  python main.py --dpsgd True --eps mixgauss1 --fedavg True
  ```
* WeiAvg algorithm experiments
  ```python
  python main.py --dpsgd True --eps mixgauss1 --weiavg True
  ```
* PFA algorithm
  ```python
  python main.py --dpsgd True --eps mixgauss1 --proj_wavg True --proj_dims 1 --lanczos_iter 256
  ```
* PFA+ algorithm
  ```python
  python main.py --dpsgd True --eps mixgauss1 --proj_wavg True --delay True --proj_dims 1 --lanczos_iter 256
  ```

<!-- CONTACT -->
## Contact

Junxu Liu - junxu_liu@ruc.edu.cn

Project Link: [https://github.com/JunxuLiu/PFA](https://github.com/JunxuLiu/PFA)



<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
* [Tensorflow/Privacy](https://github.com/tensorflow/privacy.git)
* [SAP-samples/machine-learning-diff-private-federated-learning](https://github.com/SAP-samples/machine-learning-diff-private-federated-learning.git)



