# FedDiff 
A pytorch implementation for the paper: Federated Conditional Latent Diffusion for Heterogeneous Cross-city Demand Prediction.

# Notes: 
The battery swap datasets from ten Chinese cities are now available. The FedDiff code is under organization and will be released within three weeks.

# Introduction
### Framework of FedDiff
<img src="https://github.com/UAV-Delta/FedDiff/blob/main/img/FedCrossCity.jpg" width="600" />

It involves three stages: (1) local
model training on each client; (2) global model aggregation on the
server; and (3) model convergence and demand prediction.

### Illustration of local diffusion model training on each client
<img src="https://github.com/UAV-Delta/FedDiff/blob/main/img/Framework.jpg" width="800" />

It consists of two phases: (1) the forward diffusion phase taking the latent representations extracted by the missingness-tolerant masked autoencoder as input; and (2) the reverse denoising phase conditioned on a UKG-based urban environment characterization.

# Installation
### Environment

### Dependencies
