# Synthetic-Data-Generation-for-Emotional-Depth-Faces
Synthetic Data Generation for Emotional Depth Faces: Optimizing Conditional DCGANs via Genetic Algorithms in the Latent Space and Stabilizing Training with Knowledge Distillation

### Link to the data and trained models on the Kaggle: https://kaggle.com/datasets/5382fad0083f66167d9191f9392806f8b3d4f856b7b0ce11f3c71ed8545ccfb9

Affective computing faces a significant challenge: the scarcity of high-quality, diverse depth facial datasets crucial for understanding subtle emotional expressions. This data gap prevents the development of robust models for applications ranging from human-computer interaction to healthcare. To address this, we propose a novel framework for synthetic depth face generation using an optimized Generative Adversarial Network (GAN). First, we integrate Knowledge Distillation (KD), using Exponential Moving Average (EMA) teacher models to stabilize GAN training, enhance the quality and diversity of generated images, and help prevent mode collapse. More importantly, we extend the use of Genetic Algorithms (GAs) into the GAN's latent space. By evolving these latent vectors based on image statistics (standard deviation and mean) that correlate with visual quality and diversity, the GA helps to identify latent representations that yield high-quality and varied images for a given target emotion, thereby optimizing the visual output for specific emotional categories. Thus, GA acts as post-processing, which diversifies faces even more than the KD level. This comprehensive methodology not only mitigates data scarcity but also yields diverse images and promising results compared to other benchmark techniques such as GAN, VAE, Gaussian Mixture Models (GMM), and Kernel Density Estimation (KDE). For the machine learning aspect, we extract depth-related LBP, HOG, Sobel edge histogram, and intensity histogram features and concatenate them for the classification task. Metrics responsible for comparing synthetic and original samples, plus evaluating synthetic samples in terms of quality and diversity, include Fréchet Inception Distance (FID), Inception Score (IS), Structural Similarity Index Measure (SSIM), and Peak Signal-to-Noise Ratio (PSNR). Our method achieved 94% and 96% recognition accuracy with the XGBoost classifier and surpassed other state-of-the-art methods. Furthermore, our methods outperform other methods regarding all synthetic data evaluation metrics.


<img width="10624" height="5668" alt="training_progress" src="https://github.com/user-attachments/assets/c3b7abc6-52d6-48cb-8bd8-a08211f2c011" />

<img width="2332" height="783" alt="selected results" src="https://github.com/user-attachments/assets/df880bc8-d763-461b-aed3-33b3ab7ae84a" />

<img width="7559" height="6673" alt="RF violin plot" src="https://github.com/user-attachments/assets/c0064489-fb16-4f96-bb51-74adb12c55a4" />

<img width="28465" height="7292" alt="metrics_comparison_subplots" src="https://github.com/user-attachments/assets/18387ca0-f6a8-47d0-9907-abaae19a892f" />

<img width="14200" height="4013" alt="LDA KDE" src="https://github.com/user-attachments/assets/eaecce8e-a394-47bb-8944-5e5560e20023" />

<img width="14048" height="4724" alt="GA Loss" src="https://github.com/user-attachments/assets/eb79fe9f-823c-43fa-aae2-023a4a42f6e4" />

<img width="14368" height="4174" alt="feature_correlation_heatmap" src="https://github.com/user-attachments/assets/8a05c187-f859-43d4-9500-aca4ec207840" />

<img width="18439" height="4410" alt="accuracy_comparison_lineplot" src="https://github.com/user-attachments/assets/ecfc47c1-ead0-455c-b131-ebd9959ce28c" />

![t SNE Plot](https://github.com/user-attachments/assets/5d21a179-fa8d-462d-a23d-3f577e1a5a90)

