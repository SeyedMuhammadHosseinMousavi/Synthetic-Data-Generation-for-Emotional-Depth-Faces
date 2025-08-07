import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
import shutil
import time
from datetime import datetime
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter, ImageEnhance
# KD and GA imports
from deap import base, creator, tools, algorithms
import random

# --------- CLEAN PREVIOUS OUTPUT --------- #
SAVE_DIR = "kdgacdcgan_generated"
if os.path.exists(SAVE_DIR):
    shutil.rmtree(SAVE_DIR)

# --------- CONFIGURATION --------- #
IMG_WIDTH = 256
IMG_HEIGHT = 256
BATCH_SIZE = 32

EPOCHS = 500

LATENT_DIM = 256
NUM_CLASSES = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABELS = {'Fear': 0, 'Happiness': 1, 'Neutral': 2}

# --------- KD PARAMETERS --------- #
# KD Parameters
EMA_DECAY = 0.994
LAMBDA_G_KD = 1.0
LAMBDA_D_KD = 0.5

# --------- GA (Genetic Algorithm) PARAMETERS --------- #
GA_POP_SIZE = 20
GA_GENERATIONS = 100
GA_CXPB = 0.8
GA_MUTPB = 0.2
GA_NUM_TO_GENERATE = 3   # Number of best images to save per class

# --------- DATASET --------- #
class DepthFaceDataset(Dataset):
    def __init__(self, root_dir):
        self.images = []
        self.labels = []
        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        for label_name, label_idx in LABELS.items():
            class_dir = os.path.join(root_dir, label_name)
            for filename in os.listdir(class_dir):
                if filename.endswith('.png'):
                    img_path = os.path.join(class_dir, filename)
                    image = Image.open(img_path)
                    image = transform(image)
                    self.images.append(image)
                    self.labels.append(label_idx)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

# --------- CONDITIONAL GAN --------- #
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(LATENT_DIM + NUM_CLASSES, 512 * 16 * 16),
            nn.BatchNorm1d(512 * 16 * 16),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Unflatten(1, (512, 16, 16)),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 1, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, z, labels):
        c = F.one_hot(labels, num_classes=NUM_CLASSES).float()
        x = torch.cat([z, c], dim=1)
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1 + NUM_CLASSES, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),
            nn.Linear(512 * 16 * 16, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        c = F.one_hot(labels, num_classes=NUM_CLASSES).float()
        c = c.view(labels.size(0), NUM_CLASSES, 1, 1)
        c = c.expand(-1, -1, IMG_HEIGHT, IMG_WIDTH)
        x = torch.cat([img, c], dim=1)
        return self.model(x)

# --------- WEIGHT INIT --------- #
def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

# --------- KD HELPER --------- #
def update_ema_weights(model, ema_model, decay):
    for param_student, param_ema in zip(model.parameters(), ema_model.parameters()):
        param_ema.data.mul_(decay).add_(param_student.data, alpha=1 - decay)

# --------- GA LATENT OPTIMIZATION --------- #
if not hasattr(creator, "FitnessMaxLatent"):
    creator.create("FitnessMaxLatent", base.Fitness, weights=(1.0,))
    creator.create("IndividualLatent", np.ndarray, fitness=creator.FitnessMaxLatent)
latent_toolbox = base.Toolbox()
latent_toolbox = base.Toolbox()
latent_toolbox.register("attr_latent", lambda: np.random.randn(LATENT_DIM).astype(np.float32))
latent_toolbox.register("individual", tools.initIterate, creator.IndividualLatent, latent_toolbox.attr_latent)
latent_toolbox.register("population", tools.initRepeat, list, latent_toolbox.individual)
latent_toolbox.register("mate", tools.cxBlend, alpha=0.5)
latent_toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.3)
latent_toolbox.register("select", tools.selTournament, tournsize=3)


def evaluate_latent_vector(latent_vector_np, generator_model, target_class_idx):
    """
    Evaluates a latent vector by generating an image and calculating its fitness.
    Fitness is based on image statistics (std deviation and mean) to encourage diversity and content.
    """
    generator_model.eval()
    with torch.no_grad():
        latent_tensor = torch.tensor(latent_vector_np, dtype=torch.float32).to(DEVICE).unsqueeze(0)
        label_tensor = torch.tensor([target_class_idx], dtype=torch.long).to(DEVICE)
        gen_img = generator_model(latent_tensor, label_tensor)
        std_dev = gen_img.std().item()
        mean_val = gen_img.mean().item()
        fitness = std_dev * 10 - abs(mean_val) * 5
        return (fitness,)


def run_latent_space_optimization(generator_model, target_class_name,
                                  num_to_generate=1,
                                  pop_size=5, generations=3,
                                  cxpb=0.7, mutpb=0.3):
    """
    Performs latent space exploration using GA for a target class.
    After GA, generates and saves post-processed evolved samples per class.
    """
    import random

    target_class_idx = LABELS[target_class_name]
    latent_toolbox.register("evaluate", evaluate_latent_vector,
                            generator_model=generator_model, target_class_idx=target_class_idx)
    pop = latent_toolbox.population(n=pop_size)
    hof = tools.HallOfFame(num_to_generate, similar=np.array_equal)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    print(f"\n--- Latent Space Evolution for '{target_class_name}' ---")
    pop, log = algorithms.eaSimple(
        pop, latent_toolbox,
        cxpb=cxpb, mutpb=mutpb, ngen=generations,
        stats=stats, halloffame=hof, verbose=True)

    # Save GA best fitness per generation as .txt
    ga_logfile = os.path.join(SAVE_DIR, f"ga_loss_{target_class_name}.txt")
    with open(ga_logfile, 'w') as f:
        for gen, stat in enumerate(log):
            f.write(f"{gen+1},{stat['max']:.6f}\n")
    gens, vals = [], []
    for gen, stat in enumerate(log):
        gens.append(gen+1)
        vals.append(stat['max'])
    if vals:
        plt.figure(figsize=(6, 4))
        plt.plot(gens, vals, marker='o', label=f"GA Best Fitness ({target_class_name})")
        plt.title(f"GA Best Fitness per Generation ({target_class_name})")
        plt.xlabel("Generation")
        plt.ylabel("Best Fitness")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, f"ga_loss_{target_class_name}_plot.png"))
        plt.close()

    # Generate and save evolved images with post-processing
    evolved_dir = os.path.join(SAVE_DIR, "evolved", target_class_name)
    os.makedirs(evolved_dir, exist_ok=True)
    generator_model.eval()
    with torch.no_grad():
        for idx in range(num_to_generate):
            latent_vec = torch.randn(1, LATENT_DIM, device=DEVICE)
            label_vec = torch.tensor([target_class_idx], dtype=torch.long).to(DEVICE)
            gen_img = generator_model(latent_vec, label_vec)
            gen_img = 0.5 * (gen_img + 1)  # Scale to [0,1]
            # Save temporarily to memory
            temp_path = f"{evolved_dir}/_tmp_{target_class_name}_{idx}.png"
            save_image(gen_img[0], temp_path)
            # --- Post-processing (denoise and adjust contrast) ---
            pil_img = Image.open(temp_path)
            pil_img = pil_img.filter(ImageFilter.MedianFilter(size=3))  # Denoising
            enhancer = ImageEnhance.Contrast(pil_img)
            pil_img = enhancer.enhance(1.3)  # Adjust contrast, factor can be tuned
            pil_img.save(f"{evolved_dir}/{target_class_name}_evolved_{idx}.png")
            os.remove(temp_path)
            print(f"Evolved image saved with post-processing: {evolved_dir}/{target_class_name}_evolved_{idx}.png")


# --------- TRAINING (with KD and GA) --------- #
def train():
    # === Prepare dataset and dataloader ===
    dataset = DepthFaceDataset("SDG Depth Three")
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # === Initialize models and EMA (KD teachers) ===
    generator = Generator().to(DEVICE)
    discriminator = Discriminator().to(DEVICE)
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    generator_ema = Generator().to(DEVICE)
    discriminator_ema = Discriminator().to(DEVICE)
    generator_ema.load_state_dict(generator.state_dict())
    discriminator_ema.load_state_dict(discriminator.state_dict())
    generator_ema.eval()
    discriminator_ema.eval()

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.00008, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.000015, betas=(0.5, 0.999))

    adversarial_loss = nn.BCELoss()
    kd_loss_func = nn.MSELoss()

    # === Ask user for number of images ===
    print("Enter number of normal GAN images to generate per class:")
    num_to_generate_normal = int(input())
    print("Enter number of evolved images (GA) to generate per class:")
    num_to_generate_evolved = int(input())

    os.makedirs(SAVE_DIR, exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    losses_G, losses_D, losses_GKD, losses_DKD = [], [], [], []
    start_time = time.time()

    for epoch in range(EPOCHS):
        for i, (imgs, labels) in enumerate(dataloader):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            batch_size = imgs.size(0)
            valid = torch.ones((batch_size, 1), device=DEVICE)
            fake = torch.zeros((batch_size, 1), device=DEVICE)

            # === Generator update (with KD) ===
            optimizer_G.zero_grad()
            z = torch.randn(batch_size, LATENT_DIM, device=DEVICE)
            gen_imgs = generator(z, labels)
            g_adv_loss = adversarial_loss(discriminator(gen_imgs, labels), valid)
            with torch.no_grad():
                gen_imgs_ema = generator_ema(z, labels)
            g_kd_loss = F.l1_loss(gen_imgs, gen_imgs_ema.detach())
            g_loss = g_adv_loss + LAMBDA_G_KD * g_kd_loss
            g_loss.backward()
            optimizer_G.step()
            update_ema_weights(generator, generator_ema, EMA_DECAY)

            # === Discriminator update (with KD) ===
            optimizer_D.zero_grad()
            real_out = discriminator(imgs, labels)
            fake_out = discriminator(gen_imgs.detach(), labels)
            d_real_adv_loss = adversarial_loss(real_out, valid)
            d_fake_adv_loss = adversarial_loss(fake_out, fake)
            d_adv_loss = (d_real_adv_loss + d_fake_adv_loss) / 2
            with torch.no_grad():
                real_out_ema = discriminator_ema(imgs, labels)
                fake_out_ema = discriminator_ema(gen_imgs.detach(), labels)
            d_kd_loss_real = kd_loss_func(real_out, real_out_ema.detach())
            d_kd_loss_fake = kd_loss_func(fake_out, fake_out_ema.detach())
            d_kd_loss = (d_kd_loss_real + d_kd_loss_fake) / 2
            d_loss = d_adv_loss + LAMBDA_D_KD * d_kd_loss
            d_loss.backward()
            optimizer_D.step()
            update_ema_weights(discriminator, discriminator_ema, EMA_DECAY)

            if i % 10 == 0:
                print(f"[Epoch {epoch+1}/{EPOCHS}] [Batch {i}/{len(dataloader)}] "
                      f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}] "
                      f"[G KD: {g_kd_loss.item():.4f}] [D KD: {d_kd_loss.item():.4f}]")

        # Log losses for plotting
        losses_D.append(d_loss.item())
        losses_G.append(g_loss.item())
        losses_GKD.append(g_kd_loss.item())
        losses_DKD.append(d_kd_loss.item())

        # Log to file
        with open(os.path.join(SAVE_DIR, "training_log.txt"), "a") as f:
            f.write(f"{timestamp_str} | Epoch {epoch+1} | D: {d_loss.item():.4f} | G: {g_loss.item():.4f} | G_KD: {g_kd_loss.item():.4f} | D_KD: {d_kd_loss.item():.4f}\n")

        # On last epoch, generate normal GAN samples per class (different z every time!)
        if epoch == EPOCHS - 1:
            generator.eval()
            with torch.no_grad():
                for class_name, class_idx in LABELS.items():
                    class_dir = os.path.join(SAVE_DIR, class_name)
                    os.makedirs(class_dir, exist_ok=True)
                    for i in range(num_to_generate_normal):
                        z = torch.randn(1, LATENT_DIM, device=DEVICE)  # New z every time
                        label = torch.tensor([class_idx], device=DEVICE)
                        gen_img = generator(z, label)
                        gen_img = 0.5 * (gen_img + 1)
                        save_image(gen_img[0], os.path.join(class_dir, f"{class_name}_final_{i}.png"))

    # === Save trained models ===
    torch.save(generator.state_dict(), os.path.join(SAVE_DIR, "generator.pth"))
    torch.save(discriminator.state_dict(), os.path.join(SAVE_DIR, "discriminator.pth"))

    # === Plot and save training loss curves (G, D, G_KD, D_KD) ===
    plt.figure()
    plt.plot(range(1, EPOCHS + 1), losses_D, label="D Loss")
    plt.plot(range(1, EPOCHS + 1), losses_G, label="G Loss")
    plt.plot(range(1, EPOCHS + 1), losses_GKD, label="G KD Loss")
    plt.plot(range(1, EPOCHS + 1), losses_DKD, label="D KD Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CDC-GAN+KD Training Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(SAVE_DIR, "training_loss_plot.png"))
    plt.close()

    elapsed = time.time() - start_time
    print(f"✅ Training complete. Total time: {elapsed:.2f} seconds")

    # === GA Latent Space Optimization for each class (ask user how many evolved images to save) ===
    print("\n==== Running GA Latent Space Optimization ====")
    for class_name in LABELS.keys():
        run_latent_space_optimization(
            generator, class_name,
            num_to_generate=num_to_generate_evolved,
            pop_size=GA_POP_SIZE,
            generations=GA_GENERATIONS,
            cxpb=GA_CXPB,
            mutpb=GA_MUTPB
        )
    print(f"Generated {num_to_generate_evolved} evolved samples for ALL emotions in '{SAVE_DIR}/evolved/'.")

if __name__ == "__main__":
    train()
