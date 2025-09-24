import os
import matplotlib.pyplot as plt
from datetime import datetime

class LossLogger: ## TODO: needs improvement
    def __init__(self, log_dir="logs", run_name=None):
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"{run_name or 'training_loss'}_{timestamp}.txt")
        self.epoch_losses = []
        self.batch_losses = []

    def log_batch(self, epoch, batch_idx, loss):
        self.batch_losses.append(loss)
        with open(self.log_file, 'a') as f:
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                    f"Epoch {epoch}, Batch {batch_idx + 1} - Loss: {loss:.6f}\n")

    def log_epoch(self, epoch, total_loss, dataset_size):
        epoch_avg_loss = total_loss / dataset_size
        self.epoch_losses.append(epoch_avg_loss)
        with open(self.log_file, 'a') as f:
            f.write(f"Epoch {epoch} - Avg Loss: {epoch_avg_loss:.6f}\n\n")

    def save_plot(self):
        plt.figure(figsize=(10, 5))

        if self.batch_losses:
            plt.plot(self.batch_losses, label='Batch Loss', alpha=0.4)

        if self.epoch_losses:
            x = [i * len(self.batch_losses) // len(self.epoch_losses) for i in range(len(self.epoch_losses))]
            plt.plot(x, self.epoch_losses, label='Epoch Avg Loss', linewidth=2.5, marker='o')

        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plot_path = self.log_file.replace('.txt', '.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"Loss plot saved to: {plot_path}")
