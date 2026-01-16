import lightning as L
from omegaconf import OmegaConf
import hydra
from model.metric_ground import MetricGround

class MetricGroundModule(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.model = MetricGround(**cfg.model)
        self.cfg = cfg

    def training_step(self, batch, batch_idx):
        outputs, distill_logits = self.model(
            batch['images'], batch['points'], batch['texts'], return_distill=True
        )
        loss_ce = outputs.loss
        loss_distill = F.mse_loss(distill_logits, batch['metric_targets'])
        loss = loss_ce + self.cfg.lambda_distill * loss_distill
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.cfg.lr)

@hydra.main(config_path="config", config_name="default")
def main(cfg):
    model = MetricGroundModule(cfg)
    trainer = L.Trainer(**cfg.trainer)
    trainer.fit(model)

if __name__ == "__main__":
    main()
