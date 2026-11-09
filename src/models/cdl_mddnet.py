"""CDL-MDDNet: lightweight, subject-independent, interpretable EEG model for MDD detection.

Owner: Member 1.

Handoff contract (do not change without telling everyone):
    model = CDLMDDNet(n_channels, n_samples=1000, n_classes=2)
    logits = model(x)            # x: (B, C, 1000) float32 -> logits: (B, 2)

Pipeline (follows deck slide 36):
    EEG epoch (B, C, T)
      -> spatial convolution            mixes information across electrodes
      -> multi-scale temporal conv      short / medium / long kernels (frequency bands)
      -> depthwise-separable conv       cheap high-level features
      -> attention pooling over time    weights informative time regions
      -> fully connected -> logits      softmax is applied by the loss / by predict.py

Domain generalization (default proposal): a subject-adversarial head trained
through a gradient-reversal layer (DANN). The head only exists during training.
Use `forward_train(x, lambd)` to get (class_logits, subject_logits).
"""
import math

import torch
import torch.nn as nn
from torch.autograd import Function


# --------------------------------------------------------------------------- #
# Gradient reversal (domain-generalization part)
# --------------------------------------------------------------------------- #
class _GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd: float = 1.0):
    return _GradReverse.apply(x, lambd)


def dann_lambda(progress: float, gamma: float = 10.0) -> float:
    """Standard DANN schedule. progress = current_step / total_steps in [0, 1]."""
    return 2.0 / (1.0 + math.exp(-gamma * progress)) - 1.0


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class CDLMDDNet(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_samples: int = 1000,
        n_classes: int = 2,
        n_subjects: int = 0,          # >0 enables the subject-adversarial head
        f_spatial: int = 16,
        f_branch: int = 12,           # filters per temporal branch
        kernels=(15, 31, 63),         # ~60 / 124 / 252 ms at 250 Hz
        f_point: int = 48,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.n_classes = n_classes
        self.n_subjects = n_subjects

        # 1) spatial convolution: (B,1,C,T) -> (B,F,1,T)
        self.spatial = nn.Sequential(
            nn.Conv2d(1, f_spatial, kernel_size=(n_channels, 1), bias=False),
            nn.BatchNorm2d(f_spatial),
            nn.ELU(),
        )

        # 2) multi-scale temporal convolution: three parallel branches
        self.branches = nn.ModuleList(
            [
                nn.Conv2d(f_spatial, f_branch, kernel_size=(1, k), padding=(0, k // 2), bias=False)
                for k in kernels
            ]
        )
        f_multi = f_branch * len(kernels)
        self.multi_bn = nn.BatchNorm2d(f_multi)
        self.multi_act = nn.ELU()
        self.multi_pool = nn.AvgPool2d((1, 4))
        self.multi_drop = nn.Dropout(dropout)

        # 3) depthwise-separable convolution
        self.dw = nn.Conv2d(f_multi, f_multi, kernel_size=(1, 16), padding=(0, 8),
                            groups=f_multi, bias=False)
        self.pw = nn.Conv2d(f_multi, f_point, kernel_size=1, bias=False)
        self.sep_bn = nn.BatchNorm2d(f_point)
        self.sep_act = nn.ELU()
        self.sep_pool = nn.AvgPool2d((1, 4))
        self.sep_drop = nn.Dropout(dropout)

        # 4) attention pooling over time
        self.attn = nn.Conv1d(f_point, 1, kernel_size=1)

        # 5) classifier
        self.classifier = nn.Linear(f_point, n_classes)

        # 6) domain generalization: subject-adversarial head (training only)
        if n_subjects > 0:
            self.subject_head = nn.Sequential(
                nn.Linear(f_point, 32), nn.ReLU(), nn.Linear(32, n_subjects)
            )
        else:
            self.subject_head = None

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------ #
    def features(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, T) -> (B, f_point) subject-agnostic feature vector."""
        x = x.unsqueeze(1)                                   # (B,1,C,T)
        x = self.spatial(x)                                  # (B,F,1,T)
        x = torch.cat([b(x) for b in self.branches], dim=1)  # (B,3*fb,1,T)
        x = self.multi_drop(self.multi_pool(self.multi_act(self.multi_bn(x))))
        x = self.pw(self.dw(x))
        x = self.sep_drop(self.sep_pool(self.sep_act(self.sep_bn(x))))
        x = x.squeeze(2)                                     # (B,f_point,T')
        w = torch.softmax(self.attn(x), dim=-1)              # (B,1,T')
        return (x * w).sum(dim=-1)                           # (B,f_point)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Contract: returns class logits of shape (B, n_classes)."""
        return self.classifier(self.features(x))

    def forward_train(self, x: torch.Tensor, lambd: float = 1.0):
        """Returns (class_logits, subject_logits). Used by Member 3's train.py.

        loss = CE(class_logits, y) + CE(subject_logits, subject_idx)
        The gradient reversal makes the encoder *unlearn* subject identity.
        subject_idx must be 0..n_subjects-1 (re-map the training subjects per fold).
        """
        f = self.features(x)
        cls = self.classifier(f)
        if self.subject_head is None:
            return cls, None
        return cls, self.subject_head(grad_reverse(f, lambd))


def count_parameters(model: nn.Module, include_subject_head: bool = True) -> int:
    n = 0
    for name, p in model.named_parameters():
        if not include_subject_head and name.startswith("subject_head"):
            continue
        if p.requires_grad:
            n += p.numel()
    return n


if __name__ == "__main__":
    for c in (128, 64, 32, 16):
        m = CDLMDDNet(n_channels=c, n_subjects=53)
        y = m(torch.randn(4, c, 1000))
        print(c, "channels ->", tuple(y.shape), count_parameters(m), "params",
              f"(inference only: {count_parameters(m, False)})")
