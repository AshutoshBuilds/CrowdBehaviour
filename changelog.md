IST 06-Dec-2025 17:01:54 - Added multi-backbone video model comparison (ResNet/EfficientNet/ViT/DenseNet/MobileNet/custom), F1 tracking, augmentation toggle, and benchmark summary output.
IST 06-Dec-2025 17:17:52 - Fixed loader error guard, restored ImageNet normalization for inference, and corrected augmentation jitter params.
IST 06-Dec-2025 17:28:08 - Added metadata header to validation sample predictions for automated parsing and context.
IST 06-Dec-2025 17:35:10 - Auto-generate metadata headers for sample predictions (val/test) with model, dataset, split, metrics, and purpose notes from evaluate pipeline.

IST 06-Dec-2025 17:43:22 - Re-evaluated val/test metrics with bootstrap CIs; updated docs metrics and plots.
IST 06-Dec-2025 17:45:05 - Deduplicated dataset globbing, re-ran val/test with bootstrap CIs; Panic support remains low (val n=2, test n=6).
IST 06-Dec-2025 17:53:27 - Ensured .cursor/mcp.json exists and reused n_classes when deriving resolved class names in evaluation.
