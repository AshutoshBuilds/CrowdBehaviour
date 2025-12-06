IST 06-Dec-2025 17:01:54 - Added multi-backbone video model comparison (ResNet/EfficientNet/ViT/DenseNet/MobileNet/custom), F1 tracking, augmentation toggle, and benchmark summary output.
IST 06-Dec-2025 17:17:52 - Fixed loader error guard, restored ImageNet normalization for inference, and corrected augmentation jitter params.
IST 06-Dec-2025 17:28:08 - Added metadata header to validation sample predictions for automated parsing and context.
IST 06-Dec-2025 17:35:10 - Auto-generate metadata headers for sample predictions (val/test) with model, dataset, split, metrics, and purpose notes from evaluate pipeline.

IST 06-Dec-2025 17:43:22 - Re-evaluated val/test metrics with bootstrap CIs; updated docs metrics and plots.
IST 06-Dec-2025 17:45:05 - Deduplicated dataset globbing, re-ran val/test with bootstrap CIs; Panic support remains low (val n=2, test n=6).
IST 06-Dec-2025 17:53:27 - Ensured .cursor/mcp.json exists and reused n_classes when deriving resolved class names in evaluation.
IST 06-Dec-2025 18:17:38 - Hardened eval splits (seed/logging), enabled stratified test-set bootstrap CIs, surfaced split sizes in artifacts, and raised default val/test ratios for better reliability.
IST 06-Dec-2025 18:27:51 - Guarded dataset label retrieval in build_loaders with get_labels fallback and clearer errors for missing labels.
IST 06-Dec-2025 18:33:59 - Forced Torch weight downloads into project-root models/ directory (via TORCH_HOME) and documented the new cache location.
