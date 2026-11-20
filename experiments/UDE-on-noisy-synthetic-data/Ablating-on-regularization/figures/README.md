# Noisy synthetic data: regularization study figures

Placements are provisional. The previews combine existing panels for review; they do not define the final manuscript layout. Generator paths are relative to this folder.

| Figure Name | Placement | Question | Generator file path (short) |
|---|---|---|---|
| [Early stopping and recovery](early-stopping-and-recovery.pdf) | Main Fig. 2A–B | Does early stopping change typical recovery error or mainly its upper tail? What do trajectories in each error quartile look like? | [../analysis/early-stopping-and-recovery.jl](../analysis/early-stopping-and-recovery.jl) |
| [Regularization by sampling design](regularization-by-sampling-design.pdf) | Main Fig. 2C | How does median worst-channel error vary across regularization settings and sampling designs? | [../analysis/regularization-by-sampling-design.jl](../analysis/regularization-by-sampling-design.jl) |
| [Paired recovery error](sr-paired-recovery-error.pdf) | Main Fig. 3A | Is the selected expression closer to truth than its own network on the same predicted states? | [../analysis/sr-paired-recovery-error.jl](../analysis/sr-paired-recovery-error.jl) |
| [Search versus selection](sr-search-versus-selection.pdf) | Main Fig. 3B | As network error changes, how often does any seed find the true structure on a saved front, and how often does any seed select it? | [../analysis/sr-search-versus-selection.jl](../analysis/sr-search-versus-selection.jl) |
| [Recovery outcomes](sr-recovery-outcomes.pdf) | Main Fig. 3C | Across eight SR seeds per fit and channel, was the generator absent from every saved front, found but never selected, or selected by ≥1 seed? | [../analysis/sr-recovery-outcomes.jl](../analysis/sr-recovery-outcomes.jl) |
| [Selected structures](sr-selected-structures.pdf) | Fig. 4A candidate | Which structures recur, and how accurate are they? The current bars include all selections, not just alternatives to truth. | [../analysis/sr-selected-structures.jl](../analysis/sr-selected-structures.jl) |
| [Distillation error](sr-distillation-error.pdf) | Supplement to Fig. 3A | Does approximating the network more closely also mean approximating truth more closely? | [../analysis/sr-distillation-error.jl](../analysis/sr-distillation-error.jl) |
| [Recovery regression](sr-recovery-regression.pdf) | Supplement to Fig. 3B | Which design choices are associated with finding truth, and which with selecting it among fits that found it? | [../analysis/sr-recovery-regression.jl](../analysis/sr-recovery-regression.jl) |
| [Recovery and design preview](recovery-and-design-preview.pdf) | Layout preview: Fig. 2 | How do stopping, trajectories, and design sensitivity read together? | [../analysis/compose-recovery-and-design.jl](../analysis/compose-recovery-and-design.jl) |
| [Symbolic recovery preview](sr-recovery-preview.pdf) | Layout preview: Figs. 3–4 | How do paired error, selected structures, and search versus selection read together? | [../analysis/compose-sr-recovery.jl](../analysis/compose-sr-recovery.jl) |
