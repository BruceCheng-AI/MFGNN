# Code Release Checklist

- [ ] Add a real `LICENSE` file after choosing the code license.
- [ ] Keep the code repository limited to source files, release docs, and lightweight config files.
- [ ] Exclude `cache/`, `output/`, `model*/`, `*.pt`, `*.pth`, `*.npy`, and TensorBoard logs.
- [ ] Exclude all dataset files that will be published separately.
- [ ] Keep `StreetSZ/config.json` as a schema reference if you want the code repo to show the expected dataset format.
- [ ] Check that the scripts under `notebooks/scripts/` still run after removing local data and artifacts.
- [ ] Confirm the import layout still works, since the current code expects the `notebooks/` folder layout.
- [ ] Add exact training, evaluation, and ablation commands used for the paper or report.
- [ ] Replace any machine-specific paths or comments before publishing.
- [ ] Add repository topics, citation text, and acknowledgements before making the repo public.
