# scaling-from-scratch — showcase site

A single-page, self-contained showcase of the three ML-systems projects in this
repo, with interactive [Plotly](https://plotly.com/javascript/) charts of the
real results. Built to be shared as a link (e.g. on LinkedIn).

- **`index.html`** — the whole site (HTML + CSS + charts, one file)
- **`assets/`** — static figures + the Open Graph preview image

The three projects live on their own branches:
- [`mlx-tiny-transformer`](https://github.com/jadoon200/scaling-from-scratch/tree/mlx-tiny-transformer) — transformer + Chinchilla scaling laws
- [`mlx-quantization`](https://github.com/jadoon200/scaling-from-scratch/tree/mlx-quantization) — INT8/INT4 quantization + the kernel lesson
- [`mlx-metal-kernels`](https://github.com/jadoon200/scaling-from-scratch/tree/mlx-metal-kernels) — six custom Metal kernels

## View locally
```bash
python -m http.server 8000      # from this branch's root
# open http://localhost:8000
```

## Publish on GitHub Pages
1. Push this branch: `git push -u origin site`
2. Repo **Settings → Pages → Build and deployment**: Source = *Deploy from a branch*,
   Branch = **`site`**, folder = **`/ (root)`**, Save.
3. Live at `https://jadoon200.github.io/scaling-from-scratch/` in ~1 min.

The Open Graph tags point at that URL's `assets/og.png`, so LinkedIn/Twitter
link previews show the kernel-bandwidth chart. If you publish under a different
URL, update the `og:image` / `og:url` meta tags in `index.html`.
