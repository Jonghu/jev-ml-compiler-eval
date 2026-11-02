# Local benchmark data

Store downloaded and generated benchmark data under this directory:

```text
data/
├── tenset/
└── tpugraphs/
```

Benchmark data is excluded from Git. Each workstream must document its download
source, version, checksums when available, preprocessing commands, selected
subsets, and train/test split construction in its own committed documentation or
scripts.

Do not place API keys, access tokens, or other credentials in this directory.
