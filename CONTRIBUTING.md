# Contributing

Contributions are welcome.

## Development principles

- Keep model-specific code inside `tabtester/backends/`.
- Keep the Streamlit UI backend-agnostic where practical.
- Do not commit pretrained model weights, private datasets, tokens, or local cache files.
- Preserve upstream licensing notices and do not imply that Tabtester's MIT License covers third-party weights.
- Add a lightweight test for reusable preprocessing or registry logic when possible.

## Basic checks

```bash
python -m compileall app.py tabtester scripts tests
python -m unittest discover -s tests -v
```
