# llm3090

Local LLM serving for a single RTX 3090, with a browser control panel.

A [llama.cpp](https://github.com/ggml-org/llama.cpp) engine, a web UI on
loopback that starts and stops it and downloads models, and a curated model
list where every entry has been checked to fit 24 GB with a usable context left
over.

The engine exposes **both** the OpenAI API and Anthropic's `/v1/messages`, so
Claude Code and OpenAI-compatible clients work against it without a translation
proxy.

::::{grid} 2
:gutter: 3

:::{grid-item-card} {material-regular}`school;2em` Tutorials
:link: tutorials
:link-type: doc

Start here. Install the stack and serve your first model.
:::

:::{grid-item-card} {material-regular}`directions;2em` How-to guides
:link: how-to
:link-type: doc

Recipes for specific jobs — Claude Code, remote access, your own GGUF files.
:::

:::{grid-item-card} {material-regular}`menu_book;2em` Reference
:link: reference
:link-type: doc

The CLI, the HTTP API, the model catalogue and its fields.
:::

:::{grid-item-card} {material-regular}`lightbulb;2em` Explanations
:link: explanations
:link-type: doc

Why the project is scoped to one GPU, and what actually decides
whether a model fits.
:::

::::

```{toctree}
:hidden:

tutorials
how-to
reference
explanations
```
