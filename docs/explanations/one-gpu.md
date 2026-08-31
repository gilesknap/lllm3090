# Why this is scoped to one GPU

The installer checks for compute capability 8.6 and 24 GB, and warns if it does
not find them. That looks like artificial narrowness. It is not.

Everything this project offers is a claim about a specific card:

- **"This model fits"** — computed from 24576 MiB minus fixed reserves.
- **"You get 203k of context"** — the leftover after weights, divided by that
  model's KV cost per token.
- **"About 35 tokens per second"** — 936 GB/s divided by bytes read per token,
  or measured directly on this card.
- **"Use the Vulkan build"** — because prebuilt CUDA binaries for Linux do not
  exist, and the alternative is a 4–6 GB toolkit and a local compile for a
  measured 1.3×.

On a 16 GB card every "fits" is wrong. On a 5090 every speed is wrong and the
context figures are needlessly pessimistic. On an AMD card the engine binary is
wrong. The software would run in all three cases and quietly mislead you, which
is worse than refusing.

So the scope is a promise rather than a limitation: **if you have this card,
every number here was computed or measured for your machine.** The warning is
skippable — pass through it and the tool still works — but you are then on your
own for the arithmetic.

## What would it take to widen it

Nothing structural. `lllm3090.config` holds the envelope as constants and
`lllm3090.catalog.fit` does the arithmetic from them, so supporting another card
means a second profile and a second set of verified catalogue entries. The work
is not the code; it is re-verifying every model on real hardware, which needs
someone who owns one.
