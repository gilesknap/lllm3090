# What makes it fast

Every trick in this project for making a local model quicker, explained without
assuming you already know the vocabulary. [](going-faster.md) reports what each
one measured here; this page says what they *are*.

You need one idea first, because everything else hangs off it.

## Two clocks

When you send a model a message, it does two quite different jobs.

**Reading your prompt** is called *prefill*. The model looks at everything you
sent — your question, the file you pasted, the whole conversation so far — and
builds a working memory of it. It can read the entire prompt at once, in
parallel, because all the words are already known. This is arithmetic-heavy
work, and a GPU is very good at it.

**Writing its reply** is called *decode*. Now the model has to produce one word
at a time, and it cannot start the second word until the first exists, because
each word depends on the one before. There is nothing to do in parallel.

That difference matters more than it sounds, because of *why* decode is slow. To
produce a single word, the model reads **every one of its parameters** — for the
27B model here, about 16 GB of numbers — out of the graphics card's memory. Not
computes with them. Just *reads* them. The 3090 can move roughly 936 GB per
second, so 16 GB per word puts a hard ceiling around 58 words a second no matter
how clever the arithmetic is.

So: **prefill is limited by how fast the card can calculate; decode is limited
by how fast it can read memory.** A trick that helps one usually does nothing at
all for the other, and confusing them is the most common way to waste a day.

:::{note}
This is also why the sparse models here are faster than the dense ones despite
being *bigger* — they only read a fraction of themselves per word. That is a
whole argument of its own: [](dense-vs-moe.md).
:::

## Guessing ahead: speculative decoding

Decode wastes the card. Reading 16 GB to produce one word leaves almost all the
GPU's arithmetic capacity idle — it is waiting on memory, not thinking.

So: what if we guessed?

Suppose something cheap and fast guesses the next few words. Now the big model
can check all of those guesses **in a single pass**, because checking words you
already have is the parallel kind of work — the same shape as prefill. It reads
its 16 GB once and confirms or rejects, say, four words instead of producing one.

If the guesses were right, you got four words for the price of one. If they were
wrong, you throw them away and you have lost only a little time. And critically,
**the output is identical either way** — the big model still decides every word.
Speculation never changes what the model says, only how quickly it says it. A
wrong guess is discarded, not smuggled through.

This is *speculative decoding*, and it is the single biggest lever here. The
question is only ever: where do the guesses come from?

### Where guesses come from

**The model guessing for itself (MTP).** Some models ship with a small extra
piece bolted on, trained alongside the model, whose only job is to guess the next
few words. It is called a *multi-token prediction head*. It costs almost nothing
to run and it guesses well, because it was trained on the same data as the model
it is guessing for. When a checkpoint has one, this project turns it on
automatically — it reads the file to check rather than trusting a label, because
passing the flag to a model without the head makes the engine refuse to start.

Both front ends say which models those are: an `MTP` badge on the row in the
panel, `+mtp` in the `KIND` column of `lllm3090 models`. Read from the file
where the model is downloaded and from the catalogue where it is not, which is
the same order the engine decides in. Where the two disagree — a quantiser that
stripped the head, a repo whose name says MTP and whose weights do not — the
row says so rather than picking one, because the disagreement changes both what
runs and how much context it leaves.

**A separate small model (a "draft model").** Run a much smaller model of the
same family to produce guesses, and let the big one check them. This works, but
you now hold two models in memory, and the small one's memory is memory the big
one's conversation cannot use.

**DFlash2** is a sophisticated version of this: a purpose-built guesser for one
specific model, which proposes a whole block of words at once rather than one
after another. Its published numbers are excellent. Measured here it did not beat
the free MTP head, and it costs 1.1 GB — about a fifth of the conversation
length. Good idea, wrong machine.

**Copying from what you already said (prompt lookup).** If the answer is going to
repeat a chunk of the input — you asked it to rewrite a file with one name
changed — then the guesses can just be copied out of your prompt. No model
needed. This sounds free and it is very cheap, but the guesses are only good when
the output really is repetitive; the rest of the time they are wrong and the
checking is wasted.

This one turns out to depend entirely on the backend, which is the clearest
illustration on this page of why that matters. On the Vulkan engine that ships
here it is a *slowdown on every workload tried*, including the copy-heavy one it
was designed for. On CUDA, with everything else identical, it is **1.4× on
copying** — and stacked with the MTP head it becomes the fastest configuration
measured on this hardware. Same guesses, same proportion accepted; the
difference is only what the card charges to check them.

### How many to guess: draft width

If guessing four words at a time is good, why not forty?

Because every guess you make has to be checked, and checking costs work whether
the guess was right or not. Guess too many and you spend more time checking
rubbish than you saved. There is also a compounding problem: the guesser's
fourth word depends on its own first three, so accuracy falls off the further
ahead it reaches.

The number of words guessed per round is the *draft width*, and it turns out to
matter a great deal — on this hardware, more than which guesser you use. It also
turns out to depend on the **backend**, which is the next idea: guessing seven
ahead costs 22% of your speed on Vulkan and 8% on CUDA, with *the same
proportion of guesses thrown away* on both. The wasted guessing is the guesser's
fault; the price of it is the backend's.

## How the GPU is driven: backends

A model does not talk to a graphics card directly. It goes through a *backend* —
a library that translates "multiply these matrices" into instructions the card
understands. llama.cpp has several, and two matter here:

- **CUDA** is NVIDIA's own. It is the fastest and best-tuned, but it is a 4–6 GB
  developer toolkit that has to be installed and then used to *compile* the
  engine on your machine. There is no ready-made download for Linux.
- **Vulkan** is a vendor-neutral graphics standard that works on almost any card.
  llama.cpp publishes ready-built Vulkan binaries, so it installs by downloading
  a file. This project uses it for exactly that reason.

Measured here with no guessing turned on, CUDA is about 1.3× faster on both
clocks. Worth knowing, but not the transformation it is sometimes described as —
this page's companion used to claim 3–4× on prefill, and that was wrong.

The interesting difference is subtler. **Vulkan gets no faster when you give it
more work to do at once.** Asked to read 512 words it manages ~1027 a second;
asked to read 4096 it manages ~1014 — no better. CUDA improves by 10% on the
same change.

That single fact explains the draft-width result. Checking guesses is exactly
"more work at once" — so on a backend that gets nothing from batching, wider
guessing is close to pure cost.

**Which means the 1.3× understates it, and by a lot.** Every guessing verdict
here was re-measured on CUDA to test that reasoning, and it held: guessing is
worth much more on a backend that batches well. The MTP head is worth 1.6× on
Vulkan and **2.0× on CUDA** — with the same guesses accepted at the same rate,
so the only thing that changed is the price of checking. Since the engine always
has guessing switched on, what a user would actually feel from CUDA is about
**1.6×**, not 1.3×; on copy-heavy work, where the extra levers pay too, close to
2×.

The general lesson is worth more than the specific numbers: a backend comparison
run with the features switched off can be badly wrong about the machine you
actually use, because the features and the backend multiply rather than add.

## Making the conversation fit: the KV cache

While reading your prompt, the model builds a summary of every word, and keeps
it so it does not have to re-read the conversation for each new word. That store
is the **KV cache**, and it is the reason a long conversation eventually stops
fitting: it grows with every word, and it lives in the same graphics memory as
the model.

You can shrink it by storing those numbers less precisely — *quantising* it.
This project stores them at 8 bits instead of 16, which halves the cache and
roughly doubles how long a conversation can get, for no measurable quality cost.

Going further, to 4 bits, saves a lot more — and costs speed, because the numbers
now have to be unpacked every time they are read, and that unpacking sits
directly in the slow path. **TurboQuant** is a research technique that claims to
get to 3 bits without that penalty, by computing on the compressed numbers
directly. It is not part of llama.cpp yet.

## Attention, and why it is "flash"

For each new word, the model compares it against every previous word to work out
what to pay attention to. Done naively, this means writing out a large grid of
comparisons to memory and reading it back — and, as established, memory traffic
is the thing that hurts.

**Flash attention** computes that grid in small tiles that stay in the chip's
fast local memory, never writing the whole thing out. Same answer, far less
traffic. It is on by default here.

*Cooperative matrix* extensions ("coopmat") are a related idea: a way for the
backend to reach the card's dedicated matrix-multiplication hardware — the
"tensor cores". Version 2 is substantially faster than version 1 for prefill,
and which one you get depends on your driver. On the current build there is
unfortunately no way to tell from the logs which is in use.

## Serving many people at once

**vLLM** and **SGLang** are alternative serving engines, and on paper they
demolish llama.cpp — one comparison has them eight times faster. That number is
real but it answers a question this project is not asking: it is throughput with
*ten or more people* using the model simultaneously, achieved by batching their
requests together.

For one person at a keyboard there is nothing to batch, and the advantage mostly
evaporates. This is worth knowing mainly so that benchmark headlines can be read
correctly — "8× faster" almost always means "8× more total work across many
users", not "your reply arrives 8× sooner".

## Putting it together

For a single user on one graphics card, ranked by what actually moved the needle
here:

1. **A model that reads less per word** — a sparse model, or a smaller one.
   Nothing else comes close.
2. **Speculative decoding with a good guesser**, which for these models means the
   free MTP head. Worth about 1.6× on the engine that ships, 2.0× on CUDA.
3. **CUDA over Vulkan**, worth about 1.6× *once guessing is on* — for a toolkit,
   a compiler, and a binary that only runs on the card it was built for. It
   moved up this list when it was measured against the real configuration
   instead of a bare one.
4. **The right draft width**, which is worth more than picking a different
   guesser, and depends on your backend and your workload both.
5. **A quantised KV cache**, which does not make it faster but makes long
   conversations possible at all.

And one thing that is not on the list, because no lever touches it: the first
reply to a very long prompt is slow, and stays slow. That is prefill, it is
mostly what the card costs, and the only real fixes are to send less or to keep
the cache warm between turns.
