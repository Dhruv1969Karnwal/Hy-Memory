# Multi-Speaker Dialogue Support

Hy Memory can tell the current user, other named real people and the AI assistant apart inside one dialogue. The capability uses `role` and `name` from the standard messages input and adds no participant table and no new Memory fields.

## 1. Input Conventions

The caller decides whose perspective is the "current user":

- `role="user"` is the user whose memory this is; the user's name may be supplied through `name`.
- `role="assistant"` with `name` is another named real person.
- `role="assistant"` without `name` is the AI assistant.
- For `role="tool"` the `name` is a tool name, not a speaker name.

For example, with Tim as the current user and John as the other real person in the dialogue:

```python
from hy_memory import HyMemoryClient

client = HyMemoryClient(mode="pro")
result = client.add(
    [
        {"role": "user", "name": "Tim", "content": "I work at Stripe."},
        {"role": "assistant", "name": "John", "content": "I am a nurse in Berlin."},
    ],
    user_id="tim",
)
```

Both OpenAI-style string messages and Anthropic-style content blocks keep the `name` of non-tool roles. A standalone OpenAI tool message still resolves `name` to the tool name.

## 2. Internal Rendering

An ordinary user-AI dialogue keeps the original format:

```text
[user]: I like coffee.
[assistant]: Noted.
```

A named multi-speaker dialogue renders as:

```text
The user's name is Tim
[user]: I work at Stripe.
[John]: I am a nurse in Berlin.
```

With Chinese message content the name line follows the message language and is written below as the `\uXXXX` escapes used in `hy_memory/pipelines/base.py`; the message lines keep the caller's own text and are shown here in translation:

```text
\u7528\u6237\u540d\u53ebTim
[user]: I work at Stripe.
[John]: I am a nurse in Berlin.
```

| Input | Rendered label | Semantics |
|---|---|---|
| `role=user` | `[user]` | Current user; the label stays even when `name` is set |
| `role=assistant, name=John` | `[John]` | Another named real person |
| `role=assistant` without `name` | `[assistant]` | AI assistant |
| `role=system/tool` | Original role | System or tool message |

The current user's name sits on its own line at the very start and appears exactly once per block. The language of that line is derived from the message content automatically.

## 3. Memory Ownership

The extraction chain produces facts by these rules:

- Facts about the current user take `The user ...`, or the zh equivalent, as the subject, with `owner="user"`.
- Facts about another named real person take that person's name as the subject, for example `John is a nurse in Berlin.`, with `owner="user"`.
- Content the AI assistant supplies or produces takes the assistant as the subject, with `owner="agent"`.

A named real person may arrive with `role="assistant"` in the input protocol, but that does not turn them into `owner="agent"`.

The L0 basic profile describes the current user only. The `name` of `role=user` is an authoritative input for `basic_info.name`; the occupation, location and similar details of other named real people belong in ordinary facts, not in the current user's L0.

The Reconcile prompt explicitly forbids UPDATE or SUPERSEDE between facts with different subjects. For example, `The user likes coffee` and `John likes coffee` stay as two separate facts.

## 4. History Context

What L1_RAW stores uses the rendering format above. When history is read back:

- The leading name line is recognized and deduplicated, and is never mistaken for a user message.
- Named labels such as `[John]:` are preserved.
- Multi-line messages are split on the speaker prefix, so a continuation line is never split off as a new message.
- Assistant and other named real person messages in history are truncated by the history context limit; user messages are kept in full.

## 5. Current Boundaries

- The caller must state the current user perspective through `role`; the SDK never guesses it from names.
- There is no separate identity model for named AI agents yet. `role=assistant + name` is read as a named real person.
- The speaker subject is carried inside the fact text rather than in a dedicated retrieval filter field.
- No separate L0 profile is created for other named real people. To maintain profiles for several people, write each perspective under its own `user_id`.

## 6. Implementation and Verification

Main implementation locations:

- `hy_memory/pipelines/base.py`: `ChatMessage.name`, speaker labels and dialogue rendering.
- `hy_memory/client.py`: OpenAI/Anthropic message normalization.
- `hy_memory/agent/extractor.py` and `hy_memory/agent/prompts_zh.py`: subject, owner and basic profile rules.
- `hy_memory/agent/reconciler.py`: cross-subject merge constraints.
- `hy_memory/pipelines/writer.py`: L1_RAW history parsing.

Targeted tests:

```bash
python -m pytest -q \
  tests/test_multi_speaker_format.py \
  tests/test_reconcile_fallback.py
```
