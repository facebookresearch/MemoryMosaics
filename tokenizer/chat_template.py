# Copyright (c) Meta Platforms, Inc. and affiliates.
# See file LICENSE.txt in the main directory.

"""The chat format of fine-tuning (SFT) and its evaluation: plain-text role labels.

    [BOS] User: {q1}\\n\\nAssistant: {a1}\\n\\nUser: {q2}\\n\\nAssistant: {a2}[EOS]

The loss mask is True on each assistant answer and the final EOS, False elsewhere. The
EOS teaches the model to stop and marks the end of the conversation, the document
boundary that separates packed conversations.
"""

_ROLE_LABELS = {"system": "System", "user": "User", "assistant": "Assistant"}


def _turns(messages: list[dict], tokenizer):
    """(separator + label tokens, content tokens, role) per message. Every piece is
    encoded on its own: SentencePiece tokenizes "\n\nAssistant: " differently from
    "\n\n" followed by "Assistant: ", so training and generation must cut alike."""
    for i, msg in enumerate(messages):
        role = msg["role"]
        head = (tokenizer.encode("\n\n") if i > 0 else []) \
            + tokenizer.encode(f"{_ROLE_LABELS.get(role, role.capitalize())}: ")
        yield head, tokenizer.encode(msg["content"]), role


def apply_chat_template(messages: list[dict], tokenizer) -> tuple[list[int], list[bool]]:
    """(tokens, loss mask) of a conversation, a list of {"role", "content"} ending
    with the assistant."""
    tokens: list[int] = [tokenizer.bos_id]
    mask: list[bool] = [False]
    for head, content, role in _turns(messages, tokenizer):
        tokens += head + content
        mask += [False] * len(head) + [role == "assistant"] * len(content)
    tokens.append(tokenizer.eos_id)
    mask.append(True)
    return tokens, mask


def build_generation_prompt(messages: list[dict], tokenizer) -> list[int]:
    """Tokens of a conversation up to the start of the next assistant answer, for
    generation: exactly the training tokens that precede that answer. A trailing
    assistant message, if any, is dropped (it is the answer)."""
    if messages and messages[-1]["role"] == "assistant":
        messages = messages[:-1]
    tokens: list[int] = [tokenizer.bos_id]
    for head, content, _ in _turns(messages + [{"role": "assistant", "content": ""}], tokenizer):
        tokens += head + content
    return tokens
