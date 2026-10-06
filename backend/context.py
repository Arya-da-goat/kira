"""Memory and tool context are transient input text, never training updates."""


def build_prompt(request):
    parts = []
    if request.memory:
        parts.append('User-provided memory:\n' + request.memory)
    for source in request.context:
        parts.append('User-provided tool or file context:\n' + source)
    if request.messages:
        parts.append('Conversation:\n' + '\n'.join(f'{message.role}: {message.content}' for message in request.messages))
    if parts:
        parts.append('user: ' + request.prompt + '\nassistant:')
        return '\n\n'.join(parts)
    return request.prompt
