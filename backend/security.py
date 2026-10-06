"""Explicit browser origins and optional local-backend access protection."""
import hmac
from urllib.parse import urlsplit

from starlette.responses import JSONResponse


class BackendAccess:
    def __init__(self, app, token=None, origins=()):
        self.app, self.token, self.origins = app, token, set(origins)

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or not scope['path'].startswith('/api/'):
            return await self.app(scope, receive, send)
        headers = dict(scope['headers'])
        origin = headers.get(b'origin', b'').decode('latin1')
        host = headers.get(b'host', b'').decode('latin1')
        same_origin = origin in (f'http://{host}', f'https://{host}')
        # Without a token, accept only local browser origins (also prevents DNS rebinding).
        local_origin = urlsplit(origin).hostname in ('localhost', '127.0.0.1', '::1')
        if origin and origin not in self.origins and not (same_origin and (self.token or local_origin)):
            return await JSONResponse({'detail': 'Browser origin is not allowed'}, status_code=403)(scope, receive, send)
        supplied = headers.get(b'authorization', b'')
        if self.token and not hmac.compare_digest(supplied, ('Bearer ' + self.token).encode()):
            return await JSONResponse({'detail': 'Enter the backend access token shown in its terminal'}, status_code=401)(scope, receive, send)
        await self.app(scope, receive, send)
