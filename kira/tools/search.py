"""Optional encyclopedia web search. This tool never generates model responses."""
from html import unescape
import json
import re
from urllib.parse import urlencode, quote
from urllib.request import Request, urlopen


def search_web(query: str, limit: int = 5) -> list[dict]:
    parameters = urlencode({'action': 'query', 'list': 'search', 'srsearch': query,
                            'srlimit': limit, 'format': 'json', 'utf8': 1})
    request = Request('https://en.wikipedia.org/w/api.php?' + parameters,
                      headers={'User-Agent': 'KiraLocal/0.2 (user-initiated encyclopedia search)'})
    with urlopen(request, timeout=8) as response:
        data = json.loads(response.read(1_000_000))
    return [{'title': item['title'],
             'url': 'https://en.wikipedia.org/wiki/' + quote(item['title'].replace(' ', '_')),
             'snippet': unescape(re.sub(r'<[^>]*>', '', item['snippet']))}
            for item in data['query']['search']]
