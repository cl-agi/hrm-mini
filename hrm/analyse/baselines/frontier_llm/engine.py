import json
def iter_sse(response):
    data = []
    for raw in response:
        line = raw.decode('utf-8').rstrip('\r\n')
        if line.startswith('data:'):
            value = line[5:]
            data.append(value[1:] if value.startswith(' ') else value)
        elif line == '' and data:
            yield '\n'.join(data)
            data = []
    if data:
        yield '\n'.join(data)

def build_payload(prompt, model, profile):
    settings = profile['call_settings']
    if profile.get('available') is not True:
        raise ValueError('Model profile is unavailable')
    if settings['max_completion_tokens'] > profile['max_supported_output_tokens']:
        raise ValueError('Requested output budget exceeds the selected endpoint limit')
    if settings['provider']['allow_fallbacks'] is not False or len(settings['provider']['only']) != 1:
        raise ValueError('Exactly one frozen provider is required')
    return {'model': model, 'messages': [{'role': 'user', 'content': prompt}], 'stream': True, **json.loads(json.dumps(settings)), 'debug': {'echo_upstream_body': True}}

def context_admission(prompt, profile):
    estimate = len(prompt.encode('utf-8')) + 4096
    output = profile['call_settings']['max_completion_tokens']
    fits = estimate + output <= profile['context_length']
    return {'method': 'utf8_bytes_plus_4096_framing_reserve_conservative_heuristic', 'exact_provider_token_count': False, 'input_admission_estimate': estimate, 'output_reserved': output, 'context_capacity': profile['context_length'], 'admission_passed': fits}
