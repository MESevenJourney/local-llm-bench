#!/usr/bin/env python3
"""Multi-turn, streaming TensorFold diagnostics. Standard library only."""
import argparse
import copy
import csv
import datetime
import json
import os
from pathlib import Path
import statistics
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parent


def make_request(history, model, tokens, temperature, thinking, budget, seed, draft, engine='tensorfold'):
    body = {'model': model, 'messages': copy.deepcopy(history), 'stream': True,
            'stream_options': {'include_usage': True}, 'max_tokens': tokens,
            'temperature': temperature, 'top_k': 20, 'top_p': .95, 'seed': seed,
            'chat_template_kwargs': {'enable_thinking': thinking},
            'thinking_budget': budget}
    if engine == 'tensorfold':
        body['draft'] = draft
    else:
        body['reasoning_effort'] = 'medium'
    return body


def load_replay(path):
    data = json.loads(Path(path).read_text())
    frozen = {}
    records = sorted(data['records'], key=lambda r: bool(r['metrics']['draft']))
    for record in records:
        key = (record['metrics']['scenario'], int(record['metrics']['turn']))
        messages = record['request']['messages']
        if not isinstance(messages, list) or not messages:
            raise ValueError(f'重放历史无效：{key}')
        frozen[key] = copy.deepcopy(messages)
    return frozen


def read_stream(response, started, clock=time.perf_counter):
    result = {'content': '', 'reasoning': '', 'finish_reason': None,
              'usage': {}, 'tensorfold': {}, 'speculative': {},
              'client_ttft_s': None, 'client_first_content_s': None}
    done = False
    last_generation = None
    for raw in response:
        line = raw.decode('utf-8').strip()
        if not line.startswith('data:'):
            continue
        data = line[5:].strip()
        if data == '[DONE]':
            done = True
            break
        chunk = json.loads(data)
        if chunk.get('error'):
            raise ValueError(f"服务端流式错误：{chunk['error']}")
        now = clock()
        for key in ('usage', 'tensorfold', 'speculative'):
            if chunk.get(key) is not None:
                result[key] = chunk[key]
        for choice in chunk.get('choices', []):
            delta = choice.get('delta') or {}
            content = delta.get('content') or ''
            reasoning = delta.get('reasoning_content') or ''
            if (content or reasoning) and result['client_ttft_s'] is None:
                result['client_ttft_s'] = now - started
            if content or reasoning:
                last_generation = now - started
            if content and result['client_first_content_s'] is None:
                result['client_first_content_s'] = now - started
            result['content'] += content
            result['reasoning'] += reasoning
            if choice.get('finish_reason'):
                result['finish_reason'] = choice['finish_reason']
    if not done or result['finish_reason'] is None:
        raise ValueError('响应流未完整结束；不能把部分回答计为成功。')
    result['client_seconds'] = clock() - started
    first = result['client_ttft_s']
    span = last_generation - first if first is not None and last_generation is not None else None
    count = result['usage'].get('completion_tokens')
    result['client_decode_s'] = span
    result['client_decode_tps'] = (count - 1) / span if span and span > 0 and count and count > 1 else None
    return result


def check_answer(content, expected):
    text = content.strip()
    if text.startswith('```') and text.endswith('```'):
        text = '\n'.join(text.splitlines()[1:-1])
    try:
        actual = json.loads(text)
    except ValueError:
        return {'passed': False, 'error': '最终回答不是可解析的 JSON', 'matched': 0, 'total': len(expected)}
    checks = {k: isinstance(actual, dict) and type(actual.get(k)) is type(v) and actual.get(k) == v
              for k, v in expected.items()}
    return {'passed': all(checks.values()), 'matched': sum(checks.values()),
            'total': len(checks), 'fields': checks, 'expected': expected, 'actual': actual}


def metric_row(scenario, turn, draft, result):
    usage, runtime, speculative = (result.get(k) or {} for k in ('usage', 'tensorfold', 'speculative'))
    drafted, accepted = speculative.get('drafted'), speculative.get('accepted')
    return {'scenario': scenario, 'turn': turn, 'draft': draft,
            'prompt_tokens': usage.get('prompt_tokens'),
            'cached_tokens': (usage.get('prompt_tokens_details') or {}).get('cached_tokens'),
            'completion_tokens': usage.get('completion_tokens'),
            'decode_tps': runtime.get('tokens_per_second'),
            'client_decode_tps': result.get('client_decode_tps'),
            'client_decode_s': result.get('client_decode_s'),
            'server_ttft_s': runtime.get('time_to_first_token'),
            'prefill_s': runtime.get('prefill_seconds'),
            'client_ttft_s': result['client_ttft_s'],
            'first_content_s': result['client_first_content_s'],
            'total_s': result['client_seconds'], 'accepted': accepted, 'drafted': drafted,
            'acceptance_rate': accepted / drafted if drafted and accepted is not None else None,
            'tokens_per_round': speculative.get('tokens_per_round'),
            'token_sha': runtime.get('token_sha'), 'finish_reason': result['finish_reason']}


def show(value):
    return 'N/A' if value is None else f'{value:.2f}' if isinstance(value, float) else str(value)


def save_report(folder, metadata, records, quality, pairs, error=None):
    payload = {'metadata': metadata, 'records': records, 'quality': quality, 'pairs': pairs, 'error': error}
    target = folder / 'results.json'
    temporary = folder / 'results.tmp'
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(target)
    rows = [record['metrics'] for record in records]
    if rows:
        with (folder / 'turns.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    args = metadata['arguments']
    lines = ['# 多轮场景测试', '', f"状态：{error or metadata['status']}", '',
             f"引擎：{args.get('engine', 'tensorfold')}；模式：{'固定历史重放' if args.get('replay_from') else '自然多轮'}。",
             '每轮携带完整可见对话历史；不自动压缩或截断历史。思考文本仅存档，不放入下一轮。',
             '以下是自然增长的实际 token 数，不代表已测满配置的上下文窗口。', '',
             '|场景|轮次|草稿|输入|缓存|输出|服务端 tok/s|客户端 tok/s|首事件秒|首正文秒|总耗时秒|结束|',
             '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for row in rows:
        keys = ('scenario', 'turn', 'draft', 'prompt_tokens', 'cached_tokens', 'completion_tokens',
                'decode_tps', 'client_decode_tps', 'client_ttft_s', 'first_content_s', 'total_s', 'finish_reason')
        lines.append('|' + '|'.join(show(row[k]) for k in keys) + '|')
    lines += ['', '## 汇总', '']
    for name in sorted({r['scenario'] for r in rows}):
        for draft in (True, False):
            group = [r for r in rows if r['scenario'] == name and r['draft'] == draft]
            if not group:
                continue
            speeds = [r['decode_tps'] for r in group if r['decode_tps'] is not None]
            lines.append(f"- {name} 草稿={draft}：{len(group)} 轮；解码速度中位数 "
                         f"{show(statistics.median(speeds) if speeds else None)} tok/s；"
                         f"截断 {sum(r['finish_reason'] == 'length' for r in group)} 次。")
    lines += ['', '## 结构化核对', '',
              '仅核对最后一轮的事实／诊断字段；不代表生成的补丁已通过执行测试。',
              '缩短轮数时不会执行最后一轮核对。生成代码仅保存，不自动执行。', '',
              '```json', json.dumps(quality, ensure_ascii=False, indent=2), '```', '',
              '客户端 tok/s = (completion_tokens - 1) / 首个到最后一个非空生成片段的时间。',
              '片段可能包含多个 token，因此这是流式近似值；包含思考，排除末尾 usage 的等待。',
              '跨引擎优先比较相同客户端口径；服务端指标单独保留，缺失显示 N/A。',
              '## 草稿对照与限制', '',
              '对照模式每轮固定同一历史、seed、采样和输出预算，轮流交换请求先后顺序。',
              '草稿开启的回答用于下一轮。若开关输出分歧，立即停止，避免后续比较不同对话。',
              '同一服务无法隔离两次请求的缓存：总耗时／首字速度不是严格的冷缓存 A/B。',
              '应同时查看 cached_tokens；解码对照也可能受温度、负载和执行顺序影响。',
              '未测量峰值 GPU 内存，未验证真实工具执行或真实代码仓库任务。', '',
              '```json', json.dumps(pairs, ensure_ascii=False, indent=2), '```']
    (folder / 'report.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description='连续多轮聊天与代码排查测试，连接已启动的本地服务。')
    parser.add_argument('--scenario', choices=['chat', 'coding', 'all'], default='all')
    parser.add_argument('--engine', choices=['tensorfold', 'omlx'], default='tensorfold')
    parser.add_argument('--api-key-env', default='INFERENCE_API_KEY', help='认证密钥的环境变量名，值不会存入报告')
    parser.add_argument('--replay-from', type=Path, help='从已有 results.json 冻结每轮 messages；不延续本次回答')
    parser.add_argument('--turns', type=int, default=8, help='每场景前 N 轮，1 到 8')
    parser.add_argument('--tokens', type=int, default=1024, help='每轮思考＋回答输出上限，显式覆盖服务默认值')
    parser.add_argument('--compare-drafts', action='store_true', help='每轮增加同历史的关闭草稿对照')
    parser.add_argument('--no-drafts', action='store_true', help='只运行串行解码')
    parser.add_argument('--thinking', choices=['auto', 'on', 'off'], default='auto')
    parser.add_argument('--thinking-budget', type=int, default=None)
    parser.add_argument('--temperature', type=float, default=None)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--port', type=int, default=int(os.environ.get('PORT', '8081')))
    parser.add_argument('--model', default=os.environ.get('MODEL_NAME', 'qwen38-local'))
    parser.add_argument('--timeout', type=float, default=600)
    parser.add_argument('--label', default='', help='记录本次服务器配置，如 ctx32k-cache2')
    parser.add_argument('--dry-run', action='store_true', help='只预览场景和参数，不发送请求')
    args = parser.parse_args()
    if not 1 <= args.turns <= 8 or args.tokens < 1 or args.timeout <= 0 or not 1 <= args.port <= 65535:
        parser.error('轮数需在 1 到 8，tokens/timeout 为正，端口为 1 到 65535')
    if args.compare_drafts and args.no_drafts:
        parser.error('--compare-drafts 与 --no-drafts 不能同时使用')
    if args.engine == 'omlx' and args.compare_drafts:
        parser.error('oMLX 草稿需在部署时切换并重新加载；不能使用 --compare-drafts，请分别启动 draft/serial')
    if args.thinking_budget is not None and not 0 <= args.thinking_budget < args.tokens:
        parser.error('思考预算应 >=0 且小于总输出上限')
    scenarios = json.loads((ROOT / 'scenarios.json').read_text())
    names = list(scenarios) if args.scenario == 'all' else [args.scenario]
    plan = {}
    for name in names:
        scenario = scenarios[name]
        thinking = scenario['thinking'] if args.thinking == 'auto' else args.thinking == 'on'
        budget = args.thinking_budget if args.thinking_budget is not None else scenario['thinking_budget']
        if thinking and budget >= args.tokens:
            parser.error('总输出上限必须大于场景思考预算；增加 --tokens 或使用 --thinking off')
        plan[name] = {'temperature': scenario['temperature'] if args.temperature is None else args.temperature,
                      'thinking': thinking, 'thinking_budget': budget,
                      'turns': scenario['turns'][:args.turns], 'system': scenario['system']}
    if args.dry_run:
        print(json.dumps({'arguments': vars(args), 'plan': plan}, ensure_ascii=False, indent=2, default=str))
        return 0
    frozen = load_replay(args.replay_from) if args.replay_from else {}
    if args.replay_from:
        for name in names:
            for turn in range(1, args.turns + 1):
                if (name, turn) not in frozen:
                    parser.error(f'重放文件缺少 {name} 第 {turn} 轮')
    folder = ROOT / 'logs' / ('scenarios-' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    folder.mkdir(parents=True)
    arguments = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    metadata = {'arguments': arguments, 'plan': plan, 'status': 'running',
                'started_at': datetime.datetime.now().astimezone().isoformat(),
                'server_config': '上下文、缓存、版本需用 label 和启动日志对应；/health 不提供这些完整信息。'}
    records, quality, pairs = [], {}, []
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    base = f'http://127.0.0.1:{args.port}'
    key = os.environ.get(args.api_key_env, '')
    headers = {'Content-Type': 'application/json'}
    if key:
        headers['Authorization'] = 'Bearer ' + key
    error = None
    try:
        with opener.open(urllib.request.Request(base + '/health', headers=headers), timeout=5) as response:
            metadata['health'] = json.load(response)
        with opener.open(urllib.request.Request(base + '/v1/models', headers=headers), timeout=5) as response:
            models = json.load(response)
        if args.model not in [m['id'] for m in models.get('data', [])]:
            raise ValueError(f'服务未列出模型 {args.model}，请核对 --model/config.sh')
        print(f'结果目录：{folder}', flush=True)
        for name in names:
            scenario, options = scenarios[name], plan[name]
            history = [{'role': 'system', 'content': options['system']}]
            for turn, prompt in enumerate(options['turns'], 1):
                if args.replay_from:
                    history = copy.deepcopy(frozen[(name, turn)])
                else:
                    history.append({'role': 'user', 'content': prompt})
                variants = ([True, False] if turn % 2 else [False, True]) if args.compare_drafts else [not args.no_drafts]
                replies = {}
                for draft in variants:
                    body = make_request(history, args.model, args.tokens, options['temperature'],
                                        options['thinking'], options['thinking_budget'], args.seed + turn, draft,
                                        engine=args.engine)
                    print(f'{name} 第 {turn}/{args.turns} 轮，草稿={draft}，等待输出…', flush=True)
                    metadata['pending_request'] = {'scenario': name, 'turn': turn, 'draft': draft, 'body': body}
                    save_report(folder, metadata, records, quality, pairs)
                    request = urllib.request.Request(base + '/v1/chat/completions', json.dumps(body).encode(),
                                                     headers)
                    started = time.perf_counter()
                    with opener.open(request, timeout=args.timeout) as response:
                        result = read_stream(response, started)
                    row = metric_row(name, turn, draft, result)
                    records.append({'request': body, 'response': result, 'metrics': row})
                    replies[draft] = result
                    metadata.pop('pending_request', None)
                    print(f"  输入={show(row['prompt_tokens'])} 缓存={show(row['cached_tokens'])} "
                          f"输出={show(row['completion_tokens'])} 服务tok/s={show(row['decode_tps'])} "
                          f"客户端tok/s={show(row['client_decode_tps'])} "
                          f"首正文={show(row['first_content_s'])}s 结束={row['finish_reason']}", flush=True)
                    if turn == len(scenario['turns']):
                        quality[f'{name}:draft={draft}'] = check_answer(result['content'], scenario['expected'])
                    save_report(folder, metadata, records, quality, pairs)
                if args.compare_drafts:
                    left, right = replies[True], replies[False]
                    hashes = [r['tensorfold'].get('token_sha') for r in (left, right)]
                    equal = all(left[k] == right[k] for k in ('content', 'reasoning', 'finish_reason'))
                    hash_equal = hashes[0] == hashes[1] if all(hashes) else None
                    pairs.append({'scenario': name, 'turn': turn, 'text_equal': equal,
                                  'token_hash_equal': hash_equal, 'order': variants})
                    if not equal or hash_equal is False:
                        raise ValueError(f'{name} 第 {turn} 轮草稿开关输出不同，已保存并停止对照。')
                result = replies[True] if True in replies else replies[False]
                if not result['content'].strip():
                    raise ValueError(f'{name} 第 {turn} 轮没有最终正文，无法继续自然对话；检查思考预算或输出上限。')
                history.append({'role': 'assistant', 'content': result['content']})
        metadata['status'] = 'completed'
    except KeyboardInterrupt:
        error = '用户中断；已完成轮次保留，当前请求可能尚未在服务端结束。'
    except urllib.error.HTTPError as exc:
        with exc:
            error = f'HTTP {exc.code}: {exc.read().decode(errors="replace")}'
    except (OSError, ValueError) as exc:
        error = str(exc)
    finally:
        if error:
            metadata['status'] = 'incomplete'
        metadata['finished_at'] = datetime.datetime.now().astimezone().isoformat()
        save_report(folder, metadata, records, quality, pairs, error)
    print(f"{'停止：' + error if error else '测试完成'}\n报告：{folder / 'report.md'}", flush=True)
    return 1 if error else 0


if __name__ == '__main__':
    raise SystemExit(main())
