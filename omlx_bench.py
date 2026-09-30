#!/usr/bin/env python3
"""Isolated oMLX benchmark configuration, lifecycle and common test entry."""
import argparse
import copy
import datetime
import json
import os
from pathlib import Path
import plistlib
import secrets
import signal
import socket
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
BASE = Path(os.environ.get('OMLX_BENCH_BASE', ROOT/'.omlx-benchmark'))
PORT = int(os.environ.get('OMLX_BENCH_PORT', '8002'))
CEILING = float(os.environ.get('OMLX_BENCH_CEILING_GB', '28'))
MODEL_ID = 'Qwen3.8-27B-MLX-4bit'
MODEL = Path(os.environ.get('MODEL_DIR', Path.home()/'.omlx/models/lmstudio-community'/MODEL_ID))
DRAFT = Path(os.environ.get('DRAFTER_DIR', Path.home()/'.omlx/models/z-lab/Qwen3.8-27B-DFlash2'))
CLI = Path(os.environ.get('OMLX_CLI', '/Applications/oMLX.app/Contents/MacOS/omlx-cli'))
OWNER = 'tensorfold-omlx-qwen38-benchmark'


def validate_base(base):
    resolved = Path(base).resolve()
    daily = (Path.home()/'.omlx').resolve()
    if resolved == daily or resolved in daily.parents or daily in resolved.parents:
        raise RuntimeError('基准目录不能指向日常oMLX目录、其子目录或父目录')
    return resolved


def build_config(source, base, model, drafter, draft):
    config = copy.deepcopy(source)
    config['model'] = {'model_dir': str(base/'models'), 'model_dirs': [str(base/'models')],
                       'model_fallback': False, 'hide_helper_models': True}
    config['server'] = {'host': '127.0.0.1', 'port': PORT, 'log_level': 'info',
                        'auto_start_on_launch': False, 'max_concurrent_requests': 1}
    config['sampling'] = {'max_context_window': 32768, 'max_tokens': 4096,
                          'temperature': 0, 'top_p': .95, 'top_k': 20, 'repetition_penalty': 1}
    config['memory'] = {'prefill_memory_guard': True, 'memory_guard_tier': 'custom',
                        'memory_guard_custom_ceiling_gb': CEILING, 'soft_threshold': .85,
                        'hard_threshold': .95, 'prefill_safe_zone_ratio': .8}
    config['cache'] = {'enabled': True, 'hot_cache_only': True, 'hot_cache_max_size': '2GB',
                       'hot_cache_write_through': False, 'ssd_cache_dir': str(base/'cache'),
                       'ssd_cache_max_size': '2GB', 'initial_cache_blocks': 256}
    config['logging'] = {}
    config['auth'] = {**config.get('auth', {}), 'api_key': secrets.token_urlsafe(24),
                      'allow_unauthenticated_inference': False, 'skip_api_key_verification': False}
    target = {'model_alias': 'qwen38-local', 'max_context_window': 32768, 'max_tokens': 4096,
              'temperature': 0, 'top_k': 20, 'top_p': .95, 'repetition_penalty': 1,
              'enable_thinking': True, 'thinking_budget_enabled': True, 'thinking_budget_tokens': 256,
              'chat_template_kwargs': {'reasoning_effort': 'medium'},
              'dflash_enabled': draft, 'dflash_draft_model': str(drafter),
              'dflash_draft_quant_enabled': True, 'dflash_draft_quant_weight_bits': 4,
              'dflash_draft_quant_activation_bits': 16, 'dflash_draft_quant_group_size': 64,
              'dflash_block_size': 8, 'dflash_max_ctx': None,
              'dflash_in_memory_cache': True, 'dflash_in_memory_cache_max_bytes': 2*1024**3,
              'dflash_ssd_cache': False, 'mtp_enabled': False, 'vlm_mtp_enabled': False,
              'is_default': True, 'is_pinned': False, 'model_type_override': None,
              'active_profile_name': None}
    return config, {'version': 1, 'models': {MODEL_ID: target}}


def draft_confirmed(log):
    bad = ('falling back', 'fallback from dflash', 'toggle ignored', 'drafter load raised')
    if any(s in log.lower() for s in bad):
        return False
    return ('DFlash drafter attached to engine' in log or
            'DFlash generation complete:' in log)


def server_environment(source, base):
    # oMLX applies environment overrides to file settings. Do not let the
    # user's daily API key/cache/log/port settings escape the isolated base.
    env = {k:v for k,v in source.items() if not k.startswith('OMLX_')}
    env.update({'HF_HUB_OFFLINE':'1','PYTHONUNBUFFERED':'1','OMLX_BASE_PATH':str(base)})
    return env


def write_json(path, value, private=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    if private:
        path.chmod(0o600)


def settings():
    return json.loads((BASE/'settings.json').read_text())


def request(route, body=None, timeout=5):
    key = settings().get('auth', {}).get('api_key', '')
    headers = {'Authorization': 'Bearer '+key, 'Content-Type': 'application/json'}
    req = urllib.request.Request(f'http://127.0.0.1:{PORT}'+route,
                                 None if body is None else json.dumps(body).encode(), headers)
    op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with op.open(req, timeout=timeout) as response:
        return json.load(response)


def start_time(pid):
    return subprocess.check_output(['ps', '-p', str(pid), '-o', 'lstart='], text=True).strip()


def stop(process=None):
    validate_base(BASE)
    file = BASE/'process.json'
    if not file.exists():
        return
    record = json.loads(file.read_text())
    pid = record['pid']
    try:
        if start_time(pid) != record['start_time']:
            raise RuntimeError('PID已被复用，拒绝停止其他进程')
        os.kill(pid, signal.SIGINT)
        if process is not None and process.pid == pid:
            process.wait(timeout=30)
        else:
            for _ in range(60):
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    break
                time.sleep(.5)
            else:
                raise RuntimeError('基准服务未退出，请查看日志；未使用强制终止')
    except subprocess.CalledProcessError:
        pass
    file.unlink(missing_ok=True)
    print('oMLX基准服务已停止')


def start(mode):
    validate_base(BASE)
    marker = BASE/'benchmark-owner.json'
    if any((BASE/name).exists() for name in ('settings.json','model_settings.json','model_profiles.json')):
        if not marker.exists() or json.loads(marker.read_text()).get('owner') != OWNER:
            raise RuntimeError('基准目录已有非本工具配置，拒绝覆盖；请使用新目录')
    for p in (MODEL, DRAFT):
        if not (p/'config.json').exists() or not list(p.glob('*.safetensors')):
            raise RuntimeError(f'模型文件缺失：{p}')
    stop()
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', PORT))
    # Refuse an overlapping TensorFold model process, even on a different port.
    process_list = subprocess.check_output(['ps', '-axo', 'command='], text=True)
    if any('/bin/tensorfold serve ' in line and '/bin/zsh -c' not in line for line in process_list.splitlines()):
        raise RuntimeError('请先停止TensorFold模型，再启动oMLX基准服务')
    source_path = Path.home()/'.omlx/settings.json'
    source = json.loads(source_path.read_text()) if source_path.exists() else {}
    app_port = source.get('server', {}).get('port', 8000)
    if app_port != PORT:
        app_key = source.get('auth', {}).get('api_key', '')
        try:
            req = urllib.request.Request(f'http://127.0.0.1:{app_port}/health',
                                         headers={'Authorization':'Bearer '+app_key} if app_key else {})
            op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with op.open(req,timeout=3) as response:
                app_health = json.load(response)
            if app_health.get('engine_pool', {}).get('loaded_count', 0):
                raise RuntimeError('日常oMLX App仍有模型驻留，请先卸载；原服务不会被本脚本停止')
        except urllib.error.URLError as exc:
            if getattr(exc, 'code', None):
                raise RuntimeError('无法认证日常oMLX的健康状态，请先停止日常服务') from exc
    BASE.mkdir(parents=True, exist_ok=True);BASE.chmod(0o700)
    write_json(marker,{'owner':OWNER,'schema':1},True)
    models = BASE/'models';models.mkdir(exist_ok=True)
    link = models/MODEL_ID
    if link.is_symlink():
        if link.resolve() != MODEL.resolve():link.unlink()
    elif link.exists():
        raise RuntimeError('基准模型目录存在非symlink文件，拒绝覆盖')
    if not link.exists():link.symlink_to(MODEL.resolve(), target_is_directory=True)
    cfg, mc = build_config(source, BASE, MODEL, DRAFT, mode=='draft')
    write_json(BASE/'settings.json',cfg,True)
    write_json(BASE/'model_settings.json',mc,True)
    write_json(BASE/'model_profiles.json',{'version':1,'profiles':{}},True)
    stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    logpath = ROOT/'logs'/f'omlx-compare-{mode}-{stamp}.log'
    logpath.parent.mkdir(parents=True,exist_ok=True)
    command = [str(CLI),'serve','--base-path',str(BASE),'--model-dir',str(models),
               '--host','127.0.0.1','--port',str(PORT),'--max-concurrent-requests','1',
               '--memory-guard-gb',str(CEILING),'--no-hf-cache','--log-level','info']
    env = server_environment(os.environ, BASE)
    with logpath.open('w') as log:
        process = subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                                   stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write_json(BASE/'process.json',{'pid':process.pid,'start_time':start_time(process.pid),'port':PORT})
    print(f'启动独立oMLX基准服务：{mode}，日志 {logpath}',flush=True)
    try:
        for _ in range(120):
            if process.poll() is not None:raise RuntimeError('启动失败，请查看基准日志')
            try:
                health = request('/health');break
            except OSError:time.sleep(1)
        else:raise RuntimeError('服务启动超时')
        warm = request('/v1/chat/completions',{'model':'qwen38-local','messages':[{'role':'user','content':'Reply with hello.'}],
                                             'max_tokens':16,'temperature':0,'seed':1234,
                                             'chat_template_kwargs':{'enable_thinking':False},'stream':False},timeout=240)
        text = logpath.read_text()
        if mode=='draft' and not draft_confirmed(text):
            raise RuntimeError('DFlash配置已请求，但日志未确认成功加载，拒绝继续测速')
        version = subprocess.check_output([str(CLI),'--version'],text=True).strip()
        receipt = {'engine':'omlx','mode':mode,'port':PORT,'ceiling_gb':CEILING,
                   'model':'qwen38-local','version':version,
                   'base_path':str(BASE),'model_path':str(MODEL.resolve()),'drafter_path':str(DRAFT.resolve()),
                   'log':str(logpath),'command':command,'model_settings':mc,'health':request('/health'),
                   'warmup':'16-token different prompt; model/kernel warmup, not a guaranteed cold prompt-cache test',
                   'draft_load_confirmed':mode=='draft' and draft_confirmed(text)}
        write_json(BASE/'receipt.json',receipt)
        write_json(logpath.with_suffix('.json'),receipt)
        write_json(BASE/'warm-response.json',warm)
        print('基准模型已就绪：qwen38-local；DFlash='+str(mode=='draft'),flush=True)
    except BaseException:
        stop(process=process);raise


def main():
    parser = argparse.ArgumentParser(description='隔离oMLX基准：start draft/serial，status，stop，test [场景参数]')
    parser.add_argument('action',choices=['start','status','stop','test'])
    parser.add_argument('extra',nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.action=='start':
        mode=args.extra[0] if args.extra else 'draft'
        if mode not in ('draft','serial'):parser.error('模式为draft或serial')
        start(mode)
    elif args.action=='stop':stop()
    elif args.action=='status':print(json.dumps(request('/health'),ensure_ascii=False,indent=2))
    else:
        receipt=json.loads((BASE/'receipt.json').read_text())
        request('/health')
        if '--compare-drafts' in args.extra:parser.error('oMLX要分别start draft/serial，不支持请求内切换')
        if receipt['mode']=='draft' and '--no-drafts' in args.extra:parser.error('当前服务仍启用草稿，请先start serial')
        command=[sys.executable,str(ROOT/'bench.py'),'--engine','omlx','--port',str(PORT),'--model','qwen38-local']
        if receipt['mode']=='serial':command+=['--no-drafts']
        env={**os.environ,'INFERENCE_API_KEY':settings()['auth']['api_key']}
        result=subprocess.run(command+args.extra,cwd=ROOT,env=env)
        return result.returncode
    return 0


if __name__=='__main__':
    try:raise SystemExit(main())
    except (OSError,ValueError,RuntimeError) as exc:raise SystemExit(str(exc))
