"""Behavioral checks for request-only facts, SFT supervision and the new workflow."""
import ast
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zlib
import zipfile
import nbformat
import pandas as pd
import pytest
from finetune.selective_facts.context import messages,selective_facts,FactUnavailable,PROTOCOL
from finetune.selective_facts.provenance import ROOT,verify_data


def test_no_unnecessary_facts_and_request_last():
    df=pd.DataFrame({'Age':[1.,2.,None],'backup':[4.,5.,6.]})
    request='Rename Age to years.'
    task=dict(request=request,data_file='input.csv',gold_contract='SECRET',reference_code='SECRET')
    result=messages(task,df)
    assert selective_facts(request,df)==[]
    assert 'Required input facts' not in result[-1]['content']
    assert result[-1]['content'].endswith('Request: '+request)
    assert 'SECRET' not in str(result)
    assert messages(task,df)==messages(dict(request=request,data_file='input.csv'),df)
    assert selective_facts("In backup, replace values with 'mean'.",df)==[]
    request='Fill missing Age with its median.'
    result=messages(dict(task,request=request),df)
    assert result[-1]['content'].endswith('Request: '+request)
    assert result[-1]['content'].index('Required input facts')<result[-1]['content'].index('Request:')
    assert 'Required input facts' not in messages(dict(task,request=request),df,original=True)[-1]['content']


def test_full_precision_and_conditional_scope():
    df=pd.DataFrame({'x':[1.123456789,9.987654321,None,1000.], 'zone':['east','east','east','west']})
    facts=selective_facts('Fill missing x with its full-column median.',df)
    assert facts[0]['value']==df.x.median()
    facts=selective_facts('Fill missing x only in zone = east with the median of x among rows with zone = east.',df)
    assert facts[0]['value']==df.loc[df.zone.eq('east'),'x'].median()
    assert facts[0]['scope']['where']=={'column':'zone','op':'==','value':'east'}
    assert facts[0]['value']!=round(facts[0]['value'],6)
    facts=selective_facts('Fill missing x only in zone = east with the full-column median of x.',df)
    assert facts[0]['value']==df.x.median() and facts[0]['scope']['kind']=='full_column'
    with pytest.raises(FactUnavailable):selective_facts('Fill missing x only in zone = east with its median.',df)
    numeric=pd.DataFrame({'x':[1.,3.,999.,None], 'key':[1.5,1.5,1.0,1.5]})
    assert selective_facts('Fill missing x with the median of x among rows with key = 1.5.',numeric)[0]['value']==2.
    with pytest.raises(FactUnavailable):selective_facts('Fill missing Age with its median.',pd.DataFrame({'Age':[1.,None],'age':[2.,None]}))
    with pytest.raises(FactUnavailable,match='unsupported conditional'):
        selective_facts('Fill missing x with the median of positive x values.',df)
    with pytest.raises(FactUnavailable):selective_facts('Fill missing x with the median within each zone.',df)
    with pytest.raises(FactUnavailable):selective_facts('Fill missing x with the mean of x where zone = absent.',df)
    with pytest.raises(FactUnavailable):selective_facts('Fill missing x using the median and mean.',df)
    with pytest.raises(FactUnavailable):selective_facts('Fill missing x with the median of zone.',df)
    with pytest.raises(FactUnavailable):selective_facts('Fill missing x using the median.',pd.DataFrame({'x':[None,None]}))


def test_modes_ties_missing_and_non_numeric():
    df=pd.DataFrame({'status':['z','a','z','a',None]})
    with pytest.raises(FactUnavailable,match='tied'):selective_facts('Fill missing status with its mode.',df)
    assert selective_facts('Fill missing status with its mode; choose the alphabetically first tied value.',df)[0]['value']=='a'
    with pytest.raises(FactUnavailable):selective_facts('Fill missing status with its median.',df)
    df=pd.DataFrame({'Ship Mode':['a','b',None]})
    assert selective_facts("In Ship Mode, replace 'a' with 'b', preserving missing values.",df)==[]


def test_training_build_matches_runtime_and_has_disjoint_sources():
    folder=ROOT/'finetune/data/selective_facts_v1';manifest=verify_data(folder)
    assert not set(manifest['train_sources'])&set(manifest['validation_sources'])
    examples=[json.loads(l) for l in (folder/'synthetic_examples.jsonl').read_text().splitlines()]
    indexed={e['id']:e for e in examples};seen=set()
    for split in ['train','val']:
        rows=[json.loads(l) for l in (folder/f'sft_{split}.jsonl').read_text().splitlines()]
        assert len(rows)==manifest['counts'][split]
        assert all(r['prompt_version']==PROTOCOL for r in rows)
        for row in rows:
            if row['facts']:
                from igt.agents.intent_parser import extract_contract
                contract=extract_contract(row['messages'][-1]['content'])['contract']
                facts=row['facts'][0]
                literals=[p['args'][0]['lit'] for p in contract['must_change'] if p.get('fn')=='const' and p.get('target')==facts['column']]
                assert literals and all(v==facts['value'] for v in literals)
        assert not seen&{r['id'] for r in rows};seen.update(r['id'] for r in rows)
        for r in rows:
            if r['id'] not in indexed:continue
            task=indexed[r['id']];df=pd.read_csv(ROOT/task['data_file'])
            assert messages(task,df)==r['messages'][:-1]
    assert len(manifest['synthetic_semantic_checks'])==540
    assert all(r['correct_accepted'] and r['wrong_rejected'] for r in manifest['synthetic_semantic_checks'])


def test_assistant_only_supervision_and_no_silent_truncation():
    from finetune.selective_facts.train import tokenize
    class Tokenizer:
        def apply_chat_template(self,ms,tokenize=False,add_generation_prompt=False):
            text=''.join(m['role']+':'+m['content']+'\n' for m in ms)
            return text+('assistant:' if add_generation_prompt else '')
        def __call__(self,text,add_special_tokens=False):return {'input_ids':[ord(c) for c in text]}
    row=dict(id='example',messages=[{'role':'system','content':'RULES'},{'role':'user','content':'DATA'},{'role':'assistant','content':'ANSWER'}])
    encoded=tokenize([row],Tokenizer(),100)[0]
    split=encoded['labels'].index(ord('A'))
    assert all(v==-100 for v in encoded['labels'][:split])
    assert ''.join(chr(v) for v in encoded['labels'][split:])=='ANSWER\n'
    with pytest.raises(ValueError,match='no examples are truncated'):tokenize([row],Tokenizer(),5)
    with pytest.raises(ValueError,match='one assistant'):tokenize([dict(row,messages=row['messages']+[row['messages'][-1]])],Tokenizer(),100)


def test_screening_cannot_trade_more_wrong_acceptances_for_more_passes():
    from finetune.selective_facts.evaluate import eligible
    baseline={c:dict(tasks=20,correct_accepted=15,wrong_accepted=1,joint_success=14) for c in ['synthetic','original']}
    improved=json.loads(json.dumps(baseline));improved['synthetic'].update(correct_accepted=16,joint_success=15)
    assert eligible(improved,baseline)
    assert not eligible(baseline,baseline)
    improved['original']['wrong_accepted']=2
    assert not eligible(improved,baseline)
    improved['original'].update(wrong_accepted=1,correct_accepted=14)
    assert not eligible(improved,baseline)


def test_adapters_require_matching_protocol_and_full_screening(tmp_path):
    from finetune.selective_facts.adapter import validate_adapter
    from finetune.selective_facts.config import MODEL,REVISION
    metadata=dict(protocol=PROTOCOL,base=MODEL,revision=REVISION,context_sha256=hashlib.sha256((ROOT/'finetune/selective_facts/context.py').read_bytes()).hexdigest())
    (tmp_path/'input_protocol.json').write_text(json.dumps(metadata))
    (tmp_path/'igt_prompt_version.txt').write_text(PROTOCOL)
    (tmp_path/'selection.json').write_text(json.dumps(dict(smoke=False,eligible=False)))
    with pytest.raises(ValueError,match='screening'):validate_adapter(tmp_path)
    (tmp_path/'selection.json').write_text(json.dumps(dict(smoke=True,eligible=True)))
    with pytest.raises(ValueError,match='screening'):validate_adapter(tmp_path)
    metadata['context_sha256']='wrong'
    (tmp_path/'input_protocol.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError,match='different facts'):validate_adapter(tmp_path)


def test_supported_and_unavailable_parser_paths_use_no_gold():
    from eval.stage2_selective.runner import parse_fresh,execute
    from igt.llm import ScriptedBackend
    df=pd.DataFrame({'x':[1.,2.,None]})
    task=dict(id='example',request='Fill missing x with its median within each group.',data_file='x.csv',split='test',dataset='fixture',transformation_type='missing_values')
    backend=ScriptedBackend([])
    parsed=parse_fresh(task,df,backend,'p1-v2')
    assert parsed['extracted']['status']=='input_facts_unavailable'
    result=execute(task,df,'proposed',parsed,backend,backend)
    assert result['record']['outcome']=='ESCALATE' and not backend.calls


def test_six_fresh_stage2_routes_and_strict_collection(tmp_path):
    from eval.stage2_selective.runner import run,load_tasks
    from eval.stage2_selective.config import SYSTEMS
    from eval.stage2_selective.collect import read_bundle,collect
    from igt.agents.intent_parser import contract_to_text
    from igt.llm import ScriptedBackend
    task=load_tasks()[0][1]
    def factory(role):
        text=contract_to_text(task['gold_contract']) if role in ['parser','agent'] else '```python\n'+task['reference_code']+'```'
        return ScriptedBackend([text]+['{"action":"escalate","reason":"fixture"}']*8)
    paths=[]
    for name in SYSTEMS:
        run(name,tmp_path/name,limit=1,backend_factory=factory)
        path=(tmp_path/name).with_suffix('.zip');paths.append(path)
        assert len(read_bundle(path,allow_scripted=True)[1])==1
        with pytest.raises(ValueError,match='real-model'):read_bundle(path)
    assert collect(paths,tmp_path/'summary',allow_scripted=True)['summary']['proposed']['correct']==1


@pytest.mark.parametrize('folder',['selective_facts','stage2_selective'])
def test_embedded_notebooks_are_complete_and_import_without_previous_outputs(folder,tmp_path):
    payload_files=None
    for path in (ROOT/'notebooks/kaggle'/folder).glob('*.ipynb'):
        book=nbformat.read(path,as_version=4);nbformat.validate(book)
        assert book.metadata.language_info.name=='python'
        for cell in book.cells:
            if cell.cell_type=='code':compile(cell.source,str(path),'exec')
        if folder=='selective_facts':
            assert path.stat().st_size<100_000
            with zipfile.ZipFile(path.parent/'selective_facts_training_source.zip') as z:
                manifest=json.loads(z.read('selective_facts_source/selective_facts_manifest.json'))
                payload_files={name:base64.b64encode(z.read('selective_facts_source/'+name)).decode() for name in manifest['sha256']}
                assert all(hashlib.sha256(base64.b64decode(payload_files[name])).hexdigest()==expected for name,expected in manifest['sha256'].items())
        else:
            assignments={}
            for node in ast.parse(book.cells[1].source).body:
                if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ['PAYLOAD','PAYLOAD_SHA']:
                    assignments[node.targets[0].id]=ast.literal_eval(node.value)
            raw=zlib.decompress(base64.b64decode(assignments['PAYLOAD']))
            assert hashlib.sha256(raw).hexdigest()==assignments['PAYLOAD_SHA']
            payload_files=json.loads(raw)
        assert not any(p.startswith(('results/','kaggle_outputs/','finetune/runs/')) for p in payload_files)
        if folder=='selective_facts':assert 'benchmark/contract_quality_v1/selection.jsonl' not in payload_files
        for name,value in payload_files.items():
            content=base64.b64decode(value)
            assert content==(ROOT/name).read_bytes()
            target=tmp_path/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(content)
    module='finetune.selective_facts.train' if folder=='selective_facts' else 'eval.stage2_selective.runner'
    result=subprocess.run([sys.executable,'-m',module,'--help'],cwd=tmp_path,capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_added_reference_programs_run_and_match_gold_contracts():
    from igt.sandbox import safe_run
    from igt.verifier import verify
    examples=[json.loads(l) for l in (ROOT/'finetune/data/selective_facts_v1/synthetic_examples.jsonl').read_text().splitlines()]
    examples=[e for e in examples if e['source_dataset']=='sf_train_00']
    assert len(examples)==18
    for example in examples:
        before=pd.read_csv(ROOT/example['data_file'])
        result=safe_run(example['reference_code'],before,timeout_s=30,mem_gb=2,cpu_s=20)
        assert result.ran_ok,(example['id'],result.error)
        assert verify(example['contract'],before,result.df_out)[0].status=='PASS',example['id']


def test_selected_adapter_is_bound_to_evaluated_weights(tmp_path):
    from finetune.selective_facts.adapter import validate_adapter
    from finetune.selective_facts.config import MODEL,REVISION
    metadata=dict(protocol=PROTOCOL,base=MODEL,revision=REVISION,context_sha256=hashlib.sha256((ROOT/'finetune/selective_facts/context.py').read_bytes()).hexdigest())
    (tmp_path/'input_protocol.json').write_text(json.dumps(metadata))
    (tmp_path/'igt_prompt_version.txt').write_text(PROTOCOL)
    (tmp_path/'adapter_config.json').write_text(json.dumps({'base_model_name_or_path':MODEL}))
    (tmp_path/'adapter_model.safetensors').write_bytes(b'fixture-not-a-model')
    expected={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.iterdir()}
    (tmp_path/'selection.json').write_text(json.dumps(dict(smoke=False,eligible=True,selected={'adapter_sha256':expected})))
    assert validate_adapter(tmp_path)['adapter_model.safetensors']==expected['adapter_model.safetensors']
    (tmp_path/'adapter_model.safetensors').write_bytes(b'changed')
    with pytest.raises(ValueError,match='evaluated checkpoint'):validate_adapter(tmp_path)


@pytest.mark.parametrize('layout',['zip','extracted','tampered','missing'])
def test_small_notebook_loads_dataset_bundle_and_rejects_bad_inputs(tmp_path,layout):
    import shutil
    folder=ROOT/'notebooks/kaggle/selective_facts'
    notebook=nbformat.read(folder/'selective_facts_train.ipynb',as_version=4)
    inputs=tmp_path/'input';inputs.mkdir();working=tmp_path/'working'
    if layout=='zip':shutil.copy2(folder/'selective_facts_training_source.zip',inputs)
    elif layout in ['extracted','tampered']:
        with zipfile.ZipFile(folder/'selective_facts_training_source.zip') as z:z.extractall(inputs)
        if layout=='tampered':(inputs/'selective_facts_source/requirements.txt').write_text('changed')
    setup=notebook.cells[1].source.split("PACKAGES=")[0]
    setup=setup.replace("Path('/kaggle/input')",f'Path({str(inputs)!r})').replace("Path('/kaggle/working')",f'Path({str(working)!r})')
    namespace={}
    if layout in ['tampered','missing']:
        with pytest.raises(AssertionError,match='Changed source/data|Attach the private dataset'):exec(compile(setup,'bootstrap','exec'),namespace)
    else:
        exec(compile(setup,'bootstrap','exec'),namespace)
        source=namespace['ROOT']
        assert (source/'finetune/data/selective_facts_v1/sft_train.jsonl').read_bytes()==(ROOT/'finetune/data/selective_facts_v1/sft_train.jsonl').read_bytes()
        assert (source/'finetune/selective_facts/train.py').read_bytes()==(ROOT/'finetune/selective_facts/train.py').read_bytes()


def test_transformers_520_warmup_argument_and_shared_probe_contract():
    from dataclasses import make_dataclass
    from finetune.selective_facts.config import TRAIN
    from finetune.selective_facts.training_api import argument_options,make_arguments
    # Constructor fields used here are present in the official v5.20.2 TrainingArguments dataclass.
    # Strict constructor rejects the removed warmup_ratio, unlike a permissive **kwargs mock.
    names='output_dir per_device_train_batch_size per_device_eval_batch_size gradient_accumulation_steps num_train_epochs max_steps learning_rate warmup_steps weight_decay lr_scheduler_type logging_steps eval_strategy save_strategy fp16 bf16 optim gradient_checkpointing report_to seed data_seed remove_unused_columns dataloader_num_workers use_cpu'.split()
    PinnedArguments=make_dataclass('PinnedArguments',[(name,object) for name in names])
    real=make_arguments(TRAIN,'/tmp/unused_training_arguments',factory=PinnedArguments)
    probe=make_arguments(TRAIN,'/tmp/unused_probe_arguments',probe=True,factory=PinnedArguments)
    assert real.warmup_steps==probe.warmup_steps==.03
    assert set(argument_options(TRAIN,'unused'))==set(argument_options(TRAIN,'unused',probe=True))
    assert real.gradient_accumulation_steps==16 and probe.gradient_accumulation_steps==1
    assert real.fp16 is True and real.use_cpu is False
    assert probe.fp16 is False and probe.use_cpu is True
    with pytest.raises(TypeError):PinnedArguments(warmup_ratio=.03)
    with pytest.raises(RuntimeError,match='API differs'):
        make_arguments(TRAIN,'unused',factory=lambda output_dir:None)


def test_training_validates_arguments_before_expensive_baseline_and_uses_shared_probe():
    train=ast.parse((ROOT/'finetune/selective_facts/train.py').read_text())
    probe=ast.parse((ROOT/'finetune/selective_facts/preflight.py').read_text())
    arguments=[n for n in ast.walk(train) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='make_arguments']
    baseline=[n for n in ast.walk(train) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='semantic' and any(k.arg=='original' for k in n.keywords)]
    assert len(arguments)==len(baseline)==1 and arguments[0].lineno<baseline[0].lineno
    calls=[n for n in ast.walk(probe) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='make_arguments']
    assert len(calls)==1 and any(k.arg=='probe' and ast.literal_eval(k.value) is True for k in calls[0].keywords)
    for tree in [train,probe]:
        assert not any(isinstance(n,ast.keyword) and n.arg=='warmup_ratio' for n in ast.walk(tree))
