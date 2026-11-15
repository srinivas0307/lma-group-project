"""Repair model isolation, NF4 precision checks, fresh pipeline and packaged notebook."""
import copy,hashlib,json,shutil,subprocess,zipfile
from pathlib import Path
import nbformat
import pytest
from tests.test_stage2_combined_3b import factory
from eval.stage2_combined_gold_7b.runner import run
from eval.stage2_combined_gold_7b.collect import read_bundle,collect
from eval.stage2_combined_gold_7b.precision import inspect_precision,validate_precision_record
from eval.stage2_combined_gold_7b.config import MODELS,COMMON,REPAIR_VERSION
ROOT=Path(__file__).resolve().parents[1]


def test_only_repair_configuration_changes():
    from eval.stage2_combined_gold import config as control
    assert MODELS['generator']==control.MODELS['generator']
    assert MODELS['agent']['id']=='Qwen/Qwen2.5-7B-Instruct' and MODELS['agent']['four_bit'] is True
    assert MODELS['agent']['revision']=='a09a35458c702b33eeacc393d103063234e8bc28'
    assert {k:v for k,v in COMMON.items() if k!='repair_model_policy'}=={k:v for k,v in control.COMMON.items() if k!='repair_model_policy'}
    assert REPAIR_VERSION==control.REPAIR_VERSION
    assert (ROOT/'eval/stage2_combined_gold_7b/execute.py').read_bytes()==(ROOT/'eval/stage2_combined_gold/execute.py').read_bytes()


def fake_model(quantized=False,quant_type='nf4',dtype='torch.float16',compute='torch.float16'):
    class Parameter:
        def __init__(self,dtype):self.dtype=dtype
        def is_floating_point(self):return self.dtype!='torch.uint8'
    class Params4bit(Parameter):
        def __init__(self):super().__init__('torch.uint8');self.quant_type=quant_type
    class Linear4bit:
        compute_dtype=compute
    class Model:
        is_loaded_in_4bit=quantized
        def parameters(self):return [Parameter(dtype)]+([Params4bit()] if quantized else [])
        def modules(self):return [Linear4bit()] if quantized else []
    return Model()


def test_actual_nf4_and_generator_precision_checks():
    p=inspect_precision('agent',fake_model(True));assert p['actual_quant_types']==['nf4'] and p['quantized_parameter_count']==1
    inspect_precision('generator',fake_model())
    for role,model in [('generator',fake_model(True)),('agent',fake_model()),('agent',fake_model(True,quant_type='fp4')),('agent',fake_model(True,compute='torch.float32')),('generator',fake_model(dtype='torch.float32'))]:
        with pytest.raises(ValueError):inspect_precision(role,model)
    for key,value in [('quantization','none'),('quantized_parameter_count',0),('actual_quant_types',['fp4']),('compute_dtype','torch.float32')]:
        wrong=dict(p,**{key:value})
        with pytest.raises(ValueError):validate_precision_record('agent',wrong)


def test_fresh_gold_pipeline_and_paired_control(tmp_path):
    from eval.stage2_combined_gold.runner import run as control_run
    backend,live,calls,lifecycle=factory('selective');out=run('gold',tmp_path/'new',smoke=True,backend_factory=backend)
    plan,rows=read_bundle(out.with_suffix('.zip'),allow_scripted=True)
    assert all(r['final_correct'] and r['attempts']==1 and not r['initial_correct'] and r['parser_calls']==r['parser_tokens']==0 for r in rows)
    assert not live and lifecycle==[(op,role) for role in ['generator','agent'] for op in ['load','close']]
    assert plan['models']==MODELS and not plan['adapter_hashes']
    backend,*_=factory('selective');old=control_run('gold',tmp_path/'control',smoke=True,backend_factory=backend)
    report=collect(out.with_suffix('.zip'),tmp_path/'comparison',old.with_suffix('.zip'),allow_scripted=True)
    assert not report['model_quality_result']
    assert report['paired']['same_initial_code_count']==2
    assert report['summary']['gold_7b_nf4']['correct']==report['summary']['gold_3b_fp16']['correct']==2
    with pytest.raises(ValueError,match='real-model'):read_bundle(out.with_suffix('.zip'))
    with pytest.raises(ValueError,match='fresh output'):run('gold',out,smoke=True,backend_factory=backend)
    with pytest.raises(ValueError,match='adapter'):run('gold',tmp_path/'bad',adapter=tmp_path/'unused',backend_factory=backend)


@pytest.mark.parametrize('packed',[True,False])
def test_notebook_pack_loader_and_full_100_stage(packed,tmp_path,monkeypatch):
    here=ROOT/'notebooks/kaggle/stage2_combined_gold_7b';inputs=tmp_path/'inputs';inputs.mkdir();work=tmp_path/'work'
    source=here/'stage2_combined_gold_7b_source.zip'
    if packed:shutil.copy(source,inputs/source.name)
    else:
        with zipfile.ZipFile(source) as z:z.extractall(inputs)
    book=nbformat.read(here/'stage2_combined_gold_7b_kaggle.ipynb',as_version=4)
    assert book.metadata.language_info.name=='python'
    for c in book.cells:
        if c.cell_type=='code':compile(c.source,'notebook','exec')
    assert '--adapter' not in book.cells[4].source
    assert "STAGE 3/3: full 100-task" in book.cells[5].source
    assert 'BUNDLE=execute(False)' in book.cells[5].source
    monkeypatch.setattr(subprocess,'run',lambda *a,**kw:None)
    ns={};exec(book.cells[1].source.replace("INPUT=Path('/kaggle/input')",f'INPUT=Path({str(inputs)!r})').replace("WORK=Path('/kaggle/working')",f'WORK=Path({str(work)!r})'),ns)
    exec(book.cells[2].source,ns)
    manifest=json.loads((ns['ROOT']/'_stage2_combined_gold_7b_manifest.json').read_text())
    assert all(hashlib.sha256((ns['ROOT']/p).read_bytes()).hexdigest()==sha for p,sha in manifest['sha256'].items())
    with zipfile.ZipFile(ROOT/'notebooks/kaggle/stage2_combined_gold/stage2_combined_gold_source.zip') as z:
        assert (ns['ROOT']/'_stage2_combined_gold_manifest.json').read_bytes()==z.read('stage2_combined_gold_source/manifest.json')
