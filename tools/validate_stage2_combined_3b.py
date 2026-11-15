"""Check the exact packaged combined runtime in isolation; no GPU/download required."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,tempfile,zipfile
ROOT=Path(__file__).resolve().parents[1]


def main():
    source=ROOT/'notebooks/kaggle/stage2_combined_3b/stage2_combined_3b_source.zip'
    with tempfile.TemporaryDirectory(prefix='combined_pack_') as temp:
        with zipfile.ZipFile(source) as z:
            assert z.testzip() is None
            for name in z.namelist():assert not Path(name).is_absolute() and '..' not in Path(name).parts
            z.extractall(temp)
        isolated=Path(temp)/'stage2_combined_3b_source'
        raw=(isolated/'manifest.json').read_bytes();(isolated/'_stage2_combined_3b_manifest.json').write_bytes(raw)
        env=dict(os.environ,PYTHONPATH=os.pathsep.join([str(isolated),str(isolated/'src')]))
        subprocess.run([sys.executable,'-m','eval.stage2_combined_3b.preflight'],cwd=isolated,env=env,check=True)
        # Reuse only the scripted test backend, whose reference responses are explicitly test fixtures.
        script='''import importlib.util,json
from pathlib import Path
spec=importlib.util.spec_from_file_location('integration_tests',TEST_FILE)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
from eval.stage2_combined_3b.runner import run
from eval.stage2_combined_3b.collect import collect
paths=[]
for method in ['selective','tools']:
    backend,*_=module.factory(method)
    out=run(method,Path(OUT)/method,smoke=True,backend_factory=backend)
    paths.append(out.with_suffix('.zip'))
report=collect(paths,Path(OUT)/'comparison',allow_scripted=True)
assert all(s['correct']==2 and s['incorrect_accepted']==0 for s in report['summary'].values())
assert not report['model_quality_result']
'''
        script='TEST_FILE='+repr(str(ROOT/'tests/test_stage2_combined_3b.py'))+'\nOUT='+repr(str(Path(temp)/'results'))+'\n'+script
        subprocess.run([sys.executable,'-c',script],cwd=isolated,env=env,check=True)
        summary=json.loads((Path(temp)/'results/comparison/summary.json').read_text())
        validation=dict(source_manifest_sha256=hashlib.sha256(raw).hexdigest(),packaged_parser_and_combined_preflight='passed',
                        independent_scripted_methods=2,scripted_tasks_per_method=2,collector='passed',
                        model_quality_result=False,gpu_validation='not run locally; provided Kaggle notebooks run actual inference',summary=summary['summary'])
        (ROOT/'docs/stage2_combined_3b_validation.json').write_text(json.dumps(validation,indent=2)+'\n')
        print('Exact source pack: both fresh scripted pipelines and collector passed.')
if __name__=='__main__':main()
