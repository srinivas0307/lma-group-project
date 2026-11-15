"""CPU-only validation of the exact Kaggle source ZIP in an isolated directory."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import zipfile
ROOT=Path(__file__).resolve().parents[1]


def main():
    source=ROOT/'notebooks/kaggle/stage2_repair_dev/stage2_repair_dev_source.zip'
    with tempfile.TemporaryDirectory(prefix='repair_dev_pack_') as temp:
        with zipfile.ZipFile(source) as z:
            assert z.testzip() is None
            for name in z.namelist():
                assert not Path(name).is_absolute() and '..' not in Path(name).parts
            z.extractall(temp)
        isolated=Path(temp)/'stage2_repair_dev_source'
        raw=(isolated/'manifest.json').read_bytes()
        (isolated/'_repair_dev_manifest.json').write_bytes(raw)
        env=dict(os.environ,PYTHONPATH=os.pathsep.join([str(isolated),str(isolated/'src')]))
        out=Path(temp)/'full_cpu_check'
        subprocess.run([sys.executable,'-m','eval.stage2_repair_dev.runner','--out',str(out),'--cpu-check'],cwd=isolated,env=env,check=True)
        with zipfile.ZipFile(out.with_suffix('.zip')) as z:
            seal=json.loads(z.read('bundle.json'))
            assert seal['complete']
            for name,sha in seal['sha256'].items():
                assert hashlib.sha256(z.read(name)).hexdigest()==sha
        completion=json.loads((out/'completion.json').read_text())
        assert completion['complete'] and completion['records']==68 and not completion['model_quality_result']
        result=dict(packaged_cpu_check='passed',records=completion['records'],
                    source_manifest_sha256=hashlib.sha256(raw).hexdigest(),
                    model_quality_result=False,gpu_validation='not run locally: no Torch/CUDA runtime',
                    summary=json.loads((out/'summary.json').read_text()),seal_verified=True)
        (ROOT/'docs/stage2_repair_dev_validation.json').write_text(json.dumps(result,indent=2)+'\n')
        print('ISOLATED PACKAGED CHECK PASSED; all scores above are scripted infrastructure checks.')
if __name__=='__main__':main()
