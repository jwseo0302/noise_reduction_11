"""Hidden-test standalone entrypoint. No clean/metadata/PESQ dependencies."""
import argparse,json
from pathlib import Path
import numpy as np
import soundfile as sf
from dsp_v14 import ParamsV14,enhance

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',default='input')
    p.add_argument('--output',default='output')
    p.add_argument('--config',default='best_params_v14.json')
    a=p.parse_args()
    src=Path(a.input);out=Path(a.output)
    if not src.is_dir():raise FileNotFoundError(src)
    if src.resolve()==out.resolve():raise ValueError('Input and output folders must differ')
    files=sorted(q for q in src.iterdir() if q.is_file() and q.suffix.lower()=='.wav')
    if not files:raise RuntimeError('Input folder contains no WAVs')
    cfg=ParamsV14(**json.loads(Path(a.config).read_text(encoding='utf8')))
    out.mkdir(parents=True,exist_ok=True)
    for f in files:
        info=sf.info(f)
        data,sr=sf.read(f,dtype='float64',always_2d=True)
        enhanced=enhance(data,sr,cfg)
        if enhanced.shape!=data.shape or not np.isfinite(enhanced).all():
            raise RuntimeError('Invalid enhanced shape or values: '+str(f))
        target=out/f.name
        sf.write(target,enhanced,sr,subtype='FLOAT')
        check=sf.info(target)
        if (check.samplerate,check.frames,check.channels)!=(info.samplerate,info.frames,info.channels):
            raise RuntimeError('Output metadata mismatch: '+str(target))
        print('OK',f.name,flush=True)
    print('Processed',len(files),'WAV(s)',flush=True)
if __name__=='__main__':main()
