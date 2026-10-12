"""Audit complete registered smoke, resume, gradient, freeze, memory and inference evidence."""
import argparse
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from extensions_built_in.streetscape_cnn.contract import read_json,object_sha256,validate_resume_pair,write_json_atomic,file_sha256

def records(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/workspace/streetscape'))
    p.add_argument('--tier',type=int,choices=[1,2,3],required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--inference-version',type=int,default=1)
    args=p.parse_args()
    if args.output.exists():raise ValueError('Audit is immutable; use a new output')
    result={'scope':'engineering acceptance, not generative quality','tier':args.tier,'runs':{}}
    for kind in ('base','cnn'):
        r=args.root/'runs'/f'krea2_streetscape_{kind}_smoke_1024_b1ga2_mem_p{args.tier}_v1'
        identity=read_json(r/'streetscape_identity.json')
        latest=validate_resume_pair(r,object_sha256(identity))
        status=read_json(r/'research_status.json')
        rows=records(r/'records/steps.jsonl');samples=records(r/'records/samples.jsonl')
        if status['state']!='completed' or latest['step']!=24 or status['completed_updates']!=24 or status['image_presentations']!=48:
            raise ValueError('Native completion and latest state differ')
        if [row['native_step'] for row in rows]!=list(range(24)) or len(samples)!=48:
            raise ValueError('Missing, repeated or noncanonical update records')
        for index,row in enumerate(rows):
            expected=.00015 if 8<=index<16 else .0001
            if row['learning_rates']!=[expected] or row['completed_updates']!=index+1 or row['image_presentations']!=2*(index+1):
                raise ValueError('Native schedule/counters do not match registration')
            if any(not math.isfinite(row[k]) for k in ('loss/base','loss/total','loss/cnn','gradient_norm')) or row['gradient_norm']<=0:
                raise ValueError('Non-finite loss or invalid total gradient')
            if max(row['cuda_peak_reserved'],row['sampled_process_gpu_peak'])>28*2**30:
                raise ValueError('Memory gate was not met')
            batch=[s for s in samples if s['native_step']==index]
            if len(batch)!=2 or sorted(s['microbatch_index'] for s in batch)!=[0,1]:
                raise ValueError('Native microbatch accumulation coverage differs')
            for sample in batch:
                if sample['scored']:
                    if kind!='cnn' or index<8:raise ValueError('Wrong scoring schedule')
                    for key in ('raw_scores','clipped_scores'):
                        if len(sample[key])!=12 or not all(math.isfinite(x) for x in sample[key]):raise ValueError('Invalid scores')
                    if not all(1<=x<=10 for x in sample['clipped_scores']):raise ValueError('Invalid clipping policy')
                elif sample['raw_scores'] is not None or sample['clipped_scores'] is not None:
                    raise ValueError('Unscored samples contain fabricated scores')
        quant=read_json(r/'quantization_check.json')
        if any(v['quantized_linears']==0 or v['unquantized_linears'] for v in quant.values()):
            raise ValueError('Actual FP8 quantization incomplete')
        if not any(item['native_step']==8 for item in records(r/'records/checkpoints.jsonl')):
            raise ValueError('Missing complete native step-8 interruption checkpoint')
        if kind=='cnn':
            if read_json(r/'frozen_state_check.json')!={'cnn':True,'vae':True}:raise ValueError('Frozen checks incomplete')
            if not read_json(r/'streetscape_gradient_check.json')['finite_nonzero_lora_gradient']:raise ValueError('Auxiliary gradient missing')
            if not read_json(r/'cnn_memory_actual.json')['decoder_checkpointing']:raise ValueError('VAE memory policy inactive')
        inference=read_json(args.root/'acceptance/memory_v1'/f'{kind}_export_p{args.tier}_v{args.inference_version}/export_inference_acceptance.json')
        if inference['optimizer_steps']!=0 or len(inference['image_files'])!=3 or not all(inference['finite_output_checks'].values()):
            raise ValueError('Native adapter inference incomplete')
        result['runs'][kind]={'native_updates':24,'image_presentations':48,'resume_step':8,
                             'training_revision':identity['research_commit'],
                             'process_peak_gib':max(row['sampled_process_gpu_peak'] for row in rows)/2**30,
                             'reserved_peak_gib':max(row['cuda_peak_reserved'] for row in rows)/2**30,
                             'allocated_peak_gib':max(row['cuda_peak_allocated'] for row in rows)/2**30,
                             'rss_peak_gib':max(row['cpu_rss_peak'] for row in rows)/2**30,
                             'scored_samples':sum(s['scored'] for s in samples),'latest':latest,
                             'inference_images':inference['image_files']}
    result['status']='passed'
    write_json_atomic(args.output,result)
    print(json.dumps({'status':'passed','tier':args.tier,'runs':result['runs']},ensure_ascii=False),flush=True)
