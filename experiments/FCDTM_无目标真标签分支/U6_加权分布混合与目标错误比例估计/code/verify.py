import sys,json
import numpy as np
import common as c

def unit():
    ref=np.zeros((2,2,3,8));ref[:,0,:,1]=100000;ref[:,1,:,6]=100000
    for rate in [0.,.15,.5,.85,1.]:
        h=(1-rate)*ref[:,0]+rate*ref[:,1];count,d=c.mixture(h,ref,0.)
        assert max(abs(q['error_rate']-rate) for q in d)<1e-5
        assert np.all(count>=0) and abs(count[0]+count[1]-.5)<1e-12
    equal=ref.copy();equal[:,1]=equal[:,0];_,d=c.mixture(equal.sum(1),equal,0.);assert all(q['nonidentifiable'] and abs(q['error_rate']-.5)<1e-12 for q in d)
    empty=ref.sum(1);empty[1]=0;cnt,d=c.mixture(empty,ref,0.);assert d[1]['empty_target_side'] and cnt[0]==cnt[1]==0
    print('PASS synthetic known mixture recovery, legal counts, unidentifiable and absent-side cases')

def independent(h,ref,lam):
    # Independently solve the one-coefficient bounded least-squares problem.
    q=(ref+.5)/(ref+.5).sum(-1,keepdims=True);target=(h+.5)/(h+.5).sum(-1,keepdims=True);e=[]
    for side in range(2):
        pri=(ref[side,1,0].sum()+.5)/(ref[side,:,0].sum()+1);a=(q[side,1]-q[side,0]).ravel()/np.sqrt(3);b=(target[side]-q[side,0]).ravel()/np.sqrt(3)
        if a@a<=1e-10 or not h[side].sum():e.append(pri);continue
        aa=np.r_[a,np.sqrt(lam)];bb=np.r_[b,np.sqrt(lam)*pri];e.append(float(np.clip(np.linalg.lstsq(aa[:,None],bb,rcond=None)[0][0],0,1)))
    area=h[1,0].sum()/h[:,0].sum();return c.f1([area*(1-e[1]),area*e[1],(1-area)*e[0]])

def final():
    m=c.check();c.check_seal();records=c.read(c.OUT/'predictions.csv');diff=[];scalars=0
    for t in m['tasks']:
        cls,src,tgt=t['cls'],t['src'],t['tgt'];p=json.loads((c.OUT/f'source/{cls}_{src}.json').read_text())
        with np.load(c.OUT/f'source/{cls}_{src}.npz') as z,np.load(c.OUT/f'unlabeled_cache/{cls}_{src}__{tgt}.npz') as h:
            bits=np.unpackbits(h['bits'],axis=1,count=65536);areas=bits.sum(1)
            assert np.array_equal(h['hist_weighted'][:,1,0].sum(-1),areas);assert np.array_equal(h['hist_uniform'][:,1,0].sum(-1),areas)
            for row in records:
                if (int(row['cls']),row['src'],row['tgt'])!=(cls,src,tgt):continue
                ix=np.array(json.loads(row['indices']))
                for mode in c.MODES:
                    ref=z[mode][z['fit']].sum(0);agg=h['hist_'+mode][ix].sum(0)
                    for suffix,lam in [('',p['params'][mode]['lam']),('_raw',0.)]:
                        x=independent(agg,ref,lam);y=float(row['mix_'+mode+suffix]);assert np.isfinite(x)==np.isfinite(y)
                        if np.isfinite(x):diff.append(abs(x-y))
                        scalars+=1
    assert scalars==5200 and max(diff)<1e-10
    result=dict(status='passed',records=len(records),scalar_mixture_predictions_replayed=scalars,max_difference=max(diff),target_labels_read=False)
    (c.OUT/'独立复算核验.json').write_text(json.dumps(result,indent=1));print(json.dumps(result))
if __name__=='__main__':unit() if sys.argv[1]=='unit' else final()
