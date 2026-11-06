import numpy as np

LENGTH=64600
RATE=16000


def variants(x):
    assert x.ndim==1 and len(x)>=LENGTH and np.isfinite(x).all()
    yield dict(condition='official',duration_s=4.0375,source_eighths=0,source_offset=0,input_offset=0,padding='none',payload_samples=LENGTH),x[:LENGTH]
    for duration in (1,2,4):
        size=duration*RATE
        for k,name in ((0,'start'),(4,'middle'),(8,'end')):
            offset=((len(x)-size)*k)//8;crop=x[offset:offset+size]
            for padding in ('repeat','zero'):
                y=np.tile(crop,int(np.ceil(LENGTH/size)))[:LENGTH] if padding=='repeat' else np.pad(crop,(0,LENGTH-size))
                yield dict(condition=f'{duration}s_{name}_{padding}',duration_s=duration,source_eighths=k,source_offset=offset,input_offset=0,padding=padding,payload_samples=size),np.asarray(y,dtype=np.float32)
    for duration in (1,2):
        size=duration*RATE
        for k in (1,2,3,5,6,7):
            offset=((len(x)-size)*k)//8;crop=x[offset:offset+size]
            y=np.tile(crop,int(np.ceil(LENGTH/size)))[:LENGTH]
            yield dict(condition=f'{duration}s_offset{k}of8_repeat',duration_s=duration,source_eighths=k,source_offset=offset,input_offset=0,padding='repeat',payload_samples=size),np.asarray(y,dtype=np.float32)
        for k,name in ((0,'start'),(4,'middle'),(8,'end')):
            offset=((len(x)-size)*k)//8;crop=x[offset:offset+size]
            for placement,j in (('center',(LENGTH-size)//2),('right',LENGTH-size)):
                y=np.zeros(LENGTH,dtype=np.float32);y[j:j+size]=crop
                assert np.array_equal(y[j:j+size],crop)
                yield dict(condition=f'{duration}s_{name}_zero_{placement}',duration_s=duration,source_eighths=k,source_offset=offset,input_offset=j,padding='zero',payload_samples=size),y


def repeat_name(duration,k):
    return f'{duration}s_'+({0:'start',4:'middle',8:'end'}[k] if k in (0,4,8) else f'offset{k}of8')+'_repeat'
