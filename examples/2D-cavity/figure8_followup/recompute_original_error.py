"""Recalculate figures/metrics from saved FVM predictions without retraining."""
import json

import numpy as np
from PIL import Image

from train_fvm_pirn import OUT, draw_case, draw_schematic, score


def main():
    metric_path=OUT/'metrics.json'
    report=json.loads(metric_path.read_text(encoding='utf-8'))
    report['original_project_error_formula']='abs((reference - prediction) / reference.max())'
    for re in (15,20.1,30):
        z=np.load(OUT/f'Re_{re:g}_prediction_vs_FVM.npz')
        pred=z['prediction'];ref=z['reference']
        report['cases'][str(re)]=score(pred,ref)
        draw_case(re,pred,ref,OUT/f'Figure9_FVM_original_error_Re_{re:g}.png')
        print(f'Re={re:g}',{k:{'mean':v['original_error_mean'],
                               'median':v['original_error_median'],
                               'max':v['original_error_max']} for k,v in report['cases'][str(re)].items()},flush=True)
    metric_path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    draw_schematic(OUT/'Figure9a_schematic.png')
    ims=[Image.open(OUT/'Figure9a_schematic.png').convert('RGB')]+[
        Image.open(OUT/f'Figure9_FVM_original_error_Re_{r:g}.png').convert('RGB') for r in (15,20.1,30)]
    width=max(im.width for im in ims)
    out=Image.new('RGB',(width,sum(im.height for im in ims)),'white');y=0
    for im in ims:out.paste(im,((width-im.width)//2,y));y+=im.height
    out.save(OUT/'Figure9_FVM_original_error_all_cases.png',optimize=True)


if __name__=='__main__':main()
