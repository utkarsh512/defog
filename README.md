# defog
Implementation of the paper "Single Image Defogging by Multiscale Depth Fusion" by Yuan-Kai Wang and  Ching-Tang Fan

## Installing
```bash
$ git clone https://github.com/utkarsh512/defog.git
$ cd defog
$ pip install .
```

## Running
```
usage: python3 defog [options]

defog - single image defogging by multiscale depth fusion

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        path to input image
  -o OUTPUT, --output OUTPUT
                        path for output image
  -p PATCHES [PATCHES ...], --patches PATCHES [PATCHES ...]
                        size of patches
  -r MAX_ITER, --max_iter MAX_ITER
                        maximum iteration to run
  -t THRESHOLD, --threshold THRESHOLD
                        `T` in pg. 5, eqn. 15
  -l LAMBDA_, --lambda_ LAMBDA_
                        regularization factor
  -v V_MAX, --v_max V_MAX
                        upper bound on potential
```
