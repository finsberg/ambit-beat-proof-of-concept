# Coupling fenics-beat with ambit

This repository contains a proof of concept for coupling [`fenics-beat`](https://github.com/finsberg/fenics-beat) together with [`ambit`](https://github.com/marchirschvogel/ambit).
Note that `fenics-beat` uses legacy FEniCS while `ambit` uses dolfinx so the two simulators needs to be run independently.

The initial idea would be to first run a monodomain simulation using `fenics-beat` and then saving the resulting active tension in a format that can be read by `ambit`.
The we will use that active tension to drive the mechanics simulator in `ambit`.

For the time being we use the demo and mesh provided by `ambit` in the demo [solid_flow0d](https://github.com/marchirschvogel/ambit/tree/master/demos/solid_flow0d).


## Running the Mondodomain simulator

### Set up environment
We will use a docker image for legacy FEniCS
```
docker run --name beat -w /home/shared -v $PWD:/home/shared -it ghcr.io/scientificcomputing/fenics-gmsh:2023-11-15
```
Next we install the development version of `fenics-beat` and `gotranx`
```
python3 -m pip install git+https://github.com/finsberg/fenics-beat
python3 -m pip install git+https://github.com/finsberg/gotranx
```
Note that eventually we will converge on a specific version (or the latest stable version), but for the time being we will use the development version.

### Run
```
python3 main_beat.py
```

### Converting active tension to a format that can be read by `ambit`
```
python3 convert_Ta.py
```



## Running Mechanics Simulator

### Set up environment
Ambit currently works with `dolfinx` version 0.6, so we will use a docker image with this version
e.g
```
docker run --name ambit -w /home/shared -v $PWD:/home/shared -it ghcr.io/fenics/dolfinx/dolfinx:v0.6.0
```
Next we install the development version of `ambit`
```
python3 -m pip install git+git+https://github.com/marchirschvogel/ambit
```
### Run

```
python3 main_ambit.py
```
