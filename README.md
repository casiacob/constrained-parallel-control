First clone and install the unconstrained parallel optimal control solver
Clone the repository:

```
$ git clone https://github.com/casiacob/parallel-optimal-control.git
```

Create conda environment:
```
$ conda create --name paroc python=3.10
$ conda activate paroc
$ cd parallel-optimal-control
$ pip install .
```

Clone the repository:
```
$ cd ..
$ git clone https://github.com/casiacob/constrained-parallel-control.git
$ cd constrained-parallel-control
$ pip install .
```

Run a runtime example (requires GPU)

```
$ cd examples
$ python cartpole_runtime_ip.py
```

Run the cart-pole mpc example 
```
$ python cartpole_mpc_ip.py
```
