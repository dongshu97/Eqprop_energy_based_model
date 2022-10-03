To train a model by energy-based supervised EP:
```bash
python main.py --device 0 --dataset mnist --action supervised_ep --epochs 100 --batchSize 256 --dt 0.2 --T 60 --Kmax 20 --beta 0.5 --clamped 1 --fcLayers 784 256 10 --lr 0.01 0.005 --activation_function hardsigm
```

To see the neuron dynamics:
```bash
 python main.py --device 0 --dataset mnist --action test --epochs 1 --dt 0.2 --T 60 --Kmax 20 --beta 0.5 --clamped 1 --fcLayers 784 256 10 --lr 0.01 0.005 --activation_function hardsigm
```

To test a trained model (Remark: you should change the path of trained model firstly):
```bash
python main.py --device 0 --dataset mnist --action visu --epoch 1 --batchSize 256 --dt 0.2 --T 60 --Kmax 20 --beta 0.5 --clamped 1 --fcLayers 784 256 10 --lr 0.01 0.005 --activation_function hardsigm --imWeights 1 --imShape 28 28 16 16 --display 10 10 2 5
```