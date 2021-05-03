# nohup bash -c "python3 /home/rmoine/Documents/PIRfolder/IA/enet/train_transfert.py -bs=10 -gpu=2 -opti=adam -nbImg=7500 -nbEpochs=1 &> /home/rmoine/Documents/PIRfolder/IA/enet/logs"+
# Remise au propre des essais
## Tests de taille (128)
nohup bash -c "python3 train_dataset.py -bs=10 -gpu=... -approxAccur=round -opti=adam -nbImg=7500 -nbEpochs=1 -img_w=128 &> logsAdam" &
nohup bash -c "python3 train_dataset.py -bs=10 -gpu=... -approxAccur=round -opti=sgd -nbImg=7500 -nbEpochs=1 -img_w=128 &> logsSGD" &

## Tests nb de modules taille (128)
nohup bash -c "python3 train_dataset.py -bs=10 -gpu=... -approxAccur=round -opti=adam -nbImg=7500 -nbEpochs=1 -nbMod=1 &> logsAdam" &
nohup bash -c "python3 train_dataset.py -bs=10 -gpu=... -approxAccur=round -opti=adam -nbImg=7500 -nbEpochs=1 -nbMod=2 &> logsAdam" &
nohup bash -c "python3 train_dataset.py -bs=10 -gpu=... -approxAccur=round -opti=adam -nbImg=7500 -nbEpochs=1 -nbMod=3 &> logsAdam" &

#nohup bash -c "python3 train_dataset.py -bs=10 -gpu=... -approxAccur=round -opti=sgd -nbImg=7500 -nbEpochs=1 -nbMod=1 &> logsSGD" &
#nohup bash -c "python3 train_dataset.py -bs=10 -gpu=... -approxAccur=round -opti=sgd -nbImg=7500 -nbEpochs=1 -nbMod=2 &> logsSGD" &
#nohup bash -c "python3 train_dataset.py -bs=10 -gpu=... -approxAccur=round -opti=sgd -nbImg=7500 -nbEpochs=1 -nbMod=3 &> logsSGD" &

# Dropout (0.2 et 0)

nohup bash -c "python3 train_dataset.py -bs=10 -gpu=... -approxAccur=round -opti=adam -nbImg=7500 -nbEpochs=1 -dptRate=0.5 &> logsAdam" &
nohup bash -c "python3 train_dataset.py -bs=10 -gpu=... -approxAccur=round -opti=adam -nbImg=7500 -nbEpochs=1 -dptRate=0.0 &> logsAdam" &

#nohup bash -c "python3 train_dataset.py -bs=10 -gpu=... -approxAccur=round -opti=sgd -nbImg=7500 -nbEpochs=1 -dptRate=0.5 &> logsSGD" &
#nohup bash -c "python3 train_dataset.py -bs=10 -gpu=... -approxAccur=round -opti=sgd -nbImg=7500 -nbEpochs=1 -dptRate=0.0 &> logsSGD" &

# Regularization

nohup bash -c "python3 train_dataset.py -bs=10 -gpu=... -approxAccur=round -opti=adam -nbImg=7500 -nbEpochs=1 -regMod=y &> logsAdam" &
nohup bash -c "python3 train_dataset.py -bs=10 -gpu=... -approxAccur=round -opti=adam -nbImg=7500 -nbEpochs=1 -regMod=y &> logsAdam" &

#nohup bash -c "python3 train_dataset.py -bs=10 -gpu=... -approxAccur=round -opti=sgd -nbImg=7500 -nbEpochs=1 -regMod=y &> logsSGD" &
#nohup bash -c "python3 train_dataset.py -bs=10 -gpu=... -approxAccur=round -opti=sgd -nbImg=7500 -nbEpochs=1 -regMod=y &> logsSGD" &

# Couche de passage en vecteur

nohup bash -c "python3 train_dataset.py -bs=10 -gpu=... -approxAccur=round -opti=adam -nbImg=7500 -nbEpochs=1 -redLayer=spp &> logsAdam" &
nohup bash -c "python3 train_dataset.py -bs=10 -gpu=... -approxAccur=round -opti=adam -nbImg=7500 -nbEpochs=1 -redLayer=flatten &> logsAdam" &

#nohup bash -c "python3 train_dataset.py -bs=10 -gpu=... -approxAccur=round -opti=sgd -nbImg=7500 -nbEpochs=1 -redLayer=spp &> logsSGD" &
#nohup bash -c "python3 train_dataset.py -bs=10 -gpu=... -approxAccur=round -opti=sgd -nbImg=7500 -nbEpochs=1 -redLayer=flatten &> logsSGD" &

# Fonction d'activation finale

nohup bash -c "python3 train_dataset.py -bs=10 -gpu=... -approxAccur=round -opti=adam -nbImg=7500 -nbEpochs=1 -lastAct=relu &> logsAdam" &
nohup bash -c "python3 train_dataset.py -bs=10 -gpu=... -approxAccur=round -opti=adam -nbImg=7500 -nbEpochs=1 -lastAct=softplus &> logsAdam" &
nohup bash -c "python3 train_dataset.py -bs=10 -gpu=... -approxAccur=round -opti=adam -nbImg=7500 -nbEpochs=1 -lastAct=exp &> logsAdam" &

#nohup bash -c "python3 train_dataset.py -bs=10 -gpu=... -approxAccur=round -opti=sgd -nbImg=7500 -nbEpochs=1 -lastAct=relu  &> logsSGD" &
#nohup bash -c "python3 train_dataset.py -bs=10 -gpu=... -approxAccur=round -opti=sgd -nbImg=7500 -nbEpochs=1 -lastAct=softplus  &> logsSGD" &
#nohup bash -c "python3 train_dataset.py -bs=10 -gpu=... -approxAccur=round -opti=sgd -nbImg=7500 -nbEpochs=1 -lastAct=exp  &> logsSGD" &

# epsilon lr

nohup bash -c "python3 train_dataset.py -bs=10 -gpu=... -approxAccur=round -opti=adam -nbImg=7500 -nbEpochs=1 -eps=1e-2 -lr=1e-5 &> logsAdam" &
nohup bash -c "python3 train_dataset.py -bs=10 -gpu=... -approxAccur=round -opti=adam -nbImg=7500 -nbEpochs=1 -eps=1e-7 -lr=1e-4 &> logsAdam" &
nohup bash -c "python3 train_dataset.py -bs=10 -gpu=... -approxAccur=round -opti=adam -nbImg=7500 -nbEpochs=1 -eps=1e-5 -lr=1e-3 &> logsAdam" &
nohup bash -c "python3 train_dataset.py -bs=10 -gpu=... -approxAccur=round -opti=adam -nbImg=7500 -nbEpochs=1 -eps=1e-2 -lr=1e-3 &> logsAdam" &
nohup bash -c "python3 train_dataset.py -bs=10 -gpu=... -approxAccur=round -opti=adam -nbImg=7500 -nbEpochs=1 -eps=1e-7 -lr=1e-1 &> logsAdam" &
