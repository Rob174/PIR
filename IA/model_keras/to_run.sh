# nohup bash -c "python3 /home/rmoine/Documents/PIRfolder/IA/enet/train_transfert.py -bs=10 -gpu=2 -opti=adam -nbImg=7500 -nbEpochs=1 &> /home/rmoine/Documents/PIRfolder/IA/enet/logs"+
# Remise au propre des essais
## Tests de taille (128)
nohup bash -c "python3 train_dataset.py -bs=10 -gpu=0 -opti=adam -nbImg=7500 -nbEpochs=1 -img_w=128 &> logsAdam" & # TODO : DONE 2021-05-04_00h17min48s;
nohup bash -c "python3 train_dataset.py -bs=10 -gpu=1 -opti=sgd -nbImg=7500 -nbEpochs=1 -img_w=128 &> logsSGD" & # TODO : DONE 2021-05-04_00h18min12s

nohup bash -c "python3 train_dataset.py -bs=10 -gpu=0 -opti=adam -nbImg=7500 -nbEpochs=1 -img_w=128 -approxAccur=round &> logsAdam" & # TODO : DONE 2021-05-04_13h38min12s ;
nohup bash -c "python3 train_dataset.py -bs=10 -gpu=1 -opti=sgd -nbImg=7500 -nbEpochs=1 -img_w=128 -approxAccur=round &> logsSGD" & # TODO : DONE 2021-05-04_13h39min31s

nohup bash -c "python3 train_dataset.py -bs=10 -gpu=0 -opti=adam -nbImg=7500 -nbEpochs=1 -img_w=128 -approxAccur=int &> logsAdams18" & # TODO : doing 2021-05-06_19h34min00s ;
nohup bash -c "python3 train_dataset.py -bs=10 -gpu=0 -opti=sgd -nbImg=7500 -nbEpochs=1 -img_w=128 -approxAccur=int &> logsSGD19" & # TODO : doing 2021-05-06_19h36min07s
## Tests nb de modules taille (128)
nohup bash -c "python3 train_dataset.py -bs=10 -gpu=2 -opti=adam -nbImg=7500 -nbEpochs=1 -nbMod=1 &> logsAdam1" & # TODO : DONE 2021-05-04_13h39min38s
nohup bash -c "python3 train_dataset.py -bs=10 -gpu=3 -opti=adam -nbImg=7500 -nbEpochs=1 -nbMod=2 &> logsAdam2" & # TODO : DONE 2021-05-04_13h39min45s
nohup bash -c "python3 train_dataset.py -bs=10 -gpu=3 -opti=adam -nbImg=7500 -nbEpochs=1 -nbMod=3 &> logsAdam3" & # TODO :
nohup bash -c "python3 train_dataset.py -bs=10 -gpu=0 -opti=adam -nbImg=7500 -nbEpochs=1 -nbMod=1 -approxAccur=round &> logsAdam1" & # TODO : DONE 2021-05-04_20h53min54s
nohup bash -c "python3 train_dataset.py -bs=10 -gpu=2 -opti=adam -nbImg=7500 -nbEpochs=1 -nbMod=2 -approxAccur=round &> logsAdam24" & # TODO : DONE 2021-05-04_22h59min19s
nohup bash -c "python3 train_dataset.py -bs=10 -gpu=2 -opti=adam -nbImg=7500 -nbEpochs=1 -nbMod=3 -approxAccur=round &> logsAdam34 " & # TODO : DONE 2021-05-04_23h01min18s

nohup bash -c "python3 train_dataset.py -bs=10 -gpu=0 -opti=adam -nbImg=7500 -nbEpochs=1 -nbMod=1 -approxAccur=int &> logsAdamss1" & # TODO : doing 2021-05-06_11h51min44s
nohup bash -c "python3 train_dataset.py -bs=10 -gpu=1 -opti=adam -nbImg=7500 -nbEpochs=1 -nbMod=2 -approxAccur=int &> logsAdamss2" & # TODO : doing 2021-05-06_11h51min58s
nohup bash -c "python3 train_dataset.py -bs=10 -gpu=0 -opti=adam -nbImg=7500 -nbEpochs=1 -nbMod=3 -approxAccur=int &> logsAdamss3 " & # TODO : doing 2021-05-06_11h53min30s

#nohup bash -c "python3 train_dataset.py -bs=10 -gpu=... -approxAccur=round -opti=sgd -nbImg=7500 -nbEpochs=1 -nbMod=1 &> logsSGD" &
#nohup bash -c "python3 train_dataset.py -bs=10 -gpu=... -approxAccur=round -opti=sgd -nbImg=7500 -nbEpochs=1 -nbMod=2 &> logsSGD" &
#nohup bash -c "python3 train_dataset.py -bs=10 -gpu=... -approxAccur=round -opti=sgd -nbImg=7500 -nbEpochs=1 -nbMod=3 &> logsSGD" &

# Regex :  2021-05-04_20h53min54s|2021-05-04_22h59min19s|2021-05-04_23h01min18s|2021-05-06_11h51min44s|2021-05-06_11h51min58s|2021-05-06_11h53min30s

# Tests de référence

nohup bash -c "python3 train_dataset.py -bs=10 -gpu=3 -opti=adam -nbImg=7500 -nbEpochs=1 &> logsAdam9999" & # TODO : DONE 2021-05-05_21h19min48s
nohup bash -c "python3 train_dataset.py -bs=10 -gpu=2 -opti=sgd -nbImg=7500 -nbEpochs=1 &> logsSGD1" & # TODO : DONE 2021-05-04_20h15min16s
nohup bash -c "python3 train_dataset.py -bs=10 -gpu=2 -opti=adam -nbImg=7500 -nbEpochs=1 -approxAccur=round &> logsAdam2" & # TODO : DONE 2021-05-01_23h54min26s 2021-05-04_20h15min51s
nohup bash -c "python3 train_dataset.py -bs=10 -gpu=1 -opti=sgd -nbImg=7500 -nbEpochs=1 -approxAccur=round &> logsSGD2" & # TODO : DONE 2021-05-02_19h21min15s 2021-05-04_20h49min02s
nohup bash -c "python3 train_dataset.py -bs=10 -gpu=0 -opti=adam -nbImg=7500 -nbEpochs=1 -approxAccur=int &> logsAdam31" & # TODO : REFAIRE 2021-05-04_20h28min41s --à refaire avec fonction approx matrix de conf int
nohup bash -c "python3 train_dataset.py -bs=10 -gpu=0 -opti=sgd -nbImg=7500 -nbEpochs=1 -approxAccur=int &> logsSGD31" & # TODO : REFAIRE 2021-05-04_22h52min51s --à refaire avec fonction approx matrix de conf int

# avec bonne matrice de confusion
nohup bash -c "python3 train_dataset.py -bs=10 -gpu=1 -opti=adam -nbImg=7500 -nbEpochs=1 -approxAccur=int &> logsAdamsss1" & # TODO : doing 2021-05-06_19h46min23s
nohup bash -c "python3 train_dataset.py -bs=10 -gpu=2 -opti=sgd -nbImg=7500 -nbEpochs=1 -approxAccur=int &> logsSGDss2" & # TODO : doing 2021-05-06_19h46min46s

# regex d'affichage : 2021-05-05_21h19min48s|2021-05-04_20h15min16s|2021-05-04_20h15min51s|2021-05-04_20h49min02s|2021-05-06_19h46min23s|2021-05-06_19h46min46s

# Adam 2021-05-05_21h19min48s|2021-05-04_20h15min51s|2021-05-04_20h28min41s
# SGD 2021-05-04_20h15min16s|2021-05-04_20h49min02s|2021-05-04_22h52min51s
# Code du modèle de référence : 2021-05-04_20h15min51s

# Dropout (0.2 et 0)

nohup bash -c "python3 train_dataset.py -bs=10 -gpu=1 -opti=adam -nbImg=7500 -nbEpochs=1 -dptRate=0.5 -approxAccur=round &> logsAdam4" & # TODO : DONE 2021-05-04_23h13min51s
nohup bash -c "python3 train_dataset.py -bs=10 -gpu=1 -opti=adam -nbImg=7500 -nbEpochs=2 -dptRate=0.2 -approxAccur=round &> logsAdam411" & # TODO : doing 2021-05-06_19h39min10s
nohup bash -c "python3 train_dataset.py -bs=10 -gpu=2 -opti=adam -nbImg=7500 -nbEpochs=1 -dptRate=0.0 -approxAccur=round &> logsAdam5" & # TODO : DONE 2021-05-05_10h52min37s

nohup bash -c "python3 train_dataset.py -bs=10 -gpu=0 -opti=adam -nbImg=7500 -nbEpochs=2 -dptRate=0.2 -approxAccur=int &> logsAdamz1" & # TODO : DONE 2021-05-07_07h50min31s
nohup bash -c "python3 train_dataset.py -bs=10 -gpu=0 -opti=adam -nbImg=7500 -nbEpochs=1 -dptRate=0.0 -approxAccur=int &> logsAdamz2" & # TODO : DONE 2021-05-07_07h50min38s

# Regex : 2021-05-04_23h13min51s|2021-05-06_19h39min10s|2021-05-05_10h52min37s|2021-05-07_07h50min31s|2021-05-07_07h50min38s|2021-05-06_19h46min23s

#nohup bash -c "python3 train_dataset.py -bs=10 -gpu=... -approxAccur=round -opti=sgd -nbImg=7500 -nbEpochs=1 -dptRate=0.5 &> logsSGD" &
#nohup bash -c "python3 train_dataset.py -bs=10 -gpu=... -approxAccur=round -opti=sgd -nbImg=7500 -nbEpochs=1 -dptRate=0.0 &> logsSGD" &

# Regularization

nohup bash -c "python3 train_dataset.py -bs=10 -gpu=1 -approxAccur=round -opti=adam -nbImg=7500 -nbEpochs=1 -regMod=y -approxAccur=round &> logsAdam6" & # TODO : DONE 2021-05-05_10h51min49s
nohup bash -c "python3 train_dataset.py -bs=10 -gpu=1 -approxAccur=round -opti=adam -nbImg=7500 -nbEpochs=1 -approxAccur=round &> logsAdam7" & # TODO : DONE 2021-05-05_10h58min10s
nohup bash -c "python3 train_dataset.py -bs=10 -gpu=1 -approxAccur=round -opti=adam -nbImg=7500 -nbEpochs=1 -regMod=y -approxAccur=int &> logsAdamz3" & # TODO : doing

# Regex : 2021-05-05_10h51min49s|2021-05-05_10h58min10s|2021-05-04_20h15min51s (référence)

#nohup bash -c "python3 train_dataset.py -bs=10 -gpu=... -approxAccur=round -opti=sgd -nbImg=7500 -nbEpochs=1 -regMod=y &> logsSGD" &
#nohup bash -c "python3 train_dataset.py -bs=10 -gpu=... -approxAccur=round -opti=sgd -nbImg=7500 -nbEpochs=1 -regMod=y &> logsSGD" &

# Couche de passage en vecteur

nohup bash -c "python3 train_dataset.py -bs=10 -gpu=1 -approxAccur=round -opti=adam -nbImg=7500 -nbEpochs=1 -redLayer=spp &> logsAdam8" & # TODO : DONE 2021-05-05_16h47min24s
nohup bash -c "python3 train_dataset.py -bs=10 -gpu=2 -approxAccur=round -opti=adam -nbImg=7500 -nbEpochs=1 -redLayer=flatten &> logsAdam9" & # TODO : DONE 2021-05-05_16h48min51s

# Regex : 2021-05-05_16h47min24s|2021-05-05_16h48min51s|2021-05-04_20h15min51s (référence)

#nohup bash -c "python3 train_dataset.py -bs=10 -gpu=... -approxAccur=round -opti=sgd -nbImg=7500 -nbEpochs=1 -redLayer=spp &> logsSGD" &
#nohup bash -c "python3 train_dataset.py -bs=10 -gpu=... -approxAccur=round -opti=sgd -nbImg=7500 -nbEpochs=1 -redLayer=flatten &> logsSGD" &

# Fonction d'activation finale

nohup bash -c "python3 train_dataset.py -bs=10 -gpu=0 -approxAccur=round -opti=adam -nbImg=7500 -nbEpochs=1 -lastAct=relu &> logsAdam10" & # TODO : 2021-05-05_16h50min08s
nohup bash -c "python3 train_dataset.py -bs=10 -gpu=0 -approxAccur=round -opti=adam -nbImg=7500 -nbEpochs=1 -lastAct=softplus &> logsAdams11" & # TODO : DONE 2021-05-05_20h57min21s
nohup bash -c "python3 train_dataset.py -bs=10 -gpu=1 -approxAccur=round -opti=adam -nbImg=7500 -nbEpochs=1 -lastAct=exponential &> logsAdams12" & # TODO : DONE 2021-05-05_21h01min04s

# Regex 2021-05-05_16h50min08s|2021-05-05_20h57min21s|2021-05-05_21h01min04s|2021-05-04_20h15min51s

#nohup bash -c "python3 train_dataset.py -bs=10 -gpu=... -approxAccur=round -opti=sgd -nbImg=7500 -nbEpochs=1 -lastAct=relu  &> logsSGD" &
#nohup bash -c "python3 train_dataset.py -bs=10 -gpu=... -approxAccur=round -opti=sgd -nbImg=7500 -nbEpochs=1 -lastAct=softplus  &> logsSGD" &
#nohup bash -c "python3 train_dataset.py -bs=10 -gpu=... -approxAccur=round -opti=sgd -nbImg=7500 -nbEpochs=1 -lastAct=exp  &> logsSGD" &

# epsilon lr

nohup bash -c "python3 train_dataset.py -bs=10 -gpu=2 -approxAccur=round -opti=adam -nbImg=7500 -nbEpochs=1 -eps=1e-2 -lr=1e-5 &> logsAdams13" & # TODO : DONE 2021-05-05_21h04min24s
nohup bash -c "python3 train_dataset.py -bs=10 -gpu=0 -approxAccur=round -opti=adam -nbImg=7500 -nbEpochs=1 -eps=1e-7 -lr=1e-4 &> logsAdams14" & # TODO : doing 2021-05-06_00h15min22s
nohup bash -c "python3 train_dataset.py -bs=10 -gpu=1 -approxAccur=round -opti=adam -nbImg=7500 -nbEpochs=1 -eps=1e-5 -lr=1e-3 &>  logsAdams15" & # TODO : doing 2021-05-06_00h17min08s
nohup bash -c "python3 train_dataset.py -bs=10 -gpu=2 -approxAccur=round -opti=adam -nbImg=7500 -nbEpochs=1 -eps=1e-2 -lr=1e-3 &> logsAdams16" & # TODO : doing 2021-05-06_00h17min28s
nohup bash -c "python3 train_dataset.py -bs=10 -gpu=3 -approxAccur=round -opti=adam -nbImg=7500 -nbEpochs=1 -eps=1e-7 -lr=1e-1 &> logsAdams17" & # TODO : doing 2021-05-06_00h19min21s

# Regex 2021-05-05_21h04min24s|2021-05-06_00h15min22s|2021-05-06_00h17min08s|2021-05-06_00h17min28s|2021-05-06_00h19min21s|2021-05-04_20h15min51s