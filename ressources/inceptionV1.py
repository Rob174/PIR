import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.initializers import he_normal
from scripts_personnels.scripts_personnels.models.layers.graph.graphLayers import G_Input,G_Conv2D,G_MaxPooling2D,G_Concatenate,G_Model


def create_model(input_shape):
    """Permet de créer la version 1 du module inception telle que décrite [ici]( https://towardsdatascience.com/a-simple-guide-to-the-versions-of-the-inception-network-7fc52b863202)
    # In:

    - **input_shape** : *tuple* taille des images en entrée en omettant les batchs : [longueur,largeur,channels] (3 dimensions obligatoires)
    # Out:

    - **model** : *Model keras*
    """
    input = G_Input(shape=input_shape,dtype=tf.dtypes.float32,name='image_entree')
    conv = G_Conv2D(filters=64,kernel_size=7,strides=1,kernel_initializer=he_normal(seed=1),bias_initializer="zeros",padding='SAME',activation='relu')(input)
    pool = G_MaxPooling2D(pool_size=2,strides=2,padding='SAME')(conv)
    conv = G_Conv2D(filters=64,kernel_size=1,strides=1,kernel_initializer=he_normal(seed=1),bias_initializer="zeros",padding='SAME',activation='relu')(pool)
    conv = G_Conv2D(filters=128,kernel_size=3,strides=1,kernel_initializer=he_normal(seed=1),bias_initializer="zeros",padding='SAME',activation='relu')(conv)
    pool = G_MaxPooling2D(pool_size=2,strides=2,padding='SAME',name='pool_avant_inception')(conv)

    def inception(conv1_filters,conv3_filters,conv5_filters,pool_red_filters,input,index):
        """Construit un module inception **différents des autres** (voir [couches partagées](https://keras.io/guides/functional_api/#shared-layers) avec les filtres indiqués"""
        # Construction du module
        conv1 = G_Conv2D(filters=conv1_filters,kernel_size=1,strides=1,kernel_initializer=he_normal(seed=1),bias_initializer="zeros",padding='SAME',activation='relu',name='mod%d_conv1'%index)(input)
        conv3red = G_Conv2D(filters=conv3_filters[0],kernel_size=1,strides=1,kernel_initializer=he_normal(seed=1),bias_initializer="zeros",padding='SAME',activation='relu',name='mod%d_conv3red'%index)(input)
        conv3 = G_Conv2D(filters=conv3_filters[1],kernel_size=3,strides=1,kernel_initializer=he_normal(seed=1),bias_initializer="zeros",padding='SAME',activation='relu',name='mod%d_conv3'%index)(conv3red)
        conv5red = G_Conv2D(filters=conv5_filters[0],kernel_size=1,strides=1,kernel_initializer=he_normal(seed=1),bias_initializer="zeros",padding='SAME',activation='relu',name='mod%d_conv5red'%index)(input)
        conv5 = G_Conv2D(filters=conv5_filters[1],kernel_size=5,strides=1,kernel_initializer=he_normal(seed=1),bias_initializer="zeros",padding='SAME',activation='relu',name='mod%d_conv5'%index)(conv5red)
        pool = G_MaxPooling2D(pool_size=3,strides=1,padding='SAME',name='mod%d_pool'%index)(input)
        poolred = G_Conv2D(filters=pool_red_filters,kernel_size=1,strides=1,padding='SAME',activation='relu',name='mod%d_poolred'%index)(pool)
        # Rassemble les résultats
        conv =  G_Concatenate(name='mod%d_concat'%index)([conv1,conv3,conv5,poolred])
        pool = G_MaxPooling2D(pool_size=2,strides=2,padding='SAME',name='mod%d_pool_fin'%index)(conv)
        return pool
    # Construction des modules
    inception1 = inception(conv1_filters=64,
                        conv3_filters=[48,96],
                        conv5_filters=[16,32],
                        pool_red_filters=32,input=pool,index=1)
    inception2 = inception(conv1_filters=96,
                        conv3_filters=[64,128],
                        conv5_filters=[24,48],
                        pool_red_filters=48,input=inception1,index=2)
    inception3 = inception(conv1_filters=128,
                        conv3_filters=[96,192],
                        conv5_filters=[32,64],
                        pool_red_filters=64,input=inception2,index=3)
    inception4 = inception(conv1_filters=192,
                        conv3_filters=[128,256],
                        conv5_filters=[48,96],
                        pool_red_filters=96,input=inception3,index=4)
    model = G_Model(inputs=[input],outputs=[inception4],name="Inception",color="lightblue")
    # print(model.keras_layer.summary())
    # Permet d'afficher une vue sommaire (sans les hyperparamètres des couches) du modèle
    # tf.keras.utils.plot_model(model,to_file="/home/moine/stageIAP/summary/inceptionV4.png",show_shapes=True,show_layer_names=True)
    return model