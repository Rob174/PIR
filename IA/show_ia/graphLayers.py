import types

from tensorflow.keras.layers import Input,Conv2D,MaxPooling2D,AveragePooling2D,Concatenate,Dense
from tensorflow.keras import Model
import graphviz
import inspect

from tensorflow.keras.layers import Dropout, Flatten

class G_Node:
    id = 0
    def __init__(self,graphviz_params,**layer_params):
        self.keras_layer = None
        G_Node.id += 1
        self.graphviz_params = {**graphviz_params,"style":"filled","fillcolor":"white"} \
            if "G_Model" not in self.__class__.__name__ else graphviz_params
        self.id = G_Node.id
        self.layer_params = layer_params
        self.tenseur = None
        self.parents = []# 1 layer n'a qu'un seul layer parent
        self.enfants = [] # un layer peut avoir plusieurs enfants
        self.graph_done = False
        self.output_node = False
    def __call__(self, input_node):
        if input_node.tenseur is None:
            raise Exception("No tensor input")
        else:
            input_node.enfants.append(self)
            self.parent = [input_node]
            if "G_Input" not in self.__class__.__name__:
                self.tenseur = self.keras_layer(input_node.tenseur)

            return self
    def link(self,parent_graph):
        for child in self.enfants:
            if "G_Model" in child.__class__.__name__:
                child.link_to_inputs(self,parent_graph)
            else:
                parent_graph.edge(str(self.id),str(child.id),label=str(self.tenseur.get_shape().as_list()))

        for child in self.enfants:
            child.build(parent_graph)
    def parse_str_args(self,arg_value):
        parsed_value = str(arg_value)
        if parsed_value.strip()[0] != "<":
            return parsed_value
        elif callable(arg_value):
            try:
                return arg_value.__name__
            except:
                return arg_value.__class__.__name__.split(".")[-1]
        else:
            try:
                return arg_value.name
            except Exception as e:
                print("EXCEPTION parsage en str d'un attribut")
                return " "
    def build(self,parent_graph):
        if len(list(filter(lambda p:p.graph_done is False,self.parents))) > 0:
            return
        label_attributs = " | ".join([("{%s|%s}" % (k.capitalize(), self.parse_str_args(v))) for k, v in self.layer_params.items()])
        label = "{{%s%s %d|{%s%s}}}" % (self.output()[0],self.__class__.__name__, self.id, self.output()[1],label_attributs)
        parent_graph.node(str(self.id), label=label, shape="record", **self.graphviz_params)
        self.graph_done = True
        self.link(parent_graph)

    def output(self):
        if self.output_node == True:
            return r"Output\n","{Output_shape|%s}|"%(str(self.tenseur.get_shape().as_list()))
        else:
            return "",""
class G_Conv2D(G_Node):
    def __init__(self, **kargs):
        super().__init__({"color":"blue"},**kargs)
        self.keras_layer = Conv2D(**kargs)
class G_Dense(G_Node):
    def __init__(self, **kargs):
        super().__init__({"color":"blue"},**kargs)
        self.keras_layer = Dense(**kargs)
class G_Input(G_Node):
    def __init__(self, **kargs):
        super().__init__({"color":"yellow"},**kargs)
        self.keras_layer = Input(**kargs)
        self.tenseur = self.keras_layer
class G_MaxPooling2D(G_Node):
    def __init__(self, **kargs):
        super().__init__({"color":"red"},**kargs)
        self.keras_layer = MaxPooling2D(**kargs)
class G_AveragePooling2D(G_Node):
    def __init__(self, **kargs):
        super().__init__({"color":"orange"},**kargs)
        self.keras_layer = AveragePooling2D(**kargs)
class G_Dropout(G_Node):
    def __init__(self, **kargs):
        super().__init__({"color":"orange"},**kargs)
        self.keras_layer = Dropout(**kargs)
class G_Flatten(G_Node):
    def __init__(self, **kargs):
        super().__init__({"color":"orange"},**kargs)
        self.keras_layer = Flatten(**kargs)
class G_Concatenate(G_Node):
    def __init__(self, **kargs):
        super().__init__({"color":"black"},**kargs)
        self.keras_layer = Concatenate(**kargs)
    def __call__(self, input_nodes):
        if type(input_nodes) != list:
            raise Exception("A Concatenate layer should be called on a list of inputs")
        elif len(list(filter(lambda x:x.tenseur is None,input_nodes))) > 0:
            raise Exception("No tensor input")
        else:
            for node in input_nodes:
                node.enfants.append(self)
            self.parents = input_nodes
            self.tenseur = self.keras_layer(list(map(lambda x:x.tenseur,input_nodes)))
            return self
    def build(self,parent_graph):
        if len(list(filter(lambda p:p.graph_done is False,self.parents))) > 0:
            return
        label = "{{%s%s %d|%s}}" % (self.output()[0],self.__class__.__name__, self.id,self.output()[1])
        parent_graph.node(str(self.id), label=label, shape="record", **self.graphviz_params)
        self.link(parent_graph)
        self.graph_done = True

class G_Model(G_Node):
    def __init__(self,inputs,outputs,name,color="white"):
        super().__init__({"style":"filled","fillcolor":color,"peripheries":str(0)},**{"inputs":inputs,"outputs":outputs})
        self.inputs_used = 0
        inputs = list(map(lambda x:x. tenseur,inputs))
        outputs = list(map(lambda x:x. tenseur,outputs))
        self.keras_layer = Model(inputs=inputs,outputs=outputs)
        self.graph = None
    def __call__(self, inputs):
        for input in inputs:
            input.enfants.append(self)
        self.tenseur = self.keras_layer(list(map(lambda x:x.tenseur,inputs)))
        return self
    def link_to_inputs(self,input,parent_graph):
        if self.inputs_used == len(self.layer_params["inputs"]):
            raise Exception("All inputs used")
        else:
            parent_graph.edge(str(input.id),str(self.layer_params["inputs"][self.inputs_used].id),
                              label=str(input.tenseur.get_shape().as_list()))
            self.inputs_used += 1

    def render(self,out_path):
        graph = graphviz.Digraph(name="Main",format="png")
        for x in self.layer_params["outputs"]:
            x.output_node = True
        self.build(graph)

        graph.render(out_path)
    def build(self,parent_graph):
        with parent_graph.subgraph(name='cluster_'+str(self.id)) as c:
            for k,v in self.graphviz_params.items():
                c.graph_attr[k] = v
            for input in self.layer_params["inputs"]:
                input.build(c)
        for child,prev_out in zip(self.enfants,self.layer_params["outputs"]):
            parent_graph.edge(str(prev_out.id), str(child.id),
                              label=str(prev_out.tenseur.get_shape().as_list()))
            child.build(parent_graph)
        self.graph_done = True
