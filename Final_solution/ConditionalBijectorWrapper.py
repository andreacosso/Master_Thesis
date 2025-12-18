
import tensorflow as tf
import tensorflow_probability as tfp
tfb = tfp.bijectors
tfd = tfp.distributions
from tensorflow.keras.layers import Conv2D, BatchNormalization, Add, Dense, ReLU, Layer, Conv2D, Reshape, Concatenate, Identity
from RealNVP import *

class Cond_NN(Layer):
    """
    Neural Network Architecture for calcualting s and t for Real-NVP
  
    """
    def __init__(self, input_shape, output_shape = None, n_hidden=[128,128,128], activation="relu",use_bias=True,
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros', kernel_regularizer=None,
    bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
    bias_constraint=None, data_input_layer_dim = 128, cond_input_layer_dim = 128,
    input_structure = 'separated', ncond = 0, conditional_input_layers = 'all_layers'):
        super(Cond_NN, self).__init__()

        self.input_structure = input_structure
        if input_structure == 'separated': #! *****************************************************
            ###Building the layers
            #self.cond_input = tf.keras.Input((cond_input_layer_dim,))        
            #self.data_input = tf.keras.Input((data_input_layer_dim,))
            self.cond_input = Dense(cond_input_layer_dim, activation=activation)
            self.data_input = Dense(data_input_layer_dim, activation=activation)

            self.hidden_layers = tf.keras.Sequential([
                                  Dense(hidden, activation=activation) for hidden in n_hidden])
        
            if output_shape is not None:
                self.log_s_layer = Dense(output_shape, activation="tanh")
                self.t_layer = Dense(output_shape)
            else:
                self.log_s_layer = Dense(input_shape, activation="tanh")
                self.t_layer = Dense(input_shape)


        elif input_structure == 'single': #! *****************************************************
            ###Building the layers
            if ncond == 0:
                raise ValueError("ncond kwarg not given or equals 0 in single input structure")
            # input = input_dim = ncond + rem_dims
            #output_shape = bij.tran_ndims
            rem_dims = input_shape - ncond
            if ncond <= rem_dims:
                if ncond == rem_dims:
                    print("Conditionals are equal to data dimensions, transforming conditionals")

                self.cond_input = Dense(rem_dims, activation=activation)
                self.data_input = Identity()
            else:
                self.cond_input = Identity()
                self.data_input = Dense(ncond, activation=activation)
            
            self.hidden_layers = tf.keras.Sequential([
                                  Dense(hidden, activation=activation) for hidden in n_hidden])
        
            if output_shape is not None:
                self.log_s_layer = Dense(output_shape, activation="tanh")
                self.t_layer = Dense(output_shape)
            else:
                self.log_s_layer = Dense(input_shape, activation="tanh")
                self.t_layer = Dense(input_shape)


        elif input_structure == 'SuperCalo': #! *****************************************************
            if conditional_input_layers == 'first_layer':
                hidden_input = n_hidden[0] 
                self.cond_input = Dense(hidden_input, activation=activation)
                self.data_input = Dense(hidden_input, activation=activation)

                self.hidden_layers = tf.keras.Sequential([
                                      Dense(hidden, activation=activation) for hidden in n_hidden])

                if output_shape is not None:
                    self.log_s_layer = Dense(output_shape, activation="tanh")
                    self.t_layer = Dense(output_shape)
                else:
                    self.log_s_layer = Dense(input_shape, activation="tanh")
                    self.t_layer = Dense(input_shape)
            elif conditional_input_layers == 'all_layers':
                hidden_input = n_hidden[0] 
                self.data_input = Dense(hidden_input, activation=activation)
                self.cond_input = Dense(hidden_input, activation=activation)

                self.hidden_layers = tf.keras.Sequential([
                                      Dense(hidden, activation=activation, name=f"data_hidden_{i}") for i, hidden in enumerate(n_hidden)])
                self.cond_hidden_layers = tf.keras.Sequential([
                                      Dense(hidden, activation=activation, name=f"cond_hidden_{i}") for i, hidden in  enumerate(n_hidden)])

                self.merge_add = tf.keras.layers.Add()
                if output_shape is not None:
                    self.log_s_layer = Dense(output_shape, activation="tanh")
                    self.t_layer = Dense(output_shape)
                else:
                    self.log_s_layer = Dense(input_shape, activation="tanh")
                    self.t_layer = Dense(input_shape)
            else:
                raise ValueError("conditional_input_layers must be 'first_layer' or 'all_layers'")
        else:
            raise ValueError("input_structure must be 'separated', 'single' or 'SuperCalo'")
        

    def call(self, x, conditionals = None): #implicitly called when the model is called
        #passo l'input attraverso il primo layer
        x_enc = self.data_input(x) 
        if self.input_structure == 'SuperCalo':
            cond_enc = self.cond_input(conditionals)
            merged_input = self.merge_add([x_enc, cond_enc])
            if self.cond_hidden_layers == 'all_layers':
                for data_layer, cond_layer in zip(self.hidden_layers,self.cond_hidden_layers):
                    h_data = data_layer(merged_input)
                    h_cond = cond_layer(conditionals)
                    merged_input = self.merge_add([h_data, h_cond])
        elif (self.input_structure == 'separated') or (self.input_structure == 'single'):
            cond_enc = self.cond_input(conditionals) 
            merged_input = Concatenate(axis=-1)([x_enc, cond_enc])
        else:
            merged_input = x_enc
        #concateno
        y = self.hidden_layers(merged_input)
        log_s = self.log_s_layer(y)
        t = self.t_layer(y)
        return t, log_s

class Cond_RealNVP(tfb.Bijector):
    def __init__(self, ndims, rem_dims, n_hidden=[24,24],activation='relu', forward_min_event_ndims=1,use_bias=True,
                    kernel_initializer='glorot_uniform',
                    bias_initializer='zeros', kernel_regularizer=None,
                    bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
                    bias_constraint=None, validate_args: bool = False, input_structure = None, 
                    data_input_layer_dim = 128, cond_input_layer_dim = 128,
                    conditional_input_layers = 'all_layers',  conditional_event_shape=None):


        bij = RealNVP(ndims, rem_dims,n_hidden,activation, forward_min_event_ndims,use_bias,
                            kernel_initializer,
                            bias_initializer, kernel_regularizer,
                            bias_regularizer, activity_regularizer, kernel_constraint,
                            bias_constraint, validate_args)
        
        super().__init__(
            forward_min_event_ndims=bij.forward_min_event_ndims,
            is_constant_jacobian=bij.is_constant_jacobian,
            validate_args=bij.validate_args,
            name="conditional_" + bij.name,
        )

        ncond = conditional_event_shape[-1] if conditional_event_shape is not None else 0
        self.bij = bij
        self.use_conditionals = True
        self.input_structure = input_structure

        input_dim = ncond + rem_dims
        output_shape = bij.tran_ndims


        if input_structure is None:
            nn_layer = NN(input_dim, output_shape, n_hidden,activation,use_bias,
                            kernel_initializer,
                            bias_initializer, kernel_regularizer,
                            bias_regularizer, activity_regularizer, kernel_constraint,
                            bias_constraint)

            x = tf.keras.Input((input_dim,))
            t,log_s = nn_layer(x)
            self.bij.nn = Model(x, [t, log_s])


        elif self.input_structure == 'separated':
            print("Using separated input structure")
            self.data_input_layer_dim = data_input_layer_dim
            self.cond_input_layer_dim = cond_input_layer_dim
            nn_layer = Cond_NN( input_dim, output_shape, n_hidden,activation,use_bias,
                            kernel_initializer,
                            bias_initializer, kernel_regularizer,
                            bias_regularizer, activity_regularizer, kernel_constraint,
                            bias_constraint, data_input_layer_dim = data_input_layer_dim, 
                            cond_input_layer_dim = cond_input_layer_dim)

            # One input tensor to conform to RealNVP's expectations
            combined_input = tf.keras.Input((ncond + rem_dims,))

            # Split it internally
            data_input_tensor = combined_input[:, :rem_dims]     # x_b (data for transformation)
            cond_input_tensor = combined_input[:, rem_dims:]     # conditional input
            t, log_s = nn_layer(data_input_tensor, cond_input_tensor)
            self.bij.nn = tf.keras.Model(inputs=combined_input, outputs=[t, log_s])
        

        elif input_structure == 'single':
            #trasforma solo il piu piccolo, l'altro rimane invariato
            print("Using single input structure")
            nn_layer = Cond_NN( input_dim, output_shape, n_hidden,activation,use_bias,
                            kernel_initializer,
                            bias_initializer, kernel_regularizer,
                            bias_regularizer, activity_regularizer, kernel_constraint,
                            bias_constraint, data_input_layer_dim = data_input_layer_dim, 
                            cond_input_layer_dim = cond_input_layer_dim, input_structure = 'single', ncond = ncond)
            
            combined_input = tf.keras.Input((ncond + rem_dims,))

            # Split it internally
            data_input_tensor = combined_input[:, :rem_dims]     # x_b (data for transformation)
            cond_input_tensor = combined_input[:, rem_dims:]     # conditional input

            t, log_s = nn_layer(data_input_tensor, cond_input_tensor)
            self.bij.nn = tf.keras.Model(inputs=combined_input, outputs=[t, log_s])

        elif self.input_structure == 'SuperCalo':
            print("Using SuperCalo input structure")
            nn_layer = Cond_NN( input_dim, output_shape, n_hidden,activation,use_bias,
                            kernel_initializer,
                            bias_initializer, kernel_regularizer,
                            bias_regularizer, activity_regularizer, kernel_constraint,
                            bias_constraint, input_structure= 'SuperCalo', conditional_input_layers = conditional_input_layers)

            # One input tensor to conform to RealNVP's expectations
            combined_input = tf.keras.Input((ncond + rem_dims,))

            # Split it internally
            data_input_tensor = combined_input[:, :rem_dims]     # x_b (data for transformation)
            cond_input_tensor = combined_input[:, rem_dims:]     # conditional input

            t, log_s = nn_layer(data_input_tensor, cond_input_tensor)
            self.bij.nn = tf.keras.Model(inputs=combined_input, outputs=[t, log_s])

        else:
            raise ValueError("input_structure must be None, 'separated', 'single' or 'SuperCalo'")



    def _forward(self, x):
        #x_cond = tf.concat([x, conditionals], axis=-1)
        #print("passing x through bijector in the forward direction")
        return self.bij.forward(x)
    
    def _inverse(self, y):
        #y_cond = tf.concat([y, conditionals], axis=-1)
        #print("passing y through bijector in the inverse direction")
        #tf.print("y in the Cond_RealNVP bijector:", y)
        return self.bij.inverse(y)

    def _forward_log_det_jacobian(self, x):
        #x_cond = tf.concat([x, conditionals], axis=-1)
        return self.bij.forward_log_det_jacobian(x, event_ndims=1)

    def _inverse_log_det_jacobian(self, y):
        #y_cond = tf.concat([y, conditionals], axis=-1)
        return self.bij.inverse_log_det_jacobian(y, event_ndims=1)
