# Generalization of the NFT2 repository to implement Conditional Normalising Flows 

## RealNVP
The logic is to append conditional inputs on the tail of the input vector, the full vector is then divided as the standard RealNVP provides. The first dimensions are transformed according to the remaining ones which contain the conditional inputs. `ConditionalBijectorWrapper` file replace the internal NN of the RealNVP to handle the additional conditional dimensions. 

### Conditional input
Different input structures for the internal NN are implemented. Data and conditionals are separated and different pipeline are applied: 
- separated: data and conditional are passed through separate dense layers and then concatenated.
- single: the smallest vector is passed through a dense layer with output dimension equal to the other vector. 
- Supercalo: analogus to the TensorFlow conditiona implementation, with first_layer or all_layers, the inputs are taken to the dimension of the first hidden layer and then summed. 


## MAF
Standard TensorFlow 2 functions already able to handle conditional inputs, just modified the building call inside `Bijector` file (`ChooseCondBijector`). 

## A-RQS 
Added Batch-Norm and implemented linear tails for the RQS building on standard TensorFlow 2 code. Build using custom code, with small adjustments built to the standard TensorFlow code.

## Passing conditional through the chain: 
Changed `log_prob_wrapper` function inside `Trainer.py` to handle chains of conditional bijectors. Different path for MAF, A-RQS, RealNVP.