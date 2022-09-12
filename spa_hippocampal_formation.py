import nengo
import nengo.spa as spa

import importlib
import hippocampal_formation
importlib.reload(hippocampal_formation)
#from nengo.spa.vocab import Vocabulary

class SPA_Hippocampal_Formation(spa.module.Module):
    #input_vocab = Vocabulary(
    #    'input_vocab', default=None, readonly=True)
    #output_vocab = Vocabulary(
    #    'output_vocab', default=None, readonly=True)    
    
    def __init__(self, 
                    dimensions=None,
                    seed=None,
                    n_neurons=500,
                    ECDGdrop=0,
                    ECDGdropmethod="random",
                    ECDGtrans=1,
                    MFdrop=0,
                    MFdropmethod="lowest", # which weights to drop: lowest, highest, random
                    MFtrans=1,
                    VERBOSE=False,
                    label = "HF",
                    add_to_container=None, 
                    **keys):
        super(SPA_Hippocampal_Formation, self).__init__(label=label, seed=seed,
                            add_to_container=add_to_container)
    
        with self:
            
            self.HF = hippocampal_formation.Hippocampal_Formation(
                        dimensions=dimensions,
                        seed=seed,
                        n_neurons=n_neurons,
                        ECDGdrop=ECDGdrop,
                        ECDGdropmethod=ECDGdropmethod,
                        ECDGtrans=ECDGtrans,
                        MFdrop=MFdrop,
                        MFdropmethod=MFdropmethod, # which weights to drop: lowest, highest, random
                        MFtrans=MFtrans,
                        VERBOSE=VERBOSE   
                    )

        self.input = self.HF.input


