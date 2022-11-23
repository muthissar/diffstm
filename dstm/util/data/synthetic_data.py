import torch
from util.load_data import DataPreprocessing
class SyntheticData(DataPreprocessing):
    def __init__(self, nb_classes):
        #self.nb_classes = nb_classes
        super().__init__(nb_classes)
    def variations(self, length, n_reps, n_vars_pr_motif, n_samples):
        samples = []
        for _ in range(n_samples):
            motif = torch.multinomial(1/self.nb_classes *torch.ones(self.nb_classes), length, replacement=True)
            sample = [motif]
            #Keep the last (sample indices)
            for _ in range(n_reps):
                #For now variations are not unique but we can add another while loop
                var_indices = torch.multinomial(1/(length-1)*torch.ones(length-1), n_vars_pr_motif, replacement=False)
                var = motif.clone()
                
                for index in var_indices:
                    while True:
                        var_symb = torch.multinomial(1/self.nb_classes *torch.ones(self.nb_classes), 1).squeeze(0)
                        if var_symb != var[index]:
                            var[index] = var_symb
                            break
                sample.append(var)
            sample =  torch.cat(sample)
            samples.append(sample)
        return torch.stack(samples)
                

            
        
        
