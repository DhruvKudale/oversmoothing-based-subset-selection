### Usage example
# from multi_llm import multi_llm_wrapper 
# multi_llm = multi_llm_wrapper(model_names=["microsoft/mpnet-base", 'bert-base-uncased'])
# multi_llm.get_embeddings("hi")



from transformers import AutoTokenizer, AutoModel


class multi_llm_wrapper:

    tokenizers = {}
    models = {}

    def __init__(self, model_names):
        """
        Parameters
        ----------
        model_names: List of automodel names
        ----------
        Returns None 
        """
        for i in range(len(model_names)):
            self.tokenizers[f'{i}'] = AutoTokenizer.from_pretrained(model_names[i], return_tensors="pt")
            self.models[f'{i}']= AutoModel.from_pretrained(model_names[i])
        
        
    def get_embeddings(self, text):
        """
        Parameters
        ----------
        text: string
        ----------
        Returns: List of embedding of the string from each model
        """
        
        embeddings = []
        for i in range(len(self.models)):
            tokenized_input = self.tokenizers[f'{i}'](text, return_tensors="pt")
            output =  self.models[f'{i}'](**tokenized_input)
            embeddings.append(output.pooler_output)

        return embeddings
        
