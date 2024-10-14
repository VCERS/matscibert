#!/usr/bin/python3

import torch
from torch import nn
from huggingface_hub import login
from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer
from torchcrf import CRF

class BERT_CRF(nn.Module):
    def __init__(self, num_labels = 15):
        super(BERT_CRF, self).__init__()
        login('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')
        config = AutoConfig.from_pretrained('m3rg-iitd/matscibert')
        config.num_labels = num_labels
        self.encoder = AutoModelForTokenClassification.from_config(config)
        self.crf = CRF(num_labels, batch_first=True)
    def forward(self, **inputs):
        results = self.encoder(**inputs,return_dict = True)
        labels = self.crf.decode(results.logits,inputs['attention_mask'].to(torch.bool))
        return labels

class NER(nn.Module):
  def __init__(self, ):
    super(NER, self).__init__()
    self.model = BERT_CRF()
    ckpt = torch.load('pytorch_model.bin', map_location = next(self.model.parameters()).device)
    del ckpt['encoder.bert.embeddings.position_ids']
    self.model.load_state_dict(ckpt)
    self.tokenizer = AutoTokenizer.from_pretrained('m3rg-iitd/matscibert')
    self.tags = ['B-APL', 'B-CMT', 'B-DSC', 'B-MAT', 'B-PRO', 'B-SMT', 'B-SPL', 'I-APL', 'I-CMT', 'I-DSC', 'I-MAT', 'I-PRO', 'I-SMT', 'I-SPL', 'O']
  def forward(self, text):
    encode_input = self.tokenizer(text, padding = True, return_tensors = 'pt').to(next(self.model.parameters()).device)
    output = self.model(**encode_input)
    results = list()
    for sample in output:
      results.append([self.tags[t] for t in sample])
    return results

if __name__ == "__main__":
  ner = NER().to(torch.device('cuda'))
  output = ner(['Glasses are emerging as promising and efficient solid electrolytes for all-solid-state sodium-ion batteries.',
                'The current study shows a significant enhancement in crack resistance (from 11.3 N to 32.9 N) for Na3Al1.8Si1.65P1.8O12 glass (Ag-0 glass) upon Na+-Ag+ ion-exchange (IE) due to compressive stresses generated in the glass surface while the ionic conductivity values (∼10−5 S/cm at 473 K) were retained. ',
                'In this study, magic angle spinning-nuclear magnetic resonance (MAS-NMR), molecular dynamics (MD) simulations, Vickers micro hardness, and impedance spectroscopic techniques were used to evaluate the intermediate-range structure, atomic structure, crack resistance and conductivity of the glass.',
                'Selected beam geometry allows us to suppress the bulk contribution to sum-frequency generation from crystalline quartz and use sum-frequency vibrational spectroscopy to study water/α-quartz interfaces with different bulk pH values.',
                'XRD patterns of glass-ceramics sintered at different holding times; identifying rutile TiO2 crystal grains.']
             )
  print(output)
