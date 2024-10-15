#!/usr/bin/python3

import pathlib
import torch
from torch import nn
from huggingface_hub import login
from transformers import AutoConfig, AutoModel, AutoModelForTokenClassification, AutoTokenizer
from tokenizers.normalizers import BertNormalizer
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

class Tokenizer_RC(nn.Module):
  def __init__(self, ):
    self.tokenizer = AutoTokenizer.from_pretrained('m3rg-iitd/matscibert')
    self.tokenizer.add_tokens(['[E1]', '[/E1]', '[E2]', '[/E2]'])
    self.norm = BertNormalizer(lowercase=False, strip_accents=True, clean_text=True, handle_chinese_chars=True)
    with open(os.path.join(pathlib.Path(__file__).parent.resolve(), 'vocab_mappings.txt'), 'r') as f:
      self.mappings = f.read().strip().split('\n')
  def normalize(self, text):
    text = [self.norm.normalize_str(s) for s in text.split('\n')]
    out = []
    for s in text:
        norm_s = ''
        for c in s:
            norm_s += self.mappings.get(c, ' ')
        out.append(norm_s)
    return '\n'.join(out)
  def tokenize(self, text):
    self.tokenizer.tokenize(self.normalize(text))
  def forward(self, text):
    # TODO

class BERT_RC(nn.Module):
  def __init__(self, ):
    super(BERT_RC, self).__init__()
    login('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')
    self.encoder = AutoModel.from_pretrained('m3rg-iitd/matscibert')
    self.encoder.resize_token_embeddings(len(self.tokenizer))
    self.dropout = nn.Dropout(0.1)
    self.linear = nn.Linear(2 * self.encoder.config.hidden_size, num_labels)
  def forward(self, **inputs):
    hidden_states = self.encoder(input_ids = inputs['input_ids'], attention_mask = inputs['attention_mask'])
    # TODO

class NER(nn.Module):
  def __init__(self, ):
    super(NER, self).__init__()
    self.model = BERT_CRF()
    ckpt = torch.load('models/ner/pytorch_model.bin', map_location = next(self.model.parameters()).device)
    del ckpt['encoder.bert.embeddings.position_ids']
    self.model.load_state_dict(ckpt)
    self.tokenizer = AutoTokenizer.from_pretrained('m3rg-iitd/matscibert')
    self.tags = ['B-APL', 'B-CMT', 'B-DSC', 'B-MAT', 'B-PRO', 'B-SMT', 'B-SPL', 'I-APL', 'I-CMT', 'I-DSC', 'I-MAT', 'I-PRO', 'I-SMT', 'I-SPL', 'O']
  def forward(self, text):
    assert type(text) is list
    encode_input = self.tokenizer(text, padding = True, return_tensors = 'pt', return_offsets_mapping = True).to(next(self.model.parameters()).device)
    offset_mapping = encode_input.pop('offset_mapping')
    output = self.model(**encode_input)
    results = list()
    for idx, sample in enumerate(output):
      input_ids = encode_input['input_ids'][idx]
      attention_mask = encode_input['attention_mask'][idx]
      input_ids = torch.masked_select(input_ids, attention_mask.to(torch.bool))
      offsets = offset_mapping[idx].detach().cpu().numpy()
      labels = [self.tags[t] for t in sample]
      assert len(input_ids) == len(labels)
      status = 'O'
      start = None
      end = None
      entities = list()
      for input_id, label, offset in zip(input_ids, labels, offsets):
        if status == 'O':
          if label == 'O':
            status = label
          elif label.startswith('B-'):
            status = label
            start = offset[0]
          else:
            print(f'parse error: {label} right after {status}!')
            status = 'O'
        elif status.startswith('B-'):
          if label.startswith('I-') and status[2:] == label[2:]:
            status = label
          elif label.startswith('B-'):
            end = offset[0]
            entities.append((status[2:], (start, end)))
            status = label
          elif label == 'O':
            end = offset[0]
            entities.append((status[2:], (start, end)))
            status = label
          else:
            print(f'parse error: {label} right after {status}!')
        elif status.startswith('I-'):
          if label.startswith('I-') and status[2:] == label[2:]:
            status = label
          elif label.startswith('B-'):
            end = offset[0]
            entities.append((status[2:], (start, end)))
            status = label
          elif label == 'O':
            end = offset[0]
            entities.append((status[2:], (start, end)))
            status = label
          else:
            print(f'parse error: {label} right after {status}!')
        else:
          raise Exception(f'label: {label}, status: {status}')
      results.append(entities)
    return results

if __name__ == "__main__":
  ner = NER().to(torch.device('cuda'))
  texts = ['Glasses are emerging as promising and efficient solid electrolytes for all-solid-state sodium-ion batteries.',
          'The current study shows a significant enhancement in crack resistance (from 11.3 N to 32.9 N) for Na3Al1.8Si1.65P1.8O12 glass (Ag-0 glass) upon Na+-Ag+ ion-exchange (IE) due to compressive stresses generated in the glass surface while the ionic conductivity values (∼10−5 S/cm at 473 K) were retained. ',
          'In this study, magic angle spinning-nuclear magnetic resonance (MAS-NMR), molecular dynamics (MD) simulations, Vickers micro hardness, and impedance spectroscopic techniques were used to evaluate the intermediate-range structure, atomic structure, crack resistance and conductivity of the glass.',
          'Selected beam geometry allows us to suppress the bulk contribution to sum-frequency generation from crystalline quartz and use sum-frequency vibrational spectroscopy to study water/α-quartz interfaces with different bulk pH values.',
          'XRD patterns of glass-ceramics sintered at different holding times; identifying rutile TiO2 crystal grains.']
  entities = ner(texts)
  for entity, text in zip(entities, texts):
    print([{'entity':text[e[1][0]:e[1][1]], 'type': e[0]} for e in entity])
      
