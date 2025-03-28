#!/usr/bin/python3

from os.path import join
import pathlib
import torch
from torch import nn
import numpy as np
from transformers import AutoConfig, AutoModel, AutoModelForTokenClassification, AutoTokenizer
from tokenizers.normalizers import BertNormalizer
from torchcrf import CRF

class BERT_CRF(nn.Module):
  def __init__(self, num_labels = 15):
    super(BERT_CRF, self).__init__()
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
    super(Tokenizer_RC, self).__init__()
    self.tokenizer = AutoTokenizer.from_pretrained('m3rg-iitd/matscibert')
    self.tokenizer.add_tokens(['[E1]', '[/E1]', '[E2]', '[/E2]'])
    self.norm = BertNormalizer(lowercase=False, strip_accents=True, clean_text=True, handle_chinese_chars=True)
    with open(join(pathlib.Path(__file__).parent.resolve(), 'vocab_mappings.txt'), 'r') as f:
      self.mappings = f.read().strip().split('\n')
    self.mappings = {m[0]: m[2:] for m in self.mappings}
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
    assert type(text) is str
    return self.tokenizer.tokenize(self.normalize(text))
  def forward(self, text, entity1, entity2):
    assert type(text) is str
    assert type(entity1) is tuple
    assert type(entity2) is tuple
    s1, e1 = entity1
    s2, e2 = entity2
    s1, e1, s2, e2 = (s1, e1, s2, e2) if s1 < s2 else (s2, e2, s1, e1)
    tokens = (self.tokenize(text[:s1]) if text[:s1] != '' else []) + ['[E1]'] + self.tokenize(text[s1:e1]) + ['[/E1]'] + \
             (self.tokenize(text[e1:s2]) if text[e1:s2] != '' else []) + ['[E2]'] + self.tokenize(text[s2:e2]) + ['[/E2]'] + \
             (self.tokenize(text[e2:]) if text[e2:] != '' else [])
    s1 = tokens.index('[E1]')
    e1 = tokens.index('[/E1]')
    s2 = tokens.index('[E2]')
    e2 = tokens.index('[/E2]')
    if len(tokens) <= 512 - 2:
      tokens = ['[CLS]'] + tokens + ['[SEP]']
    else:
      rem = (512 - 2) - (e2 - s1 + 1)
      assert rem >= 0
      s = max(0, s1 - rem // 2)
      e = min(len(tokens) - 1, e2 + rem // 2)
      tokens = ['[CLS]'] + tokens[s:e + 1] + ['[SEP]']
    entity_marker = [tokens.index('[E1]'), tokens.index('[E2]')]
    tokens = self.tokenizer.convert_tokens_to_ids(tokens)
    assert len(tokens) <= 512
    input_ids = tokens
    attention_mask = [1] * len(tokens)
    return {'entity_marker': entity_marker,
            'input_ids': input_ids,
            'attention_mask': attention_mask}

class BERT_RC(nn.Module):
  def __init__(self, tokenizer):
    super(BERT_RC, self).__init__()
    self.encoder = AutoModel.from_pretrained('m3rg-iitd/matscibert')
    self.encoder.resize_token_embeddings(len(tokenizer.tokenizer))
    self.dropout = nn.Dropout(0.1)
    self.linear = nn.Linear(2 * self.encoder.config.hidden_size, 16)
  def forward(self, **inputs):
    input_ids = torch.tensor(inputs['input_ids'], dtype = torch.int64).unsqueeze(dim = 0).to(next(self.parameters()).device)
    attention_mask = torch.tensor(inputs['attention_mask'], dtype = torch.int64).unsqueeze(dim = 0).to(next(self.parameters()).device)
    entity_marker = torch.tensor(inputs['entity_marker'], dtype = torch.int64).to(next(self.parameters()).device)
    hidden_states = self.encoder(input_ids = input_ids, attention_mask = attention_mask)
    outs = torch.cat([hidden_states.last_hidden_state[:, entity_marker[0]],
                      hidden_states.last_hidden_state[:, entity_marker[1]]], dim = 1) # outs.shape = (1, 768*2)
    logits = self.linear(self.dropout(outs))
    return logits

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
            end = offset[1]
          else:
            print(f'parse error: {label} right after {status}!')
            status = 'O'
        elif status.startswith('B-'):
          if label.startswith('I-') and status[2:] == label[2:]:
            status = label
            end = offset[1]
          elif label.startswith('B-'):
            entities.append((status[2:], (start, end)))
            start = offset[0]
            end = offset[1]
            status = label
          elif label == 'O':
            entities.append((status[2:], (start, end)))
            status = label
          else:
            print(f'parse error: {label} right after {status}!')
        elif status.startswith('I-'):
          if label.startswith('I-') and status[2:] == label[2:]:
            end = offset[1]
            status = label
          elif label.startswith('B-'):
            entities.append((status[2:], (start, end)))
            start = offset[0]
            end = offset[1]
            status = label
          elif label == 'O':
            entities.append((status[2:], (start, end)))
            status = label
          else:
            print(f'parse error: {label} right after {status}!')
        else:
          raise Exception(f'label: {label}, status: {status}')
      results.append(entities)
    return results

class RE(nn.Module):
  def __init__(self, ):
    super(RE, self).__init__()
    self.tokenizer = Tokenizer_RC()
    self.model = BERT_RC(self.tokenizer)
    ckpt = torch.load('models/re/pytorch_model.bin', map_location = next(self.model.parameters()).device)
    del ckpt['encoder.embeddings.position_ids']
    self.model.load_state_dict(ckpt)
    self.tags = ['Amount_Of', 'Apparatus_Attr_Of', 'Apparatus_Of', 'Atmospheric_Material', 'Brand_Of', 'Condition_Of', 'Coref_Of', 'Descriptor_Of', 'Next_Operation', 'Number_Of', 'Participant_Material', 'Property_Of', 'Recipe_Precursor', 'Recipe_Target', 'Solvent_Material', 'Type_Of']
  def forward(self, text, entity1, entity2):
    assert type(text) is str
    assert type(entity1) is tuple
    assert type(entity2) is tuple
    inputs = self.tokenizer(text, entity1, entity2)
    logits = self.model(**inputs).detach().cpu().numpy()
    pred = np.argmax(logits, axis = 1)
    return self.tags[pred[0]]

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
  re = RE().to(torch.device('cuda'))
  for i in range(len(texts)):
    for id1, e1 in enumerate(entities[i]):
      for id2, e2 in enumerate(entities[i]):
        if id1 == id2: continue
        pred = re(texts[i], e1[1], e2[1])
        print(f'({texts[i][e1[1][0]:e1[1][1]]},{pred},{texts[i][e2[1][0]:e2[1][1]]})')
