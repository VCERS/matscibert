#!/usr/bin/python3

from models import NER, RE

def main():
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

if __name__ == "__main__":
  main()
