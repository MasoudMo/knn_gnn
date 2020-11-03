from sources.data import PtbEcgDataSet
from torch.utils.data import DataLoader


dataset = PtbEcgDataSet(root_dir='H:\Workspace\ELEC 421 Project\data\ptb-diagnostic-ecg-database-1.0.0\ptb-diagnostic-ecg-database-1.0.0',
                        records='RECORDS',
                        weighted=False)

dataloader = DataLoader(dataset, shuffle=True)

for label in dataloader:
    print('hello')
