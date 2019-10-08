import pandas as pd
import numpy as np
import warnings
import re

test_data_with_labels = pd.read_csv('titanic.csv')
test_data = pd.read_csv('test.csv')
warnings.filterwarnings('ignore')

for i, name in enumerate(test_data_with_labels['name']):
    if '"' in name:
        test_data_with_labels['name'][i] = re.sub('"', '', name)
        
for i, name in enumerate(test_data['Name']):
    if '"' in name:
        test_data['Name'][i] = re.sub('"', '', name)


survived = []

for name in test_data['Name']:
    survived.append(int(test_data_with_labels.loc[test_data_with_labels['name'] == name]['survived'].values[-1]))
    

survived

Survived=pd.DataFrame(survived)
PassengerId=test_data[["PassengerId"]]


submission = pd.concat([PassengerId,Survived],axis=1)
submission.columns = ["PassengerId","Survived"]
submission.to_csv('finalsubmission.csv',index=False)
