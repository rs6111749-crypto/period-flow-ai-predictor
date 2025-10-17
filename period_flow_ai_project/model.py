import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import pickle

# Simple sample data (you should replace with larger real dataset for better accuracy)
data = {
    'cycle_day': [1,2,3,4,5,6,7,8,9,10],
    'cramps':    [1,2,3,2,1,0,0,1,2,3],
    'mood':      [3,2,1,2,3,3,2,1,2,1],
    'bloating':  [1,1,1,0,0,0,0,1,1,1],
    'prev_flow': [2,2,3,3,1,1,1,2,2,3],
    'flow':      ['Medium','Medium','Heavy','Heavy','Light','Light','Light','Medium','Medium','Heavy']
}

df = pd.DataFrame(data)
X = df[['cycle_day','cramps','mood','bloating','prev_flow']]
y = df['flow']

model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

with open('flow_model.pkl','wb') as f:
    pickle.dump(model, f)

print('Model trained and saved to flow_model.pkl')
